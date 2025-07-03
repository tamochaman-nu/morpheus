import pathlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from diffusers import ControlNetModel
from loguru import logger
from PIL import Image

from morpheus.cross_attention import (
    AutoregressiveCrossAttentionProcessor,
    compute_mesh_warped_attention_map,
)
from morpheus.ddim_partial_inverter import DDIMPartialInverter
from morpheus.diffuser_pipelines.pipeline_rgbd_controlnet_inpaint import (
    StableDiffusionRGBDControlNetInpaintPipeline,
)
from morpheus.utils.data_utils import Frame
from morpheus.utils.depth_scaling import compute_scale_and_shift
from morpheus.utils.geometry_utils import ScaleShiftDepthNormalizer
from morpheus.visualization.image_viz import colormap_image


@dataclass
class InpaintInputs:
    """Stores inputs to the controlnet"""

    input_frame: Frame
    mask: torch.Tensor
    original_frame: Frame
    xattn_ref_frames: Optional[List[Frame]] = None


class XAttnWarpRGBDControlNet:
    def __init__(
        self,
        text_prompt: str,
        negative_text_prompt: str,
        noise_strength: float,
        guidance_scale: float,
        controlnet_strength: float,
        controlnet_model_path: pathlib.Path,
        stable_diffusion_model_path: pathlib.Path,
        huggingface_cache_dir: Optional[pathlib.Path] = None,
        inversion_prompt: str = None,
        debug_path: Optional[pathlib.Path] = None,
        depth_noise_strength: Optional[float] = None,
        partial_inversion_noise_level: Optional[float] = 0.05,
        partial_inversion_depth_noise_level: Optional[float] = 0.05,
        guidance_rescale: Optional[float] = None,
        n_controlnets: Optional[int] = 1,
        feature_injection_strength: float = 0.05,
        num_inference_steps: int = 60,
        do_xattn: bool = True,
        xattn_type="none",
        extra_attn_processor_args: Optional[Dict] = None,
        extra_attn_maps_args: Optional[Dict] = None,
    ) -> None:
        self.text_prompt = text_prompt
        self.negative_text_prompt = negative_text_prompt
        self.noise_strength = noise_strength
        self.depth_noise_strength = (
            depth_noise_strength if depth_noise_strength is not None else noise_strength
        )
        self.guidance_scale = guidance_scale
        self.controlnet_strength = controlnet_strength

        self.num_inference_steps = num_inference_steps
        self.inversion_prompt = inversion_prompt
        self.partial_inversion_noise_level = partial_inversion_noise_level
        self.partial_inversion_depth_noise_level = partial_inversion_depth_noise_level

        self.device = torch.device("cuda")
        self.generator = torch.Generator(device=self.device)

        self.n_controlnets = n_controlnets
        logger.debug(f"Using {self.n_controlnets} controlnet passes")
        controlnet = ControlNetModel.from_pretrained(
            controlnet_model_path, torch_dtype=torch.float32
        )
        # This allows us to condition on multiple reference frames:
        controlnet = [controlnet for _ in range(self.n_controlnets)]

        self.pipe = StableDiffusionRGBDControlNetInpaintPipeline.from_pretrained(
            stable_diffusion_model_path,
            controlnet=controlnet,
            safety_checker=None,
            requires_safety_checker=False,
            cache_dir=huggingface_cache_dir,
            torch_dtype=torch.float32,
        ).to("cuda")

        self.guidance_rescale = guidance_rescale

        logger.debug(
            f"Using DDIMPartialInverter for inversion with {self.num_inference_steps} steps"
        )
        self.ddim_inverter = DDIMPartialInverter(
            num_inference_steps=self.num_inference_steps,
            rgbd_pipeline=self.pipe,
            prompt=self.inversion_prompt if self.inversion_prompt is not None else "",
        )

        self.debug_path = debug_path

        self.generator = torch.Generator(device=self.pipe.device)
        self.xattn_type = xattn_type
        self.do_xattn = do_xattn and self.xattn_type != "none"
        self.depth_normalizer = ScaleShiftDepthNormalizer()
        self.extra_attn_maps_args = extra_attn_maps_args

        if self.xattn_type != "none":
            logger.debug("Setting up cross-attention processor...")
            self.xattn_processor = AutoregressiveCrossAttentionProcessor(
                ref_mixing_amount=feature_injection_strength,
                **(extra_attn_processor_args or {}),
            )
            self.pipe.set_attention_processor(self.xattn_processor)
        else:
            self.xattn_processor = None

    def inpaint(
        self,
        inputs: Union[List[InpaintInputs]],
        target_frame: Frame,
        idx: int,
        seed: int = 0,
        xattn_ref_frames: Optional[List[Frame]] = None,
        noise_strength: Optional[float] = None,
        depth_noise_strength: Optional[float] = None,
        guidance_scale: Optional[float] = None,
        controlnet_strength: Optional[float] = None,
        partial_inversion_noise_level: Optional[float] = None,
        partial_inversion_depth_noise_level: Optional[float] = None,
    ):
        # Seed the generator for reproducibility
        # (We suggest using a different but deterministic seed for each frame, so that the results
        # are not identical)
        self.generator.manual_seed(seed)

        target_image_bchw = target_frame.image_bchw
        target_depth_b1hw = target_frame.depth_b1hw
        assert target_image_bchw.shape[0] == 1
        assert target_image_bchw.shape[1] == 3
        assert (
            target_image_bchw.min() >= 0.0 and target_image_bchw.max() <= 1.0
        ), "image_bchw is not normalized"

        # Inputs can be a list of InpaintInputs if we have multiple reference frames:
        if len(inputs) > 1:
            assert (
                len(inputs) == self.n_controlnets
            ), f"Expected {self.n_controlnets} inputs, got {len(inputs)}"
            logger.debug(f"Using {len(inputs)} control images")
        else:
            # If we have a single input, duplicate it for each controlnet
            # We will pass it through each and then average the results, which is the same as
            # just passing it through once (though it is a little inefficient)
            inputs = inputs * self.n_controlnets
            logger.debug(f"Using {len(inputs)} copies of the same input")

        control_image_bhw4_list = []
        for inp in inputs:
            assert isinstance(inp, InpaintInputs)

            image_bchw = inp.input_frame.image_bchw
            depth_b1hw = inp.input_frame.depth_b1hw
            mask_b1hw = 1.0 - inp.mask

            assert image_bchw.shape[0] == 1
            assert image_bchw.shape[1] == 3
            assert (
                image_bchw.min() >= 0.0 and image_bchw.max() <= 1.0
            ), "image_bchw is not normalized"

            disp_b1hw = 1 / depth_b1hw
            norm_disp_b1hw = (disp_b1hw - disp_b1hw.min()) / (disp_b1hw.max() - disp_b1hw.min())
            mask_b1hw = 1 - mask_b1hw
            control_image_4hw = torch.cat(
                (
                    image_bchw.cpu().squeeze(0),
                    norm_disp_b1hw.squeeze(0),
                    mask_b1hw.squeeze(0),
                ),
                dim=0,
            )
            control_image_hw4 = control_image_4hw.permute(1, 2, 0).numpy()
            control_image_bhw4_list.append(control_image_hw4[None, ...])

        target_disp_b1hw = 1 / target_depth_b1hw
        norm_target_disp_b1hw = (target_disp_b1hw - target_disp_b1hw.min()) / (
            target_disp_b1hw.max() - target_disp_b1hw.min()
        )
        input_image = (
            torch.cat(
                (
                    # 3 channels of RGB
                    target_image_bchw[0].cpu(),
                    # 3 channels of disparity (single channel repeated 3x)
                    norm_target_disp_b1hw[0],
                    norm_target_disp_b1hw[0],
                    norm_target_disp_b1hw[0],
                ),
                dim=0,
            )
            .permute(1, 2, 0)
            .numpy()
        )

        if self.debug_path is not None:
            input_image_rgb_pil = Image.fromarray(((input_image[:, :, :3]) * 255).astype(np.uint8))
            input_image_rgb_pil.save(self.debug_path / f"{idx:03d}_input_image_rgb.png")
            input_image_d_pil = Image.fromarray(((input_image[:, :, 3]) * 255).astype(np.uint8))
            input_image_d_pil.save(self.debug_path / f"{idx:03d}_input_image_norm_disp.png")

            for i, control_image_bhw4 in enumerate(control_image_bhw4_list):
                control_image_rgb_pil = Image.fromarray(
                    (control_image_bhw4[0, :, :, :3] * 255).astype(np.uint8)
                )
                control_image_rgb_pil.save(self.debug_path / f"{idx:03d}_control_image_rgb_{i}.png")
                control_image_d_pil = Image.fromarray(
                    (control_image_bhw4[0, :, :, 3] * 255).astype(np.uint8)
                )
                control_image_d_pil.save(
                    self.debug_path / f"{idx:03d}_control_image_norm_disp_{i}.png"
                )
                control_image_mask_pil = Image.fromarray(
                    (control_image_bhw4[0, :, :, 4] * 255).astype(np.uint8)
                )
                control_image_mask_pil.save(
                    self.debug_path / f"{idx:03d}_control_image_mask_{i}.png"
                )

        controlnet_strength = (
            controlnet_strength if controlnet_strength is not None else self.controlnet_strength
        )
        guidance_scale = guidance_scale if guidance_scale is not None else self.guidance_scale
        noise_strength = noise_strength if noise_strength is not None else self.noise_strength
        depth_noise_strength = (
            depth_noise_strength if depth_noise_strength is not None else self.depth_noise_strength
        )
        partial_inversion_noise_level = (
            partial_inversion_noise_level
            if partial_inversion_noise_level is not None
            else self.partial_inversion_noise_level
        )
        partial_inversion_depth_noise_level = (
            partial_inversion_depth_noise_level
            if partial_inversion_depth_noise_level is not None
            else self.partial_inversion_depth_noise_level
        )
        logger.debug(
            f"Controlnet strength: {controlnet_strength}, Guidance scale: {guidance_scale}, "
            f"Noise strength: {noise_strength}, Depth noise strength: {depth_noise_strength}, "
            f"Partial inversion noise level: {partial_inversion_noise_level}, "
            f"Partial inversion depth noise level: {partial_inversion_depth_noise_level}"
        )

        # Do partial DDIM inversion to get input latents ready for denoising
        inverted_latents = self.ddim_inverter.do_partial_inversion(
            input_image=input_image,
            control_image_list=control_image_bhw4_list,
            generator=self.generator,
            noise_strength=noise_strength,
            depth_noise_strength=depth_noise_strength,
            partial_inversion_noise_level=partial_inversion_noise_level,
            partial_inversion_depth_noise_level=partial_inversion_depth_noise_level,
        ).to(self.pipe.device)

        if self.do_xattn:
            do_xattn = self.xattn_processor.ref_latents is not None
        else:
            do_xattn = False

        # Define a callback to update the latents in the xattn processor
        # This will be called by the pipeline at each timestep, and allows us to set the latents
        # to the corresponding latents from that same denoising step. This way we are doing xattn
        # with reference latents that have the right amount of noise.
        def update_latents(timestep):
            # Retrieve latents from the reference frames
            ref_latents = torch.cat(
                [
                    frame.debug_dict["intermediate_latents"][int(timestep)]
                    for frame in xattn_ref_frames
                ],
                dim=0,
            )
            # set the latents in the xattn processor
            self.pipe.cross_attn_processor.set_reference_latents(ref_latents)

        output, latents = self.pipe(
            prompt=self.text_prompt,
            negative_prompt=self.negative_text_prompt,
            num_inference_steps=self.num_inference_steps,
            generator=self.generator,
            image=inverted_latents,
            latents=inverted_latents,
            control_image=(
                control_image_hw4[None, ...]
                if not control_image_bhw4_list
                else control_image_bhw4_list
            ),
            controlnet_conditioning_scale=controlnet_strength,
            mask_image=torch.ones(
                (1, 1, input_image.shape[0], input_image.shape[1]), device=self.pipe.device
            ),
            guidance_scale=guidance_scale,
            strength=noise_strength,
            depth_strength=depth_noise_strength,
            output_type="pt",
            do_xattn=do_xattn,
            return_all_latents=True,
            pre_step_callback=lambda timestep: (
                update_latents(timestep)
                if xattn_ref_frames is not None and self.pipe.cross_attn_processor is not None
                else None
            ),
        )
        output = output.images

        output_rgb_3hw = output[0][:3, :, :].cpu().float()
        output_disp_hw = output[0][3, :, :].cpu().float()

        if self.debug_path is not None:
            output_rgb_pil = Image.fromarray(
                (output_rgb_3hw * 255).permute(1, 2, 0).numpy().astype(np.uint8)
            )
            output_rgb_pil.save(self.debug_path / f"{idx:03d}_output_rgb.png")
            output_norm_disp_hw = (output_disp_hw - output_disp_hw.min()) / (
                output_disp_hw.max() - output_disp_hw.min()
            )
            output_d_pil = Image.fromarray((output_norm_disp_hw * 255).numpy().astype(np.uint8))
            output_d_pil.save(self.debug_path / f"{idx:03d}_output_norm_disp.png")

        output_norm_disp_1hw = output_disp_hw.unsqueeze(0)
        scale, shift = compute_scale_and_shift(
            output_norm_disp_1hw, target_disp_b1hw[0], torch.ones_like(output_norm_disp_1hw)
        )
        # Model can occasionally return zeros or even negative values for the disp, so we clip it to something low to be safe (1e-2)
        scaled_output_disp_1hw = torch.clip(output_norm_disp_1hw * scale + shift, 1e-2)
        output_depth_1hw = 1.0 / scaled_output_disp_1hw

        if self.debug_path is not None:
            output_d_pil = colormap_image(torch.Tensor(output_depth_1hw), vmin=0)
            output_d_pil_pil = Image.fromarray(
                (output_d_pil.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            )
            output_d_pil_pil.save(self.debug_path / f"{idx:03d}_output_depth.png")

        output_image_b3hw = output_rgb_3hw.unsqueeze(0)
        output_depth_b1hw = output_depth_1hw.unsqueeze(0)

        return output_image_b3hw, output_depth_b1hw, latents

    def prepare_xattention(self, ref_frames: List[Frame], target_frame: Frame) -> None:
        # We compute the xattention maps at resolution (H/min_downsample_factor, W/min_downsample_factor)
        # In Stablediffusion, the highest resolution at which attention is done is 8x downsampled, so we do
        # not ned our attention maps to be higher-resolution than that:
        min_downsample_factor = 8

        # Preprocessess and store ref frames in the cross attention processor
        self.set_reference_latents_from_frames(ref_frames)

        def compute_multiple_attn_maps(func, source_frames, target_frame, s, **kwargs):
            # Compute attention maps for each pair of source and target frames
            extra_attention_maps = []
            for source_frame in source_frames:
                extra_attention_maps.append(
                    func(source_frame=source_frame, target_frame=target_frame, s=s, **kwargs)
                )
            return torch.stack(extra_attention_maps, dim=0)

        # Set up extra attention maps if necessary
        if self.xattn_type == "depth_conditioned":
            logger.debug("Computing depth-conditioned attention map with mesh warping")
            self.pipe.cross_attn_processor.extra_attention_maps = compute_multiple_attn_maps(
                compute_mesh_warped_attention_map,
                ref_frames,
                target_frame,
                min_downsample_factor,
                **(self.extra_attn_maps_args or {}),
            ).to(self.pipe.device)
        elif self.xattn_type == "none":
            logger.debug("Not setting any extra attention maps because xattn_type is none")
            return
        else:
            raise ValueError(f"Invalid xattention type {self.xattn_type}")

    def set_reference_latents_from_frames(self, ref_frames: list[Frame]) -> None:
        if self.pipe.cross_attn_processor is None:
            logger.debug("No cross-attn processor, so not setting reference latents")
            return

        logger.debug(f"Setting ref latents from {len(ref_frames)} frames")
        # Extract RGBD from frames
        ref_images = torch.cat([frame.image_bchw for frame in ref_frames], dim=0)

        # Preprocess & encode
        ref_images = self.prepare_guidance_image(ref_images)

        # Extract depths
        ref_depths = torch.cat([1.0 / frame.depth_b1hw for frame in ref_frames], dim=0)

        # Preprocess & encode
        ref_depths = self.prepare_guidance_depth(ref_depths)

        # Cat together
        ref_latents = torch.cat([ref_images, ref_depths], dim=1)

        self.pipe.cross_attn_processor.set_reference_latents(ref_latents)

    def prepare_guidance_image(self, image, generator=None):
        init_image = self.pipe.image_processor.preprocess(image)
        init_image = init_image.to(self.pipe.device)
        init_image = init_image.to(dtype=torch.float32)
        init_latents = self.pipe._encode_vae_image(init_image, generator=generator)
        return init_latents

    def prepare_guidance_depth(self, depth, generator=None):
        # Preproc & encode depth
        depth_normed = self.depth_normalizer(depth)

        # depth_normed will be b1hw; we want b3hw by repeating the single channel 3 times so that
        # we can pass through the image encoder
        depth_normed = depth_normed.repeat(1, 3, 1, 1)

        # Have to call private function :(
        return self.pipe._encode_vae_image(depth_normed.to(self.pipe.device), generator=generator)
