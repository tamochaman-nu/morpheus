from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from loguru import logger
from PIL import Image

from morpheus.utils.data_utils import Frame
from morpheus.visualization.image_viz import colormap_image
from morpheus.xattn_controlnet import InpaintInputs, XAttnWarpRGBDControlNet


class Inpainter:
    def __init__(
        self,
        text_prompt: str,
        negative_text_prompt: str,
        noise_strength: float,
        guidance_scale: float,
        controlnet_strength: float,
        controlnet_model_path: Path,
        stable_diffusion_model_path: Path,
        debug_path: Optional[Path],
        n_controlnets: Optional[int] = 1,
        huggingface_cache_dir: Optional[Path] = None,
        **extra_controlnet_kwargs,
    ):
        if debug_path is not None:
            (debug_path / "controlnet_debug").mkdir(exist_ok=True)
        self.warp_controlnet = XAttnWarpRGBDControlNet(
            text_prompt=text_prompt,
            negative_text_prompt=negative_text_prompt,
            noise_strength=noise_strength,
            guidance_scale=guidance_scale,
            controlnet_strength=controlnet_strength,
            controlnet_model_path=controlnet_model_path,
            stable_diffusion_model_path=stable_diffusion_model_path,
            huggingface_cache_dir=huggingface_cache_dir,
            debug_path=debug_path / "controlnet_debug" if debug_path is not None else None,
            n_controlnets=n_controlnets,
            **extra_controlnet_kwargs,
        )

        self.debug_path = debug_path
        self.n_controlnets = n_controlnets

    def inpaint(
        self, inputs: List[InpaintInputs], target_frame: Frame, seed: int, inpainter_kwargs=None
    ) -> Frame:
        if self.debug_path is not None:
            rgb_img = Image.fromarray(
                (target_frame.image_bchw[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            )
            rgb_img.save(self.debug_path / f"{inputs[-1].input_frame.idx}_target_frame_rgb.png")

            depth_img = colormap_image(target_frame.depth_b1hw[0], vmin=0)
            depth_img = Image.fromarray(
                (depth_img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            )
            depth_img.save(self.debug_path / f"{inputs[-1].input_frame.idx}_target_frame_depth.png")

        if inputs[-1].xattn_ref_frames is not None:
            logger.debug("Preparing reference frames for xattention...")
            self.warp_controlnet.prepare_xattention(
                ref_frames=inputs[-1].xattn_ref_frames, target_frame=inputs[-1].input_frame
            )

        (
            inpainted_frame_b3hw,
            inpainted_frame_depth_b1hw,
            intermediate_latents,
        ) = self.warp_controlnet.inpaint(
            inputs=inputs,
            target_frame=target_frame,
            idx=target_frame.idx if target_frame is not None else None,
            seed=seed,
            xattn_ref_frames=(
                inputs.xattn_ref_frames
                if not isinstance(inputs, list)
                else inputs[-1].xattn_ref_frames
            ),
            **(inpainter_kwargs if inpainter_kwargs is not None else {}),
        )

        if self.debug_path is not None:
            rgb_img = Image.fromarray(
                (inpainted_frame_b3hw[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            )
            rgb_img.save(self.debug_path / f"{inputs[-1].input_frame.idx}_inpainted_rgb.png")

            depth_img = colormap_image(inpainted_frame_depth_b1hw[0], vmin=0)
            depth_img = Image.fromarray(
                (depth_img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            )
            depth_img.save(self.debug_path / f"{inputs[-1].input_frame.idx}_inpainted_depth.png")

        inpainted_frame = Frame(
            idx=(
                inputs.input_frame.idx
                if not isinstance(inputs, list)
                else inputs[-1].input_frame.idx
            ),
            image_bchw=inpainted_frame_b3hw,
            depth_b1hw=inpainted_frame_depth_b1hw,
            K_b44=(
                inputs.input_frame.K_b44
                if not isinstance(inputs, list)
                else inputs[-1].input_frame.K_b44
            ),
            invK_b44=(
                inputs.input_frame.invK_b44
                if not isinstance(inputs, list)
                else inputs[-1].input_frame.invK_b44
            ),
            cam_to_world_b44=(
                inputs.input_frame.cam_to_world_b44
                if not isinstance(inputs, list)
                else inputs[-1].input_frame.cam_to_world_b44
            ),
            world_to_cam_b44=(
                inputs.input_frame.world_to_cam_b44
                if not isinstance(inputs, list)
                else inputs[-1].input_frame.world_to_cam_b44
            ),
            debug_dict={"intermediate_latents": intermediate_latents},
        )
        return inpainted_frame


def stylise_i_frame(
    input_frame: Frame,
    inpainter: Inpainter,
    controlnet_strength: float,
    guidance_scale: float,
    noise_strength: float,
    depth_noise_strength: float,
    seed: int,
    partial_inversion_noise_level: Optional[float] = None,
    partial_inversion_depth_noise_level: Optional[float] = None,
):
    # Handles special case of making the I-frame. Here the mask is trivially zero everywhere,
    # because in the first frame we do not yet have any previously stylized frames to warp from, so
    # so we need to inpaint everything.
    inpainter_kwargs = {
        "noise_strength": noise_strength,
        "depth_noise_strength": depth_noise_strength,
        "guidance_scale": guidance_scale,
        "controlnet_strength": controlnet_strength,
    }
    if partial_inversion_noise_level is not None:
        inpainter_kwargs["partial_inversion_noise_level"] = partial_inversion_noise_level

    if partial_inversion_depth_noise_level is not None:
        inpainter_kwargs[
            "partial_inversion_depth_noise_level"
        ] = partial_inversion_depth_noise_level

    return inpainter.inpaint(
        [
            InpaintInputs(
                input_frame=input_frame,
                mask=torch.zeros_like(input_frame.image_bchw[:, 0:1]),
                original_frame=input_frame,
            )
        ],
        target_frame=input_frame,
        seed=seed,
        inpainter_kwargs=inpainter_kwargs,
    )
