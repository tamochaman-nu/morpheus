# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This model implementation is heavily inspired by https://github.com/haofanwang/ControlNet-for-Diffusers/

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from diffusers.image_processor import PipelineImageInput
from diffusers.models import AutoencoderKL, ControlNetModel, UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers.pipelines.controlnet.pipeline_controlnet_inpaint import (
    StableDiffusionControlNetInpaintPipeline,
    retrieve_latents,
)
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    rescale_noise_cfg,
)
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import deprecate, logging
from diffusers.utils.torch_utils import is_compiled_module, randn_tensor
from loguru import logger
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)


class StableDiffusionRGBDControlNetInpaintPipeline(StableDiffusionControlNetInpaintPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        controlnet: Union[
            ControlNetModel, List[ControlNetModel], Tuple[ControlNetModel], MultiControlNetModel
        ],
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection = None,
        requires_safety_checker: bool = True,
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
            requires_safety_checker=requires_safety_checker,
        )
        self.cross_attn_processor = None

    def set_attention_processor(self, attn_processor):
        self.cross_attn_processor = attn_processor

    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        if image.shape[1] == 3:
            return super()._encode_vae_image(image, generator)
        elif image.shape[1] == 6:
            if isinstance(generator, list):
                for i in range(image.shape[0]):
                    rgb_latents = retrieve_latents(
                        self.vae.encode(image[i : i + 1, :3, :, :]), generator=generator[i]
                    )
                    depth_latents = retrieve_latents(
                        self.vae.encode(image[i : i + 1, 3:, :, :]), generator=generator[i]
                    )
                    image_latents = torch.cat([rgb_latents, depth_latents], dim=1)
                image_latents = torch.cat(image_latents, dim=0)
            else:
                rgb_latents = retrieve_latents(
                    self.vae.encode(image[:, :3, :, :]), generator=generator
                )
                depth_latents = retrieve_latents(
                    self.vae.encode(image[:, 3:, :, :]), generator=generator
                )
                image_latents = torch.cat([rgb_latents, depth_latents], dim=1)

            image_latents = self.vae.config.scaling_factor * image_latents
        else:
            raise ValueError(f"Invalid image shape: {image.shape}, expected 3 or 6 channels")

        return image_latents

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
        image=None,
        timestep=None,
        is_strength_max=True,
        return_noise=False,
        return_image_latents=False,
    ):
        """
        Prepare latents for denoising.

        Notes:
        - If `latents` is `None`, random noise will be generated. If `latents` is provided, this will
        be used instead (except that it will be noised to the level specified by `timestep`).
        - A tuple will be returned containing (latents, noise, image_latents), except that `noise'
        will only be prepared if `return_noise` is True, and `image_latents` will only be
        prepared if `return_image_latents` is True.
        - If `image` has number of channels == 4 or 8, it will be interpreted as latents.
        """
        shape = (
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if (image is None or timestep is None) and not is_strength_max:
            raise ValueError(
                "Since strength < 1. initial latents are to be initialised as a combination of Image + Noise."
                "However, either the image or the noise timestep has not been provided."
            )

        if return_image_latents or (latents is None and not is_strength_max):
            image = image.to(device=device, dtype=dtype)

            if image.shape[1] == 4:
                image_latents = image
            elif image.shape[1] == 8:
                image_latents = image
            else:
                image_latents = self._encode_vae_image(image=image, generator=generator)
            image_latents = image_latents.repeat(batch_size // image_latents.shape[0], 1, 1, 1)

        if latents is None:
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            # if strength is 1. then initialise the latents to noise, else initial to image + noise
            latents = (
                noise
                if is_strength_max
                else self.scheduler.add_noise(image_latents, noise, timestep)
            )
            # if pure noise then scale the initial latents by the  Scheduler's init sigma
            latents = latents * self.scheduler.init_noise_sigma if is_strength_max else latents
        else:
            noise = latents.to(device)
            latents = noise * self.scheduler.init_noise_sigma

        outputs = (latents,)

        if return_noise:
            outputs += (noise,)

        if return_image_latents:
            outputs += (image_latents,)

        return outputs

    def prepare_mask_latents(
        self,
        mask,
        masked_image,
        batch_size,
        height,
        width,
        dtype,
        device,
        generator,
        do_classifier_free_guidance,
    ):
        """
        Convert a mask to latent space.
        If `masked_image` has number of channels == 4 or 8, we will presume that it is a latent image,
        and that the mask therefore needs downsampling by 4x along the HW axes to bring it into
        latent space to match.

        """
        # Transfer the mask and masked_image to the correct device and dtype
        mask = mask.to(device=device, dtype=dtype)
        masked_image = masked_image.to(device=device, dtype=dtype)

        # If the masked image looks like it is already latents
        if masked_image.shape[1] in (4, 8):
            mask = torch.nn.functional.interpolate(mask, size=(height, width))
            masked_image_latents = masked_image
        else:
            mask = torch.nn.functional.interpolate(
                mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
            )
            masked_image_latents = self._encode_vae_image(masked_image, generator=generator)

        # duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
        if mask.shape[0] < batch_size:
            if not batch_size % mask.shape[0] == 0:
                raise ValueError(
                    "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                    f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                    " of masks that you pass is divisible by the total requested batch size."
                )
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)
        if masked_image_latents.shape[0] < batch_size:
            if not batch_size % masked_image_latents.shape[0] == 0:
                raise ValueError(
                    "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                    f" to a total batch size of {batch_size}, but {masked_image_latents.shape[0]} images were passed."
                    " Make sure the number of images that you pass is divisible by the total requested batch size."
                )
            masked_image_latents = masked_image_latents.repeat(
                batch_size // masked_image_latents.shape[0], 1, 1, 1
            )

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = (
            torch.cat([masked_image_latents] * 2)
            if do_classifier_free_guidance
            else masked_image_latents
        )

        # aligning device to prevent device errors when concating it with the latent model input
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        return mask, masked_image_latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: PipelineImageInput = None,
        mask_image: PipelineImageInput = None,
        control_image: PipelineImageInput = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        padding_mask_crop: Optional[int] = None,
        strength: float = 1.0,
        depth_strength: Optional[float] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.FloatTensor]] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 0.5,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        timesteps_override: Optional[List[int]] = None,
        guidance_rescale: Optional[float] = None,
        do_xattn: bool = False,
        return_all_latents: bool = False,
        pre_step_callback: Optional[Callable] = None,
        min_noise_strength: Optional[float] = None,
        min_depth_noise_strength: Optional[float] = None,
        **kwargs,
    ):
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        # If a noise strength for the depth channel wasn't explicitly specified, use the same noise
        # level as the RGB channel:
        if depth_strength is None:
            depth_strength = strength

        self.guidance_rescale = guidance_rescale
        if self.guidance_rescale is not None:
            logger.debug(f"Using guidance rescale: {self.guidance_rescale}")

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )

        controlnet = (
            self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet
        )

        # align format for control guidance
        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(
            control_guidance_start, list
        ):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(
            control_guidance_end, list
        ):
            mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
            control_guidance_start, control_guidance_end = (
                mult * [control_guidance_start],
                mult * [control_guidance_end],
            )

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            control_image,
            mask_image,
            height,
            width,
            callback_steps,
            output_type,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            controlnet_conditioning_scale,
            control_guidance_start,
            control_guidance_end,
            callback_on_step_end_tensor_inputs,
            padding_mask_crop,
        )

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if padding_mask_crop is not None:
            height, width = self.image_processor.get_default_height_width(image, height, width)
            crops_coords = self.mask_processor.get_crop_region(
                mask_image, width, height, pad=padding_mask_crop
            )
            resize_mode = "fill"
        else:
            crops_coords = None
            resize_mode = "default"

        device = self._execution_device

        if isinstance(controlnet, MultiControlNetModel) and isinstance(
            controlnet_conditioning_scale, float
        ):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)

        global_pool_conditions = (
            controlnet.config.global_pool_conditions
            if isinstance(controlnet, ControlNetModel)
            else controlnet.nets[0].config.global_pool_conditions
        )
        guess_mode = guess_mode or global_pool_conditions

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            self.cross_attention_kwargs.get("scale", None)
            if self.cross_attention_kwargs is not None
            else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=self.clip_skip,
        )
        # For classifier free guidance, we need to do two forward passes, which we do in a single
        # batch:
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        # 4. Prepare image
        if isinstance(controlnet, ControlNetModel):
            control_image = self.prepare_control_image(
                image=control_image,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=controlnet.dtype,
                crops_coords=crops_coords,
                resize_mode=resize_mode,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                guess_mode=guess_mode,
            )
        elif isinstance(controlnet, MultiControlNetModel):
            control_images = []

            for control_image_ in control_image:
                control_image_ = self.prepare_control_image(
                    image=control_image_,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=controlnet.dtype,
                    crops_coords=crops_coords,
                    resize_mode=resize_mode,
                    do_classifier_free_guidance=self.do_classifier_free_guidance,
                    guess_mode=guess_mode,
                )

                control_images.append(control_image_)

            control_image = control_images
        else:
            raise ValueError(f"Unsupported controlnet type: {type(controlnet)}")

        # 4.1 Preprocess mask and image - resizes image and mask w.r.t height and width
        original_image = image
        init_image = self.image_processor.preprocess(
            image, height=height, width=width, crops_coords=crops_coords, resize_mode=resize_mode
        )
        init_image = init_image.to(dtype=torch.float32)

        # Unlike an actual inpainting-based pipelines, we don't mask out the image here, because we
        # want the model to see the whole image:
        masked_image = init_image
        _, _, height, width = init_image.shape

        # 5. Prepare timesteps
        if timesteps_override is not None:
            timesteps = timesteps_override
        else:
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps, num_inference_steps = self.get_timesteps(
                num_inference_steps=num_inference_steps,
                strength=max(strength, depth_strength),
                device=device,
            )
        # at which timestep to set the initial noise (n.b. 50% if strength is 0.5)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
        # create a boolean to check if the strength is set to 1. if so then initialise the latents with pure noise
        is_strength_max = strength == 1.0
        self._num_timesteps = len(timesteps)

        # 6. Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels * 2
        num_channels_unet = self.unet.config.in_channels
        assert num_channels_unet == 10, f"Need 10-channel unet, got {num_channels_unet}"

        latents_outputs = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
            image=init_image,
            timestep=latent_timestep,
            is_strength_max=is_strength_max,
            return_noise=True,
            return_image_latents=True,
        )

        latents, noise, image_latents = latents_outputs

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7.1 Add image embeds for IP-Adapter
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if ip_adapter_image is not None or ip_adapter_image_embeds is not None
            else None
        )

        # 7.2 Create tensor stating which controlnets to keep
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)

        # If we don't want to do cross-attention, then reset the attention processor to the default,
        # rather than our special cross-attention version:
        if not do_xattn:
            self.unet.set_attn_processor(AttnProcessor())

        # Set up cacheing of intermediate latents
        if return_all_latents:
            # dict that will store latents from each timestep (keyed by timestep)
            all_latents = {}

        num_train_timesteps = len(self.scheduler)

        original_noisy_latents = latents.clone()  # [B, 8, H, W]
        original_noisy_rgb_latents = original_noisy_latents[:, :4, :, :]
        original_noisy_depth_latents = original_noisy_latents[:, 4:, :, :]

        # we initialize the last latents to the noisy latents, these are used when we are using a dual t model
        last_rgb_latent = original_noisy_rgb_latents.clone()
        last_depth_latent = original_noisy_depth_latents.clone()

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                if pre_step_callback is not None:
                    pre_step_callback(t)

                curr_time_fraction = t / num_train_timesteps

                latent_model_input = latents
                assert (
                    latent_model_input.shape[1] == 8
                ), f"Expected 8 channels in latents, got {latent_model_input.shape[1]}"
                rgb_latent = latent_model_input[:, :4, :, :]
                depth_latent = latent_model_input[:, 4:, :, :]

                # update the last latents we used; these are used when we need to discard the update from the model
                last_rgb_latent = rgb_latent.clone()
                last_depth_latent = depth_latent.clone()

                rgb_time_mask = torch.ones_like(rgb_latent[:, 0:1, :, :]) * (
                    curr_time_fraction <= strength
                )
                rgb_time_mask = rgb_time_mask.float()
                depth_time_mask = torch.ones_like(depth_latent[:, 0:1, :, :]) * (
                    curr_time_fraction <= depth_strength
                )
                depth_time_mask = depth_time_mask.float()

                latent_model_input = torch.cat(
                    [rgb_latent, rgb_time_mask, depth_latent, depth_time_mask], dim=1
                )

                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latent_model_input] * 2)
                    if self.do_classifier_free_guidance
                    else latent_model_input
                )

                if return_all_latents:
                    all_latents[int(t)] = latent_model_input[0:1].detach()

                assert (
                    latent_model_input.shape[1] == 10
                ), f"Expected 10 channels in latents, got {latent_model_input.shape[1]}"
                latent_model_input[:, :4, :, :] = self.scheduler.scale_model_input(
                    latent_model_input[:, :4, :, :], t
                )
                latent_model_input[:, 5:9, :, :] = self.scheduler.scale_model_input(
                    latent_model_input[:, 5:9, :, :], t
                )

                # controlnet(s) inference
                if guess_mode and self.do_classifier_free_guidance:
                    # Infer ControlNet only for the conditional batch.
                    control_model_input = latents
                    control_model_input = self.scheduler.scale_model_input(control_model_input, t)
                    controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
                else:
                    control_model_input = latent_model_input
                    controlnet_prompt_embeds = prompt_embeds

                if isinstance(controlnet_keep[i], list):
                    cond_scale = [
                        c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])
                    ]
                else:
                    controlnet_cond_scale = controlnet_conditioning_scale
                    if isinstance(controlnet_cond_scale, list):
                        controlnet_cond_scale = controlnet_cond_scale[0]
                    cond_scale = controlnet_cond_scale * controlnet_keep[i]

                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    control_model_input,
                    t,
                    encoder_hidden_states=controlnet_prompt_embeds,
                    controlnet_cond=control_image,
                    conditioning_scale=cond_scale,
                    guess_mode=guess_mode,
                    return_dict=False,
                )

                if isinstance(controlnet, MultiControlNetModel):
                    # we average the residuals from the different controlnets
                    down_block_res_samples = [
                        samples / len(controlnet.nets) for samples in down_block_res_samples
                    ]
                    mid_block_res_sample /= len(controlnet.nets)

                if guess_mode and self.do_classifier_free_guidance:
                    # Inferred ControlNet only for the conditional batch.
                    # To apply the output of ControlNet to both the unconditional and conditional batches,
                    # add 0 to the unconditional batch to keep it unchanged.
                    down_block_res_samples = [
                        torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples
                    ]
                    mid_block_res_sample = torch.cat(
                        [torch.zeros_like(mid_block_res_sample), mid_block_res_sample]
                    )

                if do_xattn:
                    num_ref_frames = self.cross_attn_processor.num_ref_latents

                    # Chunk the text embedding into two parts
                    text_embed_without_ref = prompt_embeds
                    uncond_embedding, cond_embedding = text_embed_without_ref.chunk(2)

                    # Extend both tensors of embeddings by num_ref_frames
                    last_uncond_embedding = uncond_embedding[-1]
                    last_cond_embedding = cond_embedding[-1]

                    last_uncond_embedding_shape = last_uncond_embedding.shape
                    ref_uncond_embeddings = last_uncond_embedding.view(
                        1, *last_uncond_embedding_shape
                    ).expand(num_ref_frames, *last_uncond_embedding_shape)
                    ref_cond_embeddings = last_cond_embedding.view(
                        1, *last_cond_embedding.shape
                    ).expand(num_ref_frames, *last_cond_embedding.shape)

                    # Concatenate the reference embeddings to the text embeddings
                    uncond_embedding = torch.cat([ref_uncond_embeddings, uncond_embedding])
                    cond_embedding = torch.cat([ref_cond_embeddings, cond_embedding])

                    prompt_embeds_for_unet = torch.cat([uncond_embedding, cond_embedding])

                    # Build multi-image chunks by concatenating the reference latents with the
                    # target view. Each of these chunks is an independent forward pass through the
                    # model (though they are done in a single batch). The final element of each
                    # chunk is the target view - i.e. the thing we're actually generating - and
                    # the preceding elements are the reference frames that the target view will be
                    # attending to. (Both the unconditional and conditional fwd passes will attend
                    # to the same ref latents.)
                    chunks = latent_model_input.chunk(2 if self.do_classifier_free_guidance else 1)
                    chunks = [
                        torch.cat([self.cross_attn_processor.ref_latents, chunk], dim=0)
                        for chunk in chunks
                    ]

                    # Concatenate the chunks together, packing them into a single batch of shape
                    # [num_chunks * (num_ref_frames + 1), spatial_dim, channel_dim].
                    # This ensures the input matches the dimensionality expected by the UNet.
                    latent_model_input = torch.cat(chunks)

                    # Let the cross-attention processor know how many independent chunks we're
                    # giving it. It will then use this information to then unpack the input,
                    # separating out the 'chunk' and 'frame-within-in' dimensions that we just
                    # concatenated into one on the previous step.
                    self.cross_attn_processor.unet_num_chunks = (
                        2 if self.do_classifier_free_guidance else 1
                    )
                    self.unet.set_attn_processor(self.cross_attn_processor)

                    # The controlnet residuals must match the shape of the latents, so we must
                    # expand them to have the same batch size as our unet input (which has
                    # been enlarged because of the ref frames).
                    # For the controlnet residuals corresponding to the ref frames, we inject zeroes
                    # because we don't actually want to change the ref frames - they are only there
                    # so that the target view can attend to them.
                    def make_zero_ref_like(res):
                        # Will make a block of zeroes that is the same size as the ref frames and
                        # axes > 0, and with size along axis 0 equal to the number of ref frames
                        return torch.zeros_like(res[:1]).expand(num_ref_frames, *res.shape[1:])

                    down_block_res_samples_ref = [
                        make_zero_ref_like(res) for res in down_block_res_samples
                    ]
                    mid_block_res_sample_ref = make_zero_ref_like(mid_block_res_sample)

                    def cat_residuals(res, ref_res):
                        # Do the same as we do for the latents: split into chunks and cat the ref to each chunk
                        res_chunks = [
                            torch.cat([ref_res, chunk], dim=0)
                            for chunk in res.chunk(self.cross_attn_processor.unet_num_chunks)
                        ]
                        return torch.cat(res_chunks)

                    # Now cat in the ref residuals for each residual
                    down_block_res_samples = [
                        cat_residuals(res, ref_res)
                        for res, ref_res in zip(down_block_res_samples, down_block_res_samples_ref)
                    ]
                    mid_block_res_sample = cat_residuals(
                        mid_block_res_sample, mid_block_res_sample_ref
                    )

                else:
                    prompt_embeds_for_unet = prompt_embeds

                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds_for_unet,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # the first num_ref_latents from each chunk are the ref frames.
                # So now we will have a contiguous block that goes:
                # [chunk_0_ref, chunk_0_non_ref, chunk_1_ref, chunk_1_non_ref, ...]
                # Typically we will have two chunks, one for the uncond and one for the cond, so
                # that we can do CFG
                if do_xattn:
                    chunks = noise_pred.chunk(self.cross_attn_processor.unet_num_chunks)
                    noise_pred = torch.cat(
                        [chunk[self.cross_attn_processor.num_ref_latents :] for chunk in chunks]
                    )

                # Perform guidance:
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                    if self.guidance_rescale is not None and self.guidance_rescale > 0.0:
                        # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        noise_pred = rescale_noise_cfg(
                            noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale
                        )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]

                assert (
                    latents.shape[1] == 8
                ), f"Expected 8 channels in latents, got {latents.shape[1]}"

                rgb_latent = latents[:, :4, :, :]
                depth_latent = latents[:, 4:, :, :]

                # If our current timestep is before the RGB timestep, we throw away the RGB latent
                # and replace it with the original RGB latent
                curr_time_fraction = t / num_train_timesteps

                # If we are inverting, when replacing the latents we use the last inverted latents
                # that were inverted up to the required time.
                # If we are not inverting, we use the original noisy latents which were noised up to
                # the required time
                # Example of when we might hit the below statement:
                # strength=0.5, depth_strength=0.6, normal denoising (not inversion).
                # Then the denoising loop will start at a noise level of 0.6, but we will discard
                # updates to the RGB latents until we reach 0.5, because we assume that the RGB
                # latents that were fed in are already noised to a level of 0.5.
                if curr_time_fraction > strength:
                    logger.debug(
                        f"Replacing RGB latent at current timestep {t} fraction "
                        f"{curr_time_fraction} with original at time {strength}"
                    )
                    rgb_latent = last_rgb_latent
                elif min_noise_strength is not None and curr_time_fraction < min_noise_strength:
                    logger.debug(
                        f"min_noise_strength is not None, replacing RGB latent at current "
                        f"timestep {t} fraction {curr_time_fraction} with original at time "
                        f"{min_noise_strength}"
                    )
                    rgb_latent = last_rgb_latent

                # Just as above but for depth:
                if curr_time_fraction > depth_strength:
                    logger.debug(
                        f"Replacing depth latent at current timestep {t} fraction "
                        f"{curr_time_fraction} with original at time {depth_strength}"
                    )
                    depth_latent = last_depth_latent
                elif (
                    min_depth_noise_strength is not None
                    and curr_time_fraction < min_depth_noise_strength
                ):
                    logger.debug(
                        f"min_depth_noise_strength is not None, replacing depth latent at "
                        f"current timestep {t} fraction {curr_time_fraction} with original at "
                        f"time {min_depth_noise_strength}"
                    )
                    depth_latent = last_depth_latent

                # Cat the latents back together again:
                latents = torch.cat([rgb_latent, depth_latent], dim=1)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop(
                        "negative_prompt_embeds", negative_prompt_embeds
                    )

                # Call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        # If we do sequential model offloading, let's offload unet and controlnet
        # manually for max memory savings
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.unet.to("cpu")
            self.controlnet.to("cpu")
            torch.cuda.empty_cache()

        if not output_type == "latent":
            if latents.shape[1] == 4:
                image = self.vae.decode(
                    latents / self.vae.config.scaling_factor, return_dict=False, generator=generator
                )[0]
                image, has_nsfw_concept = self.run_safety_checker(
                    image, device, prompt_embeds.dtype
                )
            elif latents.shape[1] == 8:
                rgb_latents, depth_latents = latents.chunk(2, dim=1)
                rgb_image = self.vae.decode(
                    rgb_latents / self.vae.config.scaling_factor,
                    return_dict=False,
                    generator=generator,
                )[0]
                depth_image = self.vae.decode(
                    depth_latents / self.vae.config.scaling_factor,
                    return_dict=False,
                    generator=generator,
                )[0]
                image = torch.cat([rgb_image, depth_image], dim=1)
                image, has_nsfw_concept = self.run_safety_checker(
                    image, device, prompt_embeds.dtype
                )
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(
            image, output_type=output_type, do_denormalize=do_denormalize
        )

        if padding_mask_crop is not None:
            image = [
                self.image_processor.apply_overlay(mask_image, original_image, i, crops_coords)
                for i in image
            ]

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        if return_all_latents:
            return (
                StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept),
                all_latents,
            )
        else:
            return StableDiffusionPipelineOutput(
                images=image, nsfw_content_detected=has_nsfw_concept
            )
