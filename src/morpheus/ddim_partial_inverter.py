from typing import Optional

import torch
from diffusers.models.attention_processor import AttnProcessor
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers.schedulers import DDIMScheduler
from loguru import logger

from morpheus.diffuser_pipelines.schedulers import PartialDDIMInverseScheduler


class DDIMPartialInverter:
    def __init__(self, num_inference_steps: int, rgbd_pipeline, prompt=""):
        self.num_inference_steps = num_inference_steps
        self.pipe = rgbd_pipeline
        self.ddim_scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.ddim_inverter = PartialDDIMInverseScheduler.from_config(self.pipe.scheduler.config)
        self.ddim_inverter.set_timesteps(
            num_inference_steps=self.num_inference_steps, device=rgbd_pipeline.device
        )
        self.prompt = prompt

        logger.info(f"Using inversion prompt: {self.prompt} with {self.num_inference_steps} steps")

    def do_partial_inversion(
        self,
        input_image,
        control_image_list,
        generator,
        noise_strength,
        depth_noise_strength,
        partial_inversion_noise_level: float = 0.0,
        partial_inversion_depth_noise_level: Optional[float] = None,
    ):

        # Set attention procesor to be the default one, rather than our modified one that does
        # cross-attention with reference frames. For inversion, we don't need this.
        self.pipe.unet.set_attn_processor(processor=AttnProcessor())
        if isinstance(self.pipe.controlnet, MultiControlNetModel):
            for net in self.pipe.controlnet.nets:
                net.set_attn_processor(processor=AttnProcessor())
        else:
            self.pipe.controlnet.set_attn_processor(processor=AttnProcessor())

        if partial_inversion_depth_noise_level is None:
            partial_inversion_depth_noise_level = partial_inversion_noise_level

        # The passed *_noise_level args are defined as floats in [0, 1]; we want to find the number
        # of inference steps that corresponds to in practice:
        random_noise_for_first_n_steps = int(
            partial_inversion_noise_level * self.num_inference_steps
        )
        random_depth_noise_for_first_n_steps = int(
            partial_inversion_depth_noise_level * self.num_inference_steps
        )

        # Set timesteps for partial inversion
        self.ddim_inverter.set_timesteps(
            num_inference_steps=self.num_inference_steps, device=self.pipe.device
        )
        timesteps = self.ddim_inverter.timesteps

        # Consistency check: we have a two-part inversion process, first adding random noise to
        # reach some initial noise level, then inverting until we reach the target noise level.
        # That first noise level - defined by random_noise_for_first_n_steps - must therefore be
        # less than the final noise level, defined by noise_strength and depth_noise_strength.
        assert (
            random_noise_for_first_n_steps <= noise_strength * self.num_inference_steps
        ), f"random_noise_for_first_n_steps: {random_noise_for_first_n_steps} must be less than noise_strength: {noise_strength} * num_inference_steps: {self.num_inference_steps}"
        assert (
            random_depth_noise_for_first_n_steps <= depth_noise_strength * self.num_inference_steps
        ), f"random_depth_noise_for_first_n_steps: {random_depth_noise_for_first_n_steps} must be less than depth_noise_strength: {depth_noise_strength} * num_inference_steps: {self.num_inference_steps}"

        # Note that because we are inverting, our tsteps start at 1 (the clean end) and go up to the
        # noisy end. So for DDIM inversion up to some noise level short of the maximum we keep the
        # first num_tsteps_to_keep timesteps, not the last (as we would with a normal scheduler)
        if noise_strength >= depth_noise_strength:
            num_tsteps_to_keep = int(noise_strength * self.num_inference_steps)
        else:
            num_tsteps_to_keep = int(depth_noise_strength * self.num_inference_steps)
        tsteps = timesteps[:num_tsteps_to_keep]

        # Swap out the scheduler in the diffusion pipeline for our inversion one:
        self.pipe.scheduler = self.ddim_inverter

        # Preprocess the image into latents
        init_image = self.pipe.image_processor.preprocess(input_image[None, ...])
        init_image = init_image.to(self.pipe.device)
        init_image = init_image.to(dtype=torch.float32)
        latents = self.pipe._encode_vae_image(init_image, generator=generator)

        noise = torch.randn(
            latents.shape,
            device=self.pipe.device,
            generator=generator,
        )  # [B, 8, h, w]

        rgb_latents = latents[:, :4, :, :]
        depth_latents = latents[:, 4:, :, :]

        if random_noise_for_first_n_steps > 0:
            logger.debug(
                f"Adding noise to rgb latents to get to tstep {tsteps[random_noise_for_first_n_steps]}"
            )
            rgb_latents = self.ddim_scheduler.add_noise(
                rgb_latents,
                noise[:, :4, :, :],
                tsteps[random_noise_for_first_n_steps : random_noise_for_first_n_steps + 1],
            )

        if random_depth_noise_for_first_n_steps > 0:
            logger.debug(
                f"Adding noise to depth latents to get to tstep {tsteps[random_depth_noise_for_first_n_steps]}"
            )
            depth_latents = self.ddim_scheduler.add_noise(
                depth_latents,
                noise[:, 4:, :, :],
                tsteps[
                    random_depth_noise_for_first_n_steps : random_depth_noise_for_first_n_steps + 1
                ],
            )

        latents = torch.cat([rgb_latents, depth_latents], dim=1)

        if random_noise_for_first_n_steps > 0 or random_depth_noise_for_first_n_steps > 0:
            steps_to_chop_off = (
                min(random_noise_for_first_n_steps, random_depth_noise_for_first_n_steps) + 1
            )
            # Chop off the timesteps that we don't need to do any more because we added noise:
            tsteps = tsteps[steps_to_chop_off:]
            logger.debug(f"Timesteps for partial inversion are: {tsteps}")

        output = self.pipe(
            prompt=self.prompt,
            num_inference_steps=self.num_inference_steps,
            generator=generator,
            image=latents,
            latents=latents,
            control_image=control_image_list,
            controlnet_conditioning_scale=0.0,
            mask_image=torch.ones(
                (1, 1, input_image.shape[0], input_image.shape[1]), device=self.pipe.device
            ),
            guidance_scale=1,  # no cfg
            strength=noise_strength,
            depth_strength=depth_noise_strength,
            output_type="latent",
            timesteps_override=tsteps[:-1],
            min_noise_strength=partial_inversion_noise_level,
            min_depth_noise_strength=partial_inversion_depth_noise_level,
        ).images

        inverted_latents = output.to(self.pipe.device)

        # Restore the original scheduler (which we swapped out with our inversion one):
        self.pipe.scheduler = self.ddim_scheduler

        return inverted_latents
