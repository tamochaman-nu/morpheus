from typing import List, Optional, Union

import numpy as np
import torch
from diffusers.schedulers import DDIMInverseScheduler


class PartialDDIMInverseScheduler(DDIMInverseScheduler):
    """
    Custom version of the DDIMInverseScheduler that allows for partial inversion - i.e., inversion
    starting at a timestep after the first one. We need this because in Morpheus add random noise to
    get to some noise level, and then do inversion from there.
    """

    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        timesteps: List[int] = None,
        device: Union[str, torch.device] = None,
    ):
        # Unlike the base class, this has an extra `timesteps` argument, allowing us to explicitly
        # pass in a list of timesteps that need not start at timestep 1, which allows us to do
        # partial inversion.

        if timesteps is not None:
            self.timesteps = torch.tensor(timesteps, device=device)
            return

        if num_inference_steps > self.config.num_train_timesteps:
            raise ValueError(
                f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.config.train_timesteps`:"
                f" {self.config.num_train_timesteps} as the unet model trained with this scheduler can only handle"
                f" maximal {self.config.num_train_timesteps} timesteps."
            )

        self.num_inference_steps = num_inference_steps

        # "leading" and "trailing" corresponds to annotation of Table 2. of https://arxiv.org/abs/2305.08891
        if self.config.timestep_spacing == "leading":
            step_ratio = self.config.num_train_timesteps // self.num_inference_steps
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            timesteps = (
                (np.arange(0, num_inference_steps) * step_ratio).round().copy().astype(np.int64)
            )
            timesteps += self.config.steps_offset
        elif self.config.timestep_spacing == "trailing":
            step_ratio = self.config.num_train_timesteps / self.num_inference_steps
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            timesteps = np.round(
                np.arange(self.config.num_train_timesteps, 0, -step_ratio)[::-1]
            ).astype(np.int64)
            timesteps -= 1
        else:
            raise ValueError(
                f"{self.config.timestep_spacing} is not supported. Please make sure to choose one of 'leading' or 'trailing'."
            )

        self.timesteps = torch.from_numpy(timesteps).to(device=device)
