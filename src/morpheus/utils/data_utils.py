from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class Frame:
    """Represents a single posed frame."""

    idx: int
    image_bchw: torch.Tensor
    depth_b1hw: torch.Tensor
    K_b44: torch.Tensor
    invK_b44: torch.Tensor
    cam_to_world_b44: torch.Tensor
    world_to_cam_b44: torch.Tensor
    debug_dict: dict = field(default_factory=dict)

    def to(self, device):
        """Convenience function to move all tensors in the Frame to a specific device."""
        debug_dict = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in self.debug_dict.items()
        }

        return Frame(
            idx=self.idx,
            image_bchw=self.image_bchw.to(device),
            depth_b1hw=self.depth_b1hw.to(device),
            K_b44=self.K_b44.to(device),
            invK_b44=self.invK_b44.to(device),
            cam_to_world_b44=self.cam_to_world_b44.to(device),
            world_to_cam_b44=self.world_to_cam_b44.to(device),
            debug_dict=debug_dict,
        )
