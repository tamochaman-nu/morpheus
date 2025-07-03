from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from loguru import logger
from PIL import Image

from morpheus.forward_warp_compositor import ForwardWarpCompositor
from morpheus.inpainter import Inpainter
from morpheus.utils.data_utils import Frame
from morpheus.visualization.image_viz import colormap_image
from morpheus.xattn_controlnet import InpaintInputs


class KeyframeSelector:
    """
    This class is used to select source frames to use for compositing and reference frames to use
    for xattn.
    """

    def __init__(self, idxs: list[int] = None):
        """idxs are the indices of frames to return"""
        self.idxs = idxs

    def __call__(self, target_frame: Frame, frames: List[Frame]) -> List[int]:
        # Length-zero special case, carried over from old function
        logger.info(f"Selecting frames with idxs: {self.idxs}")

        if self.idxs is None:
            return list(range(len(frames)))

        if len(self.idxs) == 0:
            return []

        frames_idxs = list(range(len(frames)))
        chosen_idxs = [frames_idxs[idx] for idx in self.idxs]

        return list(set(chosen_idxs))


def make_frame_selector(rule: str) -> KeyframeSelector:
    if rule == "all":
        return KeyframeSelector(None)
    elif rule == "first":
        return KeyframeSelector([0])
    elif rule == "last":
        return KeyframeSelector([-1])
    elif rule == "first-and-last":
        return KeyframeSelector([0, -1])
    else:
        raise ValueError(f"Unknown frame selection rule: {rule}")


class PFrameGenerator:
    def __init__(
        self,
        compositor: ForwardWarpCompositor,
        inpainter: Inpainter,
        debug_path: Optional[Path] = None,
        src_frame_selector: str = "all",
        xattn_ref_frame_selector: str = "first-and-last",
    ):
        self.compositor = compositor
        self.inpainter = inpainter
        self.debug_path = debug_path
        logger.debug(f"Got src frame selector arg: {src_frame_selector}")
        logger.debug(f"Got xattn ref frame selector arg: {xattn_ref_frame_selector}")
        logger.debug(
            f"src_frame_selector: {src_frame_selector}, xattn_ref_frame_selector: {xattn_ref_frame_selector}"
        )
        self.src_frame_selector = make_frame_selector(src_frame_selector)
        self.xattn_ref_frame_selector = make_frame_selector(xattn_ref_frame_selector)

    def generate_p_frame(
        self,
        src_frames: list[Frame],
        src_frames_stylised: list[Frame],
        target_frame: Frame,
        seed: int,
    ):
        # Sanity-check src_frames and src_frames_stylised match up (refer to the same frames)
        assert len(src_frames) == len(src_frames_stylised)
        for src_frame, src_frame_stylised in zip(src_frames, src_frames_stylised):
            assert src_frame.idx == src_frame_stylised.idx

        # Pick src frames to use for compositing...
        src_frame_idxs = self.src_frame_selector(target_frame, src_frames)
        src_frames = [src_frames[idx] for idx in src_frame_idxs]
        src_frames_stylised = [src_frames_stylised[idx] for idx in src_frame_idxs]
        logger.debug(f"src frame idxs: {src_frame_idxs}")

        # ...and ref frames to use for xattn
        xattn_ref_frame_idxs = self.xattn_ref_frame_selector(target_frame, src_frames)
        xattn_ref_frames = [src_frames_stylised[idx] for idx in xattn_ref_frame_idxs]
        logger.debug(f"xattn ref frame idxs: {xattn_ref_frame_idxs}")

        if len(src_frames) < self.inpainter.n_controlnets:
            if len(src_frames) == 1:
                logger.warning(
                    f"Only got 1 src frame, duplicating it {self.inpainter.n_controlnets} times"
                )
                # Duplicate src frames
                src_frames = src_frames * self.inpainter.n_controlnets
                src_frames_stylised = src_frames_stylised * self.inpainter.n_controlnets
            else:
                raise ValueError(
                    f"Number of src frames ({len(src_frames)}) is less than the number of controlnets ({self.inpainter.n_controlnets})"
                )

        assert (
            len(src_frames) == self.inpainter.n_controlnets
        ), f"Number of src frames ({len(src_frames)}) is not equal to the number of controlnets ({self.inpainter.n_controlnets})"

        # Split src frames into groups for each controlnet
        src_frames_groups = np.array_split(src_frames, self.inpainter.n_controlnets)
        src_frames_stylised_groups = np.array_split(
            src_frames_stylised, self.inpainter.n_controlnets
        )
        assert (
            len(src_frames_groups)
            == len(src_frames_stylised_groups)
            == self.inpainter.n_controlnets
        ), f"Number of src frames groups ({len(src_frames_groups)}) is not equal to the number of controlnets ({self.inpainter.n_controlnets})"

        inpainting_inputs_list = []
        for i in range(self.inpainter.n_controlnets):
            input_src_frames = src_frames_groups[i]
            input_src_frames_stylised = src_frames_stylised_groups[i]
            # Composite
            inpainting_inputs = self.compositor.composite_for_inpainting(
                src_frames_raw=input_src_frames,
                src_frames_stylised=input_src_frames_stylised,
                target_frame_raw=target_frame,
            )

            # Dump composited frame to debug path
            if self.debug_path is not None:
                composited_image = Image.fromarray(
                    (
                        inpainting_inputs.input_frame.image_bchw[0].permute(1, 2, 0).cpu().numpy()
                        * 255
                    ).astype(np.uint8)
                )
                composited_image.save(self.debug_path / f"{target_frame.idx}_composited_{i}.jpg")

                depth_img = colormap_image(inpainting_inputs.input_frame.depth_b1hw[0], vmin=0)
                depth_img = Image.fromarray(
                    (depth_img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                )
                depth_img.save(self.debug_path / f"{target_frame.idx}_composited_depth_{i}.jpg")

                disp = 1.0 / inpainting_inputs.input_frame.depth_b1hw[0]
                disp_img = colormap_image(disp, flip=False)
                disp_img = Image.fromarray(
                    (disp_img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                )
                disp_img.save(self.debug_path / f"{target_frame.idx}_composited_disp_{i}.jpg")

            # Insert original frame & xattn ref frame into inpaint inputs
            logger.debug(f"Inserting original frame into inpaint inputs")
            inpainting_inputs.original_frame = target_frame
            logger.debug(f"Inserting {len(xattn_ref_frames)} xattn ref frames into inpaint inputs")
            inpainting_inputs.xattn_ref_frames = xattn_ref_frames

            # Sanity-check composited frame: image, mask and depth should be in valid ranges, and all must be finite
            sanity_check_inpaint_inputs(inpainting_inputs)
            logger.debug(
                f"Composited depth range: {inpainting_inputs.input_frame.depth_b1hw.min()} - {inpainting_inputs.input_frame.depth_b1hw.max()}"
            )

            inpainting_inputs_list.append(inpainting_inputs)

        # Inpaint
        inpainted_frame = self.inpainter.inpaint(inpainting_inputs_list, target_frame, seed)

        # Sanity-check inpainted image
        sanity_check_frame(inpainted_frame)

        # Dump inpainted frame & depth to debug path
        if self.debug_path is not None:
            inpainted_image = Image.fromarray(
                (inpainted_frame.image_bchw[0].permute(1, 2, 0).cpu().numpy() * 255).astype(
                    np.uint8
                )
            )
            inpainted_image.save(self.debug_path / f"{target_frame.idx}_inpainted.jpg")

            # inpainted_depth = Image.fromarray((inpainted_frame.depth_b1hw[0, 0].numpy() * 255).astype(np.uint8))
            inpainted_depth_img = colormap_image(inpainted_frame.depth_b1hw[0], vmin=0)
            inpainted_depth_img = Image.fromarray(
                (inpainted_depth_img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            )
            inpainted_depth_img.save(self.debug_path / f"{target_frame.idx}_inpainted_depth.jpg")

            # also show inverse depth (ie disp)
            disp = 1.0 / inpainted_frame.depth_b1hw[0]
            disp_img = colormap_image(disp, flip=False)
            disp_img = Image.fromarray(
                (disp_img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            )
            disp_img.save(self.debug_path / f"{target_frame.idx}_inpainted_disp.jpg")

        return inpainted_frame


def sanity_check_inpaint_inputs(inpainting_inputs: InpaintInputs):
    assert inpainting_inputs.mask.min() >= 0.0
    assert inpainting_inputs.mask.max() <= 1.0
    sanity_check_frame(inpainting_inputs.input_frame)


def sanity_check_frame(frame: Frame):
    assert torch.isfinite(frame.image_bchw).all()
    assert torch.isfinite(frame.depth_b1hw).all()
    assert frame.image_bchw.min() >= 0.0
    assert frame.image_bchw.max() <= 1.0
    assert frame.depth_b1hw.min() >= 0.0
