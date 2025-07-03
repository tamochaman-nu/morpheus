import copy
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from loguru import logger
from PIL import Image

from morpheus.utils.data_utils import Frame
from morpheus.utils.filtering import median_filter
from morpheus.utils.mesh_utils import MeshWarp
from morpheus.visualization.image_viz import colormap_image
from morpheus.xattn_controlnet import InpaintInputs


class ForwardWarpCompositor:
    """
    Compositor that uses fwd warping to go from ref frames to current frames.
    """

    def __init__(
        self,
        mesh_warper: MeshWarp,
        composite_inv_temperature: float = 10.0,
        frame_oldness_weight: float = 5e-5,
        depth_weight: float = 10.0,
        debug_path: Optional[Path] = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.forward_warper = mesh_warper
        self.device = device
        self.debug_path = debug_path
        self.composite_inv_temperature = composite_inv_temperature
        self.frame_oldness_weight = frame_oldness_weight
        self.depth_weight = depth_weight

    def composite_for_inpainting(
        self,
        src_frames_raw: list[Frame],
        src_frames_stylised: list[Frame],
        target_frame_raw: Frame,
    ) -> InpaintInputs:
        """
        Composites the source frames together using the compositing scores. Overall this function will:
        - Forward-warp the source frames to the target frame, bringing the target frame and source frames
            into the same space
        - Composite together the warped source frames and the target frame. Where a given pixel could be
            filled by multiple source frames, compositing scores will be used to determine which frame
            should be used. The scores take into account e.g. occlusions and the likelihood that the warp
            has introduced artifacts.

        Args:
            src_frames_raw (list[Frame]): The source frames to be used for compositing
            src_frames_stylised (list[Frame]): The source frames to be used for compositing
            target_frame_raw (Frame): The target frame to be used for compositing
        Returns:
            InpaintInputs: The composited frame and the inpainting mask
        """
        # Move everything to device
        target_frame_raw = target_frame_raw.to(self.device)
        src_frames_raw = [frame.to(self.device) for frame in src_frames_raw]
        src_frames_stylised = [frame.to(self.device) for frame in src_frames_stylised]

        if self.debug_path is not None:
            for src_frame in src_frames_raw:
                image = Image.fromarray(
                    (src_frame.image_bchw[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                )
                image.save(
                    self.debug_path / f"{target_frame_raw.idx}_{src_frame.idx}_src_start.jpg"
                )

        # Sanity check - verify that the i-th stylized frame corresponds to the i-th raw frame
        for src_frame, src_frame_stylised in zip(src_frames_raw, src_frames_stylised):
            assert src_frame.idx == src_frame_stylised.idx

        # Filter on the depth map before warping
        def filter_depth_for_frame(frame: Frame) -> Frame:
            new_frame = copy.deepcopy(frame)
            new_frame.depth_b1hw = median_filter(new_frame)
            return new_frame

        src_frames_stylised = [filter_depth_for_frame(frame) for frame in src_frames_stylised]

        # Dump filtered colourised depths to disk, using colormap_image
        if self.debug_path is not None:
            for src_frame in src_frames_stylised:
                image = Image.fromarray(
                    (
                        colormap_image(src_frame.depth_b1hw[0]).cpu().permute(1, 2, 0).numpy() * 255
                    ).astype(np.uint8)
                )
                image.save(
                    self.debug_path
                    / f"{target_frame_raw.idx}_{src_frame.idx}_src_filtered_depth.png"
                )

        # Warp src frames to target
        src_frames_warped_to_target = [
            self.forward_warper(target_frame_raw.to("cuda"), src_frame_stylised.to("cuda")).to(
                self.device
            )
            for src_frame_stylised in src_frames_stylised
        ]

        # Compute compositing scores for each source frame, which will tell us which frames should
        # win when multiple frames warp to the same pixel:
        compositing_scores_b1hw = [
            self.compute_compositing_score(
                src_frame_warped_to_target=src_frame_warped_to_target,
                source_frame=src_frame_stylised,
                target_frame=target_frame_raw,
            )
            for src_frame_stylised, src_frame_warped_to_target in zip(
                src_frames_stylised, src_frames_warped_to_target
            )
        ]

        # Use the inpainting mask to composite the warped stylised reference frame with the raw RGB of the target frame
        composited_frame, final_inpainting_mask = self.composite_frames_with_compositing_scores(
            target_frame=target_frame_raw,
            compositing_scores_b1hw=compositing_scores_b1hw,
            src_frames_stylised=src_frames_warped_to_target,
        )

        # Pack the composited RGB and depth into a single frame object
        composited_frame = Frame(
            idx=target_frame_raw.idx,
            image_bchw=composited_frame.image_bchw,
            depth_b1hw=composited_frame.depth_b1hw,
            K_b44=target_frame_raw.K_b44,
            invK_b44=target_frame_raw.invK_b44,
            cam_to_world_b44=target_frame_raw.cam_to_world_b44,
            world_to_cam_b44=target_frame_raw.world_to_cam_b44,
        )

        assert composited_frame.image_bchw.min() >= 0.0
        # Occasional floating point issue so adding tolerance to <=1 check.
        eps = 1e-4
        assert composited_frame.image_bchw.max() <= 1.0 + eps
        composited_frame.image_bchw = torch.clamp(composited_frame.image_bchw, 0.0, 1.0)

        if self.debug_path is not None:
            input_image = Image.fromarray(
                (composited_frame.image_bchw[0].cpu().permute(1, 2, 0).numpy() * 255).astype(
                    np.uint8
                )
            )
            input_image.save(self.debug_path / f"{target_frame_raw.idx}_input.jpg")

            inpainting_mask = Image.fromarray(
                (final_inpainting_mask[0, 0].cpu().numpy() * 255).astype(np.uint8)
            )
            inpainting_mask.save(self.debug_path / f"{target_frame_raw.idx}_inpainting_mask.jpg")

        return InpaintInputs(
            input_frame=Frame(
                idx=target_frame_raw.idx,
                image_bchw=composited_frame.image_bchw,
                depth_b1hw=composited_frame.depth_b1hw,
                K_b44=target_frame_raw.K_b44,
                invK_b44=target_frame_raw.invK_b44,
                cam_to_world_b44=target_frame_raw.cam_to_world_b44,
                world_to_cam_b44=target_frame_raw.world_to_cam_b44,
            ),
            mask=final_inpainting_mask,
            original_frame=target_frame_raw,
        )

    def composite_frames_with_compositing_scores(
        self,
        target_frame: Frame,
        compositing_scores_b1hw: list[torch.Tensor],
        src_frames_stylised: list[Frame],
    ) -> Tuple[Frame, torch.Tensor]:
        # Each pixel in the composited image will be a weighted sum of the pixel from the target
        # frame and warped pixels from the source frames (where available).
        # The compositing scores are used to weight the contribution of frame to the composite.

        # Give the target frame a constant score of zero everywhere:
        target_compositing_score = torch.ones_like(compositing_scores_b1hw[0]) * 0.0
        compositing_scores_b1hw.append(target_compositing_score)

        # To simplify things, add a copy of the target frame to the list of source frames. This will
        # allow the target frame to contribute to the composite without needing any special handling.
        target_pseudo_frame = copy.deepcopy(target_frame).to(
            src_frames_stylised[0].image_bchw.device
        )
        src_frames_stylised.append(target_pseudo_frame)

        # Make compositing scores into a tensor, and take the softmax over the img dimension to get
        # weights for each source frame:
        compositing_scores_b1hw = torch.cat(compositing_scores_b1hw, dim=0)
        alphas_b1hw = torch.nn.functional.softmax(
            self.composite_inv_temperature * compositing_scores_b1hw, dim=0
        )

        # Dump alphas for each image as greyscale images
        if self.debug_path is not None:
            for idx, alpha in enumerate(alphas_b1hw):
                alpha_image = Image.fromarray((alpha[0].cpu().numpy() * 255).astype(np.uint8))
                alpha_image.save(self.debug_path / f"{target_frame.idx}_composite_alpha_{idx}.png")

        # Smooth the compositing scores so that we don't get noise where they vary a lot from pixel to pixel
        validity_mask = torch.isfinite(compositing_scores_b1hw)

        # For all frames but the target frame, set the validity mask to zero in areas where the target frame has the
        # highest compositing score
        # This helps keep a hard boundary between the target frame and the other frames
        target_compositing_score = compositing_scores_b1hw[-1]

        # Maintain a hard edge between target & non-target:
        # where target frame has alpha > 0.5, invalidate all the other frames;
        # and where the target frame has alpha < 0.5, invalidate the target frame.
        # This helps to avoid situations where, if target & non-target happen to score similarly, we
        # get an ugly and incoherent mix of both of them as input.
        target_frame_wins = alphas_b1hw[-1] > 0.5
        validity_mask[:-1] = validity_mask[:-1] & ~target_frame_wins
        validity_mask[-1] = validity_mask[-1] & target_frame_wins

        # Dump smoothed alphas
        if self.debug_path is not None:
            for idx, alpha in enumerate(alphas_b1hw):
                alpha_image = Image.fromarray((alpha[0].cpu().numpy() * 255).astype(np.uint8))
                alpha_image.save(
                    self.debug_path / f"{target_frame.idx}_composite_alpha_{idx}_filtered.png"
                )

        # Move to the device of the src frames
        compositing_scores_b1hw = alphas_b1hw.to(src_frames_stylised[0].image_bchw.device)

        # Now create a composite frame using the compositing scores
        composited_rgb = torch.zeros_like(
            src_frames_stylised[0].image_bchw, device=src_frames_stylised[0].image_bchw.device
        )
        composited_depth = torch.zeros_like(
            src_frames_stylised[0].depth_b1hw, device=src_frames_stylised[0].depth_b1hw.device
        )
        for idx, (src_frame_stylised, compositing_score) in enumerate(
            zip(src_frames_stylised, compositing_scores_b1hw)
        ):
            if self.debug_path is not None:
                # Save the src frame backwarped to the target, with the mask applied
                src_image = compositing_score * src_frame_stylised.image_bchw
                src_image = Image.fromarray(
                    (src_image[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                )
                src_image.save(self.debug_path / f"{target_frame.idx}_composite_src_{idx}.jpg")

                # Also save the depth
                src_depth = compositing_score * src_frame_stylised.depth_b1hw
                src_depth = Image.fromarray((src_depth[0, 0].cpu().numpy() * 255).astype(np.uint8))
                src_depth.save(
                    self.debug_path / f"{target_frame.idx}_composite_src_depth_{idx}.jpg"
                )

            # Make sanitised versions of the depth & rgb where values for which the alpha weight is zero are set to zero
            # unsqueeze to (b, 1, h, w) for broadcasting
            validity_mask = (compositing_score > 0.0).unsqueeze(0)
            # now to (b, 3, h, w) for broadcasting
            sanitised_rgb = src_frame_stylised.image_bchw.clone()
            sanitised_rgb[~validity_mask.expand_as(sanitised_rgb)] = 0.0
            sanitised_depth = src_frame_stylised.depth_b1hw.clone()
            sanitised_depth[~validity_mask] = 0.0
            composited_rgb += compositing_score * sanitised_rgb
            composited_depth += compositing_score * sanitised_depth

        composited_frame = Frame(
            idx=target_frame.idx,
            image_bchw=composited_rgb,
            depth_b1hw=composited_depth,
            K_b44=target_frame.K_b44,
            invK_b44=target_frame.invK_b44,
            cam_to_world_b44=target_frame.cam_to_world_b44,
            world_to_cam_b44=target_frame.world_to_cam_b44,
        )

        # Also compute & return the summed weights for the stylized frames, i.e. the inpainting mask
        # This will be fed into the controlnet (although it will be binarized first)
        final_inpainting_mask = 1.0 - compositing_scores_b1hw[-1:]

        return composited_frame, final_inpainting_mask

    def compute_compositing_score(
        self, src_frame_warped_to_target: Frame, source_frame: Frame, target_frame: Frame
    ) -> torch.Tensor:
        """
        Compute a compositing score for each pixel in a frame. This score will then be used by the forward
        warper to determine which frames to use in the composite. This score will reflect multiple
        factors, including:
        - The angle of the mesh warper at each pixel (i.e. how much the pixel is warped);
            less is better, because more extreme warps cause more artifacts
        - The depth of the pixel in the source frame relative to the target frame;
            closer pixels should occlude more distant ones
        - The age of the source frame relative to the target frame;
            older frames should be preferred to newer ones, to avoid repeated warping of material

        Args:
            src_frame_warped_to_target (Frame): The source frame warped to the target frame
            source_frame (Frame): The original (unwarped) source frame
            target_frame (Frame): The target frame
        Returns:
            compositing_scores_b1hw (torch.Tensor): The compositing scores for each pixel in the
                source frame
        """
        # Assign the compositing score for each px
        mesh_warper_face_angles_deg_b1hw = src_frame_warped_to_target.debug_dict[
            "mesh_warper_angles_b1hw"
        ].to(self.device)
        mesh_warper_face_angles_rad_b1hw = mesh_warper_face_angles_deg_b1hw * np.pi / 180.0
        compositing_scores_b1hw = torch.abs(torch.sin(mesh_warper_face_angles_rad_b1hw))

        # Also incorporate a contribution from depth - closer things should occlude more distant ones
        # (Nans may get introduced here if the warp has produced non-positive definite depth values)
        # Measure this relative to the target depth, i.e. frames get increasingly penalised if they are behind the tgt
        compositing_scores_b1hw -= self.depth_weight * (
            src_frame_warped_to_target.depth_b1hw / target_frame.depth_b1hw - 1.0
        )

        # Put in contribution from frame oldness - older frames are slightly preferred so as to avoid repeated warping of material
        # technically idx should be normalised by num previously generated frames, but in practice it's always 20...
        logger.info(f"Target frame idx: {target_frame.idx}, Source frame idx: {source_frame.idx}")
        compositing_scores_b1hw += self.frame_oldness_weight * (target_frame.idx - source_frame.idx)

        # Impose validity of the depth map - this compels softmax to assign weights of zero to px that are invalid
        frame_validity = get_validity_for_frame(src_frame_warped_to_target)
        compositing_scores_b1hw[~frame_validity] = -np.inf

        if self.debug_path is not None:
            if True:
                # Dump a bunch of helpful debug images to disk
                warped_image = Image.fromarray(
                    (
                        src_frame_warped_to_target.image_bchw[0].cpu().permute(1, 2, 0).numpy()
                        * 255
                    ).astype(np.uint8)
                )
                warped_image.save(
                    self.debug_path
                    / f"{target_frame.idx}_{src_frame_warped_to_target.idx}_warped.png"
                )
                # Save the target frame
                target_image = Image.fromarray(
                    (target_frame.image_bchw[0].cpu().permute(1, 2, 0).numpy() * 255).astype(
                        np.uint8
                    )
                )
                target_image.save(self.debug_path / f"{target_frame.idx}_target.png")
                # Save the warped depth
                warped_depth_image = Image.fromarray(
                    (src_frame_warped_to_target.depth_b1hw[0, 0].cpu().numpy() * 255).astype(
                        np.uint8
                    )
                )
                warped_depth_image.save(
                    self.debug_path
                    / f"{target_frame.idx}_{src_frame_warped_to_target.idx}_warped_depth.png"
                )
                # Colormap the compositing scores
                compositing_score_image = Image.fromarray(
                    (
                        colormap_image(compositing_scores_b1hw[0])
                        .cpu()
                        .permute(1, 2, 0)
                        .cpu()
                        .numpy()
                        * 255
                    ).astype(np.uint8)
                )
                compositing_score_image.save(
                    self.debug_path
                    / f"{target_frame.idx}_{src_frame_warped_to_target.idx}_compositing_score.png"
                )

        return compositing_scores_b1hw


def get_validity_for_frame(frame: Frame) -> torch.Tensor:
    # Frames are defined as valid if their depth map is finite and positive-definite
    # Any pixels where this isn't true are invalid
    # Where a frame is the result of a forward warp, invalid pixels may appear where nothing warped
    # to a particular pixel, and this function allows us to ensure that we don't use these pixels in
    # the composite.
    return torch.isfinite(frame.depth_b1hw) & (frame.depth_b1hw > 0.0)
