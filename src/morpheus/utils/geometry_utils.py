import logging
from pathlib import Path

import kornia
import torch
import torch.jit as jit
import torch.nn.functional as F
from pytorch3d.renderer import (
    AlphaCompositor,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
)
from pytorch3d.structures import Pointclouds
from pytorch3d.utils import cameras_from_opencv_projection
from torch import Tensor, nn

from morpheus.utils.data_utils import Frame


@torch.jit.script
def to_homogeneous(input_tensor: Tensor, dim: int = 0) -> Tensor:
    """
    Converts tensor to homogeneous coordinates by adding ones to the specified
    dimension
    """
    ones = torch.ones_like(input_tensor.select(dim, 0).unsqueeze(dim))
    output_bkN = torch.cat([input_tensor, ones], dim=dim)
    return output_bkN


class BackprojectDepth(jit.ScriptModule):
    """
    Projects points from 2D camera to 3D space. The 3D points are represented in homogeneous
    coordinates.
    """

    def __init__(self, height: int, width: int):
        super().__init__()

        self.height = height
        self.width = width

        xx, yy = torch.meshgrid(
            torch.arange(self.width),
            torch.arange(self.height),
            indexing="xy",
        )
        pix_coords_2hw = torch.stack((xx, yy), axis=0) + 0.5

        pix_coords_13N = (
            to_homogeneous(
                pix_coords_2hw,
                dim=0,
            )
            .flatten(1)
            .unsqueeze(0)
        )

        # make these tensors into buffers so they are put on the correct GPU
        # automatically
        self.register_buffer("pix_coords_13N", pix_coords_13N)

    @jit.script_method
    def forward(self, depth_b1hw: Tensor, invK_b44: Tensor) -> Tensor:
        """
        Backprojects spatial points in 2D image space to world space using
        invK_b44 at the depths defined in depth_b1hw.
        """
        cam_points_b3N = torch.matmul(
            invK_b44[:, :3, :3].to(self.pix_coords_13N.device), self.pix_coords_13N
        )
        cam_points_b3N = (
            depth_b1hw.flatten(start_dim=2).to(self.pix_coords_13N.device) * cam_points_b3N
        )
        cam_points_b4N = to_homogeneous(cam_points_b3N, dim=1)
        return cam_points_b4N.to(depth_b1hw.device)


class Project3D(jit.ScriptModule):
    """
    Layer that projects 3D points into the 2D camera
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()

        self.register_buffer("eps", torch.tensor(eps).view(1, 1, 1))

    @jit.script_method
    def forward(self, points_b4N: Tensor, K_b44: Tensor, cam_T_world_b44: Tensor) -> Tensor:
        """
        Projects spatial points in 3D world space to camera image space using
        the extrinsics matrix cam_T_world_b44 and intrinsics K_b44.
        """
        P_b44 = K_b44 @ cam_T_world_b44

        cam_points_b3N = P_b44[:, :3] @ points_b4N

        # from Kornia and OpenCV, https://kornia.readthedocs.io/en/latest/_modules/kornia/geometry/conversions.html#convert_points_from_homogeneous
        mask = torch.abs(cam_points_b3N[:, 2:]) > self.eps
        depth_b1N = cam_points_b3N[:, 2:] + self.eps
        scale = torch.where(mask, 1.0 / depth_b1N, torch.tensor(1.0, device=depth_b1N.device))

        pix_coords_b2N = cam_points_b3N[:, :2] * scale

        return torch.cat([pix_coords_b2N, depth_b1N], dim=1)


class BackWarp(nn.Module):
    def __init__(self, height: int, width: int):
        super().__init__()

        self.height = height
        self.width = width

        self.backproject_depth = BackprojectDepth(height, width)
        self.project_3d = Project3D()

    def forward(self, target_frame: Frame, src_frame: Frame, padding_mode="zeros") -> Frame:
        target_to_src_b44 = torch.matmul(src_frame.world_to_cam_b44, target_frame.cam_to_world_b44)

        cam_points_b4N = self.backproject_depth(target_frame.depth_b1hw, target_frame.invK_b44)
        pix_coords_b2N = self.project_3d(cam_points_b4N, src_frame.K_b44, target_to_src_b44)[:, :2]

        pix_coords_b2N[:, 0] = 2.0 * pix_coords_b2N[:, 0] / self.width - 1.0
        pix_coords_b2N[:, 1] = 2.0 * pix_coords_b2N[:, 1] / self.height - 1.0

        pix_coords_bhw2 = pix_coords_b2N.permute(0, 2, 1).view(-1, self.height, self.width, 2)

        # project src depths into target frame to allow for computing occliusion mask
        src_to_target_b44 = torch.inverse(target_to_src_b44)
        src_points_b4N = self.backproject_depth(src_frame.depth_b1hw, src_frame.invK_b44)
        src_depth_in_target_b1hw = self.project_3d(
            src_points_b4N, src_frame.K_b44, src_to_target_b44
        )[:, 2:3].view(1, 1, self.height, self.width)

        # sample the src_frame for rgb and depth
        src_rgbd_b4hw = torch.cat(
            [src_frame.image_bchw, src_depth_in_target_b1hw],
            dim=1,
        )
        warped_rgbd_b4hw = F.grid_sample(
            src_rgbd_b4hw,
            pix_coords_bhw2,
            mode="bilinear",
            padding_mode=padding_mode,
            align_corners=False,
        )

        return Frame(
            idx=target_frame.idx,
            image_bchw=warped_rgbd_b4hw[:, :3],
            depth_b1hw=warped_rgbd_b4hw[:, 3:4],
            K_b44=target_frame.K_b44,
            invK_b44=target_frame.invK_b44,
            cam_to_world_b44=target_frame.cam_to_world_b44,
            world_to_cam_b44=target_frame.world_to_cam_b44,
        )


class ForwardWarp(nn.Module):
    def __init__(self, height: int, width: int, device):
        super().__init__()

        self.height = height
        self.width = width

        self.backproject_depth = BackprojectDepth(height, width).to(device)
        self.project_3d = Project3D().to(device)

        self.raster_settings = PointsRasterizationSettings(
            image_size=(self.height, self.width), radius=0.006, points_per_pixel=10
        )

        self.device = device

    def backproject_and_save_point_cloud(self, frame: Frame, output_path: Path):
        # Backprojects frame into 3D, transforms into the world frame, and saves a point cloud
        cam_points_b4N = self.backproject_depth(frame.depth_b1hw, frame.invK_b44)
        world_points_b4N = torch.matmul(frame.cam_to_world_b44, cam_points_b4N)

        points_bN3 = world_points_b4N[:, :3, :].permute(0, 2, 1)
        features_bN3 = frame.image_bchw.permute(0, 2, 3, 1).reshape(-1, 3).unsqueeze(0)

        # Save as ply
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_bN3[0].cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(features_bN3[0].cpu().numpy())
        o3d.io.write_point_cloud(str(output_path), pcd)

    def forward(self, target_frame: Frame, src_frame: Frame) -> Frame:
        src_to_target_b44 = torch.matmul(target_frame.world_to_cam_b44, src_frame.cam_to_world_b44)

        cam_points_b4N = self.backproject_depth(src_frame.depth_b1hw, src_frame.invK_b44)
        pix_coords_b2N = self.project_3d(cam_points_b4N, target_frame.K_b44, src_to_target_b44)[
            :, :2
        ]

        pix_coords_b2N[:, 0] = 2.0 * pix_coords_b2N[:, 0] / self.width - 1.0
        pix_coords_b2N[:, 1] = 2.0 * pix_coords_b2N[:, 1] / self.height - 1.0

        # project src depths into target frame to allow for computing occliusion mask
        src_points_b4N = self.backproject_depth(src_frame.depth_b1hw, src_frame.invK_b44)
        src_depth_in_target_b1hw = self.project_3d(
            src_points_b4N, src_frame.K_b44, src_to_target_b44
        )[:, 2:3].view(1, 1, self.height, self.width)

        def make_pytorch3d_camera(K=None, R=None, T=None):
            return cameras_from_opencv_projection(
                R=R, tvec=T, camera_matrix=K, image_size=torch.tensor([[self.height, self.width]])
            )

        target_cameras = make_pytorch3d_camera(
            K=target_frame.K_b44, R=src_to_target_b44[:, :3, :3], T=src_to_target_b44[:, :3, 3]
        )

        # Make a point cloud using backprojected depths as the positions and the src frame as the colours
        # pytorch3d wants shape [N, 3], so we have to lop off the extra 1 and permute
        points_bN3 = cam_points_b4N[:, :3, :].permute(0, 2, 1)
        # Also make some features, which will be the colours of each point (hence 3 dims)
        features_bN3 = (
            src_frame.image_bchw.permute(0, 2, 3, 1).contiguous().view(-1, 3).unsqueeze(0)
        )

        # cat the depths w/ the RGBs to make a bN4 feature map
        depths_bN1 = src_depth_in_target_b1hw.view(-1, 1).unsqueeze(0)
        features_bN4 = torch.cat([features_bN3, depths_bN1], dim=2)

        point_cloud = Pointclouds(points=points_bN3, features=features_bN4).to(self.device)

        # Create a points renderer by compositing points using an weighted compositor (3D points are
        # weighted according to their distance to a pixel and accumulated using a weighted sum)
        rasterizer = PointsRasterizer(cameras=target_cameras, raster_settings=self.raster_settings)
        renderer = PointsRenderer(
            rasterizer=rasterizer,
            compositor=AlphaCompositor(background_color=(0.0, 0.0, 1.0, torch.nan)),
        ).to(self.device)

        rendered_bhw4 = renderer(point_cloud)

        rendered_b4hw = rendered_bhw4.permute(0, 3, 1, 2)

        # extract rgb & depths
        rendered_rgb_bchw = rendered_b4hw[:, :3, :, :]
        rendered_depth_b1hw = rendered_b4hw[:, 3:4, :, :]

        return Frame(
            idx=target_frame.idx,
            image_bchw=rendered_rgb_bchw,
            depth_b1hw=rendered_depth_b1hw,
            K_b44=target_frame.K_b44,
            invK_b44=target_frame.invK_b44,
            cam_to_world_b44=target_frame.cam_to_world_b44,
            world_to_cam_b44=target_frame.world_to_cam_b44,
        )


class NormalGenerator(nn.Module):
    def __init__(
        self,
        height: int,
        width: int,
    ):
        """
        Estimates normals from depth maps.
        """
        super().__init__()
        self.height = height
        self.width = width

        self.backproject = BackprojectDepth(self.height, self.width)

    def forward(self, depth_b1hw: torch.Tensor, invK_b44: torch.Tensor) -> torch.Tensor:
        cam_points_b4N = self.backproject(depth_b1hw, invK_b44)
        cam_points_b3hw = cam_points_b4N[:, :3].view(-1, 3, self.height, self.width)

        gradients_b32hw = kornia.filters.spatial_gradient(cam_points_b3hw)

        return F.normalize(
            torch.cross(
                gradients_b32hw[:, :, 0],
                gradients_b32hw[:, :, 1],
                dim=1,
            ),
            dim=1,
        )


class DepthNormalizerBase:
    is_absolute = None
    far_plane_at_max = None

    def __init__(
        self,
        norm_min=-1.0,
        norm_max=1.0,
    ) -> None:
        self.norm_min = norm_min
        self.norm_max = norm_max
        raise NotImplementedError

    def __call__(self, depth, valid_mask=None, clip=None):
        raise NotImplementedError

    def denormalize(self, depth_norm, **kwargs):
        # For metric depth: convert prediction back to metric depth
        # For relative depth: convert prediction to [0, 1]
        raise NotImplementedError


class ScaleShiftDepthNormalizer(DepthNormalizerBase):
    """
    Use near and far plane to linearly normalize depth,
        i.e. d' = d * s + t,
        where near plane is mapped to `norm_min`, and far plane is mapped to `norm_max`
    Near and far planes are determined by taking quantile values.
    """

    is_absolute = False
    far_plane_at_max = True

    def __init__(self, norm_min=-1.0, norm_max=1.0, min_max_quantile=0.02, clip=True) -> None:
        self.norm_min = norm_min
        self.norm_max = norm_max
        self.norm_range = self.norm_max - self.norm_min
        self.min_quantile = min_max_quantile
        self.max_quantile = 1.0 - self.min_quantile
        self.clip = clip

    def __call__(self, depth_linear, valid_mask=None, clip=None):
        clip = clip if clip is not None else self.clip

        if valid_mask is None:
            valid_mask = torch.ones_like(depth_linear).bool()
        valid_mask = valid_mask & (depth_linear > 0)

        # Torch quantile is limited to 16 million elements
        # Handle case where valid mask is all False (which otherwise crashes torch.quantile)
        if not valid_mask.any():
            logging.warning("All elements are invalid in depth map.")
            _min, _max = torch.tensor([0.0, 1.0])

        elif depth_linear.numel() > 16e6:
            logging.warning(f"Too many elements ({depth_linear.numel()}) for torch.quantile. ")
            # Subsample both depth & mask by 4x (should be enough unless the image is stupidly big)
            subsample = 4
            depth_subsampled = depth_linear[::subsample, ::subsample]
            valid_mask_subsampled = valid_mask[::subsample, ::subsample]
            _min, _max = torch.quantile(
                depth_subsampled[valid_mask_subsampled],
                torch.tensor([self.min_quantile, self.max_quantile]),
            )

        else:
            # Take quantiles as min and max
            _min, _max = torch.quantile(
                depth_linear[valid_mask],
                torch.tensor([self.min_quantile, self.max_quantile]),
            )

        # If _max == _min, then artificially widen their range to prevent numerical issues
        if _max == _min:
            _max += 0.1
            _min -= 0.1

        depth_norm_linear = (depth_linear - _min) / (_max - _min) * self.norm_range + self.norm_min

        if clip:
            depth_norm_linear = torch.clip(depth_norm_linear, self.norm_min, self.norm_max)

        return depth_norm_linear

    def scale_back(self, depth_norm):
        # scale to [0, 1]
        depth_linear = (depth_norm - self.norm_min) / self.norm_range
        return depth_linear

    def denormalize(self, depth_norm, **kwargs):
        logging.warning(f"{self.__class__} is not revertible without GT")
        return self.scale_back(depth_norm=depth_norm)
