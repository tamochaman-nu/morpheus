from pathlib import Path

import pytorch3d
import pytorch3d.renderer
import torch
from pytorch3d.renderer import MeshRendererWithFragments
from pytorch3d.utils import cameras_from_opencv_projection
from torch import nn

from morpheus.utils.data_utils import Frame
from morpheus.utils.geometry_utils import BackprojectDepth, Project3D


def check_face_validity(
    faces_verts_f3N: torch.Tensor, faces_3N: torch.Tensor, min_angle: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Allows us to identify faces that are being seen very obliquely. We want to discard these faces,
    so we identify them as 'valid' in the returned mask if their angle to the ray from the
    camera to the face is greater than min_angle.
    Args:
        faces_verts_f3N: (3, 3, N) tensor of face vertices in camera space
        faces_3N: (3, N) tensor of face indices
        min_angle: minimum angle in degrees
    Returns:
        faces_3N: (3, M) tensor of face indices, discarding invalid faces
        angle_N: (M) tensor of angles in degrees, discarding invalid faces
    """
    v1_3N = faces_verts_f3N[0, :, :] - faces_verts_f3N[1, :, :]
    v2_3N = faces_verts_f3N[0, :, :] - faces_verts_f3N[2, :, :]
    n_3N = torch.cross(v1_3N, v2_3N, axis=0)
    n_3N /= torch.linalg.norm(n_3N, axis=0)

    center_3N = faces_verts_f3N.mean(0)
    u_3N = center_3N / torch.linalg.norm(center_3N, axis=0)
    dot_prod_N = (u_3N * n_3N).sum(0)
    angle_N = torch.rad2deg(torch.arcsin(torch.abs(dot_prod_N)))
    mask_N = angle_N > min_angle
    return faces_3N[:, mask_N], angle_N[mask_N]


def depth_to_mesh(
    depth_1hw: torch.Tensor, K_inv_44: torch.Tensor, image_3hw: torch.Tensor, min_angle: float = 1.0
) -> tuple[pytorch3d.structures.Meshes, torch.Tensor]:
    """
    Converts a depth map to a mesh using the camera intrinsics and the image.

    Args:
        depth_1hw: (1, H, W) tensor of depth values
        K_inv_44: (4, 4) tensor of camera intrinsics
        image_3hw: (3, H, W) tensor of image colors
        min_angle: minimum angle in degrees for face validity
    Returns:
        meshes: pytorch3d Meshes object
        valid_angles_N: (N) tensor of angles in degrees for each face
    """

    depth_hw = depth_1hw[0]

    height, width = depth_hw.shape
    backprojector = BackprojectDepth(height=height, width=width).to(depth_1hw.device)

    # cam points are our verts
    cam_points_3N = backprojector(depth_hw[None, None], K_inv_44[None])[0][:3]

    # ij for the vectorized for loop's indices into vertices down below.
    ii, jj = torch.tensor(list(range(height - 1))).to(depth_1hw.device), torch.tensor(
        list(range(width - 1))
    ).to(depth_1hw.device)
    II, JJ = torch.meshgrid(ii, jj, indexing="ij")

    faces_verts1_f3N = torch.stack(
        [
            cam_points_3N[:, (II * width + JJ).flatten()],
            cam_points_3N[:, ((II + 1) * width + JJ).flatten()],
            cam_points_3N[:, (II * width + (JJ + 1)).flatten()],
        ],
        0,
    )
    faces1_3N = torch.stack(
        [
            (II * width + JJ).flatten(),
            ((II + 1) * width + JJ).flatten(),
            (II * width + (JJ + 1)).flatten(),
        ],
        0,
    )
    faces_verts2_f3N = torch.stack(
        [
            cam_points_3N[:, (II * width + JJ + 1).flatten()],
            cam_points_3N[:, ((II + 1) * width + JJ).flatten()],
            cam_points_3N[:, ((II + 1) * width + (JJ + 1)).flatten()],
        ],
        0,
    )
    faces2_3N = torch.stack(
        [
            (II * width + JJ + 1).flatten(),
            ((II + 1) * width + JJ).flatten(),
            ((II + 1) * width + (JJ + 1)).flatten(),
        ],
        0,
    )

    faces_verts_f3N = torch.cat([faces_verts1_f3N, faces_verts2_f3N], dim=2)
    _faces_3N = torch.cat([faces1_3N, faces2_3N], dim=1)

    # trim away invalid faces by checking if their normals are pointing in a direction
    # perpendicular to the camera's lookat direction.
    valid_faces_3N, valid_angles_N = check_face_validity(faces_verts_f3N, _faces_3N, min_angle)

    vert_colors_3N = image_3hw.flatten(1)

    # to pytorch3d mesh
    textures = pytorch3d.renderer.TexturesVertex(verts_features=vert_colors_3N.permute(1, 0)[None])
    meshes = pytorch3d.structures.Meshes(
        verts=[cam_points_3N.permute(1, 0)], faces=[valid_faces_3N.permute(1, 0)], textures=textures
    )

    return meshes, valid_angles_N


class MeshWarp(nn.Module):
    """
    Warps a frame to a new camera view by converting the RGBD to a mesh and rendering it into the
    new view.
    """

    def __init__(self, height: int, width: int, device, min_angle=8.0, deterministic=True):
        """
        Initialise the MeshWarp.
        Args:
            height: height of the image
            width: width of the image
            device: device to use for rendering
            min_angle: minimum angle in degrees for face validity
            deterministic: If False, pytorch3d will use a rendering strategy that is faster but not
                deterministic due to parallelism.
        """
        super().__init__()

        self.height = height
        self.width = width

        self.backproject_depth = BackprojectDepth(height, width)
        self.project_3d = Project3D()

        self.raster_settings = pytorch3d.renderer.RasterizationSettings(
            image_size=(self.height, self.width),
            blur_radius=0.00001,
            faces_per_pixel=1,
            cull_backfaces=True,
            **({"bin_size": 0} if deterministic else {}),
        )

        self.device = device
        self.min_angle = min_angle

    def forward(self, target_frame: Frame, src_frame: Frame) -> Frame:
        # Move to device
        src_frame = src_frame.to(self.device)
        target_frame = target_frame.to(self.device)
        src_to_target_b44 = torch.matmul(target_frame.world_to_cam_b44, src_frame.cam_to_world_b44)

        meshes, valid_angles_N = depth_to_mesh(
            depth_1hw=src_frame.depth_b1hw[0],
            K_inv_44=src_frame.invK_b44[0],
            image_3hw=src_frame.image_bchw[0],
            min_angle=self.min_angle,
        )

        def make_pytorch3d_camera(K=None, R=None, T=None):
            return cameras_from_opencv_projection(
                R=R, tvec=T, camera_matrix=K, image_size=torch.tensor([[self.height, self.width]])
            )

        target_cameras = make_pytorch3d_camera(
            K=target_frame.K_b44, R=src_to_target_b44[:, :3, :3], T=src_to_target_b44[:, :3, 3]
        )
        # Create a mesh renderer
        renderer = MeshRendererWithFragments(
            rasterizer=pytorch3d.renderer.MeshRasterizer(
                cameras=target_cameras, raster_settings=self.raster_settings
            ),
            shader=pytorch3d.renderer.SoftPhongShader(
                device=self.device,
                cameras=target_cameras,
                lights=pytorch3d.renderer.lighting.AmbientLights().to(self.device),
            ),
        )

        rendered_bhw4, fragments = renderer(meshes)
        depth_bhw1 = fragments.zbuf

        rendered_b4hw = rendered_bhw4.permute(0, 3, 1, 2)
        depth_b1hw = depth_bhw1.permute(0, 3, 1, 2)

        # extract rgb, depth, and alpha mask.
        rendered_rgb_bchw = rendered_b4hw[:, :3, :, :]
        rendered_depth_b1hw = depth_b1hw
        rendered_alpha_b1hw = rendered_b4hw[:, 3:, :, :]

        # mirroring forward warper. add blue for invalid
        rendered_rgb_bchw[:, 0:1, :, :][(rendered_depth_b1hw == -1)] = 0
        rendered_rgb_bchw[:, 1:2, :, :][(rendered_depth_b1hw == -1)] = 0
        rendered_rgb_bchw[:, 2:3, :, :][(rendered_depth_b1hw == -1)] = 1

        rendered_depth_b1hw[rendered_depth_b1hw == -1] = torch.nan

        # Get face idxs for each rendered px
        face_idxs_bhwc = fragments.pix_to_face

        assert (face_idxs_bhwc < valid_angles_N.shape[0]).all()

        # Expect valid angles to be in range [0, 180]
        assert (valid_angles_N >= 0).all()

        valid_angles_N = valid_angles_N[face_idxs_bhwc.flatten()].reshape(*face_idxs_bhwc.shape)

        # Where there are no faces, set angle to 0 degrees
        valid_angles_N[face_idxs_bhwc == -1] = 0

        # Take a mean over faces where one px has multiple faces
        valid_angles_N = torch.mean(valid_angles_N, dim=-1, keepdim=True)

        return Frame(
            idx=target_frame.idx,
            image_bchw=rendered_rgb_bchw,
            depth_b1hw=rendered_depth_b1hw,
            K_b44=target_frame.K_b44,
            invK_b44=target_frame.invK_b44,
            cam_to_world_b44=target_frame.cam_to_world_b44,
            world_to_cam_b44=target_frame.world_to_cam_b44,
            debug_dict={
                "mesh_warper_rendered_alpha_b1hw": rendered_alpha_b1hw,
                "mesh_warper_angles_b1hw": valid_angles_N.permute(0, 3, 1, 2),
            },
        )

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
