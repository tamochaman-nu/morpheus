import pathlib

import open3d as o3d
import torch

from morpheus.utils.data_utils import Frame
from morpheus.utils.geometry_utils import BackprojectDepth


class ResplattingPointCloud:
    """Point cloud used for initialisation of gaussians during re-splatting."""

    def __init__(self, target_width: int, target_height: int) -> None:
        """Initialise the point cloud.

        :param target_width: width of the target image (used by back-projection)
        :param target_height: height of the target image (used by back-projection)
        """
        self.backproject_depth = BackprojectDepth(
            width=target_width,
            height=target_height,
        )
        self.point_cloud = o3d.geometry.PointCloud()
        self.point_cloud.points = o3d.utility.Vector3dVector([])
        self.point_cloud.colors = o3d.utility.Vector3dVector([])

    def add_frame(self, frame: Frame) -> None:
        """Add a frame to the point cloud.

        :param frame: frame to be added to the point cloud
        """
        cam_depth_b4n = self.backproject_depth(frame.depth_b1hw, frame.invK_b44)
        world_depth_b4n = torch.matmul(frame.cam_to_world_b44, cam_depth_b4n)

        points_n3 = world_depth_b4n[0, :3].numpy().transpose(1, 0)
        self.point_cloud.points.extend(points_n3.astype("float64"))

        colors_n3 = frame.image_bchw[0].numpy().reshape(3, -1).transpose(1, 0)
        self.point_cloud.colors.extend(colors_n3.astype("float64"))

    def export_to_ply(
        self,
        output_data_path: pathlib.Path,
        voxel_size: float = 0.015,
        outliers_removal_nb_neighbors: int = 20,
        outliers_removal_std_ratio: float = 2.0,
    ) -> None:
        """Export the point cloud to a PLY file.

        :param output_data_path: path to the output data directory
        :param voxel_size: voxel size for down-sampling
        :param outliers_removal_nb_neighbors: number of neighbors for outlier removal
        :param outliers_removal_std_ratio: standard deviation ratio for outlier removal
        """
        down_sampled_pcd = self.point_cloud.voxel_down_sample(voxel_size=voxel_size)
        cleaned_pcd, _ = down_sampled_pcd.remove_statistical_outlier(
            nb_neighbors=outliers_removal_nb_neighbors,
            std_ratio=outliers_removal_std_ratio,
        )
        o3d.io.write_point_cloud(str(output_data_path / "point_cloud.ply"), cleaned_pcd)
