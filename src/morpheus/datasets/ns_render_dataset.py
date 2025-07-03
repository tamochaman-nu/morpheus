import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from morpheus.utils.data_utils import Frame


class NerfStudioRenderDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        renders_path: Path,
        json_path: Path,
        poses_path: Path,
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
        square_images: bool = False,
        num_leading_zeros: int = 5,
        end_index: Optional[int] = None,
        permute_frames_by: int = 0,
    ):
        """Dataset for loading images, depths, poses and intrinsics from the nerf studio renders.
        Args:
            renders_path (Path): Path to the renders folder.
            json_path (Path): Path to the json file containing the transform.
            poses_path (Path): Path to the json file containing the camera poses.
            target_width (int, optional): Target width for the images. Defaults to None.
            target_height (int, optional): Target height for the images. Defaults to None.
            square_images (bool, optional): If True, the images are resized to be square.
                Will use target_height. Defaults to False.
            num_leading_zeros (int, optional): Number of leading zeros in the image names. Defaults to 5.
            end_index (int, optional): If not None, only load images up to this index. Defaults to None.
        """
        self.num_leading_zeros = num_leading_zeros

        assert renders_path.exists(), f"renders_path {renders_path} does not exist"
        assert json_path.exists(), f"json_path {json_path} does not exist"
        assert poses_path.exists(), f"poses_path {poses_path} does not exist"

        self.renders_path = renders_path
        self.json_path = json_path
        self.poses_path = poses_path

        self.transform_json = json.load(self.json_path.open("r"))
        self.transform_144 = torch.tensor(self.transform_json["transform"]).unsqueeze(0)
        self.scale = self.transform_json["scale"]

        self.render_json = json.load(self.poses_path.open("r"))
        num_poses = len(self.render_json["camera_path"])

        # get number of images in renders path
        num_rgb_images = len(list(renders_path.glob("*_rgb.png")))
        num_depths = len(list(renders_path.glob("*.npy")))

        assert (
            num_rgb_images == num_poses
        ), f"num_rgb_images {num_rgb_images} != num_poses {num_poses}. num_depths = {num_depths}"
        assert (
            num_depths == num_poses
        ), f"num_depths {num_depths} != num_poses {num_poses}. num_rgb_images = {num_rgb_images}"

        if end_index is not None:
            assert end_index < len(
                self.render_json["camera_path"]
            ), f"end_index {end_index} is greater than or equal to the number of poses {len(self.render_json['camera_path'])}"
            self.render_json["camera_path"] = self.render_json["camera_path"][: end_index + 1]

        self.target_width = target_width
        self.target_height = target_height
        self.square_images = square_images
        self.original_height = int(self.render_json["render_height"])
        self.original_width = int(self.render_json["render_width"])

        if self.target_height is None:
            self.target_height = int(self.render_json["render_height"])

        if self.target_width is None:
            self.target_width = int(self.render_json["render_width"])

        if self.square_images:
            self.target_width = self.target_height

        self.permute_frames_by = permute_frames_by

    def load_pose(self, idx):
        """Get pose for idx from the camera path json file in opencv format."""
        cam_to_world = self.render_json["camera_path"][idx]["camera_to_world"]
        cam_to_world_44 = torch.tensor(cam_to_world).reshape(4, 4)

        gl_to_cv = torch.FloatTensor([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        cam_to_world_44 = cam_to_world_44 @ gl_to_cv

        world_to_cam_44 = torch.inverse(cam_to_world_44)

        return cam_to_world_44, world_to_cam_44

    def load_intrinsics(self, idx):
        """Load the intrinsics matrix for idx from the json file."""

        fov_x = self.render_json["camera_path"][idx]["fov"]
        fov_x = fov_x * (np.pi / 180)

        # fov to focal length
        f = self.target_height / (2 * torch.tan(torch.tensor(fov_x) / 2))

        K_44 = torch.eye(4)
        K_44[0, 0] = f
        K_44[1, 1] = f
        K_44[0, 2] = 0.5
        K_44[1, 2] = 0.5

        K_44[0, 2] *= self.target_width
        K_44[1, 2] *= self.target_height

        invK_44 = torch.inverse(K_44)

        return K_44, invK_44

    def crop_image(self, img_chw):
        if self.square_images:
            if img_chw.shape[2] > img_chw.shape[1]:
                # crop the image on width
                crop_width = img_chw.shape[1]
                crop_start = (img_chw.shape[2] - crop_width) // 2
                img_chw = img_chw[:, :, crop_start : crop_start + crop_width]

            elif img_chw.shape[1] > img_chw.shape[2]:
                # crop the image on height
                crop_height = img_chw.shape[2]
                crop_start = (img_chw.shape[1] - crop_height) // 2
                img_chw = img_chw[:, crop_start : crop_start + crop_height, :]

            else:
                # already square, leave the same.
                pass

        return img_chw

    def load_image(self, idx):
        """Load the image for idx from the renders path. Normalized to [0, 1]."""
        img_path = self.renders_path / f"{idx:0{self.num_leading_zeros}d}_rgb.png"
        img = Image.open(img_path)
        # to torch
        img_chw = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0

        img_chw = self.crop_image(img_chw)

        # resize the image
        if self.target_width != img_chw.shape[2] or self.target_height != img_chw.shape[1]:
            img_chw = torch.nn.functional.interpolate(
                img_chw.unsqueeze(0),
                size=(self.target_height, self.target_width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        return img_chw

    def load_raw_depth(self, idx):
        """Load the depth map for idx from the renders path."""
        depth_path = self.renders_path / f"{idx:0{self.num_leading_zeros}d}_depth.npy"
        depth_1hw = torch.tensor(np.load(depth_path)).unsqueeze(0).squeeze(-1)

        depth_1hw = self.crop_image(depth_1hw)

        # resize the depth
        if self.target_width != depth_1hw.shape[2] or self.target_height != depth_1hw.shape[1]:
            depth_1hw = torch.nn.functional.interpolate(
                depth_1hw.unsqueeze(0),
                size=(self.target_height, self.target_width),
                mode="bilinear",
            ).squeeze(0)

        return depth_1hw

    @lru_cache(maxsize=None)
    def load_depth(self, idx):
        # Load raw depth and convert to numpy
        # Cached because filtering is a little slow
        depth = self.load_raw_depth(idx)
        depth = depth.squeeze(0).numpy()

        return torch.tensor(depth).unsqueeze(0)

    def __len__(self):
        return len(self.render_json["camera_path"])

    def __getitem__(self, idx):
        if self.permute_frames_by > 0:
            idx = (idx + self.permute_frames_by) % len(self)
            # Must still raise an error if the index is out of bounds
            if idx >= len(self):
                raise IndexError(f"Index {idx} out of bounds for dataset of length {len(self)}")
        elem_dict = {"idx": idx}

        elem_dict["image_bchw"] = self.load_image(idx)
        elem_dict["depth_b1hw"] = self.load_depth(idx)

        K_44, invK_44 = self.load_intrinsics(idx)
        elem_dict["K_b44"] = K_44
        elem_dict["invK_b44"] = invK_44

        cam_to_world_44, world_to_cam_44 = self.load_pose(idx)
        elem_dict["cam_to_world_b44"] = cam_to_world_44
        elem_dict["world_to_cam_b44"] = world_to_cam_44

        return elem_dict

    def get_frame(self, idx):
        elem_dict = self.__getitem__(idx)

        return Frame(
            idx=idx,
            image_bchw=elem_dict["image_bchw"].unsqueeze(0),
            depth_b1hw=elem_dict["depth_b1hw"].unsqueeze(0),
            K_b44=elem_dict["K_b44"].unsqueeze(0),
            invK_b44=elem_dict["invK_b44"].unsqueeze(0),
            cam_to_world_b44=elem_dict["cam_to_world_b44"].unsqueeze(0),
            world_to_cam_b44=elem_dict["world_to_cam_b44"].unsqueeze(0),
        )
