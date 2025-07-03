import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset

from regsplatfacto.data.metric3d_predictor import Metric3DPredictor


class Metric3dDataset(InputDataset):
    """
    A nerfstudio Dataset which additionally returns depth and normal estimates from Metric3D.

    This will attempt to load the estimates from disk, but if no estimates can be found, it will
    predict them and then save to disk for future iterations.
    """

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0) -> None:
        super().__init__(dataparser_outputs=dataparser_outputs, scale_factor=scale_factor)

        self.depth_and_normal_predictor: Metric3DPredictor | None = None

    def _predict_depth_and_normal(self, image_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Predicts a depth and normal map for a single image in the dataset.

        Args:
            image_idx (int): The index of the image to predict for.

        Returns:
            tuple[np.ndarray, np.ndarray]: The predicted depth and normal as numpy arrays.
        """

        if self.depth_and_normal_predictor is None:
            self.depth_and_normal_predictor = Metric3DPredictor()

        image_13hw = self.get_image_float32(image_idx).permute(2, 0, 1).unsqueeze(0).cuda()
        depth_11hw, normal_11hw = self.depth_and_normal_predictor.run_inference(image_13hw * 255.0)

        return depth_11hw.cpu().numpy().squeeze(0), normal_11hw.cpu().numpy().squeeze(0)

    def _get_depth_and_normal_estimate(self, image_idx: int) -> dict[str, torch.Tensor]:
        """
        Returns the depth and normal estimates for a single image in the dataset. This will
        attempt to load from disk, but if no estimates are found, it will predict them and save
        to disk.

        Args:
            image_idx (int): The index of the image to get the estimates for.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing the depth and normal estimates, ready
                to be inserted into the dictionary returned from `super().get_data(...)`.
        """
        image_path = Path(self._dataparser_outputs.image_filenames[image_idx])
        depth_path = image_path.parent.parent / "regsplatfacto_depths" / f"{image_path.stem}.npz"

        if depth_path.exists():
            data = np.load(depth_path)
            depth_11hw = data["depth"]
            normal_13hw = data["normal"]

        else:
            warnings.warn(
                f"Depth and normal estimates were not found in {depth_path.parent}.\n"
                f"We will now run inference and save to disk at {depth_path.parent}."
            )
            depth_11hw, normal_13hw = self._predict_depth_and_normal(image_idx)
            depth_path.parent.mkdir(exist_ok=True, parents=True)
            np.savez(depth_path, depth=depth_11hw, normal=normal_13hw)

        # convert normal to 0-1 and nerfstudio convention
        normal_13hw = 1 - (normal_13hw + 1) / 2

        return {"depth": torch.tensor(depth_11hw), "normal": torch.tensor(normal_13hw)}

    def get_data(self, image_idx: int, image_type: Literal["uint8", "float32"] = "float32") -> dict:
        """Returns the ImageDataset data as a dictionary.

        Args:
            image_idx: The image index in the dataset.
            image_type: the type of images returned
        """
        data = super().get_data(image_idx=image_idx, image_type=image_type)
        data.update(self._get_depth_and_normal_estimate(image_idx))
        return data
