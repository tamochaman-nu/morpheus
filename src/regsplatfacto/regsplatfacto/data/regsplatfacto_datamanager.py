from dataclasses import dataclass, field
from typing import Type

from nerfstudio.data.datamanagers.full_images_datamanager import (
    FullImageDatamanager,
    FullImageDatamanagerConfig,
)

from regsplatfacto.data.metric3d_dataset import Metric3dDataset


@dataclass
class RegSplatfactoDatamanagerConfig(FullImageDatamanagerConfig):
    _target: Type = field(default_factory=lambda: RegSplatfactoDatamanager)


class RegSplatfactoDatamanager(FullImageDatamanager):
    """
    Data manager for RegSplatfacto, whis is used to force the dataset to be Metric3dDataset.

    This data manager should be used in the config class when running RegSplatfacto.
    """

    config: RegSplatfactoDatamanagerConfig
    train_dataset: Metric3dDataset
    eval_dataset: Metric3dDataset

    @property
    def dataset_type(self):
        return Metric3dDataset
