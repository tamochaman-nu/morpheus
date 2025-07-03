# Code in this file is based on code from https://github.com/YvanYin/Metric3D/blob/main/hubconf.py
# which is licensed under the BSD 2-Clause License. The license is included below.

# BSD 2-Clause License

# Copyright (c) 2024, Wei Yin and Mu Hu

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import torch
import torch.nn.functional as F


class Metric3DPredictor:
    """
    Class to run metric3D vit large on an image, including preprocessing and postprocessing.

    This is based on the torchhub code from https://github.com/YvanYin/Metric3D/blob/main/hubconf.py
    """

    long_size: int = 1064
    short_size: int = 616

    padding_values = torch.tensor([123.675, 116.28, 103.53]).float().view(1, 3, 1, 1)
    std = torch.tensor([58.395, 57.12, 57.375]).float().view(1, 3, 1, 1)

    def __init__(self, model_name: str = "metric3d_vit_large") -> None:
        self.model = torch.hub.load("yvanyin/metric3d", model_name, pretrain=True).cuda()

    def _pad_and_resize_image(self, image_b3hw: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """
        Preprocessing for model inference

        The tranformer expects an image of size 1064x616, so we first resize the image to the long
        size and then pad it to the target size.

        Args:
            image_b3hw (torch.Tensor): The image to process, with shape Bx3xHxW

        Returns:
            padded_image_b3hw (torch.Tensor): The padded and resized image.
            pad_h_half (int): The amount of padding on the top of the image.
            pad_w_half (int) The amount of padding on the left of the image.
        """
        batch_size, _, orig_h, orig_w = image_b3hw.shape

        if orig_h >= orig_w:
            target_h = self.long_size
            target_w = self.short_size
        else:
            target_h = self.short_size
            target_w = self.long_size

        scale = min(target_h / orig_h, target_w / orig_w)

        image_b3hw = F.interpolate(
            image_b3hw,
            size=(int(orig_h * scale), int(orig_w * scale)),
            mode="bilinear",
            align_corners=False,
        )

        padding = self.padding_values.to(image_b3hw.device)
        pad_h = target_h - image_b3hw.shape[2]
        pad_w = target_w - image_b3hw.shape[3]
        pad_h_half = pad_h // 2
        pad_w_half = pad_w // 2

        padded_image_b3hw = (
            torch.zeros(
                (batch_size, 3, target_h, target_w),
                device=image_b3hw.device,
                dtype=image_b3hw.dtype,
            )
            + padding
        )
        padded_image_b3hw[
            ...,
            pad_h_half : pad_h_half + image_b3hw.shape[2],
            pad_w_half : pad_w_half + image_b3hw.shape[3],
        ] = image_b3hw

        return padded_image_b3hw, pad_h_half, pad_w_half

    def _standardize_image(self, image_b3hw: torch.Tensor) -> torch.Tensor:
        """
        Standardize the image to have a mean of 0 and std of 1
        """
        mean = self.padding_values.to(image_b3hw.device)
        std = self.std.to(image_b3hw.device)
        return (image_b3hw - mean) / std

    def _crop_padded_pixels(
        self, padded_image_bnhw: torch.Tensor, pad_h_half: int, pad_w_half: int
    ) -> torch.Tensor:
        """
        Remove the padded pixels from the prediction

        Args:
            padded_image_bnhw (torch.Tensor): The padded prediction image (depth or normals).
            pad_h_half (int): The amount of padding on the top of the image.
            pad_w_half (int) The amount of padding on the left of the image.

        Returns:
            torch.Tensor: The original prediction with the padding removed.
        """
        if pad_h_half > 0:
            padded_image_bnhw = padded_image_bnhw[..., pad_h_half:-pad_h_half, :]
        if pad_w_half > 0:
            padded_image_bnhw = padded_image_bnhw[..., pad_w_half:-pad_w_half]
        return padded_image_bnhw

    def run_inference(self, image_b3hw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Runs inference on the image, including pre- and post-processing.

        The image tensor are expected to be in the range 0-255.

        Args:
            image_b3hw (torch.Tensor): The image to predict for in the range 0-255.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The predicted depth and normal, with shapes
                Bx1xHxW and Bx3xHxW respectively. Note that these predictions are *not* resized
                back to the size of the input image, but are left at the raw network resolution.
        """
        padded_image_b3hw, pad_h_half, pad_w_half = self._pad_and_resize_image(image_b3hw)
        padded_image_b3hw = self._standardize_image(padded_image_b3hw)

        pred_depth, _, output_dict = self.model.inference({"input": padded_image_b3hw})
        pred_normal = output_dict["prediction_normal"][:, :3, :, :]

        if pad_h_half > 0 or pad_w_half > 0:
            pred_depth = self._crop_padded_pixels(pred_depth, pad_h_half, pad_w_half)
            pred_normal = self._crop_padded_pixels(pred_normal, pad_h_half, pad_w_half)

        return pred_depth, pred_normal
