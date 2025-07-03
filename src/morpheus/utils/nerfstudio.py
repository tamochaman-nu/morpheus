import json
import pathlib
import subprocess
from typing import Optional

import numpy as np
import torch
from loguru import logger
from PIL import Image

from morpheus.utils.data_utils import Frame
from morpheus.utils.geometry_utils import NormalGenerator
from morpheus.visualization.image_viz import colormap_image


def generate_transform_json(
    input_camera_paths_path: pathlib.Path,
    output_data_path: pathlib.Path,
    width: int,
    height: int,
    end_index: Optional[int] = None,
    selected_frame_ids: Optional[list[int]] = None,
    view_idxs: Optional[list[int]] = None,
):
    with open(input_camera_paths_path, "r") as fp:
        camera_paths = json.load(fp)

    nerfstudio_frames = []
    frame_ids = selected_frame_ids if selected_frame_ids else len(camera_paths["camera_path"])
    for idx, frame_idx in enumerate(frame_ids):
        if end_index is not None and frame_idx > end_index:
            break

        camera_idx = frame_idx if view_idxs is None else view_idxs[idx]
        camera = camera_paths["camera_path"][camera_idx]
        fov_x = camera["fov"]
        fov_x = fov_x * (np.pi / 180)
        fl = height / (2 * torch.tan(torch.tensor(fov_x) / 2))

        nerfstudio_frames.append(
            {
                "file_path": f"frame_{frame_idx:08d}.png",
                "transform_matrix": np.array(camera["camera_to_world"])
                .astype(float)
                .reshape(4, 4)
                .tolist(),
                "w": width,
                "h": height,
                "fl_x": fl.item(),
                "fl_y": fl.item(),
                "cx": width / 2.0,
                "cy": height / 2.0,
                "is_fisheye": False,
            }
        )

    with open(output_data_path / "transforms.json", "w") as fp:
        json.dump(
            {
                "ply_file_path": "point_cloud.ply",
                "frames": nerfstudio_frames,
            },
            fp,
        )


def save_depth_for_regsplatfacto(
    output_data_path: pathlib.Path,
    stylised_frame: Frame,
    normal_generator: NormalGenerator,
    debug_path: Optional[pathlib.Path] = None,
) -> None:
    # regsplatfacto caches its predicted depth & normal maps in the 'regsplatfacto_depths'
    # directory. We're going to save these in the same format as regsplatfacto's predictions,
    # so that it can use them during training.
    regsplatfacto_depths = output_data_path / "regsplatfacto_depths"
    regsplatfacto_depths.mkdir(exist_ok=True, parents=True)

    # Estimate normals from the depth map and normalise it to regsplatfacto's convention
    normal_b3hw = normal_generator(stylised_frame.depth_b1hw, stylised_frame.invK_b44)
    predictions_path = regsplatfacto_depths / f"frame_{stylised_frame.idx:08d}.npz"
    np.savez(
        predictions_path,
        depth=stylised_frame.depth_b1hw.squeeze(0).cpu().numpy(),
        normal=normal_b3hw.squeeze(0).cpu().numpy(),
    )

    if debug_path is not None:
        depth_colormap = colormap_image(stylised_frame.depth_b1hw[0], vmin=0)
        depth_pil = Image.fromarray(
            (depth_colormap.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        )
        depth_pil.save(debug_path / f"{stylised_frame.idx:03d}_depth.png")
        normals_pil = Image.fromarray(
            ((normal_b3hw[0].permute(1, 2, 0).cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
        )
        normals_pil.save(debug_path / f"{stylised_frame.idx:03d}_normals.png")


def run_regsplatfacto(
    output_data_path: pathlib.Path,
    input_camera_paths_path: pathlib.Path,
    num_iterations: int = 15000,
    depth_regularisation_weight: float = 0.5,
    tvl1_regularisation_weight: float = 0.05,
    normal_regularisation_weight: float = 0.001,
) -> None:
    subprocess.run(
        [
            "ns-train",
            "regsplatfacto",
            "--max-num-iterations",
            str(num_iterations),
            "--pipeline.model.regularisation-first-step",
            str(3000),  # Default value in regsplatfacto: 0 steps
            "--pipeline.model.depth-regularisation-weight",
            str(depth_regularisation_weight),
            "--pipeline.model.tvl1_regularisation_weight",
            str(tvl1_regularisation_weight),
            "--pipeline.model.normal-regularisation-weight",
            str(normal_regularisation_weight),
            "--viewer.quit-on-train-completion",
            "True",
            "--logging.local-writer.max-log-size",
            "0",
            "nerfstudio-data",
            "--auto-scale-poses",
            "False",
            "--orientation-method",
            "none",
            "--center-method",
            "none",
            "--train-split-fraction",
            "1",  # Force training on all frames
            "--downscale-factor",
            "1",  # Don't downscale the images
        ],
        cwd=str(output_data_path),
        check=True,
    )

    # There should be only one config file in the output directory
    config_yaml = list(output_data_path.glob("outputs/unnamed/regsplatfacto/*/config.yml"))[0]

    # Render MP4 & RGBD from the resplatted scene
    subprocess.run(
        [
            "ns-render",
            "camera-path",
            "--load-config",
            str(config_yaml.relative_to(output_data_path)),
            "--camera-path-filename",
            str(input_camera_paths_path.absolute()),
            "--output-path",
            "./resplatting_render.mp4",
        ],
        cwd=str(output_data_path),
        check=True,
    )
    subprocess.run(
        [
            "ns-render",
            "camera-path",
            "--load-config",
            str(config_yaml.relative_to(output_data_path)),
            "--camera-path-filename",
            str(input_camera_paths_path.absolute()),
            "--output-path",
            "./resplatting_render/",
            "--rendered-output-names",
            "rgb",
            "depth",
            "--output-format",
            "images",
        ],
        cwd=str(output_data_path),
        check=True,
    )

    # Render eval training views for evaluation (if we have the camera paths)
    eval_camera_paths_path = pathlib.Path(
        input_camera_paths_path.parent / "train_ours_interp_views.json"
    )
    if eval_camera_paths_path.exists():
        subprocess.run(
            [
                "ns-render",
                "camera-path",
                "--load-config",
                str(config_yaml.relative_to(output_data_path)),
                "--camera-path-filename",
                str(eval_camera_paths_path.absolute()),
                "--output-path",
                "./rendered_train_ours_interp_views/",
                "--rendered-output-names",
                "rgb",
                "depth",
                "--output-format",
                "images",
                "--image-format",
                "png",
            ],
            cwd=str(output_data_path),
            check=True,
        )
    else:
        logger.warning(
            f"Could not find eval training views camera path at '{eval_camera_paths_path}'."
            f" Skipping rendering for evaluation..."
        )

    # Render circular 'wiggles' for use in visualisation & user study (if we have the camera paths)
    circle_eval_camera_paths_path = pathlib.Path(input_camera_paths_path.parent / "circle_cams")
    if circle_eval_camera_paths_path.exists():
        # get all .json files in the directory
        circle_cam_paths = list(circle_eval_camera_paths_path.glob("*.json"))
        for circle_cam_path in circle_cam_paths:
            subprocess.run(
                [
                    "ns-render",
                    "camera-path",
                    "--load-config",
                    str(config_yaml.relative_to(output_data_path)),
                    "--camera-path-filename",
                    str(circle_cam_path.absolute()),
                    "--output-path",
                    f"./rendered_circle_cams/{circle_cam_path.stem}",
                    "--rendered-output-names",
                    "rgb",
                    "depth",
                    "--output-format",
                    "images",
                    "--image-format",
                    "png",
                ],
                cwd=str(output_data_path),
                check=True,
            )
            # Render MP4, rendered images are of form circle_cam_path.stem/00000_rgb.png, use ffmpeg to create mp4
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-r",
                    "30",
                    "-i",
                    f"rendered_circle_cams/{circle_cam_path.stem}/%05d_rgb.png",
                    "-vcodec",
                    "libx264",
                    "-crf",
                    "17",
                    "-pix_fmt",
                    "yuv420p",
                    f"rendered_circle_cams/{circle_cam_path.stem}.mp4",
                ],
                cwd=str(output_data_path),
                check=True,
            )
    else:
        logger.warning(
            f"Could not find circle camera paths at '{circle_eval_camera_paths_path}'."
            f" Skipping rendering for circle camera paths..."
        )
