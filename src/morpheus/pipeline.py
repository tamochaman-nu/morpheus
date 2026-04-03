import pathlib
import pprint
from argparse import Namespace
from typing import Optional

import click
import torch
import yaml
from click.core import ParameterSource
from loguru import logger
from tqdm import tqdm

from morpheus.datasets.ns_render_dataset import NerfStudioRenderDataset
from morpheus.forward_warp_compositor import ForwardWarpCompositor
from morpheus.inpainter import Inpainter, stylise_i_frame
from morpheus.p_frame_generation import PFrameGenerator
from morpheus.point_cloud import ResplattingPointCloud
from morpheus.utils.geometry_utils import ForwardWarp, NormalGenerator
from morpheus.utils.mesh_utils import MeshWarp
from morpheus.utils.ns_utils import (
    generate_transform_json,
    run_regsplatfacto,
    save_depth_for_regsplatfacto,
)
from morpheus.video import generate_video_from_frames, save_frame_to_output

DEFAULT_GROUP_OF_PICTURES_SIZE = 20


@torch.no_grad()
def run_pipeline(
    prompt: str,
    negative_prompt: str,
    input_renders_path: pathlib.Path,
    input_transform_json_path: pathlib.Path,
    input_camera_paths_path: pathlib.Path,
    output_data_path: pathlib.Path,
    group_of_pictures_size: int = DEFAULT_GROUP_OF_PICTURES_SIZE,
    initial_seed=1,
    resolution=None,
    square_image=True,
    i_frame_controlnet_strength: Optional[float] = None,
    i_frame_noise_strength: Optional[float] = None,
    i_frame_depth_noise_strength: Optional[float] = None,
    i_frame_guidance_scale: Optional[float] = None,
    noise_strength: float = 0.95,
    depth_noise_strength: Optional[float] = None,
    controlnet_strength: float = 0.5,
    guidance_scale: float = 12.5,
    start_index: int = 0,
    debug: bool = False,
    end_index: Optional[int] = None,
    i_frame_only: Optional[bool] = False,
    extra_inpainter_args: Optional[dict] = None,
    stable_diffusion_model_path: pathlib.Path = None,
    controlnet_model_path: pathlib.Path = None,
    huggingface_cache_dir: pathlib.Path = None,
    run_resplatting: bool = False,
    extra_p_frame_generator_args: Optional[dict] = None,
    extra_compositor_args: Optional[dict] = None,
    permute_frames_by: int = 0,
    extra_nerfstudio_args: Optional[dict] = None,
) -> None:
    # Validate options
    if not square_image:
        logger.warning("Ignoring resolution as square_image is set to False")
        resolution = None
    if resolution is not None:
        assert square_image, "Resolution can only be set if square_image is set to True"

    # Load the dataset into memory
    logger.info("Loading dataset from:")
    logger.info(f" input_renders_path={input_renders_path}")
    logger.info(f" input_transform_json_path={input_transform_json_path}")
    logger.info(f" input_camera_paths_path={input_camera_paths_path}")
    dataset = NerfStudioRenderDataset(
        renders_path=input_renders_path,
        json_path=input_transform_json_path,
        poses_path=input_camera_paths_path,
        target_height=resolution,
        square_images=square_image,
        end_index=end_index,
        permute_frames_by=permute_frames_by,
    )
    assert start_index < len(
        dataset
    ), f"start_index ({start_index}) is greater than the dataset length ({len(dataset)})"

    # Instantiate geometry/warping/compositing classes:
    device = torch.device("cuda")

    fwd_warper = ForwardWarp(
        height=dataset.target_height,
        width=dataset.target_width,
        device=device,
    )

    mesh_warper = MeshWarp(
        height=dataset.target_height,
        width=dataset.target_width,
        device=device,
        deterministic=True,
    )

    normal_generator = NormalGenerator(
        width=dataset.target_height,
        height=dataset.target_width,
    ).to(device)

    output_data_path.mkdir(parents=True, exist_ok=True)

    (output_data_path / "compositor_debug").mkdir(exist_ok=True)
    compositor = ForwardWarpCompositor(
        mesh_warper=mesh_warper,
        debug_path=output_data_path / "compositor_debug" if debug else None,
        **(extra_compositor_args or {}),
    )

    (output_data_path / "inpaint_debug").mkdir(exist_ok=True)
    inpainter = Inpainter(
        text_prompt=prompt,
        negative_text_prompt=negative_prompt,
        controlnet_strength=controlnet_strength,
        noise_strength=noise_strength,
        depth_noise_strength=depth_noise_strength,
        guidance_scale=guidance_scale,
        stable_diffusion_model_path=stable_diffusion_model_path,
        controlnet_model_path=controlnet_model_path,
        huggingface_cache_dir=huggingface_cache_dir,
        debug_path=output_data_path / "inpaint_debug" if debug else None,
        **(extra_inpainter_args or {}),
    )

    (output_data_path / "pframe_debug").mkdir(exist_ok=True)
    pframe_generator = PFrameGenerator(
        compositor=compositor,
        inpainter=inpainter,
        debug_path=output_data_path / "pframe_debug" if debug else None,
        **(extra_p_frame_generator_args or {}),
    )

    gop_start_indices = list(range(start_index, len(dataset), group_of_pictures_size))

    logger.debug(f"Frame group starts: {gop_start_indices}")

    if start_index > 0:
        # add all raw RGB frames till this point
        for i in tqdm(range(start_index), desc=f"Saving raw frames up to {start_index}"):
            frame = dataset.get_frame(i)
            save_frame_to_output(frame, output_data_path)

    # Perform full style transfer on the first frame in the dataset (called "I-frame" in the paper)
    logger.info("Performing full style transfer on the first I-frame")

    # Stylise the first frame (the i-frame)
    # Possibly override the strength for the first frame if this has been requested via i_frame_strength
    i_frame = dataset.get_frame(start_index)
    i_frame_stylised = stylise_i_frame(
        input_frame=i_frame,
        inpainter=inpainter,
        seed=initial_seed,
        controlnet_strength=(
            i_frame_controlnet_strength
            if i_frame_controlnet_strength is not None
            else controlnet_strength
        ),
        guidance_scale=(
            i_frame_guidance_scale if i_frame_guidance_scale is not None else guidance_scale
        ),
        noise_strength=(
            i_frame_noise_strength if i_frame_noise_strength is not None else noise_strength
        ),
        depth_noise_strength=(
            i_frame_depth_noise_strength
            if i_frame_depth_noise_strength is not None
            else depth_noise_strength
        ),
    )
    save_frame_to_output(i_frame_stylised, output_data_path)

    if i_frame_only:
        # If in i_frame_only mode, we only generate the first frame and
        # then exit the pipeline. This is useful for debugging
        logger.info("i_frame only mode: Exiting pipeline after generating the first frame")
        return

    if debug:
        debug_path = output_data_path / "debug"
        debug_path.mkdir(exist_ok=True)
        point_cloud_debug_path = output_data_path / "point_cloud_debug"
        point_cloud_debug_path.mkdir(exist_ok=True)
        fwd_warper.backproject_and_save_point_cloud(
            frame=i_frame, output_path=point_cloud_debug_path / f"i_frame_{i_frame.idx}_raw.ply"
        )
        fwd_warper.backproject_and_save_point_cloud(
            frame=i_frame_stylised,
            output_path=point_cloud_debug_path / f"i_frame_{i_frame.idx}_stylised.ply",
        )
    else:
        debug_path = None

    # Initialise list storing all available P-frames and their stylised versions
    # (For now this is just the first frame)
    all_p_frames = [i_frame]
    all_p_frames_stylised = [i_frame_stylised]

    # Initialise the point cloud used for re-splatting with the initial i-frame
    resplatting_point_cloud = ResplattingPointCloud(
        target_width=dataset.target_width,
        target_height=dataset.target_height,
    )
    resplatting_point_cloud.add_frame(i_frame_stylised)

    # Save the initial i-frame's depth and normal maps for the later regsplatfacto training
    (output_data_path / "regsplatfacto_debug").mkdir(exist_ok=True)
    save_depth_for_regsplatfacto(
        output_data_path=output_data_path,
        stylised_frame=i_frame_stylised,
        normal_generator=normal_generator,
        debug_path=output_data_path / "regsplatfacto_debug" if debug else None,
    )

    # Iterate over the dataset in groups of pictures
    for gop_ind, gop_start_idx in enumerate(gop_start_indices):
        # Extract the group of pictures from the dataset
        if gop_ind + 1 < len(gop_start_indices):
            gop_end_idx = gop_start_indices[gop_ind + 1]
        else:
            gop_end_idx = len(dataset) - 1

        logger.debug(f"gop_start_idx = {gop_start_idx}, gop_end_idx = {gop_end_idx}")

        group_of_pictures = [
            dataset.get_frame(idx) for idx in range(gop_start_idx, gop_end_idx + 1)
        ]
        if not group_of_pictures:
            continue
        p_frame = group_of_pictures[-1]
        logger.debug(
            f"Running processing for GoP (I-frame idx: {i_frame.idx}, "
            f"P-frame idx: {p_frame.idx})"
        )

        if i_frame.idx == p_frame.idx:
            logger.debug("i_frame.idx == p_frame.idx, skipping this step")
            continue

        # Generate the next stylised P-frame
        logger.debug(f"Generating the next P-frame (P-frame idx: {p_frame.idx})")

        src_frames = all_p_frames
        src_frames_stylised = all_p_frames_stylised

        p_frame_stylised = pframe_generator.generate_p_frame(
            src_frames=src_frames,
            target_frame=p_frame,
            src_frames_stylised=src_frames_stylised,
            seed=initial_seed * p_frame.idx,
        )
        all_p_frames.append(p_frame)
        all_p_frames_stylised.append(p_frame_stylised)

        # Also dump point clouds of the raw & stylised new frame
        if debug:
            point_cloud_debug_path = output_data_path / "point_cloud_debug"
            point_cloud_debug_path.mkdir(exist_ok=True)
            # use the fwd warp's backproject_and_save_point_cloud method
            fwd_warper.backproject_and_save_point_cloud(
                frame=p_frame, output_path=point_cloud_debug_path / f"p_frame_{p_frame.idx}_raw.ply"
            )
            fwd_warper.backproject_and_save_point_cloud(
                frame=p_frame_stylised,
                output_path=point_cloud_debug_path / f"p_frame_{p_frame.idx}_stylised.ply",
            )

        # Append the P-frame to the output frames and point cloud used for re-splatting init
        save_frame_to_output(p_frame_stylised, output_data_path)
        resplatting_point_cloud.add_frame(p_frame_stylised)

        # Save the depth and normal maps for the later regsplatfacto training
        save_depth_for_regsplatfacto(
            output_data_path=output_data_path,
            stylised_frame=p_frame_stylised,
            normal_generator=normal_generator,
            debug_path=output_data_path / "regsplatfacto_debug" if debug else None,
        )

        # Prepare I-frames for the next GoP (group of pictures)
        i_frame = p_frame
        i_frame_stylised = p_frame_stylised

    # Generate the final video by combining the output frames
    logger.info("Generating the final video out of all generated frames")
    generate_video_from_frames(output_data_path)

    # Export the point cloud for re-splatting init in NerfStudio
    logger.info("Exporting the point cloud for re-splatting initialisation")
    resplatting_point_cloud.export_to_ply(output_data_path)

    # if we did permute_frames_by, we now need to fixup the frame indices
    # In particular if a frame has idx i, then its true idx w.r.t. the original dataset
    # will be (i + permute_frames_by) % len(dataset)
    if permute_frames_by != 0:
        logger.info(f"Computing true (unpermuted) frame idxs")
        selected_frame_ids = [f.idx for f in all_p_frames_stylised]
        view_idxs = [(idx + permute_frames_by) % len(dataset) for idx in selected_frame_ids]
    else:
        view_idxs = None

    # Generate the transform json file for re-splatting in NerfStudio
    logger.info("Generating transform.json file for re-splatting in NerfStudio")
    generate_transform_json(
        input_camera_paths_path=input_camera_paths_path,
        output_data_path=output_data_path,
        width=dataset.target_width,
        height=dataset.target_height,
        end_index=end_index,
        selected_frame_ids=[f.idx for f in all_p_frames_stylised],
        view_idxs=view_idxs,
    )

    # Run the regsplatfacto training on the stylised frames
    if run_resplatting:
        run_regsplatfacto(
            output_data_path=output_data_path,
            input_camera_paths_path=input_camera_paths_path,
            **(extra_nerfstudio_args or {}),
        )


@click.command()
@click.pass_context
@click.option("--config-path", default=None, type=click.Path())
@click.option(
    "--prompt",
    type=str,
    help="Text prompt for scene editing",
)
@click.option(
    "--negative-prompt",
    type=str,
    help="Negative text prompt for scene editing",
    default="",
)
@click.option(
    "--input-renders-path",
    type=click.Path(readable=True, path_type=pathlib.Path),
    help="Path to the Nerfstudio render rgb & depth images",
)
@click.option(
    "--input-transform-json-path",
    type=click.Path(readable=True, path_type=pathlib.Path),
    help="Path to the Nerfstudio transform JSON file of the trained splat",
)
@click.option(
    "--input-camera-paths-path",
    type=click.Path(readable=True, path_type=pathlib.Path),
    help="Path to the Nerfstudio camera paths JSON file with the trajectory",
)
@click.option(
    "--output-data-path",
    type=click.Path(writable=True, path_type=pathlib.Path),
    help="Path to the output directory with the pipeline results",
)
@click.option(
    "--group-of-pictures-size",
    type=int,
    help="Size of the group of pictures (GoP) used in the pipeline",
    default=DEFAULT_GROUP_OF_PICTURES_SIZE,
)
@click.option(
    "--initial-seed",
    type=int,
    help="Seed for the RNG",
    default=1,
)
@click.option(
    "--resolution",
    type=int,
    help="Target resolution in height",
    default=None,
)
@click.option(
    "--square-image/--full-res",
    is_flag=True,
    help="Crop the image to a square",
    default=True,
)
@click.option(
    "--i-frame-controlnet-strength",
    type=float,
    help="Strength of the inpainting for the I-frame",
    default=None,
)
@click.option(
    "--i-frame-noise-strength",
    type=float,
    help="Strength of the inpainting for the I-frame",
    default=None,
)
@click.option(
    "--i-frame-depth-noise-strength",
    type=float,
    help="Strength of the inpainting for the I-frame",
    default=None,
)
@click.option(
    "--i-frame-guidance-scale",
    type=float,
    help="Guidance scale for the inpainting of the first frame",
    default=None,
)
@click.option(
    "--i-frame-only",
    is_flag=True,
    help="Only generate the first frame",
    default=False,
)
@click.option(
    "--controlnet-strength",
    type=float,
    help="Strength of the inpainting",
    default=0.5,
)
@click.option(
    "--guidance-scale",
    type=float,
    help="Guidance scale for the inpainting",
    default=12.5,
)
@click.option(
    "--noise-strength",
    type=float,
    help="Guidance scale for the inpainting",
    default=0.95,
)
@click.option(
    "--depth-noise-strength",
    type=float,
    help="Guidance scale for the inpainting",
    default=None,
)
@click.option(
    "--debug",
    is_flag=True,
    help="Debug mode",
    default=False,
)
@click.option(
    "--start-index",
    type=int,
    help="Start index of the frame to start stylizing.",
    default=0,
)
@click.option(
    "--end-index",
    type=int,
    help="end index of the frame to end",
)
@click.option(
    "--extra-inpainter-args",
    type=dict,
    help="Extra inpainter arguments, passed in as kwargs",
)
@click.option(
    "--stable-diffusion-model-path",
    type=click.Path(readable=True, path_type=pathlib.Path),
    help="Path to the Stable Diffusion model",
)
@click.option(
    "--controlnet-model-path",
    type=click.Path(readable=True, path_type=pathlib.Path),
    help="Path to the ControlNet model",
)
@click.option(
    "--huggingface-cache-dir",
    type=click.Path(readable=True, path_type=pathlib.Path),
    help="Path to the HuggingFace Cache directory",
    default=None,
)
@click.option(
    "--run-resplatting/--no-run-resplatting",
    is_flag=True,
    help="Run the re-splatting step once the pipeline is done",
    default=False,
)
@click.option(
    "--extra-p-frame-generator-args",
    type=dict,
    help="Extra P-frame generator arguments, passed in as kwargs",
)
@click.option(
    "--extra-compositor-args",
    type=dict,
    help="Extra compositor arguments, passed in as kwargs",
)
@click.option(
    "--permute-frames-by",
    type=int,
    help="Permute the frames by this amount",
    default=0,
)
@click.option(
    "--extra-nerfstudio-args",
    type=dict,
    help="Extra NerfStudio arguments, passed in as kwargs",
)
def pipeline_cli(
    ctx: click.Context,
    config_path: pathlib.Path,
    prompt: str,
    negative_prompt: str,
    input_renders_path: pathlib.Path,
    input_transform_json_path: pathlib.Path,
    input_camera_paths_path: pathlib.Path,
    output_data_path: pathlib.Path,
    group_of_pictures_size: int = DEFAULT_GROUP_OF_PICTURES_SIZE,
    initial_seed: int = 1,
    resolution: int = None,
    square_image: bool = True,
    i_frame_controlnet_strength: float = None,
    i_frame_noise_strength: float = None,
    i_frame_depth_noise_strength: float = None,
    i_frame_guidance_scale: float = None,
    i_frame_only: bool = False,
    noise_strength: float = 0.95,
    depth_noise_strength: float = None,
    controlnet_strength: float = 0.5,
    guidance_scale: float = 12.5,
    start_index: int = 0,
    debug: bool = False,
    end_index: int = None,
    extra_inpainter_args: dict = None,
    stable_diffusion_model_path: pathlib.Path = None,
    controlnet_model_path: pathlib.Path = None,
    huggingface_cache_dir: pathlib.Path = None,
    run_resplatting: bool = False,
    extra_p_frame_generator_args: dict = None,
    extra_compositor_args: dict = None,
    permute_frames_by: int = 0,
    extra_nerfstudio_args: dict = None,
) -> None:
    """Don't use any of these variables directly. Instead reference using opts"""
    # Load in the config file
    if config_path:
        logger.info(f"Loading config from {config_path}")
        with open(config_path, "r") as f:
            config = yaml.load(f.read(), Loader=yaml.Loader)

        # Loop over supplied config and override the default values
        for key, value in config.items():
            # Handle cases where we have a value from the CLI and a value from the config file.
            # We keep the CLI value iff it was explicitly set by the user (i.e. not just the default
            # value)
            if ctx.get_parameter_source(key) == ParameterSource.DEFAULT:
                # Override with the config file's value:
                ctx.params[key] = value
            else:
                pass

    print(ctx.params)
    opts = Namespace(**ctx.params)

    logger.info(f"Running pipeline with options: ")
    pprint.pp(vars(opts))

    assert opts.input_renders_path, "input_renders_path is required"
    assert opts.input_transform_json_path, "input_transform_json_path is required"
    assert opts.input_camera_paths_path, "input_camera_paths_path is required"
    assert opts.output_data_path, "output_data_path is required"
    assert opts.prompt is not None, "prompt is required"

    opts.output_data_path = pathlib.Path(opts.output_data_path)
    opts.input_renders_path = pathlib.Path(opts.input_renders_path)
    opts.input_transform_json_path = pathlib.Path(opts.input_transform_json_path)
    opts.input_camera_paths_path = pathlib.Path(opts.input_camera_paths_path)
    opts.output_data_path = pathlib.Path(opts.output_data_path)

    opts.output_data_path.mkdir(parents=True, exist_ok=True)

    # Save the context to a yaml file in the outputdir
    logger.info(f"Saving pipeline params to {opts.output_data_path / 'pipeline_params.yaml'}")
    with open(opts.output_data_path / "pipeline_params.yaml", "w") as f:
        if config_path:
            # remove the old config path from the saved yaml
            ctx.params["config_path"] = None
        yaml.dump(ctx.params, f)

    # Discard the config path from the options (the pipeline doesn't need it)
    opts = vars(opts)
    opts.pop("config_path", None)

    run_pipeline(**opts)


if __name__ == "__main__":
    pipeline_cli()
