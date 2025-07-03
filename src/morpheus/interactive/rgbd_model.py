import pathlib

import gradio as gr
import torch
import yaml
from loguru import logger

from morpheus.datasets.ns_render_dataset import NerfStudioRenderDataset
from morpheus.inpainter import Inpainter, stylise_i_frame
from morpheus.visualization.image_viz import colormap_image

config: dict = None
dataset: NerfStudioRenderDataset = None
inpainter: Inpainter = None
current_config_path = pathlib.Path("")
current_stable_diffusion_model_path = pathlib.Path("")


def gradio_demo(
    config_yaml_path: str,
    prompt: str,
    negative_prompt: str,
    inversion_prompt: str,
    guidance_scale: float,
    rgb_noise_strength: float,
    depth_noise_strength: float,
    partial_inversion_noise_level: float,
    partial_inversion_depth_noise_level: float,
    frame_idx: int,
    seed: int,
):
    global current_config_path, config, dataset
    if current_config_path != config_yaml_path:
        current_config_path = config_yaml_path
        with open(config_yaml_path, "r") as f:
            config = yaml.load(f.read(), Loader=yaml.Loader)

        logger.info(f"Loading dataset from {config['input_renders_path']}")
        dataset = NerfStudioRenderDataset(
            renders_path=pathlib.Path(config["input_renders_path"]),
            json_path=pathlib.Path(config["input_transform_json_path"]),
            poses_path=pathlib.Path(config["input_camera_paths_path"]),
            target_height=config["resolution"],
            square_images=config["square_image"],
        )

    if "stable_diffusion_model_path" not in config:
        raise ValueError("No stable_diffusion_model_path defined in the YAML config file")
    if "controlnet_model_path" not in config:
        raise ValueError("No controlnet_model_path defined in the YAML config file")
    controlnet_model_path = pathlib.Path(config["controlnet_model_path"])
    stable_diffusion_model_path = pathlib.Path(config["stable_diffusion_model_path"])

    guidance_scale = float(guidance_scale)
    rgb_noise_strength = float(rgb_noise_strength)
    depth_noise_strength = float(depth_noise_strength)

    global inpainter, current_stable_diffusion_model_path, current_controlnet_model_path
    if (
        inpainter is None
        or current_stable_diffusion_model_path != stable_diffusion_model_path
        or current_controlnet_model_path != controlnet_model_path
    ):
        logger.info(f"Reloading model...")
        inpainter_args = dict(
            text_prompt=prompt,
            negative_text_prompt=negative_prompt,
            inversion_prompt=inversion_prompt,
            partial_inversion_noise_level=partial_inversion_noise_level,
            partial_inversion_depth_noise_level=partial_inversion_depth_noise_level,
            noise_strength=rgb_noise_strength,
            depth_noise_strength=depth_noise_strength,
            guidance_scale=guidance_scale,
            stable_diffusion_model_path=stable_diffusion_model_path,
            controlnet_model_path=controlnet_model_path,
            controlnet_strength=0.0,  # Disabled on initial frame
            debug_path=None,
        )
        inpainter_args.update(config.get("extra_inpainter_args") or {})
        inpainter = Inpainter(**inpainter_args)

    frame = dataset.get_frame(frame_idx)
    inpainter.warp_controlnet.text_prompt = prompt
    inpainter.warp_controlnet.negative_text_prompt = negative_prompt
    inpainter.warp_controlnet.ddim_inverter.prompt = inversion_prompt

    with torch.no_grad():
        stylised_frame = stylise_i_frame(
            input_frame=frame,
            inpainter=inpainter,
            controlnet_strength=0.0,
            guidance_scale=guidance_scale,
            noise_strength=rgb_noise_strength,
            depth_noise_strength=depth_noise_strength,
            partial_inversion_noise_level=partial_inversion_noise_level,
            partial_inversion_depth_noise_level=partial_inversion_depth_noise_level,
            seed=seed,
        )

    depth_img = colormap_image(stylised_frame.depth_b1hw[0], vmin=0)
    return [
        (255.0 * stylised_frame.image_bchw[0].permute(1, 2, 0).cpu().numpy()).astype("uint8"),
        (255.0 * depth_img.permute(1, 2, 0).cpu().numpy()).astype("uint8"),
    ]


def main():
    demo = gr.Interface(
        gradio_demo,
        inputs=[
            gr.Textbox(
                label="config_yaml_path",
                lines=0,
                value="./data/configs/fangzhou/a_photo_of_a_human_skeleton.yaml",
            ),
            gr.Textbox(
                label="prompt",
                lines=0,
                value="a photo of a human skeleton",
            ),
            gr.Textbox(
                label="negative_prompt",
                lines=0,
                value="a photo of a face of a man lowres, grainy, blurry",
            ),
            gr.Textbox(
                label="inversion_prompt",
                lines=0,
                value="a photo of a face of a man",
            ),
            gr.Slider(0.0, 20.0, value=10.0, label="guidance_scale"),
            gr.Slider(0.0, 1.0, value=0.8, label="rgb_noise_strength"),
            gr.Slider(0.0, 1.0, value=0.8, label="depth_noise_strength"),
            gr.Slider(0.0, 1.0, value=0.05, label="partial_inversion_noise_level"),
            gr.Slider(0.0, 1.0, value=0.05, label="partial_inversion_depth_noise_level"),
            gr.Number(0, label="frame_idx"),
            gr.Number(1, label="seed"),
        ],
        outputs=[
            gr.Image(label="Stylised image"),
            gr.Image(label="Stylised depth"),
        ],
    )
    demo.launch()


if __name__ == "__main__":
    main()
