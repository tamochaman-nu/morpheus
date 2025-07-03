import pathlib
import subprocess

from torchvision.utils import save_image

from morpheus.utils.data_utils import Frame


def save_frame_to_output(frame: Frame, output_path: pathlib.Path) -> None:
    image_path = output_path / f"frame_{frame.idx:08d}.png"
    save_image(frame.image_bchw, image_path)


def generate_video_from_frames(output_path: pathlib.Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            "3",
            "-pattern_type",
            "glob",
            "-i",
            f"{output_path / 'frame_*.png'}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            f"{output_path / 'output.mp4'}",
        ]
    )
