import json
import os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from nerfstudio.utils.eval_utils import eval_setup

def preprocess_case1_data(config_path: str, output_dir: str):
    """
    Renders RGB and Depth from a trained NerfStudio model and saves them in Morpheus format.
    """
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {config_path}...")
    config, pipeline, checkpoint_path, step = eval_setup(config_path)
    pipeline.eval()

    datamanager = pipeline.datamanager
    dataset = datamanager.train_dataset
    
    print(f"Rendering {len(dataset)} images...")
    
    camera_path = []
    
    # Use safe resolution (multiples of 64 to avoid SD/Metric3D errors)
    orig_h = int(dataset.cameras.height[0])
    orig_w = int(dataset.cameras.width[0])
    aspect = orig_w / orig_h
    
    # Target height 1024 (closest to 960 that is mult of 64)
    target_h = 1024 
    target_w = 576 # 1024 * 0.5625
    
    print(f"Using safe resolution: {target_w}x{target_h} (original: {orig_w}x{orig_h})")

    for i in tqdm(range(len(dataset))):
        camera = dataset.cameras[i:i+1].to(pipeline.device)
        # Update camera resolution for rendering
        camera.width[0] = target_w
        camera.height[0] = target_h
        
        outputs = pipeline.model.get_outputs_for_camera(camera)
        
        # RGB
        rgb = outputs["rgb"].cpu().numpy()
        rgb = (rgb * 255).astype(np.uint8)
        rgb_img = Image.fromarray(rgb)
        rgb_img.save(output_dir / f"{i:05d}_rgb.png")
        
        # Depth
        depth = outputs["depth"].cpu().numpy() # [H, W, 1]
        np.save(output_dir / f"{i:05d}_depth.npy", depth.squeeze(-1))
        
        # Collect camera info for camera_path.json
        c2w = camera.camera_to_worlds[0].cpu().numpy() # [3, 4]
        # Convert to 4x4
        c2w_44 = np.eye(4)
        c2w_44[:3, :] = c2w
        
        # Convert focal length to FOV in degrees
        focal_length = float(camera.fx[0].cpu().item())
        height = float(camera.height[0].cpu().item())
        fov_deg = 2 * np.arctan(height / (2 * focal_length)) * (180 / np.pi)
        
        camera_path.append({
            "camera_to_world": c2w_44.flatten().tolist(),
            "fov": float(fov_deg),
            "aspect": float(target_w / target_h),
            "seconds": i / 24.0 # Default 24 fps
        })

    # Save camera_path.json
    fps = 24
    with open(output_dir.parent / "camera_path.json", "w") as f:
        json.dump({
            "camera_path": camera_path,
            "render_width": target_w,
            "render_height": target_h,
            "fps": fps,
            "seconds": len(dataset) / fps
        }, f, indent=2)

    # Copy transforms.json as dataparser_transforms.json
    # We need to adjust it to match Morpheus's expectation if needed
    source_transforms = config.data / "transforms.json"
    if source_transforms.exists():
        with open(source_transforms, "r") as f:
            t = json.load(f)
            # Morpheus expects "transform" and "scale" at top level
            with open(output_dir.parent / "dataparser_transforms.json", "w") as f_out:
                json.dump({
                    "transform": t.get("transform", np.eye(4).tolist()),
                    "scale": t.get("scale", 1.0)
                }, f_out, indent=2)
                
    print(f"Preprocessing complete. Data saved to {output_dir}")

if __name__ == "__main__":
    # Use the latest config found
    import glob
    configs = glob.glob("outputs/**/config.yml", recursive=True)
    if configs:
        latest_config = max(configs, key=os.path.getmtime)
        preprocess_case1_data(latest_config, "data/scenes/Case1/3dgs")
    else:
        print("No config.yml found in outputs/Case1")
