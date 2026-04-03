import os
from pathlib import Path
from PIL import Image
from rembg import remove
from tqdm import tqdm
import argparse

def process_images(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.jpeg"))
    print(f"Found {len(image_paths)} images in {input_dir}")

    for img_path in tqdm(image_paths):
        with open(img_path, "rb") as i:
            input_data = i.read()
            # Remove background (returns image with alpha channel)
            output_data = remove(input_data)
            
            # Convert to PIL Image
            from io import BytesIO
            img = Image.open(BytesIO(output_data)).convert("RGBA")
            
            # Create background (default to white as requested)
            bg_color = (255, 255, 255, 255) if args.bg_color == "white" else (0, 0, 0, 255)
            background = Image.new("RGBA", img.size, bg_color)
            
            # Paste the foreground onto the background
            final_img = Image.alpha_composite(background, img).convert("RGB")
            
            # Save
            final_img.save(output_dir / img_path.name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="data/input/Case1")
    parser.add_argument("--output_dir", type=str, default="data/input/Case1_white")
    parser.add_argument("--bg_color", type=str, default="white", choices=["white", "black"])
    args = parser.parse_args()
    
    process_images(args.input_dir, args.output_dir)
