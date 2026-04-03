import os
import subprocess
from pathlib import Path
import argparse
import sys

# Attempt to import rembg, install if missing
try:
    from rembg import remove
except ImportError:
    print("rembg is not installed. Attempting to install...")
    subprocess.run([sys.executable, "-m", "pip", "install", "rembg==2.0.52"], check=True)
    from rembg import remove

def run_command(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def main():
    parser = argparse.ArgumentParser(description="Morpheus 3DGS Data preparation pipeline")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing raw images")
    parser.add_argument("--case_name", type=str, required=True, help="Name of the case (e.g. Case1)")
    parser.add_argument("--remove-bg", action="store_true", help="Remove background using rembg")
    parser.add_argument("--bg-color", type=str, default="white", choices=["white", "black"], help="Background color (white or black)")
    parser.add_argument("--train", action="store_true", help="Run ns-train after processing")
    parser.add_argument("--render", action="store_true", help="Run rendering script after training")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    case_name = args.case_name
    
    # 1. Background Removal
    process_dir = input_dir
    if args.remove_bg:
        print(f"--- Step 1: Removing Background (Color: {args.bg_color}) ---")
        masked_dir = input_dir.parent / f"{input_dir.name}_{args.bg_color}"
        cmd = f"python src/morpheus/utils/remove_background.py --input_dir {input_dir} --output_dir {masked_dir} --bg_color {args.bg_color}"
        run_command(cmd)
        process_dir = masked_dir
    
    # 2. ns-process-data
    print("--- Step 2: Processing Data (COLMAP) ---")
    ns_data_dir = Path(f"data/nerfstudio/{case_name}")
    if args.remove_bg:
        ns_data_dir = Path(f"data/nerfstudio/{case_name}_masked")
        
    cmd = f"ns-process-data images --data {process_dir} --output-dir {ns_data_dir} --no-gpu"
    run_command(cmd)
    
    # 3. ns-train (Optional)
    if args.train:
        print("--- Step 3: Training 3DGS ---")
        output_dir = Path(f"outputs/{case_name}")
        if args.remove_bg:
            output_dir = Path(f"outputs/{case_name}_masked")
            
        cmd = f"ns-train regsplatfacto --data {ns_data_dir} --output-dir {output_dir} --viewer.quit-on-train-completion True --max-num-iterations 10000"
        run_command(cmd)
        
    # 4. Render (Optional)
    if args.render:
        print("--- Step 4: Rendering RGBD for Morpheus ---")
        # This assumes the training finished and we can find the config
        cmd = f"python src/morpheus/utils/preprocess_custom_data.py"
        run_command(cmd)

if __name__ == "__main__":
    main()
