import argparse
import csv
import torch
from diffusers import StableDiffusion3Pipeline
from ras.utils.stable_diffusion_3.update_pipeline_sd3 import update_sd3_pipeline
from ras.utils import ras_manager
from ras.utils.ras_argparser import parse_args  # <-- Keep this
import os
from datasets import load_dataset
from tqdm import tqdm
import sys

def local_parse_args():
    """
    Local argument parser for script-specific settings.
    It returns a parser object, but doesn't parse yet.
    """
    parser = argparse.ArgumentParser(description="Local script settings", add_help=False)
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="/DATA/chetan/RAS/outputs/mscoco_prompts", 
        help="Directory to save generated images."
    )
    
    # --- CHANGE THIS ---
    parser.add_argument(
        "--start_index",  # Changed from --range_start
        type=int, 
        default=0, 
        help="The starting index (inclusive) of the dataset to process."
    )
    
    # --- AND CHANGE THIS ---
    parser.add_argument(
        "--end_index",  # Changed from --range_end
        type=int, 
        default=None, 
        help="The ending index (exclusive) of the dataset to process. If None, goes to the end."
    )
    
    parser.add_argument(
        "--csv_file", 
        type=str, 
        default="longest_captions_sample.csv", 
        help="Path to the CSV file containing 'file_name' and 'longest_caption'."
    )
    parser.add_argument(
        "--baseline", 
        action='store_true',
        help="if true runs the baseline model."
    )
    
    return parser

# --- NO CHANGE to evaluate_on_csv ---
def evaluate_on_csv(args):
    """
    Main function to run the evaluation pipeline.
    It now loads a specific RANGE of data from the specified CSV file.
    It saves images using the 'file_name' from the CSV.
    """
    # 1. SETUP: Load model and create output directory
    print("### Setting up the pipeline... ###")
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        torch_dtype=torch.float16
    )
    pipeline.to("cuda")
    
    # --- This 'if' logic now works perfectly ---
    if not args.baseline:
        print("Applying RAS pipeline update...")
        pipeline = update_sd3_pipeline(pipeline)
    else:
        print("Running baseline pipeline (no RAS update).")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory created at: {args.output_dir}")

    # 2. DATASET: Load data from CSV file
    print(f"### Loading data from CSV: {args.csv_file} ... ###")
    
    prompts_data = []
    try:
        with open(args.csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader) # Skip the header row
            file_name_idx = header.index('file_name')
            caption_idx = header.index('longest_caption')
            
            for row in reader:
                if row: 
                    file_name = row[file_name_idx]
                    prompt = row[caption_idx]
                    prompts_data.append([file_name, prompt])
                    
    except FileNotFoundError:
        print(f"ERROR: CSV file not found at {args.csv_file}")
        return
    except Exception as e:
        print(f"ERROR: Could not read CSV file. Is it formatted correctly? {e}")
        return

    total_prompts = len(prompts_data)
    print(f"Dataset loaded with {total_prompts} total prompts.")

    # 3. Handle Range Slicing
    start_index = args.start_index
    end_index = args.end_index if args.end_index is not None else total_prompts
    start_index = max(0, start_index)
    end_index = min(total_prompts, end_index)

    if start_index >= end_index:
        print(f"Start index ({start_index}) is >= end index ({end_index}). No images to generate. Exiting.")
        return

    print(f"### Processing prompts in range: {start_index} to {end_index - 1} ###")
    prompts_to_process = prompts_data[start_index:end_index]
    num_to_generate = len(prompts_to_process)

    # 4. GENERATION LOOP
    print(f"### Starting image generation for {num_to_generate} images... ###")
    generator = torch.Generator("cuda").manual_seed(args.seed) if args.seed is not None else None
    
    progress_bar = tqdm(
        prompts_to_process,
        total=num_to_generate,
        desc=f"Generating Images {start_index}-{end_index - 1}"
    )
    
    for item in progress_bar:
        file_name, prompt = item
        
        base_name = os.path.splitext(file_name)[0]
        output_filename = f"{base_name}.png"
        output_path = os.path.join(args.output_dir, output_filename)

        if os.path.exists(output_path):
            continue

        image = pipeline(
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=7.0,
            height=args.height,
            width=args.width,
            generator=generator,
        ).images[0]

        image.save(output_path)

    print(f"### Generation complete for range {start_index}-{end_index - 1}! ###")

if __name__ == "__main__":
    
    local_parser = local_parse_args()
    local_args, remaining_args = local_parser.parse_known_args()
    
    sys.argv = [sys.argv[0]] + remaining_args
    
    args = parse_args()
    
    vars(args).update(vars(local_args))

    if args.baseline:
        print("Running baseline model...")
    else:
        print("Setting RAS parameters...")
        ras_manager.MANAGER.set_parameters(args)
        
    evaluate_on_csv(args)