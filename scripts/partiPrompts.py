import argparse
import torch
from diffusers import StableDiffusion3Pipeline
from ras.utils.stable_diffusion_3.update_pipeline_sd3 import update_sd3_pipeline
from ras.utils import ras_manager
from ras.utils.ras_argparser import parse_args
import os
from datasets import load_dataset
from tqdm import tqdm

# sd3_inf function remains the same...
def sd3_inf(args):
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        torch_dtype=torch.float16
    )

    pipeline.to("cuda")
    pipeline = update_sd3_pipeline(pipeline)
    generator = torch.Generator("cuda").manual_seed(args.seed) if args.seed is not None else None
    numsteps = args.num_inference_steps
    image = pipeline(
                         generator=generator,
                         num_inference_steps=numsteps,
                         prompt=args.prompt,
                         negative_prompt=args.negative_prompt,
                         height=args.height,
                         width=args.width,
                         guidance_scale=7.0,
                         ).images[0]
    image.save(args.output)


def evaluate_on_parti(args):
    """
    Main function to run the evaluation pipeline.
    It loads the model, loops through the PartiPrompts dataset,
    and saves each generated image.
    **It now skips images that have already been generated.**
    """
    # 1. SETUP: Load model and create output directory
    print("### Setting up the pipeline... ###")
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        torch_dtype=torch.float16
    )
    pipeline.to("cuda")
    pipeline = update_sd3_pipeline(pipeline)
    output_dir="/DATA/rohit/extra/RAS/outputs/28Step_RAS_50"
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created at: {output_dir}")

    # 2. DATASET: Load the PartiPrompts dataset
    print("### Loading PartiPrompts dataset... ###")
    parti_prompts_dataset = load_dataset("nateraw/parti-prompts", split="train")
    prompts_list = parti_prompts_dataset['Prompt']
    print(f"Dataset loaded with {len(prompts_list)} prompts.")

    # 3. GENERATION LOOP
    print("### Starting image generation... ###")
    generator = torch.Generator("cuda").manual_seed(args.seed) if args.seed is not None else None
    
    # Use tqdm for a progress bar
    for i, prompt in enumerate(tqdm(prompts_list, desc="Generating Images")):
        filename = f"image_{i}.png"
        output_path = os.path.join(output_dir, filename)

        # >>> CHANGE: Check if the file already exists <<<
        if os.path.exists(output_path):
            # If it exists, skip to the next iteration
            continue

        # Generate the image
        image = pipeline(
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=7.0,
            height=args.height,
            width=args.width,
            generator=generator,
        ).images[0]

        # Save the image
        image.save(output_path)
        # Reducing print frequency to avoid clutter, tqdm already shows progress
        # print(f"Saved image {i+1}/{len(prompts_list)}: {output_path}")

    print(f"### Evaluation complete! {len(prompts_list)} images are in {output_dir} ###")


if __name__ == "__main__":
    args = parse_args()
    ras_manager.MANAGER.set_parameters(args)
    evaluate_on_parti(args)