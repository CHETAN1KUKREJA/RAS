import argparse
import torch
from diffusers import StableDiffusion3Pipeline
from ras.utils.stable_diffusion_3.update_pipeline_sd3 import update_sd3_pipeline
from ras.utils import ras_manager
from ras.utils.ras_argparser import parse_args
from fvcore.nn import FlopCountAnalysis # <-- 1. Import the tool
import tqdm # For a nice progress bar

def sd3_inf(args):
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        torch_dtype=torch.float16
    )
    pipeline.to("cuda")
    pipeline = update_sd3_pipeline(pipeline)
    generator = torch.Generator("cuda").manual_seed(args.seed) if args.seed is not None else None
    
    # --- FLOP Counting Setup ---
    total_transformer_flops = 0
    
    # --- Manual Pipeline Replication ---
    """
        By feeding the same prompt to all three, you leverage their combined strengths:
            - The two CLIP models provide a strong understanding of the visual concepts and styles you're asking for.
            - The powerful T5 language model ensures the model correctly interprets complex sentences, spatial relationships (like "a cat sitting on top of a book"), and accurately spells out text.
    """
    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipeline.encode_prompt(
        prompt=args.prompt, negative_prompt=args.negative_prompt,
        prompt_2=args.prompt,
        prompt_3=args.prompt
    )
    # 2. Set timesteps
    pipeline.scheduler.set_timesteps(args.num_inference_steps)
    timesteps = pipeline.scheduler.timesteps
    
    # 3. Prepare latents
    latents = pipeline.prepare_latents(
        batch_size=1, 
        num_channels_latents=16, 
        height=args.height, 
        width=args.width, 
        dtype=torch.float16, 
        device="cuda", 
        generator=generator
    )

    # 4. Denoising loop
    for i, t in tqdm.tqdm(enumerate(timesteps)):
        latent_model_input = torch.cat([latents] * 2) # For classifier-free guidance
        
        # Prepare inputs for the transformer
        transformer_inputs = {
            "hidden_states": latent_model_input.to("cuda"),
            "encoder_hidden_states": torch.cat([prompt_embeds, negative_prompt_embeds]).to("cuda"),
            "pooled_projections": torch.cat([pooled_prompt_embeds, negative_pooled_prompt_embeds]).to("cuda"),
            "timestep": t.unsqueeze(0).to("cuda")
        }
        
        # ---> 2. Analyze FLOPs for the transformer on THIS step <---
        flop_counter = FlopCountAnalysis(pipeline.transformer, transformer_inputs)
        step_flops = flop_counter.total()
        total_transformer_flops += step_flops
        
        # Log the FLOPs for the current step
        active_ratio = ras_manager.MANAGER.get_active_token_ratio()
        print(f"Step {i:02d} | Active Tokens: {active_ratio:.1%} | GFLOPs: {step_flops / 1e9:.2f}")

        # ---> 3. Run the actual model prediction <---
        noise_pred = pipeline.transformer(**transformer_inputs).sample
        
        # Perform guidance
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # Scheduler step
        latents = pipeline.scheduler.step(noise_pred, t, latents).prev_sample

    # 5. Decode the final image
    image = pipeline.vae.decode(latents / pipeline.vae.config.scaling_factor).sample
    image = pipeline.image_processor.postprocess(image, output_type="pil")[0]
    
    # ---> 4. Print the final results <---
    total_gflops = total_transformer_flops / 1e9
    print("\n--- FLOP Count Summary ---")
    print(f"Total Transformer GFLOPs for {args.num_inference_steps} steps: {total_gflops:.2f} GFLOPs")
    print("--------------------------\n")
    
    image.save(args.output)


if __name__ == "__main__":
    args = parse_args()
    # You may need a helper in ras_manager to get the active token ratio
    ras_manager.MANAGER.set_parameters(args)
    sd3_inf(args)