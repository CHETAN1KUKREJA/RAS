#!/bin/bash

# --- Script to run Stable Diffusion 3 inference with multiple prompts ---

# Define an array of prompts
# 5 Detailed Prompts
PROMPTS=(
    "A photorealistic image of a majestic lion with a flowing mane, standing on a rocky outcrop at sunrise, dramatic lighting, misty mountains in the background, hyperdetailed, 8k resolution."
    "An oil painting of a cozy, cluttered artist's studio, sunlight streaming through a large window, canvases and paintbrushes scattered around, a half-finished masterpiece on an easel, style of Vermeer."
    "Macro photograph of a single dewdrop on a vibrant red rose petal, reflecting the morning sky, intricate details, sharp focus, bokeh background, award-winning photography."
    "A whimsical fantasy cityscape built inside a giant crystal cave, glowing magical rivers flowing between buildings, mythical creatures flying in the sky, style of Studio Ghibli, cinematic lighting."
    "A 1950s retro-futuristic diner on Mars, chrome and neon lights, a classic car parked outside under a red sky with two moons, astronauts drinking milkshakes, photorealistic, cinematic."
    # 5 Simple Prompts
    "a green car on a street"
    "a cat sitting on a chair"
    "a tall tree in a field"
    "a red apple on a table"
    "a boat on the water at sunset"
)

# Initialize a counter for the output folder name
FOLDER_COUNTER=1

# Loop through each prompt in the array
for prompt in "${PROMPTS[@]}"; do
    echo "-----------------------------------------------------"
    echo "Running inference for prompt ${FOLDER_COUNTER}: \"${prompt}\""
    echo "-----------------------------------------------------"
    
    # Create a variable for the output directory name for clarity
    OUTPUT_DIR_NAME="${FOLDER_COUNTER}"

    python ./src/ras/model_inference/stable_diffusion_3_inference.py \
        --prompt "${prompt}" \
        --output "output.png" \
        --num_inference_steps 28 \
        --seed 29 \
        --sample_ratio 0.5 \
        --replace_with_flash_attn \
        --error_reset_steps "28" \
        --metric "std" \
        --scheduler_start_step 28 \
        --scheduler_end_step 28 \
        --patch_size 2 \
        --starvation_scale 1 \
        --std_experiment \
        --high_ratio 0.3 \
        --name_folder "${OUTPUT_DIR_NAME}"

    # --- NEW CODE: Move and rename the generated image ---
    # Check if the image was created before trying to move it
    if [ -f "/DATA/rohit/extra/RAS/output.png" ]; then
        # Move the temporary output.png into the correct folder with a permanent name
        mv "/DATA/rohit/extra/RAS/output.png" "std_analysis_${OUTPUT_DIR_NAME}/final_image.png"
        echo "Saved final image to ${OUTPUT_DIR_NAME}/final_image.png"
    else
        echo "Warning: output.png was not created for prompt ${FOLDER_COUNTER}."
    fi
    # ---------------------------------------------------
    
    python ./scripts/heatmap.py "std_analysis_${OUTPUT_DIR_NAME}" "std_analysis_${OUTPUT_DIR_NAME}/final_image.png" --std_heatmap --mean_heatmap
    # Increment the counter for the next run
    ((FOLDER_COUNTER++))
done

echo "All prompts have been processed."
