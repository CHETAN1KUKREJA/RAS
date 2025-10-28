python ./cocoInference.py \
    --num_inference_steps 28 \
    --seed 29 \
    --sample_ratio 0.5 \
    --replace_with_flash_attn \
    --error_reset_steps "12,20" \
    --metric "std" \
    --method "default" \
    --scheduler_start_step 4 \
    --scheduler_end_step 28 \
    --patch_size 2 \
    --starvation_scale 1 \
    --high_ratio 0.3 \
    --start_index 0 \
    --end_index 10 \
    --output_dir ../outputs/ms_coco_RAS_sampleRatio_50_totalSteps_28_fullInf_6/ \
    --csv_file ../additional_scripts/longest_captions_sample.csv \
    # --std_experiment \
    # --enable_index_fusion \
    # --skip_num_step 256 \
    # --skip_num_step_length 4 \
