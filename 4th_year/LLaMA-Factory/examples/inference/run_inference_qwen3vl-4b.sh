#!/bin/bash

set -x

MODEL_PATH=Qwen/Qwen3-VL-8B-Instruct
EVAL_MODEL_PATH=TRAINED_MODEL_DIR_HERE

llamafactory-cli train \
    --model_name_or_path ${MODEL_PATH} \
    --adapter_name_or_path ${EVAL_MODEL_PATH} \
    --trust_remote_code \
    --stage sft \
    --do_predict \
    --predict_with_generate \
    --finetuning_type lora \
    --infer_backend huggingface \
    --lora_rank 32 \
    --lora_target all \
    --eval_dataset ambiguousqa_ambiguity_eval \
    --template qwen3_vl \
    --cutoff_len 8172 \
    --max_samples 2000 \
    --overwrite_cache \
    --preprocessing_num_workers 24 \
    --dataloader_num_workers 24 \
    --output_dir SAVE_OUTPUT_DIR_HERE \
    --plot_loss \
    --overwrite_output_dir \
    --report_to none \
    --per_device_eval_batch_size 8 \
    --bf16 \
    --ddp_timeout 180000000
