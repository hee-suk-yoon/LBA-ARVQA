#!/bin/bash

set -x

# MODEL_PATH=/data/kakao/workspace/models/Qwen3-VL-4B-Instruct
# EVAL_MODEL_PATH=/data2/esyoon_hdd/workspace/videoqa-log/qwen3vl/lora/sft-4000-lr5e-5/checkpoint-40
MODEL_PATH=/data2/esyoon_hdd/workspace/models/Qwen3-VL-8B-Instruct
EVAL_MODEL_PATH=/data2/esyoon_hdd/workspace/videoqa-log/qwen3vl-8b/lora/sft-lr1e-6/checkpoint-79

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
    --output_dir /data2/esyoon_hdd/workspace/videoqa-log/qwen3vl-8b/lora/8b-sft-lr1e-6-step79-eval-ambiguity\
    --plot_loss \
    --overwrite_output_dir \
    --report_to none \
    --per_device_eval_batch_size 8 \
    --bf16 \
    --ddp_timeout 180000000
