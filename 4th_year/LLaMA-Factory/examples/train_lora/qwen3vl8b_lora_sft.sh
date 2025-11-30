#!/bin/bash

set -x

MODEL_PATH=Qwen/Qwen3-VL-8B-Instruct

llamafactory-cli train \
    --model_name_or_path ${MODEL_PATH} \
    --trust_remote_code \
    --stage sft \
    --do_train \
    --finetuning_type lora \
    --lora_rank 128 \
    --lora_target all \
    --dataset videoqa_train_data \
    --eval_dataset ambiguousqa_eval \
    --template qwen3_vl \
    --cutoff_len 8172 \
    --max_samples 10000 \
    --overwrite_cache \
    --preprocessing_num_workers 30 \
    --dataloader_num_workers 30 \
    --output_dir OUTDIR_HERE \
    --logging_steps 5 \
    --save_steps 10 \
    --plot_loss \
    --overwrite_output_dir \
    --save_only_model true \
    --report_to none \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-5 \
    --num_train_epochs 1.0 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16 \
    --ddp_timeout 180000000
