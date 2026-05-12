#!/bin/bash
# Qwen3-4B-Instruct-2507 + diverse_sft, обычный hypernet (per_layer_activations).
# Train+eval на CE (датасет без logprobs_vals).
# GPUs: 4,5

set -euo pipefail

export CPATH=${CPATH:+$CPATH:}/home/jovyan/.mlspace/envs/torch/include/python3.10
export CUDA_HOME=${CUDA_HOME:-/home/jovyan/davydenko/cuda-12.1}

export COMET_API_KEY=${COMET_API_KEY:-sqS8EDaUqxp02LYC7lKyrrxoZ}
export COMET_PROJECT_NAME=${COMET_PROJECT_NAME:-d2l_qwen3_4b_diverse_sft}
export COMET_TAGS=${COMET_TAGS:-qwen3-4b,diverse_sft,no_random_repr,per_layer_activations}

export D2L_USE_RANDOM_REPR=0

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5}
port=29053

uv run accelerate launch --config_file accelerate_config.yaml --main_process_port $port \
    --num_processes=2 --gpu_ids all train.py \
    configs/main_exp/qwen/diverse_sft.yaml \
    --model_name_or_path=Qwen/Qwen3-4B-Instruct-2507 \
    --target_modules=down_proj --lora_r=8 --report_to=comet_ml \
    --eval_strategy=steps --eval_steps=50 --logging_steps=5 --save_steps=200 \
    --max_qas_len=2048 --max_qas_per_sample=1 --max_val_samples_per_ds=100 \
    --per_rank_gen=True --per_layer_processing=True --gen_lora_l1_reg_coef=0.0 \
    --max_steps=3000 --gradient_accumulation_steps=8 \
    --max_packed_inp_len=4096 --max_packed_ctx_len=6144 \
    --use_per_ctx_average_loss=True --use_kl_loss=False \
    --quantize_ctx_encoder=True \
    --learning_rate=4e-5 --seed=42 \
    "$@"
