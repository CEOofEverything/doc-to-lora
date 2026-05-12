#!/bin/bash
# Qwen3-4B-Instruct-2507 + diverse_sft, экспериментальная random-repr Perceiver-ветка.
# Train+eval на CE (датасет без logprobs_vals).
# Per CLAUDE.md: random-repr требует ctx_encoder_type=early_exit и per_rank_gen=False
# (per_layer_activations OOM-ит на 80GB).
# GPUs: 6,7

set -euo pipefail

export CPATH=${CPATH:+$CPATH:}/home/jovyan/.mlspace/envs/torch/include/python3.10
export CUDA_HOME=${CUDA_HOME:-/home/jovyan/davydenko/cuda-12.1}

export COMET_API_KEY=${COMET_API_KEY:-sqS8EDaUqxp02LYC7lKyrrxoZ}
export COMET_PROJECT_NAME=${COMET_PROJECT_NAME:-d2l_qwen3_4b_diverse_sft}
export COMET_TAGS=${COMET_TAGS:-qwen3-4b,diverse_sft,random_repr,early_exit}

export D2L_USE_RANDOM_REPR=1

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5,6,7}
port=29054

uv run accelerate launch --config_file accelerate_config.yaml --main_process_port $port \
    --num_processes=3 --gpu_ids all train.py \
    configs/main_exp/qwen/diverse_sft.yaml \
    --model_name_or_path=Qwen/Qwen3-4B-Instruct-2507 \
    --target_modules=down_proj --lora_r=8 --report_to=comet_ml \
    --eval_strategy=steps --eval_steps=50 --logging_steps=5 --save_steps=200 \
    --max_qas_len=2048 --max_qas_per_sample=1 --max_val_samples_per_ds=100 \
    --ctx_encoder_type=early_exit \
    --per_rank_gen=False --per_layer_processing=True --gen_lora_l1_reg_coef=0.0 \
    --max_steps=3000 --gradient_accumulation_steps=8 \
    --max_packed_inp_len=4096 --max_packed_ctx_len=6144 \
    --use_per_ctx_average_loss=True --use_kl_loss=True \
    --quantize_ctx_encoder=True \
    --learning_rate=4e-5 --seed=42 \
    "$@"
