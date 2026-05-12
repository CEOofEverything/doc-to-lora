#!/bin/bash

# Triton/Inductor JIT-компилит cuda_utils.so через gcc и подключает -I/usr/include/python3.10,
# но в системном python3.10 нет headers (sudo apt install python3.10-dev недоступен).
# CPATH добавляет фоллбек include-путь, gcc подхватит Python.h оттуда.
export CPATH=${CPATH:+$CPATH:}/home/jovyan/.mlspace/envs/torch/include/python3.10
export CUDA_HOME=${CUDA_HOME:-/home/jovyan/davydenko/cuda-12.1}

export COMET_API_KEY="sqS8EDaUqxp02LYC7lKyrrxoZ"
export COMET_PROJECT_NAME="d2l_skills_gemma2b"

#PY_ENV=/home/jovyan/.mlspace/envs/prag_req
#ACCEL="${PY_ENV}/bin/accelerate"

port=29052
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
uv run accelerate launch --config_file accelerate_config.yaml --main_process_port $port \
--num_processes=1 --gpu_ids all train.py \
configs/main_exp/test.yaml \
--model_name_or_path=google/gemma-2-2b-it \
--target_modules=down_proj --lora_r=8 --report_to=comet_ml \
--eval_strategy=steps --eval_steps=5 --max_qas_len=2048 --max_qas_per_sample=1 --max_val_samples_per_ds=10 \
--per_rank_gen=True --per_layer_processing=True --gen_lora_l1_reg_coef=0.1 \
--max_steps=80 --gradient_accumulation_steps=8 --max_packed_inp_len=4096 \
--max_packed_ctx_len=6144 --use_per_ctx_average_loss=True --use_kl_loss=True \
--quantize_ctx_encoder=True


