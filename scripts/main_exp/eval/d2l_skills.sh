# main results
# batched
export D2L_USE_RANDOM_REPR=1
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled uv run run_eval.py --checkpoint_path train_outputs/runs/May08_08-51-19_parchiev-synthetic-data-0_c548e024/checkpoint-3000/pytorch_model.bin --datasets diverse_sft --split validation --max_ctx_chunk_len 6144 --eval_batch_size_gen 1 --max_val_samples_per_ds 50 --output_dir eval_results/qwen3_4b_skills_d2l

# iterative
#WANDB_MODE=disabled uv run run_eval.py --checkpoint_path train_outputs/runs/May08_08-51-19_parchiev-synthetic-data-0_c548e024/checkpoint-2600/pytorch_model.bin --datasets diverse_sft --split validation --max_ctx_chunk_len 6144 --eval_batch_size_gen 1 --use_iterative_mode
#Режим «iterative» имеет смысл только для чекпоинта, обученного с per_layer_activations
