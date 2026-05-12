# no truncation
CUDA_VISIBLE_DEVICES=5 WANDB_MODE=disabled uv run run_eval.py --model_name_or_path Qwen/Qwen3-4B-Instruct-2507 --datasets diverse_sft --split validation --eval_batch_size_gen 1 --max_new_tokens 512 --output_dir eval_results/qwen3_4b_skills_full --max_val_samples_per_ds 100


# no context
CUDA_VISIBLE_DEVICES=6 WANDB_MODE=disabled uv run run_eval.py --model_name_or_path Qwen/Qwen3-4B-Instruct-2507 --datasets diverse_sft --split validation --eval_batch_size_gen 1 --remove_context --max_new_tokens 512 --output_dir eval_results/qwen3_4b_skills_no_ctx --max_val_samples_per_ds 100

