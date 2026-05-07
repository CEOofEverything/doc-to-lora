import gc
import json
import os

from datasets import Dataset
from tqdm import tqdm

SRC_FULL  = "data/raw_datasets/diverse_sft/raw_tasks.jsonl"
SRC_TRAIN = "data/raw_datasets/diverse_sft/raw_tasks_train.jsonl"
OUT_DIR   = "data/raw_datasets/diverse_sft"


def to_sample(d):
    return {
        "context": d["context"],
        "prompts": d["tasks"],
        "responses": d["responses"],
        "id": d["id"],
    }


def save(samples, split):
    save_path = f"{OUT_DIR}/{split}/ds.parquet"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    n_qa = sum(len(s["prompts"]) for s in samples)
    print(f"[{split}] {len(samples)} contexts, {n_qa} qa pairs -> {save_path}")
    Dataset.from_list(samples).to_parquet(save_path)


if __name__ == "__main__":
    train_ids = set()
    with open(SRC_TRAIN) as f:
        for line in f:
            train_ids.add(json.loads(line)["id"])
    print(f"raw_tasks_train ids: {len(train_ids)}")

    train_samples, val_samples = [], []
    with open(SRC_FULL) as f:
        for line in tqdm(f, desc="reading raw_tasks"):
            d = json.loads(line)
            (train_samples if d["id"] in train_ids else val_samples).append(
                to_sample(d)
            )

    print(f"split: train={len(train_samples)}, validation={len(val_samples)}")
    assert len(val_samples) > 0, "validation split is empty"

    save(train_samples, "train")
    save(val_samples, "validation")

    del train_samples, val_samples
    gc.collect()
