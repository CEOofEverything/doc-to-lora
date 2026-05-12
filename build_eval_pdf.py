"""Builds eval_results/qwen3_4b_skills_summary.pdf — metrics tables + curves only."""

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib.styles import getSampleStyleSheet


ROOT = Path("/home/jovyan/yakubenko/doc-to-lora")

RUNS = [
    ("Base + full ICL",
     ROOT / "eval_results/Qwen/Qwen3-4B-Instruct-2507/20260508-155229_a52f3fdf",
     "_no_context"),
    ("Base, no context",
     ROOT / "eval_results/Qwen/Qwen3-4B-Instruct-2507/20260508-155234_934cd511",
     "_no_context"),
    ("D2L (random_repr)",
     ROOT / "eval_results/qwen3_4b_skills_d2l/20260508-153015_10c0a709",
     ""),
]

CURVES_DIR = ROOT / "eval_results/Qwen/Qwen3-4B-Instruct-2507/20260508-155229_a52f3fdf"


def load(p: Path) -> dict:
    with open(p / "all_results.json") as f:
        return json.load(f)


def find_key(d: dict, *needles: str) -> str | float:
    for k, v in d.items():
        if all(n in k for n in needles):
            return v
    return "—"


def fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    if isinstance(v, int):
        return str(v)
    return str(v)


def main() -> None:
    out_path = ROOT / "eval_results/qwen3_4b_skills_summary.pdf"
    styles = getSampleStyleSheet()
    h2 = styles["Heading2"]

    doc = SimpleDocTemplate(
        str(out_path), pagesize=A4,
        leftMargin=1.5 * cm, rightMargin=1.5 * cm,
        topMargin=1.5 * cm, bottomMargin=1.5 * cm,
    )
    story = []

    # === Config ===
    story.append(Paragraph("Training config (qwen_diverse_sft_random.sh)", h2))
    cfg_rows = [
        ["model_name_or_path", "Qwen/Qwen3-4B-Instruct-2507"],
        ["target_modules / lora_r", "down_proj / 8"],
        ["ctx_encoder_type", "early_exit"],
        ["per_rank_gen", "False"],
        ["per_layer_processing", "True"],
        ["gen_lora_l1_reg_coef", "0.0"],
        ["use_kl_loss", "False"],
        ["use_per_ctx_average_loss", "True"],
        ["quantize_ctx_encoder", "True"],
        ["max_steps", "3000"],
        ["gradient_accumulation_steps", "8"],
        ["num_processes (GPUs)", "3"],
        ["max_packed_inp_len / max_packed_ctx_len", "4096 / 6144"],
        ["max_qas_len / max_qas_per_sample", "2048 / 1"],
        ["max_val_samples_per_ds", "100"],
        ["learning_rate", "4e-5"],
        ["seed", "42"],
        ["D2L_USE_RANDOM_REPR", "1"],
    ]
    t = Table(cfg_rows, colWidths=[6.5 * cm, 9.5 * cm])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.4 * cm))

    # === Headline metric ===
    story.append(Paragraph("ROUGE-L F1 (validation, n=50, seed=42)", h2))
    rows = [["Run", "ROUGE-L F1", "n", "Runtime (s)", "samples/s"]]
    for label, run_dir, _ in RUNS:
        d = load(run_dir)
        f1 = find_key(d, "rougeL.f1")
        # exclude num_samples_ and len_ keys
        f1 = next(
            v for k, v in d.items()
            if k.endswith("_rougeL.f1") and "num_samples" not in k
        )
        n = next(
            v for k, v in d.items() if "num_samples_rougeL.f1" in k and "len_" not in k
        )
        runtime = next(v for k, v in d.items() if k.endswith("_runtime"))
        sps = next(v for k, v in d.items() if k.endswith("_samples_per_second"))
        rows.append([label, fmt(f1), fmt(n), fmt(runtime), fmt(sps)])

    t = Table(rows, colWidths=[4.5 * cm, 3 * cm, 1.5 * cm, 3 * cm, 3 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dddddd")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.4 * cm))

    # === Length-binned ===
    story.append(Paragraph("ROUGE-L F1 by input length (tokens)", h2))
    bins = [("512-1023", 7), ("1024-2047", 18), ("2048-4095", 25), ("0-8191", 50)]
    headers = ["Run"] + [f"{b}\n(n={n})" for b, n in bins]
    rows = [headers]
    for label, run_dir, _ in RUNS:
        d = load(run_dir)
        row = [label]
        for b, _ in bins:
            # match key ending with _rougeL.f1_len_<b>
            v = next(
                (v for k, v in d.items()
                 if k.endswith(f"_rougeL.f1_len_{b}") and "num_samples" not in k),
                "—",
            )
            row.append(fmt(v))
        rows.append(row)

    t = Table(rows, colWidths=[4.5 * cm, 3 * cm, 3 * cm, 3 * cm, 3 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dddddd")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.6 * cm))

    # === Curves ===
    story.append(Paragraph("Train loss vs step", h2))
    train_img = CURVES_DIR / "train_loss VS step.jpeg"
    story.append(Image(str(train_img), width=16 * cm, height=8 * cm))
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("Eval loss (diverse_sft) vs step", h2))
    eval_img = CURVES_DIR / "eval_diverse_sft_loss VS step.jpeg"
    story.append(Image(str(eval_img), width=16 * cm, height=8 * cm))

    doc.build(story)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
