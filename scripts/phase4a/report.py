#!/usr/bin/env python3
"""Write Phase 4A-Data markdown report from assemble + text-only metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, ROOT, load_json


def pct(x) -> str:
    if x is None:
        return "n/a"
    return f"{100 * x:.1f}%"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stats", default=str(DATA_DIR / "assemble_stats.json"))
    p.add_argument("--text-only", default=str(DATA_DIR / "text_only_metrics.json"))
    p.add_argument("--out", default=str(ROOT / "eval_results/qwen/phase4a/phase4a_data_report.md"))
    p.add_argument("--out-root", default=str(ROOT / "phase4a_data_result.md"))
    return p.parse_args()


def render(stats: dict, text: dict | None) -> str:
    lines = [
        "# Phase 4A-Data 结果：Clean Counterfactual Negative Construction",
        "",
        "本阶段不训练模型。用 InternVL3.5-8B 为 Phase 3A SRO3000 重写 linguistically plausible、visually false 的 counterfactual negative。",
        "",
        "## 产量",
        "",
        f"- 输入 attempts：{stats.get('n_attempts')}",
        f"- Clean 保留：{stats.get('n_clean')}",
        f"- 丢弃：{stats.get('n_dropped')}",
        f"- Keep rate：{pct(stats.get('keep_rate'))}",
        "",
        "丢弃原因：",
        "",
        "| reason | n |",
        "|---|---:|",
    ]
    for k, v in sorted((stats.get("drop_reasons") or {}).items(), key=lambda kv: -kv[1]):
        lines.append(f"| {k} | {v} |")
    lines += [
        "",
        f"- unique clean negative relations：{stats.get('unique_negative_relations')}",
        f"- unique old negative relations：{stats.get('unique_old_negative_relations')}",
        f"- clean 与 old 相同：{stats.get('n_negative_equals_old')}",
        f"- old grammar-smell rate：{pct(stats.get('old_negative_grammar_smell_rate'))}",
        f"- clean grammar-smell rate：{pct(stats.get('clean_negative_grammar_smell_rate'))}",
        "",
        "Candidate rank（选中第几个 InternVL candidate）：",
        "",
        "| rank | n |",
        "|---|---:|",
    ]
    for k, v in sorted((stats.get("candidate_rank") or {}).items(), key=lambda kv: kv[0]):
        lines.append(f"| {k} | {v} |")
    lines += [
        "",
        "Top clean negative relations：",
        "",
        "| relation | n |",
        "|---|---:|",
    ]
    for rel, n in stats.get("top_negative_relations") or []:
        lines.append(f"| {rel} | {n} |")
    lines += [
        "",
        "## Text-only shortcut QC",
        "",
    ]
    if not text:
        lines.append("尚未运行 `text_only_shortcut.py`。")
    else:
        old = (text.get("old_negative") or {}).get("implausible_rate")
        clean = (text.get("clean_negative") or {}).get("implausible_rate")
        pos = (text.get("positive") or {}).get("implausible_rate")
        lines += [
            "InternVL text-only：statement 是否 linguistically/semantically plausible（不看图）。",
            "将 Implausible 视为预测 Negative、Plausible 视为 Positive。",
            "",
            f"- Positive implausible rate：{pct(pos)}",
            f"- Old negative implausible rate：{pct(old)}",
            f"- Clean negative implausible rate：{pct(clean)}",
            f"- Paired old text-only Acc：{pct((text.get('paired_old_textonly_acc') or {}).get('acc'))} "
            f"(n={(text.get('paired_old_textonly_acc') or {}).get('n')})",
            f"- Paired clean text-only Acc：{pct((text.get('paired_clean_textonly_acc') or {}).get('acc'))} "
            f"(n={(text.get('paired_clean_textonly_acc') or {}).get('n')})",
            "",
            "计划期望：old Acc 明显更高（75%+），clean 接近 50%–60%。",
            "",
        ]
    lines += [
        "## 人工 QC",
        "",
        "分层抽样表：`data/phase4a/qc/review_300.tsv`（100 spatial / 100 contact / 100 other，按 GT positive relation family）；浏览页：`data/phase4a/qc/review.html`。",
        "",
        "**2026-09-16：跳过人工填写。** 不统计 overall clean rate，不作为训练门槛。后续以自动 filter + text-only shortcut 为准。",
        "",
        "## 本阶段问题",
        "",
        "> 能否利用 InternVL 根据当前图像和真实 SRO，稳定产生语言上自然、语义上合理、但当前图像中不成立的 counterfactual relation，从而消除原有 random negative 中明显的 linguistic shortcut？",
        "",
        "训练对照（Old-S3000 vs Clean-S3000）按计划为可选 sanity check，本报告不包含训练数字。",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    stats = load_json(Path(args.stats))
    text = None
    tp = Path(args.text_only)
    if tp.exists():
        text = load_json(tp)
    md = render(stats, text)
    for out in (args.out, args.out_root):
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(md, encoding="utf-8")
        print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
