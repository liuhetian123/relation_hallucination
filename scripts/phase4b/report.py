#!/usr/bin/env python3
"""Write Phase 4B markdown report from stratified analysis + falsity recheck."""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "eval_results/qwen/phase4b"
SEED = 42
PRIMARY = (
    ("rbench", "s3000_vs_base"),
    ("mmrel_adv", "s3000_vs_base"),
    ("rbench", "old_vs_base"),
    ("mmrel_adv", "old_vs_base"),
    ("rbench", "clean_vs_base"),
    ("mmrel_adv", "clean_vs_base"),
)


def loadj(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pct(x, digits=2) -> str:
    if x is None:
        return ""
    return f"{100.0 * float(x):.{digits}f}"


def pp(x) -> str:
    if x is None:
        return ""
    return f"{100.0 * float(x):+.2f}"


def ci_str(ci) -> str:
    if not ci or ci[0] is None:
        return ""
    return f"[{100*ci[0]:+.2f}, {100*ci[1]:+.2f}]"


def layer_row(name, rec, a, b) -> str:
    da = rec.get("delta") or {}
    ma = rec.get(a) or {}
    mb = rec.get(b) or {}
    flag = " **功效不足**" if rec.get("low_power") else ""
    return (
        f"| {name}{flag} | {rec.get('n','')} | {pct(ma.get('Accuracy'))} | {pct(mb.get('Accuracy'))} | "
        f"{pp(da.get('delta'))} | {ci_str(da.get('ci95'))} | {(rec.get('mcnemar') or {}).get('p_value', '')} | "
        f"{pct(ma.get('Yes_ratio'))} | {pct(mb.get('Yes_ratio'))} | {ma.get('FP','')}/{ma.get('FN','')} | "
        f"{mb.get('FP','')}/{mb.get('FN','')} |"
    )


def classify_contrast(levels: dict) -> str:
    lemma = (levels or {}).get("lemma") or {}
    seen = (lemma.get("seen") or {}).get("delta") or {}
    unseen = (lemma.get("unseen") or {}).get("delta") or {}
    did = lemma.get("did") or {}
    ds, du, d = seen.get("delta"), unseen.get("delta"), did.get("did")
    cs, cu, cd = seen.get("ci95") or [None, None], unseen.get("ci95") or [None, None], did.get("ci95") or [None, None]
    if ds is None or du is None:
        return "incomplete"
    seen_pos = (cs[0] is not None and cs[0] > 0) or (ds >= 0.02 and d is not None and d * ds > 0)
    unseen_flat = (cu[0] is not None and cu[0] <= 0 <= cu[1]) or abs(du) < 0.01
    both_flat = (
        abs(ds) < 0.01
        and abs(du) < 0.01
        and (cs[0] is None or cs[0] <= 0 <= cs[1])
        and (cu[0] is None or cu[0] <= 0 <= cu[1])
        and (d is None or abs(d) < 0.01 or (cd[0] is not None and cd[0] <= 0 <= cd[1]))
    )
    if du > ds + 0.005 and abs(du) >= 0.01:
        return "C"
    if seen_pos and unseen_flat:
        return "A"
    if both_flat:
        return "B"
    # weak / mixed: treat as B if CIs contain 0
    if (cs[0] is None or cs[0] <= 0 <= cs[1]) and (cu[0] is None or cu[0] <= 0 <= cu[1]):
        return "B"
    return "mixed"


def overall_case(blob: dict) -> tuple[str, list[str]]:
    votes = []
    for bench, contrast in PRIMARY:
        levels = ((blob.get("contrasts") or {}).get(bench) or {}).get(contrast, {}).get("levels")
        votes.append((f"{contrast}/{bench}", classify_contrast(levels)))
    labels = [v for _, v in votes]
    if labels.count("C") >= 2:
        case = "C"
    elif labels.count("A") >= 3:
        case = "A"
    elif labels.count("B") + labels.count("mixed") >= 4:
        case = "B"
    else:
        case = "B" if labels.count("A") < labels.count("B") else "mixed"
    return case, [f"{k}={v}" for k, v in votes]


def main() -> None:
    blob = loadj(OUT / "metrics/phase4b_stratified_analysis.json")
    if not blob:
        raise SystemExit("missing stratified analysis json")
    falsity = loadj(OUT / "metrics/falsity_recheck_100.json")
    extracted = load_jsonl(OUT / "metrics/relation_extraction.jsonl")
    meta = blob.get("extraction_meta") or {}

    case, votes = overall_case(blob)
    next_step = {
        "A": "覆盖度问题。下一步做数据多样性扩展（新图源 + relation 词表对齐外部分布），再进 Phase 4C。",
        "B": "量级/能力问题。下一步优先 Qwen2.5-VL-7B 或加训练 exposure（2 epoch），不再叠数据技巧。",
        "C": "unseen 增益大于 seen。优先怀疑 relation 抽取或 synonym 匹配；若人工核对无 bug，按 H2 处理。",
        "mixed": "各对比不完全一致，报告以 lemma 分层表为准，默认按 H2 谨慎解读。",
    }[case]

    lines = [
        "# Phase 4B 结果：外部平坦的归因诊断（OOD 覆盖度 vs 能力不足）",
        "",
        "不训练、不对 base/s3000/old/clean 做新推理。任务 A 在冻结 unique questions 上按训练 relation 词表分层；任务 B 对 100 条 clean negative 做独立 InternVL visual falsity 复验。",
        "",
        "## 对账",
        "",
        "重算 unique-question Acc，须与 `phase4a_train_result.md` / `phase3a_result.md` Confusion 表相差 ≤ 0.05 pp。",
        "",
        "| Model | R-Bench Acc | 发布值 | MMRel Acc | 发布值 |",
        "|---|---:|---:|---:|---:|",
    ]
    recon = blob.get("reconcile") or {}
    for name in ("base", "s3000", "old", "clean"):
        r = recon.get(name) or {}
        lines.append(
            f"| {name} | {r.get('rbench','')} | "
            f"{ {'base':83.11,'s3000':83.58,'old':83.26,'clean':82.95}[name] } | "
            f"{r.get('mmrel_adv','')} | "
            f"{ {'base':69.01,'s3000':71.09,'old':70.31,'clean':70.83}[name] } |"
        )
    lines += [
        "",
        "## Relation 抽取",
        "",
        f"- R-Bench n={meta.get('n_rbench')}，unparsed={meta.get('rbench_unparsed')} "
        f"({pct(meta.get('rbench_unparsed_rate'))}%)",
        f"- MMRel n={meta.get('n_mmrel')}，unparsed={meta.get('mmrel_unparsed')} "
        f"({pct(meta.get('mmrel_unparsed_rate'))}%)",
        f"- old/s3000 词表大小：{meta.get('vocab_old_n')}；clean 词表：{meta.get('vocab_clean_n')}",
        f"- lemma-seen（old 词表）：R-Bench {((meta.get('seen_lemma_old') or {}).get('rbench'))} / "
        f"MMRel {((meta.get('seen_lemma_old') or {}).get('mmrel_adv'))}",
        f"- lemma-seen（clean 词表）：R-Bench {((meta.get('seen_lemma_clean') or {}).get('rbench'))} / "
        f"MMRel {((meta.get('seen_lemma_clean') or {}).get('mmrel_adv'))}",
        "",
        "AMBER-dr-fast 跳过：题面几乎全是 “Is there direct contact between X and Y?”，无法抽取多样 relation。",
        "",
        "Parse reasons：",
        "",
        "```json",
        json.dumps(meta.get("parse_reasons") or {}, indent=2, ensure_ascii=False),
        "```",
        "",
    ]
    rb_rate = meta.get("rbench_unparsed_rate") or 0
    if rb_rate > 0.15:
        fails = [r for r in extracted if r["benchmark"] == "rbench" and not r["parse_ok"]]
        rng = random.Random(SEED)
        sample = fails if len(fails) <= 20 else rng.sample(fails, 20)
        lines += ["R-Bench unparsed > 15%，抽样 20 条：", "", "| reason | statement |", "|---|---|"]
        for r in sample:
            stmt = (r.get("matched_statement") or "").replace("|", "/")
            lines.append(f"| {r.get('parse_reason')} | {stmt} |")
        lines.append("")

    lines += [
        "## 主分析（lemma seen/unseen）",
        "",
        "seen = 外部题 relation 在对应训练词表上 exact 或 lemma 命中。s3000/old 用 old 词表，clean 用 clean 词表。",
        "",
    ]
    for bench in ("rbench", "mmrel_adv"):
        lines.append(f"### {bench}")
        lines.append("")
        for contrast in ("s3000_vs_base", "old_vs_base", "clean_vs_base", "clean_vs_old"):
            rec = ((blob.get("contrasts") or {}).get(bench) or {}).get(contrast) or {}
            levels = rec.get("levels") or {}
            lemma = levels.get("lemma") or {}
            a, b = contrast.split("_vs_")
            lines.append(f"#### {contrast}（vocab={rec.get('vocab')}）")
            lines.append("")
            lines.append(
                "| layer | n | Acc_A | Acc_B | ΔAcc | 95% CI | McNemar p | Yes_A | Yes_B | FP/FN_A | FP/FN_B |"
            )
            lines.append("|---|---:|---:|---:|---:|---|---:|---:|---:|---|---|")
            for layer in ("seen", "unseen", "unparsed"):
                lines.append(layer_row(layer, lemma.get(layer) or {}, a, b))
            did = lemma.get("did") or {}
            lines.append("")
            lines.append(
                f"DiD Δ(seen)−Δ(unseen) = {pp(did.get('did'))} pp，CI {ci_str(did.get('ci95'))}；"
                f"n_seen={did.get('n_seen')} n_unseen={did.get('n_unseen')}。"
            )
            lines.append("")
            lines.append("敏感性（exact / synonym-family）：")
            lines.append("")
            lines.append("| level | seen n | seen ΔAcc | seen CI | unseen n | unseen ΔAcc | unseen CI | DiD | DiD CI |")
            lines.append("|---|---:|---:|---|---:|---:|---|---:|---|")
            for level in ("exact", "lemma", "synonym-family"):
                lv = levels.get(level) or {}
                s, u, d = lv.get("seen") or {}, lv.get("unseen") or {}, lv.get("did") or {}
                lines.append(
                    f"| {level} | {s.get('n','')} | {pp((s.get('delta') or {}).get('delta'))} | "
                    f"{ci_str((s.get('delta') or {}).get('ci95'))} | {u.get('n','')} | "
                    f"{pp((u.get('delta') or {}).get('delta'))} | {ci_str((u.get('delta') or {}).get('ci95'))} | "
                    f"{pp(d.get('did'))} | {ci_str(d.get('ci95'))} |"
                )
            lines.append("")
        if bench == "mmrel_adv":
            lines.append("### MMRel 图片领域（lemma）")
            lines.append("")
            rec = ((blob.get("contrasts") or {}).get("mmrel_adv") or {}).get("clean_vs_base") or {}
            for domain, drec in (rec.get("domain") or {}).items():
                lv = (drec or {}).get("lemma") or {}
                lines.append(f"**clean_vs_base / {domain}**")
                lines.append("")
                lines.append("| layer | n | Acc_base | Acc_clean | ΔAcc | 95% CI |")
                lines.append("|---|---:|---:|---:|---:|---|")
                for layer in ("seen", "unseen"):
                    x = lv.get(layer) or {}
                    da = x.get("delta") or {}
                    lines.append(
                        f"| {layer} | {x.get('n','')} | {pct((x.get('base') or {}).get('Accuracy'))} | "
                        f"{pct((x.get('clean') or {}).get('Accuracy'))} | {pp(da.get('delta'))} | {ci_str(da.get('ci95'))} |"
                    )
                lines.append("")

    lines += [
        "## 任务 A 判定",
        "",
        f"各对比 lemma 标签：{', '.join(votes)}",
        "",
        f"**Case {case}。** {next_step}",
        "",
        "## 任务 B：Clean Negative 标签噪声（100 条）",
        "",
    ]
    if not falsity:
        lines.append("InternVL 复验尚未完成。")
    else:
        lines += [
            f"- n={falsity.get('n')}",
            f"- InternVL Yes（负例其实成立）：{pct(falsity.get('yes_rate'))}%",
            f"- Uncertain：{pct(falsity.get('uncertain_rate'))}%",
            f"- No（负例确实不成立）：{pct(falsity.get('no_rate'))}%",
            f"- 噪声上限（Yes 或 Uncertain）：{pct(falsity.get('noise_upper_yes_or_uncertain'))}%",
            f"- 自动判定：`{falsity.get('verdict')}`",
            "",
            "人工列留空，不阻塞本报告。页面：`data/phase4a/qc/review_100_falsity.html`。",
            "",
        ]
        nu = falsity.get("noise_upper_yes_or_uncertain") or 0
        if nu <= 0.05:
            lines.append("噪声率 ≤ 5%：clean 数据质量可信，train neg Acc 83.5% 主要反映难度。")
        elif nu <= 0.15:
            lines.append("噪声率 5%–15%：可用，Phase 4C 必须引用该噪声率作为解释边界。")
        else:
            lines.append("噪声率 > 15%：Phase 4C 前应加一轮全量 visual falsity filter。")
    lines += [
        "",
        "## 本阶段问题",
        "",
        "> 微调带来的关系判别增益，在外部 benchmark 上是否集中于训练 relation 词表覆盖到的题目？（顺带：clean negative 的视觉假性标签噪声有多大？）",
        "",
        f"**回答：Case {case}。** {next_step}",
        "",
    ]
    text = "\n".join(lines) + "\n"
    (OUT / "phase4b_report.md").write_text(text, encoding="utf-8")
    (ROOT / "phase4b_result.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"wrote {OUT / 'phase4b_report.md'}")


if __name__ == "__main__":
    main()
