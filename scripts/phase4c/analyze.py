#!/usr/bin/env python3
"""Phase 4C multi-format vs clean vs base: paired tests and Case A/B/C."""

from __future__ import annotations

import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/qwen_eval"))
sys.path.insert(0, str(ROOT / "scripts/phase2b"))
from score_fast import score_amber, score_mmrel, score_rbench  # noqa: E402
from score_yesno import gold_yes, load_json_or_jsonl, metrics, pred_yes  # noqa: E402

OUT = ROOT / "eval_results/qwen/phase4c"
P4A = ROOT / "eval_results/qwen/phase4a"
FAST = ROOT / "eval_results/qwen/phase2b/fast_subsets"
HELDOUT_Q = ROOT / "eval_results/qwen/phase2/heldout_verification.jsonl"
HELDOUT_F2_Q = ROOT / "data/phase4c/heldout_f2.jsonl"
N_BOOT = 2000
SEED = 42

BASE_FAST = {
    "rbench": ROOT / "eval_results/qwen/rbench/answers/qwen25vl3b_base_image-level.jsonl",
    "mmrel_adv": ROOT / "eval_results/qwen/mmrel/answers/qwen25vl3b_base_mmrel_adv.jsonl",
    "amber_dr": ROOT / "eval_results/qwen/amber/answers/qwen25vl3b_base_amber_dr.jsonl",
}
CLEAN_FAST = {
    "rbench": P4A / "answers/clean_rbench_fast.jsonl",
    "mmrel_adv": P4A / "answers/clean_mmrel_adv_fast.jsonl",
    "amber_dr": P4A / "answers/clean_amber_dr_fast.jsonl",
}
MF_FAST = {
    "rbench": OUT / "answers/mf_rbench_fast.jsonl",
    "mmrel_adv": OUT / "answers/mf_mmrel_adv_fast.jsonl",
    "amber_dr": OUT / "answers/mf_amber_dr_fast.jsonl",
}
PUBLISHED_CLEAN_VS_BASE = {
    "rbench": {"delta_pp": -0.16, "p": 0.7705, "n": 1924},
    "mmrel_adv": {"delta_pp": 1.82, "p": 0.1456, "n": 384},
}


def loadj(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def qid_aliases(qid):
    yield qid
    if isinstance(qid, int):
        yield str(qid)
    elif isinstance(qid, str) and qid.isdigit():
        yield int(qid)


def answer_preds(path: Path) -> dict:
    amap = {}
    if not path.exists():
        return amap
    for row in load_json_or_jsonl(path):
        qid = row.get("question_id", row.get("id"))
        text = row.get("text") or row.get("response") or ""
        pred = 1 if pred_yes(text) else 0
        for key in qid_aliases(qid):
            amap[key] = pred
    return amap


def lookup(amap: dict, qid):
    for key in qid_aliases(qid):
        if key in amap:
            return amap[key]
    return None


def mcnemar(a_ok: list[int], b_ok: list[int]) -> dict:
    b = c = 0
    for x, y in zip(a_ok, b_ok):
        if x and not y:
            b += 1
        elif (not x) and y:
            c += 1
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "chi2_cc": 0.0, "p_value": 1.0, "n_discordant": 0}
    chi2 = (abs(b - c) - 1) ** 2 / n
    p = math.erfc(math.sqrt(chi2 / 2))
    return {"b": b, "c": c, "chi2_cc": chi2, "p_value": p, "n_discordant": n}


def bootstrap_delta(a_ok: list[int], b_ok: list[int], n_boot=N_BOOT, seed=SEED) -> dict:
    rng = random.Random(seed)
    n = len(a_ok)
    if not n:
        return {"delta": 0.0, "ci95": [0.0, 0.0], "n": 0, "a_acc": 0.0, "b_acc": 0.0, "n_boot": n_boot}
    acc_a = sum(a_ok) / n
    acc_b = sum(b_ok) / n
    deltas = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        da = sum(a_ok[i] for i in idx) / n
        db = sum(b_ok[i] for i in idx) / n
        deltas.append(db - da)
    deltas.sort()
    return {
        "a_acc": acc_a,
        "b_acc": acc_b,
        "delta": acc_b - acc_a,
        "ci95": [deltas[int(0.025 * n_boot)], deltas[int(0.975 * n_boot)]],
        "n": n,
        "n_boot": n_boot,
    }


def item_ok_lists(questions: list[dict], pred_a: dict, pred_b: dict):
    a_ok, b_ok = [], []
    for q in questions:
        qid = q.get("question_id", q.get("id"))
        pa, pb = lookup(pred_a, qid), lookup(pred_b, qid)
        if pa is None or pb is None:
            continue
        gold = 1 if gold_yes(q.get("label")) else 0
        a_ok.append(int(pa == gold))
        b_ok.append(int(pb == gold))
    return a_ok, b_ok


def confusion(questions: list[dict], preds: dict) -> dict:
    tp = fp = tn = fn = 0
    for q in questions:
        qid = q.get("question_id", q.get("id"))
        p = lookup(preds, qid)
        if p is None:
            continue
        g = 1 if gold_yes(q.get("label")) else 0
        if p and g:
            tp += 1
        elif p and not g:
            fp += 1
        elif (not p) and (not g):
            tn += 1
        else:
            fn += 1
    n = tp + fp + tn + fn
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "n": n,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "Accuracy": (tp + tn) / n if n else 0.0,
        "Recall": rec,
        "Precision": prec,
        "F1": f1,
        "Yes_ratio": (tp + fp) / n if n else 0.0,
        "gold_yes_ratio": (tp + fn) / n if n else 0.0,
    }


def gold_rate(questions: list[dict]) -> float:
    if not questions:
        return 0.0
    return sum(1 for q in questions if gold_yes(q.get("label"))) / len(questions)


def pct(x) -> str:
    if x is None:
        return ""
    return f"{100.0 * float(x):.2f}"


def heldout_row(name: str, blob: dict | None) -> str:
    if not blob:
        return f"| {name} | | | | | | | |"
    a = blob["all"]
    pos = blob.get("positive") or {}
    rnd = blob.get("random_negative") or {}
    hard = blob.get("hard_negative") or {}
    return (
        f"| {name} | {pct(a.get('Accuracy'))} | {pct(pos.get('Accuracy'))} | "
        f"{pct(rnd.get('Accuracy'))} | {pct(hard.get('Accuracy'))} | "
        f"{pct(hard.get('hard_fp_rate'))} | {pct(a.get('Yes_ratio'))} | {a.get('n','')} |"
    )


def train_neg(blob: dict | None) -> dict:
    if not blob:
        return {}
    return blob.get("clean_negative") or blob.get("random_negative") or {}


def paired_cell(v: dict) -> str:
    bs = v["bootstrap"]
    p = v["mcnemar"]["p_value"]
    ci = bs["ci95"]
    return (
        f"| {100 * bs['delta']:+.2f} pp | "
        f"[{100 * ci[0]:+.2f}, {100 * ci[1]:+.2f}] | {p:.4g} | {bs['n']} |"
    )


def ci_contains_zero(ci: list[float]) -> bool:
    return ci[0] <= 0.0 <= ci[1]


def closer_to_gold(yes_a: float, yes_b: float, gold: float) -> bool:
    return abs(yes_b - gold) + 1e-12 < abs(yes_a - gold)


def score_train_by_format(questions: list[dict], preds: dict) -> dict:
    groups = defaultdict(lambda: {"preds": [], "labels": []})
    for q in questions:
        qid = q.get("question_id", q.get("id"))
        p = lookup(preds, qid)
        if p is None:
            continue
        fmt = q.get("prompt_format") or "unk"
        groups[fmt]["preds"].append(p)
        groups[fmt]["labels"].append(1 if gold_yes(q.get("label")) else 0)
        subset = q.get("subset") or "unk"
        key = f"{fmt}:{subset}"
        groups[key]["preds"].append(p)
        groups[key]["labels"].append(1 if gold_yes(q.get("label")) else 0)
    return {k: metrics(v["preds"], v["labels"]) for k, v in groups.items()}


def case_verdict(blob: dict) -> str:
    paired = blob.get("paired") or {}
    cm = blob.get("fast_cm") or {}
    held = blob.get("heldout") or {}
    f2 = blob.get("paired_heldout_f2") or {}
    lines = []
    main = []
    for bench in ("rbench", "mmrel_adv"):
        rec = (paired.get(bench) or {}).get("mf_vs_clean")
        if not rec:
            continue
        dpp = 100 * rec["bootstrap"]["delta"]
        p = rec["mcnemar"]["p_value"]
        ci = rec["bootstrap"]["ci95"]
        c0 = (cm.get("clean") or {}).get(bench) or {}
        c1 = (cm.get("mf") or {}).get(bench) or {}
        gold = c0.get("gold_yes_ratio")
        if gold is None:
            gold = c1.get("gold_yes_ratio")
        yes_ok = False
        if c0 and c1 and gold is not None:
            yes_ok = closer_to_gold(c0.get("Yes_ratio", 0), c1.get("Yes_ratio", 0), gold)
        fp_delta = (c1.get("FP", 0) - c0.get("FP", 0)) if c0 and c1 else None
        fn_delta = (c1.get("FN", 0) - c0.get("FN", 0)) if c0 and c1 else None
        no_bias_gain = bool(c0 and c1 and c1.get("FP", 0) < c0.get("FP", 0) and c1.get("FN", 0) > c0.get("FN", 0) and dpp > 0)
        main.append(
            {
                "bench": bench,
                "delta_pp": dpp,
                "p": p,
                "ci": ci,
                "yes_closer": yes_ok,
                "no_bias_gain": no_bias_gain,
                "fp_delta": fp_delta,
                "fn_delta": fn_delta,
            }
        )
    mf_h = (held.get("mf") or {}).get("all") or {}
    clean_h = (held.get("clean") or {}).get("all") or {}
    held_delta = None
    if mf_h.get("Accuracy") is not None and clean_h.get("Accuracy") is not None:
        held_delta = 100 * (mf_h["Accuracy"] - clean_h["Accuracy"])
    held_ok = held_delta is None or held_delta >= -1.0

    a_hits = []
    for row in main:
        if row["delta_pp"] >= 1.0 and row["p"] < 0.05 and row["yes_closer"] and not row["no_bias_gain"]:
            a_hits.append(row["bench"])
    c_external = any(row["delta_pp"] <= -1.0 and row["p"] < 0.05 for row in main)
    c_held = held_delta is not None and held_delta <= -1.5
    all_ci0 = bool(main) and all(ci_contains_zero(row["ci"]) for row in main)

    f2_rec = (f2.get("mf_vs_clean") or {})
    f2_delta = None
    f2_p = None
    if f2_rec:
        f2_delta = 100 * f2_rec["bootstrap"]["delta"]
        f2_p = f2_rec["mcnemar"]["p_value"]
    f2_mf_better = f2_delta is not None and f2_delta >= 1.0 and (f2_p is not None and f2_p < 0.05)

    if a_hits and held_ok:
        lines.append(
            "**Case A：形式绑定成立。** mf 相对 clean 在至少一个外部官方 prompt bench 上 ΔAcc ≥ +1 pp 且 McNemar 显著，"
            "Yes ratio 更接近 gold，且不是靠 No-bias 涨分；held-out 无明显退化。"
            "后续训练应改用多格式渲染。"
        )
        lines.append("触发 bench: " + ", ".join(a_hits))
    elif c_external or c_held:
        lines.append(
            "**Case C：mf 更差。** 格式混合可能稀释了每种格式的曝光（每种约 1/3）。"
            "记录后不回退 Phase 4A 结论；可考虑多格式 + 2 epoch 补一枪。"
        )
    elif all_ci0:
        if f2_mf_better:
            lines.append(
                "**Case B：形式不是瓶颈。** 外部官方 prompt 上 mf ≈ clean（CI 含 0），"
                "但 mf 在 F2 held-out 上明显好于 clean——模型能学会跨格式，外部仍不迁移，瓶颈在内容/量级。"
            )
        else:
            lines.append(
                "**Case B：形式不是瓶颈。** 外部官方 prompt 上 mf ≈ clean（CI 含 0）。"
                "F2 held-out 上 mf 也没有明显优于 clean，单一训练模板不是外部迁移失败的充分原因。"
            )
    else:
        lines.append("**结果落在 A/B/C 边界，按保守口径记为倾向 B（外部未出现显著且干净的提升）。**")

    lines.append("")
    if main:
        lines.append(
            "mf − clean ΔAcc (paired, pp): "
            + ", ".join(
                f"{r['bench']}={r['delta_pp']:+.2f} (p={r['p']:.4g}, Yes-closer={r['yes_closer']}, "
                f"FPΔ={r['fp_delta']}, FNΔ={r['fn_delta']})"
                for r in main
            )
        )
    if held_delta is not None:
        lines.append(f"Held-out F1 Acc: clean={pct(clean_h.get('Accuracy'))}  mf={pct(mf_h.get('Accuracy'))}  Δ={held_delta:+.2f} pp")
    if f2_delta is not None:
        lines.append(f"Held-out F2 mf_vs_clean ΔAcc={f2_delta:+.2f} pp, McNemar p={f2_p:.4g}")
    return "\n".join(lines)


def add_fast(blob: dict, name: str, paths: dict, rbench_q, mmrel_q):
    rec, cm = {}, {}
    if paths["rbench"].exists():
        rec["rbench"] = score_rbench(paths["rbench"])
        cm["rbench"] = confusion(rbench_q, answer_preds(paths["rbench"]))
    if paths["mmrel_adv"].exists():
        rec["mmrel_adv"] = score_mmrel(paths["mmrel_adv"])
        cm["mmrel_adv"] = confusion(mmrel_q, answer_preds(paths["mmrel_adv"]))
    if paths["amber_dr"].exists():
        rec["amber_dr"] = score_amber(paths["amber_dr"])
    blob["fast"][name] = rec
    blob["fast_cm"][name] = cm


def main() -> None:
    rbench_q = load_json_or_jsonl(FAST / "rbench_questions.jsonl")
    mmrel_q = load_json_or_jsonl(FAST / "mmrel_adv_questions.jsonl")
    heldout_q = load_json_or_jsonl(HELDOUT_Q)
    heldout_f2_q = load_json_or_jsonl(HELDOUT_F2_Q) if HELDOUT_F2_Q.exists() else []
    train_q = load_json_or_jsonl(OUT / "questions/mf_train.jsonl") if (OUT / "questions/mf_train.jsonl").exists() else []

    blob = {
        "heldout": {},
        "heldout_f2": {},
        "train": {},
        "train_by_format": {},
        "fast": {},
        "fast_cm": {},
        "paired": {},
        "paired_heldout": {},
        "paired_heldout_f2": {},
        "gold_yes": {
            "rbench": gold_rate(rbench_q),
            "mmrel_adv": gold_rate(mmrel_q),
            "heldout": gold_rate(heldout_q),
        },
        "checksum_clean_vs_base": {},
    }
    build = loadj(ROOT / "data/phase4c/mf_build_stats.json")
    blob["train_build"] = build

    blob["heldout"]["base"] = loadj(ROOT / "eval_results/qwen/phase2/metrics/base_heldout.json")
    blob["heldout"]["clean"] = loadj(P4A / "metrics/clean_heldout.json")
    blob["heldout"]["mf"] = loadj(OUT / "metrics/mf_heldout.json")
    blob["heldout_f2"]["clean"] = loadj(OUT / "metrics/clean_heldout_f2.json")
    blob["heldout_f2"]["mf"] = loadj(OUT / "metrics/mf_heldout_f2.json")
    blob["train"]["clean"] = loadj(P4A / "metrics/clean_train.json")
    blob["train"]["mf"] = loadj(OUT / "metrics/mf_train.json")

    add_fast(blob, "base", BASE_FAST, rbench_q, mmrel_q)
    add_fast(blob, "clean", CLEAN_FAST, rbench_q, mmrel_q)
    add_fast(blob, "mf", MF_FAST, rbench_q, mmrel_q)

    preds_fast = {
        "base": {b: answer_preds(p) for b, p in BASE_FAST.items()},
        "clean": {b: answer_preds(p) for b, p in CLEAN_FAST.items()},
        "mf": {b: answer_preds(p) for b, p in MF_FAST.items()},
    }
    pairs = [("clean", "mf"), ("base", "mf"), ("base", "clean")]
    for bench, questions in (("rbench", rbench_q), ("mmrel_adv", mmrel_q)):
        blob["paired"][bench] = {}
        for a, b in pairs:
            pa, pb = preds_fast.get(a, {}).get(bench, {}), preds_fast.get(b, {}).get(bench, {})
            a_ok, b_ok = item_ok_lists(questions, pa, pb)
            blob["paired"][bench][f"{b}_vs_{a}"] = {
                "mcnemar": mcnemar(a_ok, b_ok),
                "bootstrap": bootstrap_delta(a_ok, b_ok),
            }

    held_preds = {
        "base": answer_preds(ROOT / "eval_results/qwen/phase2/answers/base_heldout.jsonl"),
        "clean": answer_preds(P4A / "answers/clean_heldout.jsonl"),
        "mf": answer_preds(OUT / "answers/mf_heldout.jsonl"),
        "clean_f2": answer_preds(OUT / "answers/clean_heldout_f2.jsonl"),
        "mf_f2": answer_preds(OUT / "answers/mf_heldout_f2.jsonl"),
    }
    for a, b in (("base", "clean"), ("base", "mf"), ("clean", "mf")):
        a_ok, b_ok = item_ok_lists(heldout_q, held_preds[a], held_preds[b])
        blob["paired_heldout"][f"{b}_vs_{a}"] = {
            "mcnemar": mcnemar(a_ok, b_ok),
            "bootstrap": bootstrap_delta(a_ok, b_ok),
        }
    if heldout_f2_q:
        for a, b, pa, pb, qs in (
            ("clean_f2", "mf_f2", held_preds["clean_f2"], held_preds["mf_f2"], heldout_f2_q),
            ("clean_f1", "clean_f2", held_preds["clean"], held_preds["clean_f2"], heldout_q),
            ("mf_f1", "mf_f2", held_preds["mf"], held_preds["mf_f2"], heldout_q),
        ):
            a_ok, b_ok = item_ok_lists(qs, pa, pb)
            blob["paired_heldout_f2"][f"{b}_vs_{a}"] = {
                "mcnemar": mcnemar(a_ok, b_ok),
                "bootstrap": bootstrap_delta(a_ok, b_ok),
            }
        # alias used by verdict
        blob["paired_heldout_f2"]["mf_vs_clean"] = blob["paired_heldout_f2"].get("mf_f2_vs_clean_f2")

    if train_q:
        blob["train_by_format"] = score_train_by_format(train_q, answer_preds(OUT / "answers/mf_train.jsonl"))

    checksum = {}
    for bench, pub in PUBLISHED_CLEAN_VS_BASE.items():
        rec = (blob["paired"].get(bench) or {}).get("clean_vs_base") or {}
        bs = rec.get("bootstrap") or {}
        p = (rec.get("mcnemar") or {}).get("p_value")
        dpp = 100 * bs["delta"] if bs else None
        ok = (
            dpp is not None
            and abs(dpp - pub["delta_pp"]) < 0.02
            and p is not None
            and abs(p - pub["p"]) < 0.01
        )
        checksum[bench] = {
            "published": pub,
            "recomputed_delta_pp": None if dpp is None else round(dpp, 2),
            "recomputed_p": p,
            "match": ok,
        }
    blob["checksum_clean_vs_base"] = checksum

    def drop_folds(obj):
        if isinstance(obj, dict):
            return {k: drop_folds(v) for k, v in obj.items() if k != "folds"}
        if isinstance(obj, list):
            return [drop_folds(x) for x in obj]
        return obj

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics").mkdir(exist_ok=True)
    slim_path = OUT / "metrics" / "phase4c_analysis.json"
    slim_path.write_text(json.dumps(drop_folds(blob), indent=2) + "\n", encoding="utf-8")

    md = []
    md.append("# Phase 4C 训练结果：多格式 Prompt vs Clean 单一模板")
    md.append("")
    n_pairs = (build or {}).get("n_pairs", 2156)
    assign = (build or {}).get("assignment", "")
    md.append(
        f"Qwen2.5-VL-3B LoRA（r=16, α=32），1 epoch，seed=42，同一 {n_pairs} 对 / 4312 样本。"
        "唯一变量是训练 prompt 形式：clean 全为 F1 statement-verification；"
        "mf 按 `md5(sample_id) % 3` 混合 F1 / F2（R-Bench 风格直接疑问句）/ F3（MMRel 风格 one-word 后缀）。"
        f"分配方式：`{assign}`。"
    )
    md.append("")
    md.append("本阶段只回答：训练 prompt 形式的单一性，是否是 clean 数据外部迁移失败的原因之一？")
    md.append("")
    if build:
        md.append("## 数据构建")
        md.append("")
        md.append(f"- 样本顺序与 `train_clean.json` 一致：{build.get('same_ids_order')}")
        md.append(f"- 格式计数：{build.get('format_counts')}")
        md.append(f"- QC 失败数：{build.get('n_qc_fail')}")
        hf = (build.get("heldout_f2") or {})
        md.append(
            f"- Held-out F2 解析：n={hf.get('n')}  canonical={ (hf.get('parse_mode') or {}).get('canonical', 0) }"
            f"  fallback={(hf.get('n_noncanonical'))}"
        )
        md.append("- 渲染抽检：`eval_results/qwen/phase4c/mf_render_sample.txt`")
        md.append("")
        md.append("| Format | n | pos | neg | pos rate |")
        md.append("|---|---:|---:|---:|---:|")
        for fmt, rec in (build.get("final_rates") or {}).items():
            md.append(
                f"| {fmt} | {rec.get('n')} | {rec.get('n_pos')} | {rec.get('n_neg')} | {pct(rec.get('pos_rate'))} |"
            )
        md.append("")
    md.append("## clean_vs_base 对账（必须与 Phase 4A 已发布数字一致）")
    md.append("")
    md.append("| Bench | 已发布 ΔAcc | 复算 ΔAcc | 已发布 p | 复算 p | match |")
    md.append("|---|---:|---:|---:|---:|---|")
    for bench, rec in checksum.items():
        pub = rec["published"]
        dpp = rec["recomputed_delta_pp"]
        pval = rec["recomputed_p"]
        dpp_s = "" if dpp is None else f"{dpp:+.2f} pp"
        p_s = "" if pval is None else f"{pval:.4g}"
        md.append(
            f"| {bench} | {pub['delta_pp']:+.2f} pp | {dpp_s} | {pub['p']:.4g} | {p_s} | {rec['match']} |"
        )
    md.append("")
    md.append("## 训练自检（mf 用混合格式原样评；clean 为 Phase 4A F1 自检）")
    md.append("")
    md.append("| Model | Acc | Pos Acc | Neg Acc | Yes | n |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for name in ("clean", "mf"):
        rec = blob["train"].get(name) or {}
        allm = rec.get("all") or {}
        pos = rec.get("positive") or {}
        neg = train_neg(rec)
        md.append(
            f"| {name} | {pct(allm.get('Accuracy'))} | {pct(pos.get('Accuracy'))} | "
            f"{pct(neg.get('Accuracy'))} | {pct(allm.get('Yes_ratio'))} | {allm.get('n','')} |"
        )
    md.append("")
    if blob["train_by_format"]:
        md.append("### mf 按训练格式分解")
        md.append("")
        md.append("| Slice | Acc | Yes | n |")
        md.append("|---|---:|---:|---:|")
        for key in ("F1", "F2", "F3", "F1:positive", "F1:clean_negative", "F2:positive", "F2:clean_negative", "F3:positive", "F3:clean_negative"):
            rec = blob["train_by_format"].get(key)
            if not rec:
                continue
            md.append(f"| {key} | {pct(rec.get('Accuracy'))} | {pct(rec.get('Yes_ratio'))} | {rec.get('n','')} |")
        md.append("")
    md.append("## Gate 1：冻结 393 held-out verification（原 F1 模板，不允许改题）")
    md.append("")
    md.append("| Model | Acc | Pos Acc | Random Neg Acc | Hard Neg Acc | Hard FP | Yes | n |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name in ("base", "clean", "mf"):
        md.append(heldout_row(name, blob["heldout"].get(name)))
    md.append("")
    md.append("### Held-out F1 paired")
    md.append("")
    md.append("| Contrast | ΔAcc | 95% CI | McNemar p | n |")
    md.append("|---|---:|---|---:|---:|")
    for k, v in (blob.get("paired_heldout") or {}).items():
        md.append(f"| {k} {paired_cell(v)}")
    md.append("")
    md.append("## Gate 2：冻结 external fast subsets（官方 prompt，主判据）")
    md.append("")
    md.append("| Model | R-Bench Acc | R-Bench F1 | MMRel Acc | MMRel F1 | AMBER Acc |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for name in ("base", "clean", "mf"):
        rec = blob["fast"].get(name) or {}
        rb, mm, am = rec.get("rbench") or {}, rec.get("mmrel_adv") or {}, rec.get("amber_dr") or {}
        md.append(
            f"| {name} | {pct(rb.get('Accuracy'))} | {pct(rb.get('F1'))} | "
            f"{pct(mm.get('Accuracy'))} | {pct(mm.get('F1'))} | {pct(am.get('Accuracy'))} |"
        )
    md.append("")
    md.append(f"Gold Yes ratio：R-Bench unique={pct(blob['gold_yes']['rbench'])}，MMRel-Adv={pct(blob['gold_yes']['mmrel_adv'])}。")
    md.append("")
    md.append("R-Bench 表内 Acc 是 5-fold 均值；paired / confusion 用 unique questions。主判据以 paired ΔAcc 与 McNemar 为准。")
    md.append("")
    md.append("### Confusion（unique questions）")
    md.append("")
    md.append("| Model | Bench | Acc | FP | FN | Recall | Yes | Gold Yes |")
    md.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for name in ("base", "clean", "mf"):
        for bench in ("rbench", "mmrel_adv"):
            c = (blob["fast_cm"].get(name) or {}).get(bench) or {}
            if not c:
                continue
            md.append(
                f"| {name} | {bench} | {pct(c.get('Accuracy'))} | {c.get('FP','')} | "
                f"{c.get('FN','')} | {pct(c.get('Recall'))} | {pct(c.get('Yes_ratio'))} | "
                f"{pct(c.get('gold_yes_ratio'))} |"
            )
    md.append("")
    md.append("## Paired tests（same frozen items）")
    md.append("")
    for bench, recs in blob["paired"].items():
        md.append(f"### {bench}")
        md.append("")
        md.append("| Contrast | ΔAcc | 95% CI | McNemar p | n |")
        md.append("|---|---:|---|---:|---:|")
        for k, v in recs.items():
            md.append(f"| {k} {paired_cell(v)}")
        md.append("")
    md.append("## 格式内分解：同一 393 held-out 的 F2 渲染")
    md.append("")
    md.append("Held-out 原句是 RelSim 原始 caption，不是 renderer 的 `A {s} is {rel} a {o}`。F2 只做问句包装（A/An → `Is a/an …?`，其余句首小写后加 `Is`），不删内部 copula。clean 与 mf 都在这份 F2 题面上评。")
    md.append("")
    md.append("| Model | Acc | Pos Acc | Random Neg Acc | Hard Neg Acc | Yes | n |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for name in ("clean", "mf"):
        rec = blob["heldout_f2"].get(name) or {}
        a = rec.get("all") or {}
        pos = rec.get("positive") or {}
        rnd = rec.get("random_negative") or {}
        hard = rec.get("hard_negative") or {}
        md.append(
            f"| {name} F2 | {pct(a.get('Accuracy'))} | {pct(pos.get('Accuracy'))} | "
            f"{pct(rnd.get('Accuracy'))} | {pct(hard.get('Accuracy'))} | "
            f"{pct(a.get('Yes_ratio'))} | {a.get('n','')} |"
        )
    md.append("")
    md.append("| Contrast | ΔAcc | 95% CI | McNemar p | n |")
    md.append("|---|---:|---|---:|---:|")
    for k, v in (blob.get("paired_heldout_f2") or {}).items():
        if k == "mf_vs_clean":
            continue
        md.append(f"| {k} {paired_cell(v)}")
    md.append("")
    md.append("## 计划第 4 节判定")
    md.append("")
    md.append(case_verdict(blob))
    md.append("")
    md.append("## 本阶段问题")
    md.append("")
    md.append("> 训练 prompt 形式的单一性，是否是 clean 数据外部迁移失败的原因之一？")
    md.append("")
    text_md = "\n".join(md) + "\n"
    (OUT / "phase4c_report.md").write_text(text_md, encoding="utf-8")
    (ROOT / "phase4c_result.md").write_text(text_md, encoding="utf-8")
    print(text_md)
    print(f"wrote {slim_path}")
    print(f"wrote {OUT / 'phase4c_report.md'}")


if __name__ == "__main__":
    main()
