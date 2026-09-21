#!/usr/bin/env python3
"""Text-only (no image) greedy Yes/No inference for Qwen2.5-VL.

Measures language prior P(relation | subject, object) by answering the official
question text without visual input. Optionally records first-token Yes/No logprobs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

FALLBACK_LINE = (
    "There is no image. Answer based on what is typically most plausible. Answer Yes or No."
)


def load_existing(path: Path) -> set:
    done = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            qid = rec.get("question_id", rec.get("id"))
            if qid is not None:
                done.add(qid)
    return done


def load_questions(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("questions") or data.get("data") or list(data.values())
    return data


def question_text(item: dict) -> str:
    raw = (
        item.get("text")
        or item.get("question")
        or item.get("query")
        or ""
    )
    return raw.replace("<image>", "").strip()


def question_id(item: dict, idx: int):
    if "question_id" in item:
        return item["question_id"]
    if "id" in item:
        return item["id"]
    return idx


def single_token_ids(tokenizer, words: tuple[str, ...]) -> list[int]:
    ids = []
    seen = set()
    for w in words:
        pieces = tokenizer.encode(w, add_special_tokens=False)
        if len(pieces) == 1 and pieces[0] not in seen:
            seen.add(pieces[0])
            ids.append(pieces[0])
    return ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--question_file", default=None)
    parser.add_argument("--answers_file", default=None)
    parser.add_argument(
        "--jobs",
        default=None,
        help="JSON list of {question_file, answers_file} to share one loaded model.",
    )
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--fallback",
        action="store_true",
        help="Append the official fallback line after the original question.",
    )
    return parser.parse_args()


def load_jobs(args: argparse.Namespace) -> list[dict]:
    jobs = []
    if args.jobs:
        spec = json.loads(Path(args.jobs).read_text(encoding="utf-8"))
        if not isinstance(spec, list):
            raise SystemExit("--jobs must be a JSON list")
        jobs.extend(spec)
    if args.question_file:
        if not args.answers_file:
            raise SystemExit("--answers_file is required with --question_file")
        jobs.append({"question_file": args.question_file, "answers_file": args.answers_file})
    if not jobs:
        raise SystemExit("provide --question_file/--answers_file or --jobs")
    return jobs


def build_prompt(item: dict, fallback: bool) -> str:
    prompt = question_text(item)
    if fallback:
        prompt = prompt + "\n" + FALLBACK_LINE
    return prompt


@torch.inference_mode()
def yesno_logprobs(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, yes_ids: list[int], no_ids: list[int]) -> dict:
    if not yes_ids or not no_ids:
        return {}
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = out.logits[0, -1].float()
    logp = torch.log_softmax(logits, dim=-1)
    yes_lp = max(float(logp[i]) for i in yes_ids)
    no_lp = max(float(logp[i]) for i in no_ids)
    return {
        "yes_logprob": yes_lp,
        "no_logprob": no_lp,
        "logprob_margin": yes_lp - no_lp,
    }


def run_job(model, processor, tokenizer, model_id: str, job: dict, args: argparse.Namespace, yes_ids: list[int], no_ids: list[int]) -> None:
    qpath = Path(job["question_file"])
    out_path = Path(job["answers_file"])
    questions = load_questions(qpath)
    if args.limit is not None:
        questions = questions[: args.limit]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_existing(out_path)
    remaining = [q for i, q in enumerate(questions) if question_id(q, i) not in done]
    print(
        f"job {qpath.name} questions={len(questions)} already={len(done)} remaining={len(remaining)} fallback={args.fallback}",
        flush=True,
    )
    with out_path.open("a", encoding="utf-8") as fout:
        for i, item in enumerate(remaining, 1):
            qid = question_id(item, i)
            prompt = build_prompt(item, args.fallback)
            messages = [{"role": "user", "content": prompt}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            encoded = tokenizer(text, return_tensors="pt")
            input_ids = encoded["input_ids"].to(model.device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(model.device)
            lp = {}
            try:
                lp = yesno_logprobs(model, input_ids, attention_mask, yes_ids, no_ids)
            except Exception as exc:
                lp = {"logprob_error": repr(exc)}
            gen_kwargs = {
                "input_ids": input_ids,
                "do_sample": False,
                "max_new_tokens": args.max_new_tokens,
                "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            }
            if attention_mask is not None:
                gen_kwargs["attention_mask"] = attention_mask
            output_ids = model.generate(**gen_kwargs)
            gen = output_ids[:, input_ids.shape[1] :]
            answer = tokenizer.batch_decode(gen, skip_special_tokens=True)[0].strip()
            rec = {
                "question_id": qid,
                "prompt": prompt,
                "text": answer,
                "model_id": model_id,
                "text_only": True,
                "fallback": bool(args.fallback),
            }
            rec.update(lp)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            if i % 20 == 0 or i == 1:
                print(f"[{i}/{len(remaining)}] {qid} {answer[:80]!r}", flush=True)
    print(f"Wrote {out_path}", flush=True)


def main() -> None:
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    jobs = load_jobs(args)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )
    model_id = Path(args.adapter_path).name if args.adapter_path else "qwen25vl3b_base"
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)
        model_id = Path(args.adapter_path).name
    model.eval()
    yes_ids = single_token_ids(tokenizer, ("Yes", "yes"))
    no_ids = single_token_ids(tokenizer, ("No", "no"))
    print(f"yes_ids={yes_ids} no_ids={no_ids} device={model.device}", flush=True)
    for job in jobs:
        run_job(model, processor, tokenizer, model_id, job, args, yes_ids, no_ids)


if __name__ == "__main__":
    main()
