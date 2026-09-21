#!/usr/bin/env python3
"""LoRA SFT for Qwen2.5-VL on matched RelSim captions.

Vision encoder is frozen. Only LLM LoRA adapters are trained.
Typed and Explicit runs must use identical hyperparameters.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from PIL import Image
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

TRAIN_PROMPT = (
    "Describe the primary relation shown in this image in one short sentence."
)
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_caption(sample: dict) -> str:
    for turn in reversed(sample.get("conversations", [])):
        if turn.get("from") in {"gpt", "assistant"}:
            return turn.get("value", "").strip()
    return (sample.get("caption") or "").strip()


def extract_prompt(sample: dict) -> str:
    for turn in sample.get("conversations", []):
        if turn.get("from") in {"human", "user"}:
            text = re.sub(r"<image>\s*", "", turn.get("value") or "", flags=re.I).strip()
            if text:
                return text
    return TRAIN_PROMPT


class RelSimSFTDataset(Dataset):
    def __init__(self, samples: list[dict], image_root: Path):
        self.samples = samples
        self.image_root = image_root

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image_path = self.image_root / sample["image"]
        return {
            "id": sample["id"],
            "image_path": str(image_path),
            "prompt": extract_prompt(sample),
            "caption": extract_caption(sample),
        }


def find_lora_targets(model) -> list[str]:
    names = []
    for name, module in model.named_modules():
        if not name.endswith(LORA_TARGETS):
            continue
        if any(part in name for part in ("visual", "vision", "merger", "patch_embed")):
            continue
        names.append(name)
    if not names:
        raise RuntimeError("No LoRA target modules found on the language model.")
    return names


class QwenCollator:
    def __init__(self, processor, max_pixels: int, min_pixels: int):
        self.processor = processor
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.pad_id = processor.tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = processor.tokenizer.eos_token_id

    def __call__(self, features: list[dict]) -> dict:
        full_texts = []
        prompt_texts = []
        images = []
        for feat in features:
            image = Image.open(feat["image_path"]).convert("RGB")
            user_content = [
                {"type": "image", "image": image},
                {"type": "text", "text": feat.get("prompt") or TRAIN_PROMPT},
            ]
            prompt_messages = [{"role": "user", "content": user_content}]
            full_messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": feat["caption"]},
            ]
            prompt_texts.append(
                self.processor.apply_chat_template(
                    prompt_messages, tokenize=False, add_generation_prompt=True
                )
            )
            full_texts.append(
                self.processor.apply_chat_template(
                    full_messages, tokenize=False, add_generation_prompt=False
                )
            )
            image_inputs, _video_inputs = process_vision_info(full_messages)[:2]
            images.append(image_inputs[0] if image_inputs else image)

        size = getattr(self.processor.image_processor, "size", None)
        if size is not None:
            size["longest_edge"] = self.max_pixels
            size["shortest_edge"] = self.min_pixels
        full = self.processor(
            text=full_texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )
        prompt = self.processor(
            text=prompt_texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )
        labels = full["input_ids"].clone()
        prompt_lens = prompt["attention_mask"].sum(dim=1)
        for i, plen in enumerate(prompt_lens.tolist()):
            labels[i, : int(plen)] = -100
        labels[full["input_ids"] == self.pad_id] = -100
        full["labels"] = labels
        return full


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--image_root", default="data/relsim_images")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--max_pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--min_pixels", type=int, default=16 * 28 * 28)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_strategy", default="steps", choices=("steps", "epoch", "no"))
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="If >0, overrides num_train_epochs (used by Phase 3A step-matched run).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    set_seed(args.seed)

    samples = json.loads(Path(args.data_path).read_text(encoding="utf-8"))
    dataset = RelSimSFTDataset(samples, Path(args.image_root))
    print(f"Loaded {len(dataset)} samples from {args.data_path}")

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    if torch.cuda.is_available():
        model.to("cuda")
    model.config.use_cache = False
    if hasattr(model, "visual"):
        for p in model.visual.parameters():
            p.requires_grad = False

    target_modules = find_lora_targets(model)
    print(f"LoRA targets ({len(target_modules)}): {target_modules[:8]} ...")
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    steps_per_epoch = max(
        1,
        (len(dataset) + args.per_device_train_batch_size - 1)
        // args.per_device_train_batch_size
        // max(1, args.gradient_accumulation_steps),
    )
    warmup_steps = args.warmup_steps
    planned_steps = steps_per_epoch * max(1, int(args.num_train_epochs))
    if args.max_steps and args.max_steps > 0:
        planned_steps = args.max_steps
    if warmup_steps is None:
        warmup_steps = max(1, int(args.warmup_ratio * planned_steps))
    print(
        f"steps_per_epoch≈{steps_per_epoch} warmup_steps={warmup_steps} "
        f"max_steps={args.max_steps} planned_steps={planned_steps}"
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        weight_decay=0.0,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
        max_grad_norm=1.0,
        optim="adamw_torch",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=QwenCollator(processor, args.max_pixels, args.min_pixels),
        processing_class=processor.tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    state_src = Path(args.output_dir) / "trainer_state.json"
    if not state_src.exists():
        ckpts = sorted(
            Path(args.output_dir).glob("checkpoint-*/trainer_state.json"),
            key=lambda p: int(p.parent.name.split("-")[-1]),
        )
        if ckpts:
            state_src.write_text(ckpts[-1].read_text(encoding="utf-8"), encoding="utf-8")
    meta = {
        "model_path": args.model_path,
        "data_path": args.data_path,
        "n_samples": len(dataset),
        "seed": args.seed,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "learning_rate": args.learning_rate,
        "epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "global_step": int(getattr(trainer.state, "global_step", 0) or 0),
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_pixels": args.max_pixels,
        "default_train_prompt": TRAIN_PROMPT,
        "prompt_from_data": True,
    }
    Path(args.output_dir, "train_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved LoRA to {args.output_dir}")


if __name__ == "__main__":
    main()
