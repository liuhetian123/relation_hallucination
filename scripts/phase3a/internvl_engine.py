"""InternVL3.5-8B-HF load / greedy generate, matching Phase 2 filter_hard."""

from __future__ import annotations

import os
from pathlib import Path

from PIL import Image


def load_internvl(model_path: str, gpu: str | None = None, max_patches: int = 4):
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    model.eval()
    device = next(model.parameters()).device
    print(
        f"InternVL loaded path={model_path} device={device} "
        f"cuda={torch.cuda.is_available()} max_patches={max_patches}",
        flush=True,
    )
    if torch.cuda.is_available():
        free_gb = torch.cuda.mem_get_info()[0] / 1024**3
        print(f"GPU {torch.cuda.get_device_name(0)} free_gb={free_gb:.2f}", flush=True)

    def generate(
        prompt: str,
        image: Image.Image | None = None,
        max_new_tokens: int = 128,
        do_sample: bool = False,
        temperature: float | None = None,
    ) -> str:
        if image is None:
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            proc_kwargs = {"text": [text], "return_tensors": "pt"}
        else:
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": prompt}],
                }
            ]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            proc_kwargs = {
                "text": [text],
                "images": [image],
                "return_tensors": "pt",
                "crop_to_patches": True,
                "max_patches": max_patches,
            }
            if max_patches <= 1:
                proc_kwargs["crop_to_patches"] = False
                proc_kwargs.pop("max_patches", None)
        inputs = processor(**proc_kwargs)
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        gen_kwargs = {"do_sample": do_sample, "max_new_tokens": max_new_tokens}
        if do_sample:
            gen_kwargs["temperature"] = 0.7 if temperature is None else temperature
        with torch.inference_mode():
            output_ids = model.generate(**inputs, **gen_kwargs)
        gen = output_ids[:, inputs["input_ids"].shape[1] :]
        return processor.batch_decode(gen, skip_special_tokens=True)[0].strip()

    def infer(image: Image.Image, prompt: str, max_new_tokens: int = 128, **kwargs) -> str:
        return generate(prompt, image=image, max_new_tokens=max_new_tokens, **kwargs)

    infer.generate = generate
    return infer


def open_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")
