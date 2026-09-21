#!/usr/bin/env python3
"""Load Qwen2.5-VL-3B-Instruct and run a tiny image+text generation."""

import argparse

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--image", default=None, help="Optional image path; default is a synthetic RGB patch.")
    args = parser.parse_args()

    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu", torch.cuda.get_device_name(0), "free_gb", round(torch.cuda.mem_get_info()[0] / 1024**3, 2))

    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
    )

    if args.image:
        image = Image.open(args.image).convert("RGB")
    else:
        image = Image.new("RGB", (224, 224), color=(40, 120, 200))

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this image in one short sentence."},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)[:2]
    inputs = processor(text=[text], images=images, videos=videos, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=32)
    generated = output_ids[:, inputs["input_ids"].shape[1] :]
    answer = processor.batch_decode(generated, skip_special_tokens=True)[0]
    print("SMOKE_OK")
    print("ANSWER:", answer.strip())


if __name__ == "__main__":
    main()
