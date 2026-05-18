import os
import json
import time
import requests
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset

OUT_IMAGE_DIR = "data/relsim_images"
OUT_JSON = "data/relsim_llava_1k.json"
NUM_SAMPLES = 1000

os.makedirs(OUT_IMAGE_DIR, exist_ok=True)

dataset = load_dataset(
    "thaoshibe/anonymous-captions-114k",
    split=f"train[:{NUM_SAMPLES}]"
)

samples = []

def download_image(url, save_path, timeout=15):
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
        img.save(save_path)
        return True
    except Exception as e:
        print(f"[Skip] {url}: {e}")
        return False

for item in tqdm(dataset):
    image_hash = item["image_hash"]
    url = item["url_link"]
    caption = item["caption"]

    image_name = f"{image_hash}.jpg"
    image_path = os.path.join(OUT_IMAGE_DIR, image_name)

    if not os.path.exists(image_path):
        ok = download_image(url, image_path)
        time.sleep(0.05)
        if not ok:
            continue

    samples.append({
        "id": f"relsim_{image_hash}",
        "image": image_name,
        "conversations": [
            {
                "from": "human",
                "value": "<image>\nDescribe the underlying relational logic of this image. Focus on abstract relations rather than concrete object names, colors, or surface appearance."
            },
            {
                "from": "gpt",
                "value": caption
            }
        ]
    })

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(samples, f, ensure_ascii=False, indent=2)

print(f"Saved {len(samples)} samples to {OUT_JSON}")
print(f"Images saved to {OUT_IMAGE_DIR}")
