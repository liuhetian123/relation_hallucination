import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--yes-no-prompt", action="store_true")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as fout:
        for i, item in enumerate(data):
            qid = item.get("question_id", item.get("id", i))
            image = item.get("image", item.get("image_path", item.get("file_name")))
            text = item.get("text", item.get("question", item.get("prompt")))

            if image is None or text is None:
                raise ValueError(f"Cannot parse item: {item}")

            if args.yes_no_prompt:
                text = text.rstrip() + "\nAnswer with only 'yes' or 'no'."

            record = {
                "question_id": qid,
                "image": image,
                "text": text,
            }

            if "label" in item:
                record["label"] = item["label"]

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Loaded {len(data)} samples from {args.input}")
    print(f"Saved JSONL to {args.output}")


if __name__ == "__main__":
    main()
