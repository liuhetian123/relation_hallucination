import argparse
import json
from pathlib import Path


def normalize_yes_no(text):
    text = text.strip()
    lower = text.lower()

    if lower.startswith("yes"):
        return "Yes"
    if lower.startswith("no"):
        return "No"

    # Robust fallback.
    if "yes" in lower and "no" not in lower:
        return "Yes"
    if "no" in lower and "yes" not in lower:
        return "No"

    # AMBER discriminative evaluation expects Yes/No.
    # For ambiguous outputs, map to No conservatively.
    return "No"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llava-answer", required=True)
    parser.add_argument("--out-file", required=True)
    args = parser.parse_args()

    outputs = []

    with open(args.llava_answer, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            item = json.loads(line)
            qid = item.get("question_id", item.get("id"))
            answer = item.get("text", item.get("answer", item.get("response", "")))

            outputs.append({
                "id": qid,
                "response": normalize_yes_no(answer)
            })

    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)

    with open(args.out_file, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(outputs)} AMBER responses to {args.out_file}")


if __name__ == "__main__":
    main()
