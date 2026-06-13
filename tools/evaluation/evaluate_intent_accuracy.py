import argparse
import json
import re


INTENT_PATTERN = re.compile(r"<\|intent_(\d{3})\|>")


def load_predictions(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def extract_intent_id(text):
    match = INTENT_PATTERN.search(text)
    if match is None:
        return None
    return int(match.group(1))


def main():
    parser = argparse.ArgumentParser(description="Evaluate latent intent token accuracy.")
    parser.add_argument("--pred_path", required=True)
    parser.add_argument("--token_path", required=True)
    parser.add_argument("--intent_labels", default="create_data/intent_labels_k64.json")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    args = parser.parse_args()

    preds = load_predictions(args.pred_path)
    token_data = json.load(open(args.token_path, "r"))
    labels = json.load(open(args.intent_labels, "r"))["labels"][args.split]

    total = 0
    correct = 0
    missing = 0

    for pred, sample in zip(preds, token_data):
        token = sample["id"]
        if token not in labels:
            continue

        total += 1
        pred_intent = extract_intent_id(pred["predict"])
        if pred_intent is None:
            missing += 1
            continue

        if pred_intent == int(labels[token]):
            correct += 1

    accuracy = correct / total if total else 0.0
    missing_rate = missing / total if total else 0.0

    print(f"Intent total: {total}")
    print(f"Intent correct: {correct}")
    print(f"Intent accuracy: {accuracy:.4f}")
    print(f"Intent missing rate: {missing_rate:.4f}")


if __name__ == "__main__":
    main()
