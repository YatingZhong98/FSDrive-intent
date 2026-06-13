import argparse
import json
import re


AGENT_TOKEN_PATTERN = re.compile(r"<\|agent_(?:intent_(\d{3})|none)\|>")


def load_predictions(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def extract_agent_intents(text, top_k):
    intents = []
    for match in AGENT_TOKEN_PATTERN.finditer(text):
        intent = match.group(1)
        intents.append(None if intent is None else int(intent))
        if len(intents) == top_k:
            break

    missing = max(0, top_k - len(intents))
    intents.extend([None] * missing)
    return intents, missing


def build_expected_slots(label_items, top_k):
    slots = [None] * top_k
    for item in label_items:
        slot = int(item["slot"])
        if 0 <= slot < top_k:
            slots[slot] = int(item["intent"])
    return slots


def main():
    parser = argparse.ArgumentParser(description="Evaluate top-k surrounding agent intent token accuracy.")
    parser.add_argument("--pred_path", required=True)
    parser.add_argument("--token_path", required=True)
    parser.add_argument("--agent_intent_labels", default="create_data/agent_intent_labels_k64_top4.json")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--top_k", type=int, default=None)
    args = parser.parse_args()

    preds = load_predictions(args.pred_path)
    token_data = json.load(open(args.token_path, "r", encoding="utf-8"))
    label_file = json.load(open(args.agent_intent_labels, "r", encoding="utf-8"))
    labels = label_file["labels"][args.split]
    top_k = args.top_k or int(label_file.get("metadata", {}).get("top_k", 4))

    sample_total = 0
    exact_match = 0
    slot_total = 0
    slot_correct = 0
    agent_total = 0
    agent_correct = 0
    none_total = 0
    none_correct = 0
    pred_missing = 0

    for pred, sample in zip(preds, token_data):
        token = sample["id"]
        if token not in labels:
            continue

        expected = build_expected_slots(labels[token], top_k)
        predicted, missing = extract_agent_intents(pred["predict"], top_k)

        sample_total += 1
        pred_missing += missing
        if predicted == expected:
            exact_match += 1

        for pred_intent, expected_intent in zip(predicted, expected):
            slot_total += 1
            if pred_intent == expected_intent:
                slot_correct += 1

            if expected_intent is None:
                none_total += 1
                if pred_intent is None:
                    none_correct += 1
            else:
                agent_total += 1
                if pred_intent == expected_intent:
                    agent_correct += 1

    exact_accuracy = exact_match / sample_total if sample_total else 0.0
    slot_accuracy = slot_correct / slot_total if slot_total else 0.0
    agent_accuracy = agent_correct / agent_total if agent_total else 0.0
    none_accuracy = none_correct / none_total if none_total else 0.0
    missing_rate = pred_missing / slot_total if slot_total else 0.0

    print(f"Agent intent samples: {sample_total}")
    print(f"Agent intent top_k: {top_k}")
    print(f"Agent intent exact match: {exact_match}")
    print(f"Agent intent exact accuracy: {exact_accuracy:.4f}")
    print(f"Agent intent slot total: {slot_total}")
    print(f"Agent intent slot correct: {slot_correct}")
    print(f"Agent intent slot accuracy: {slot_accuracy:.4f}")
    print(f"Agent intent non-empty total: {agent_total}")
    print(f"Agent intent non-empty correct: {agent_correct}")
    print(f"Agent intent non-empty accuracy: {agent_accuracy:.4f}")
    print(f"Agent none total: {none_total}")
    print(f"Agent none correct: {none_correct}")
    print(f"Agent none accuracy: {none_accuracy:.4f}")
    print(f"Agent intent missing token rate: {missing_rate:.4f}")


if __name__ == "__main__":
    main()
