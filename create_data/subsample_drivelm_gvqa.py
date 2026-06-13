import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Subsample DriveLM GVQA ShareGPT data with task-balanced sampling.")
    parser.add_argument("--input", default="./LlamaFactory/data/train_drivelm_gvqa.json")
    parser.add_argument("--output", default="./LlamaFactory/data/train_drivelm_gvqa_50k.json")
    parser.add_argument("--num-samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strategy", choices=["proportional", "balanced"], default="proportional")
    return parser.parse_args()


def task_from_id(sample_id):
    marker = "_perception_"
    for task in ["perception", "prediction", "planning", "behavior"]:
        if f"_{task}_" in sample_id:
            return task
    return "unknown"


def allocate_counts(groups, total, strategy):
    available = {task: len(items) for task, items in groups.items()}
    if total >= sum(available.values()):
        return available

    if strategy == "balanced":
        base = total // len(groups)
        allocation = {task: min(base, count) for task, count in available.items()}
    else:
        full_total = sum(available.values())
        allocation = {task: int(total * count / full_total) for task, count in available.items()}

    while sum(allocation.values()) < total:
        candidates = [task for task, count in available.items() if allocation[task] < count]
        if not candidates:
            break
        task = max(candidates, key=lambda key: available[key] - allocation[key])
        allocation[task] += 1

    while sum(allocation.values()) > total:
        task = max(allocation, key=allocation.get)
        allocation[task] -= 1

    return allocation


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    with open(args.input, "r") as f:
        data = json.load(f)

    groups = defaultdict(list)
    for sample in data:
        groups[task_from_id(sample.get("id", ""))].append(sample)

    allocation = allocate_counts(groups, args.num_samples, args.strategy)
    selected = []
    for task in sorted(groups):
        items = groups[task]
        rng.shuffle(items)
        selected.extend(items[: allocation.get(task, 0)])

    rng.shuffle(selected)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(selected, f, indent=4, ensure_ascii=False)

    print(f"Input samples: {len(data)}")
    print(f"Output samples: {len(selected)} -> {output_path}")
    print("Input task counts:")
    for task in sorted(groups):
        print(f"  {task}: {len(groups[task])}")
    print("Output task counts:")
    out_counts = defaultdict(int)
    for sample in selected:
        out_counts[task_from_id(sample.get("id", ""))] += 1
    for task in sorted(out_counts):
        print(f"  {task}: {out_counts[task]}")


if __name__ == "__main__":
    main()
