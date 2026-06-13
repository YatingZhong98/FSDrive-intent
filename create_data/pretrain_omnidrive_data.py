import argparse
import json
import pickle
from pathlib import Path


SYSTEM = (
    "You're an autonomous driving assistant for visual question answering. "
    "Use the current six camera images to understand the driving scene, reason about "
    "traffic participants, road layout, safety risks, and answer the user's question accurately. "
    "Coordinates use the ego vehicle as origin: X-axis is lateral, Y-axis is forward, and units are meters."
)

CAMERA_TYPES = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

IMAGE_PROMPT = (
    "Current six camera images: "
    "'CAM_FRONT': <image>\n"
    ", 'CAM_FRONT_LEFT': <image>\n"
    ", 'CAM_FRONT_RIGHT': <image>\n"
    ", 'CAM_BACK': <image>\n"
    ", 'CAM_BACK_LEFT': <image>\n"
    ", 'CAM_BACK_RIGHT': <image>\n"
    "Answer the following driving-scene question based on these images.\n"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert OmniDrive-nuScenes desc, conv, and vqa annotations to LLaMA-Factory ShareGPT format."
    )
    parser.add_argument("--source-root", default="./LlamaFactory/data/nuscenes")
    parser.add_argument("--info-path", default="./create_data/cached_nuscenes_info.pkl")
    parser.add_argument("--output", default="./LlamaFactory/data/pretrain_omnidrive_data.json")
    parser.add_argument("--splits", nargs="+", default=["train"], choices=["train", "val"])
    parser.add_argument("--tasks", nargs="+", default=["desc", "conv", "vqa"], choices=["desc", "conv", "vqa"])
    parser.add_argument("--max-files", type=int, default=None, help="Limit files per task/split for smoke tests.")
    return parser.parse_args()


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def get_images(info, token):
    images = []
    cams = info[token]["cams"]
    for cam in CAMERA_TYPES:
        path = cams[cam]["data_path"].replace("/localdata_ssd/nuScenes", "data/nuscenes", 1)
        images.append(path)
    return images


def make_sample(sample_id, images, user_value, assistant_value):
    return {
        "id": sample_id,
        "images": images,
        "system": SYSTEM,
        "conversations": [
            {
                "from": "human",
                "value": IMAGE_PROMPT + user_value.strip() + "\n",
            },
            {
                "from": "gpt",
                "value": assistant_value.strip() + "\n<|endoftext|><|im_end|>",
            },
        ],
    }


def convert_desc(path, token, images):
    data = load_json(path)
    description = data.get("description", "").strip()
    action = data.get("action", "").strip()
    if not description and not action:
        return []

    answer = []
    if description:
        answer.append(f"Scene description: {description}")
    if action:
        answer.append(f"Recommended action: {action}")

    return [
        make_sample(
            sample_id=f"desc_{token}",
            images=images,
            user_value="Describe the driving scene and recommend the next safe driving action.",
            assistant_value="\n".join(answer),
        )
    ]


def convert_qa_list(path, token, images, task):
    data = load_json(path)
    samples = []
    if not isinstance(data, list):
        return samples

    for idx, item in enumerate(data):
        question = item.get("question", "").strip()
        answer = item.get("answer", "").strip()
        if not question or not answer:
            continue

        samples.append(
            make_sample(
                sample_id=f"{task}_{token}_{idx:02d}",
                images=images,
                user_value=question,
                assistant_value=answer,
            )
        )
    return samples


def convert_task(source_root, info, split, task, max_files=None):
    task_dir = source_root / task / split
    if not task_dir.exists():
        print(f"Warning: missing directory {task_dir}")
        return [], 0, 0

    samples = []
    missing_info = 0
    bad_files = 0

    paths = sorted(task_dir.glob("*.json"))
    if max_files is not None:
        paths = paths[:max_files]

    for path in paths:
        token = path.stem
        if token not in info:
            missing_info += 1
            continue

        try:
            images = get_images(info, token)
            if task == "desc":
                samples.extend(convert_desc(path, token, images))
            else:
                samples.extend(convert_qa_list(path, token, images, task))
        except Exception as exc:
            bad_files += 1
            print(f"Warning: failed to convert {path}: {exc}")

    return samples, missing_info, bad_files


def main():
    args = parse_args()
    source_root = Path(args.source_root)

    with open(args.info_path, "rb") as f:
        info = pickle.load(f)

    all_samples = []
    for split in args.splits:
        for task in args.tasks:
            samples, missing_info, bad_files = convert_task(source_root, info, split, task, args.max_files)
            all_samples.extend(samples)
            print(
                f"{task}/{split}: {len(samples)} samples, "
                f"{missing_info} files without cached info, {bad_files} failed files"
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_samples, f, indent=4)

    print(f"Saved {len(all_samples)} samples to {output_path}")


if __name__ == "__main__":
    main()
