import argparse
import json
from pathlib import Path


SYSTEM = (
    "You're an autonomous driving assistant for graph visual question answering. "
    "Use the current six camera images to understand the scene, reason about perception, "
    "prediction, planning, behavior, and answer the driving question accurately."
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
    "Answer the following DriveLM graph visual question based on these images.\n"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert DriveLM-nuScenes GVQA data to LLaMA-Factory ShareGPT format.")
    parser.add_argument("--source", default="./DriveLM/v1_1_train_nus.json")
    parser.add_argument("--output", default="./LlamaFactory/data/train_drivelm_gvqa.json")
    parser.add_argument("--media-root", default="./LlamaFactory", help="Root used to verify relative image paths.")
    parser.add_argument("--max-scenes", type=int, default=None, help="Limit scenes for smoke tests.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit converted QA samples for smoke tests.")
    parser.add_argument("--include-empty-answers", action="store_true", help="Keep QA nodes without answers, useful for q-only val data.")
    parser.add_argument("--allow-missing-images", action="store_true", help="Keep samples even if an image path is missing locally.")
    return parser.parse_args()


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def normalize_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = []
        for item in value:
            text = normalize_text(item)
            if text:
                parts.append(text)
        return "\n".join(parts).strip()
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def normalize_image_path(path_value):
    path = str(path_value).strip().replace("\\", "/")
    marker = "nuscenes/"
    if marker in path:
        path = path[path.index(marker) + len(marker):]
    path = path.lstrip("./")
    return f"data/nuscenes/{path}"


def get_images(frame):
    image_paths = frame.get("image_paths", {})
    images = []
    for cam in CAMERA_TYPES:
        if cam not in image_paths:
            return []
        images.append(normalize_image_path(image_paths[cam]))
    return images


def images_exist(images, media_root):
    root = Path(media_root)
    return all((root / image).is_file() for image in images)


def format_question(question, choices):
    question = normalize_text(question)
    choices = normalize_text(choices)
    if not choices:
        return question
    if choices in question:
        return question
    return f"{question}\nOptions:\n{choices}"


def iter_qa_nodes(obj, path=()):
    if isinstance(obj, dict):
        if "Q" in obj or "question" in obj:
            yield path, obj
        for key, value in obj.items():
            if isinstance(value, (dict, list)):
                yield from iter_qa_nodes(value, path + (str(key),))
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            if isinstance(value, (dict, list)):
                yield from iter_qa_nodes(value, path + (str(idx),))


def make_sample(sample_id, images, question, answer):
    return {
        "id": sample_id,
        "images": images,
        "system": SYSTEM,
        "conversations": [
            {
                "from": "human",
                "value": IMAGE_PROMPT + question.strip() + "\n",
            },
            {
                "from": "gpt",
                "value": answer.strip() + "\n<|endoftext|><|im_end|>",
            },
        ],
    }


def convert(data, media_root, max_scenes=None, max_samples=None, include_empty_answers=False, allow_missing_images=False):
    samples = []
    missing_images = 0
    skipped_empty_answers = 0
    bad_frames = 0

    scenes = list(data.items())
    if max_scenes is not None:
        scenes = scenes[:max_scenes]

    for scene_idx, (scene_token, scene) in enumerate(scenes):
        key_frames = scene.get("key_frames", {}) if isinstance(scene, dict) else {}
        for frame_token, frame in key_frames.items():
            images = get_images(frame)
            if len(images) != len(CAMERA_TYPES):
                bad_frames += 1
                continue
            if not allow_missing_images and not images_exist(images, media_root):
                missing_images += 1
                continue

            qa_root = frame.get("QA", {})
            for qa_path, qa in iter_qa_nodes(qa_root):
                question = format_question(qa.get("Q", qa.get("question", "")), qa.get("C"))
                answer = normalize_text(qa.get("A", qa.get("answer", "")))
                if not question:
                    continue
                if not answer and not include_empty_answers:
                    skipped_empty_answers += 1
                    continue

                qa_id = "_".join(part.replace("/", "_") for part in qa_path) or "qa"
                sample_id = f"drivelm_{scene_token}_{frame_token}_{qa_id}_{len(samples):06d}"
                samples.append(make_sample(sample_id, images, question, answer))
                if max_samples is not None and len(samples) >= max_samples:
                    return samples, missing_images, skipped_empty_answers, bad_frames, scene_idx + 1

    return samples, missing_images, skipped_empty_answers, bad_frames, len(scenes)


def main():
    args = parse_args()
    data = load_json(args.source)
    samples, missing_images, skipped_empty_answers, bad_frames, processed_scenes = convert(
        data=data,
        media_root=args.media_root,
        max_scenes=args.max_scenes,
        max_samples=args.max_samples,
        include_empty_answers=args.include_empty_answers,
        allow_missing_images=args.allow_missing_images,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(samples, f, indent=4, ensure_ascii=False)

    print(f"Processed scenes: {processed_scenes}")
    print(f"Saved samples: {len(samples)} to {output_path}")
    print(f"Frames skipped for missing/bad camera paths: {bad_frames}")
    print(f"Frames skipped for missing local images: {missing_images}")
    print(f"QA nodes skipped for empty answers: {skipped_empty_answers}")


if __name__ == "__main__":
    main()
