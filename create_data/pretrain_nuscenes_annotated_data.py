import argparse
import json
import math
import pickle
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


SYSTEM = (
    "You're an autonomous vehicle's brain. Coordinates: X-axis is perpendicular, "
    "and Y-axis is parallel to the direction you're facing. You're at point (0,0). "
    "Units: meters. Based on the provided camera images, predict future driving "
    "scene annotation information in image format."
)

TASK_TEXT = {
    "combined": {
        "name": "lane divider and 3D detection",
        "id": "nuscenes_annotated",
        "manifest": "./LlamaFactory/data/pretrain_nuscenes_annotated_manifest.json",
        "output": "./LlamaFactory/data/pretrain_nuscenes_annotated_data.json",
    },
    "lane": {
        "name": "lane divider",
        "id": "nuscenes_lane",
        "manifest": "./LlamaFactory/data/pretrain_nuscenes_lane_manifest.json",
        "output": "./LlamaFactory/data/pretrain_nuscenes_lane_data.json",
    },
    "bbox": {
        "name": "3D detection",
        "id": "nuscenes_3d_detection",
        "manifest": "./LlamaFactory/data/pretrain_nuscenes_3d_detection_manifest.json",
        "output": "./LlamaFactory/data/pretrain_nuscenes_3d_detection_data.json",
    },
}

CAMERA_TYPES = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

IMAGE_PROMPT = (
    "Here are current six images from the car: "
    "'CAM_FRONT': <image>\n"
    ", 'CAM_FRONT_LEFT': <image>\n"
    ", 'CAM_FRONT_RIGHT': <image>\n"
    ", 'CAM_BACK': <image>\n"
    ", 'CAM_BACK_LEFT': <image>\n"
    ", 'CAM_BACK_RIGHT': <image>\n"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render nuScenes official lane/object annotations and optionally build visual-token pretrain data."
    )
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--task", default="combined", choices=["combined", "lane", "bbox"])
    parser.add_argument("--info-path", default="./create_data/cached_nuscenes_info.pkl")
    parser.add_argument("--lane-obj-path", default="./LlamaFactory/data/nuscenes/lane_obj_train.pkl")
    parser.add_argument("--render-dir", default="./LlamaFactory/data/nuscenes_annotated_targets")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--target-token-path", default=None, help="MoVQGAN token json for rendered target PNGs.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--extent", type=float, default=50.0, help="BEV range in meters for x/y axes.")
    parser.add_argument("--skip-render", action="store_true")
    return parser.parse_args()


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def to_rel_data_path(path):
    return str(path).replace("LlamaFactory/", "", 1)


def get_images(info, token):
    images = []
    cams = info[token]["cams"]
    for cam in CAMERA_TYPES:
        images.append(cams[cam]["data_path"].replace("/localdata_ssd/nuScenes", "data/nuscenes", 1))
    return images


def world_to_pixel(x, y, size, extent):
    # x: lateral, y: forward. Image origin is top-left; forward is upward.
    px = int(round((x + extent) / (2 * extent) * (size - 1)))
    py = int(round((extent - y) / (2 * extent) * (size - 1)))
    return px, py


def in_bounds(pt, size):
    x, y = pt
    return 0 <= x < size and 0 <= y < size


def box_corners(x, y, dx, dy, yaw):
    corners = []
    for ox, oy in [(dx / 2, dy / 2), (dx / 2, -dy / 2), (-dx / 2, -dy / 2), (-dx / 2, dy / 2)]:
        rx = ox * math.cos(yaw) + oy * math.sin(yaw)
        ry = -ox * math.sin(yaw) + oy * math.cos(yaw)
        corners.append((x + rx, y + ry))
    return corners


def draw_polyline(draw, pts, size, extent, color, width):
    pixels = [world_to_pixel(float(p[0]), float(p[1]), size, extent) for p in pts]
    if len(pixels) >= 2:
        draw.line(pixels, fill=color, width=width, joint="curve")


def draw_box(draw, box, size, extent, color, width):
    x, y, _z, dx, dy, _dz, yaw = [float(v) for v in box]
    pixels = [world_to_pixel(cx, cy, size, extent) for cx, cy in box_corners(x, y, dx, dy, yaw)]
    if any(in_bounds(p, size) for p in pixels):
        draw.line(pixels + [pixels[0]], fill=color, width=width)


def render_target(annotation, out_path, size, extent, task):
    image = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    grid = (230, 230, 230)
    for meter in range(int(-extent), int(extent) + 1, 10):
        x0, y0 = world_to_pixel(-extent, meter, size, extent)
        x1, y1 = world_to_pixel(extent, meter, size, extent)
        draw.line([(x0, y0), (x1, y1)], fill=grid, width=1)
        x0, y0 = world_to_pixel(meter, -extent, size, extent)
        x1, y1 = world_to_pixel(meter, extent, size, extent)
        draw.line([(x0, y0), (x1, y1)], fill=grid, width=1)

    # Ego vehicle marker.
    ex, ey = world_to_pixel(0, 0, size, extent)
    draw.ellipse([(ex - 3, ey - 3), (ex + 3, ey + 3)], fill=(0, 0, 0))

    if task in {"combined", "lane"}:
        for lane in annotation.get("all_lane_pts", []):
            arr = np.asarray(lane)
            if arr.ndim == 2 and arr.shape[0] >= 2:
                draw_polyline(draw, arr[:, :2], size, extent, color=(20, 110, 220), width=2)

    if task in {"combined", "bbox"}:
        lane_objects = annotation.get("lane_objects", {})
        colors = [(220, 20, 20), (0, 150, 80), (190, 90, 20), (130, 20, 180)]
        for idx, (_lane_id, objects) in enumerate(sorted(lane_objects.items(), key=lambda item: str(item[0]))):
            color = colors[idx % len(colors)]
            for item in objects:
                if len(item) < 2:
                    continue
                box = item[1]
                draw_box(draw, box, size, extent, color=color, width=2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def make_visual_tokens(token_value):
    token_value = str(token_value).replace(" ", "")
    numbers = token_value.strip("[]").split(",")
    return "".join(f"<|{num}|>" for num in numbers if num != "")


def token_lookup(tokens, target_rel_path, token):
    candidates = [target_rel_path, Path(target_rel_path).name, token]
    for key in candidates:
        if key in tokens:
            return tokens[key]
    return None


def make_sample(token, input_images, target_tokens, task):
    task_name = TASK_TEXT[task]["name"]
    return {
        "id": f"{TASK_TEXT[task]['id']}_{token}",
        "images": input_images,
        "system": SYSTEM,
        "conversations": [
            {
                "from": "human",
                "value": IMAGE_PROMPT + f"Please predict the future {task_name} result in image-token format.\n",
            },
            {
                "from": "gpt",
                "value": target_tokens + f" These are the visual tokens of the future {task_name} image.\n<|endoftext|><|im_end|>",
            },
        ],
    }


def main():
    args = parse_args()
    manifest_path = Path(args.manifest or TASK_TEXT[args.task]["manifest"])
    output_path = Path(args.output or TASK_TEXT[args.task]["output"])
    info = load_pickle(args.info_path)
    lane_obj = load_pickle(args.lane_obj_path)
    target_tokens = load_json(args.target_token_path) if args.target_token_path else None

    if args.task == "combined":
        render_root = Path(args.render_dir) / args.split
    else:
        render_root = Path(args.render_dir) / args.task / args.split
    manifest = []
    train_samples = []

    tokens = [token for token in lane_obj.keys() if token in info]
    if args.max_samples is not None:
        tokens = tokens[: args.max_samples]

    missing_token_targets = 0
    for idx, token in enumerate(tokens):
        target_path = render_root / f"{token}.png"
        target_rel_path = to_rel_data_path(target_path)

        if not args.skip_render:
            render_target(lane_obj[token], target_path, args.image_size, args.extent, args.task)

        input_images = get_images(info, token)
        manifest.append(
            {
                "id": token,
                "images": input_images,
                "target_image": target_rel_path,
            }
        )

        if target_tokens is not None:
            raw_tokens = token_lookup(target_tokens, target_rel_path, token)
            if raw_tokens is None:
                missing_token_targets += 1
                continue
            train_samples.append(make_sample(token, input_images, make_visual_tokens(raw_tokens), args.task))

        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1}/{len(tokens)} samples")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)

    print(f"Saved manifest with {len(manifest)} samples to {manifest_path}")

    if target_tokens is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(train_samples, f, indent=4)
        print(f"Saved train data with {len(train_samples)} samples to {output_path}")
        print(f"Missing tokenized targets: {missing_token_targets}")
    else:
        print("No --target-token-path provided; rendered target images only. Tokenize them with MoVQGAN before building train JSON.")


if __name__ == "__main__":
    main()
