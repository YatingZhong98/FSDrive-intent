#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def patch_rope_block(obj):
    changed = False
    if isinstance(obj, dict):
        if obj.get("type") == "mrope":
            obj["rope_type"] = "default"
            del obj["type"]
            changed = True
        elif obj.get("rope_type") == "mrope" and "mrope_section" in obj:
            obj["rope_type"] = "default"
            changed = True

        for value in obj.values():
            changed = patch_rope_block(value) or changed
    elif isinstance(obj, list):
        for value in obj:
            changed = patch_rope_block(value) or changed

    return changed


def patch_qwen2vl_top_level_text_config(config):
    if config.get("model_type") != "qwen2_vl":
        return False

    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        return False

    changed = False
    for key in ("vocab_size",):
        if key not in config and key in text_config:
            config[key] = text_config[key]
            changed = True

    return changed


def patch_config(config_path):
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    changed = patch_rope_block(config)
    changed = patch_qwen2vl_top_level_text_config(config) or changed

    if not changed:
        return False

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        f.write("\n")

    return True


def iter_config_paths(model_path):
    if model_path.is_file():
        if model_path.name != "config.json":
            raise ValueError(f"Expected config.json file, got: {model_path}")
        yield model_path
    else:
        yield from sorted(model_path.rglob("config.json"))


def main():
    parser = argparse.ArgumentParser(description="Patch saved Qwen2-VL RoPE configs for vLLM.")
    parser.add_argument("model_path", help="Model directory or config.json to patch before vLLM inference.")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    patched = []
    for config_path in iter_config_paths(model_path):
        if patch_config(config_path):
            patched.append(config_path)

    if patched:
        print("Patched vLLM config compatibility in:")
        for path in patched:
            print(f"  {path}")
    else:
        print("No vLLM config compatibility changes needed.")


if __name__ == "__main__":
    main()
