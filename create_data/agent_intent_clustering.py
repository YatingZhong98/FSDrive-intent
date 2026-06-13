import argparse
import json
import pickle

import numpy as np


def standardize(features):
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (features - mean) / std, mean, std


def kmeans(features, num_clusters, max_iter=100, seed=42):
    if features.shape[0] < num_clusters:
        raise ValueError(f"Only {features.shape[0]} features, fewer than {num_clusters} clusters.")

    rng = np.random.default_rng(seed)
    init_indices = rng.choice(features.shape[0], size=num_clusters, replace=False)
    centers = features[init_indices].copy()
    labels = np.zeros(features.shape[0], dtype=np.int64)

    for _ in range(max_iter):
        distances = ((features[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = distances.argmin(axis=1)
        if np.array_equal(labels, new_labels):
            break

        labels = new_labels
        for cluster_id in range(num_clusters):
            cluster_features = features[labels == cluster_id]
            if len(cluster_features) == 0:
                centers[cluster_id] = features[rng.integers(0, features.shape[0])]
            else:
                centers[cluster_id] = cluster_features.mean(axis=0)

    return labels, centers


def assign_labels(features, centers):
    distances = ((features[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return distances.argmin(axis=1).astype(np.int64)


def is_vehicle(name):
    return str(name).startswith("vehicle.")


def build_agent_feature(sample, agent_index, steps=6, min_valid_steps=6):
    traj = np.asarray(sample["gt_agent_fut_trajs"][agent_index], dtype=np.float32)
    mask = np.asarray(sample["gt_agent_fut_masks"][agent_index], dtype=np.float32)

    if traj.shape[0] < steps * 2 or mask.shape[0] < steps:
        raise ValueError("agent trajectory is shorter than requested steps")

    if int(mask[:steps].sum()) < min_valid_steps:
        raise ValueError("agent has too few valid future steps")

    traj = traj[: steps * 2].reshape(steps, 2)
    if not np.isfinite(traj).all():
        raise ValueError("agent trajectory contains non-finite values")

    return traj.reshape(-1)


def select_agents(sample, args):
    boxes = np.asarray(sample["gt_boxes"], dtype=np.float32)
    names = np.asarray(sample["gt_names"])
    selected = []

    for agent_index, name in enumerate(names):
        if not is_vehicle(name):
            continue

        try:
            feature = build_agent_feature(
                sample,
                agent_index,
                steps=args.steps,
                min_valid_steps=args.min_valid_steps,
            )
        except Exception:
            continue

        if agent_index >= boxes.shape[0]:
            continue

        x, y = float(boxes[agent_index, 0]), float(boxes[agent_index, 1])
        distance = float(np.hypot(x, y))

        if not args.include_behind and y <= 0:
            continue
        if abs(x) > args.max_lateral:
            continue
        if distance > args.max_distance:
            continue

        selected.append(
            {
                "agent_index": agent_index,
                "name": str(name),
                "x": x,
                "y": y,
                "distance": distance,
                "feature": feature,
            }
        )

    selected.sort(key=lambda item: (item["distance"], abs(item["x"])))
    return selected[: args.top_k]


def collect_features(data, tokens, args):
    sample_tokens = []
    agent_indices = []
    features = []

    for token in tokens:
        sample = data[token]
        for agent in select_agents(sample, args):
            sample_tokens.append(token)
            agent_indices.append(agent["agent_index"])
            features.append(agent["feature"])

    if not features:
        raise ValueError("No valid agent trajectory features were collected.")

    return sample_tokens, agent_indices, np.stack(features, axis=0)


def build_label_entries(data, tokens, centers_norm, mean, std, args):
    labels = {}
    for token in tokens:
        sample = data[token]
        entries = []
        selected_agents = select_agents(sample, args)
        if selected_agents:
            features = np.stack([agent["feature"] for agent in selected_agents], axis=0)
            labels_norm = assign_labels((features - mean) / std, centers_norm)

            for slot, (agent, intent_id) in enumerate(zip(selected_agents, labels_norm)):
                entries.append(
                    {
                        "slot": slot,
                        "agent_index": int(agent["agent_index"]),
                        "name": agent["name"],
                        "box_xy": [round(agent["x"], 4), round(agent["y"], 4)],
                        "distance": round(agent["distance"], 4),
                        "intent": int(intent_id),
                    }
                )

        labels[token] = entries

    return labels


def main():
    parser = argparse.ArgumentParser(description="Cluster selected surrounding vehicle trajectories into intent ids.")
    parser.add_argument("--info_path", default="./create_data/cached_nuscenes_info.pkl")
    parser.add_argument("--split_path", default="./create_data/full_split.json")
    parser.add_argument("--num_clusters", type=int, default=64)
    parser.add_argument("--top_k", type=int, default=4)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--min_valid_steps", type=int, default=6)
    parser.add_argument("--max_distance", type=float, default=50.0)
    parser.add_argument("--max_lateral", type=float, default=15.0)
    parser.add_argument("--include_behind", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument("--output_labels", default="./create_data/agent_intent_labels_k64_top4.json")
    parser.add_argument("--output_centers", default="./create_data/agent_intent_centers_k64.npy")
    args = parser.parse_args()

    data = pickle.load(open(args.info_path, "rb"))
    split = json.load(open(args.split_path, "r"))

    train_sample_tokens, train_agent_indices, train_features = collect_features(data, split["train"], args)
    train_features_norm, mean, std = standardize(train_features)
    train_feature_labels, centers_norm = kmeans(
        train_features_norm,
        num_clusters=args.num_clusters,
        max_iter=args.max_iter,
        seed=args.seed,
    )

    centers = centers_norm * std + mean
    np.save(args.output_centers, centers.reshape(args.num_clusters, args.steps, 2))

    label_payload = {
        "metadata": {
            "num_clusters": args.num_clusters,
            "top_k": args.top_k,
            "steps": args.steps,
            "min_valid_steps": args.min_valid_steps,
            "max_distance": args.max_distance,
            "max_lateral": args.max_lateral,
            "include_behind": args.include_behind,
            "seed": args.seed,
            "train_agent_count": len(train_agent_indices),
            "train_sample_count": len(set(train_sample_tokens)),
            "val_sample_count": len(split["val"]),
        },
        "labels": {
            "train": build_label_entries(data, split["train"], centers_norm, mean, std, args),
            "val": build_label_entries(data, split["val"], centers_norm, mean, std, args),
        },
    }

    with open(args.output_labels, "w") as f:
        json.dump(label_payload, f, indent=4)

    counts = np.bincount(train_feature_labels, minlength=args.num_clusters)
    print(f"Saved labels to {args.output_labels}")
    print(f"Saved centers to {args.output_centers}")
    print(f"Train selected agent count: {len(train_agent_indices)}")
    print(f"Train cluster count min/mean/max: {counts.min()}/{counts.mean():.1f}/{counts.max()}")


if __name__ == "__main__":
    main()
