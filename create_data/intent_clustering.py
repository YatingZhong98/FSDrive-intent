import argparse
import json
import pickle

import numpy as np


def build_feature(data, token, steps=6):
    traj = np.asarray(data[token]["gt_ego_fut_trajs"], dtype=np.float32)
    if traj.shape[0] < steps + 1:
        raise ValueError(f"{token} has only {traj.shape[0]} future trajectory points")

    # Skip the current point and use the next 3 seconds at 0.5s intervals.
    return traj[1 : steps + 1, :2].reshape(-1)


def standardize(features):
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (features - mean) / std, mean, std


def kmeans(features, num_clusters, max_iter=100, seed=42):
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


def collect_features(data, tokens, steps):
    valid_tokens = []
    features = []
    for token in tokens:
        try:
            features.append(build_feature(data, token, steps=steps))
            valid_tokens.append(token)
        except Exception:
            continue

    if not features:
        raise ValueError("No valid trajectory features were collected.")

    return valid_tokens, np.stack(features, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Cluster future ego trajectories into latent intent ids.")
    parser.add_argument("--info_path", default="./create_data/cached_nuscenes_info.pkl")
    parser.add_argument("--split_path", default="./create_data/full_split.json")
    parser.add_argument("--num_clusters", type=int, default=64)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument("--output_labels", default="./create_data/intent_labels_k64.json")
    parser.add_argument("--output_centers", default="./create_data/intent_centers_k64.npy")
    args = parser.parse_args()

    data = pickle.load(open(args.info_path, "rb"))
    split = json.load(open(args.split_path, "r"))

    train_tokens, train_features = collect_features(data, split["train"], args.steps)
    val_tokens, val_features = collect_features(data, split["val"], args.steps)

    train_features_norm, mean, std = standardize(train_features)
    val_features_norm = (val_features - mean) / std

    train_labels, centers_norm = kmeans(
        train_features_norm,
        num_clusters=args.num_clusters,
        max_iter=args.max_iter,
        seed=args.seed,
    )
    val_labels = assign_labels(val_features_norm, centers_norm)

    centers = centers_norm * std + mean
    np.save(args.output_centers, centers.reshape(args.num_clusters, args.steps, 2))

    label_payload = {
        "metadata": {
            "num_clusters": args.num_clusters,
            "steps": args.steps,
            "seed": args.seed,
            "train_count": len(train_tokens),
            "val_count": len(val_tokens),
        },
        "labels": {
            "train": {token: int(label) for token, label in zip(train_tokens, train_labels)},
            "val": {token: int(label) for token, label in zip(val_tokens, val_labels)},
        },
    }

    with open(args.output_labels, "w") as f:
        json.dump(label_payload, f, indent=4)

    counts = np.bincount(train_labels, minlength=args.num_clusters)
    print(f"Saved labels to {args.output_labels}")
    print(f"Saved centers to {args.output_centers}")
    print(f"Train cluster count min/mean/max: {counts.min()}/{counts.mean():.1f}/{counts.max()}")


if __name__ == "__main__":
    main()
