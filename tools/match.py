import json
import os
import argparse

def load_pred_trajs_from_json(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]

parser = argparse.ArgumentParser(description="Process trajectories and generate evaluation JSON.")
parser.add_argument("--pred_trajs_path", type=str, required=True, 
                    help="Path to the generated predictions JSONL file (e.g., generated_predictions.jsonl)")
parser.add_argument("--token_traj_path", type=str, required=True, 
                    help="Path to the token trajectories JSON file (e.g., val_cot_motion_single.json)")
parser.add_argument("--output_dir", type=str, 
                    help="Directory to save the output file. If not provided, uses the same directory as pred_trajs_path.")
parser.add_argument("--output_path", type=str,
                    help="Full path to save the output file. Overrides output_dir if provided.")
args = parser.parse_args()

pred_trajs = load_pred_trajs_from_json(args.pred_trajs_path)
token_traj = json.load(open(args.token_traj_path, 'r'))


if len(pred_trajs) > len(token_traj):
    raise ValueError(
        f"More predictions than token trajectories: {len(pred_trajs)} > {len(token_traj)}"
    )

if len(pred_trajs) < len(token_traj):
    print(
        f"Warning: only {len(pred_trajs)} predictions for {len(token_traj)} trajectories; "
        f"matching the first {len(pred_trajs)} entries."
    )

eval_traj = {}
for pred, traj in zip(pred_trajs, token_traj):
    eval_traj[traj['id']] = pred['predict']


if args.output_path:
    output_path = args.output_path
else:
    output_dir = args.output_dir if args.output_dir else os.path.dirname(args.pred_trajs_path)
    output_path = os.path.join(output_dir, "eval_traj.json")

output_dir = os.path.dirname(output_path)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)

with open(output_path, "w") as f:
    json.dump(eval_traj, f, indent=4)

print(f"Evaluation trajectories saved to {output_path}")
