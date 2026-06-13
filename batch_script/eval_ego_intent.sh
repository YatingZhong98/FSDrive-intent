#!/bin/bash
#SBATCH --job-name=fsdrive_eval_ego_intent
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/eval_ego_intent_%j.out
#SBATCH --error=logs/eval_ego_intent_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=m15024431998@163.com
set -e

source /anvme/workspace/b305bb10-zyt/FSDrive/activate_fsdrive.sh

cd /anvme/workspace/b305bb10-zyt/FSDrive
mkdir -p logs
mkdir -p LlamaFactory/eval_results

module load cuda/12.8.1
NVCC_PATH="$(command -v nvcc)"
export CUDA_HOME="$(dirname "$(dirname "$NVCC_PATH")")"
export CUDA_ROOT="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

PRED_PATH=LlamaFactory/eval_results/results_sft_ego_intent.jsonl
EVAL_TRAJ_PATH=LlamaFactory/eval_results/eval_traj_sft_ego_intent_val_cot_motion_ego_intent.json

python tools/match.py \
  --pred_trajs_path "$PRED_PATH" \
  --token_traj_path LlamaFactory/data/val_cot_motion_ego_intent.json \
  --output_path "$EVAL_TRAJ_PATH"

python tools/evaluation/evaluate_intent_accuracy.py \
  --pred_path "$PRED_PATH" \
  --token_path LlamaFactory/data/val_cot_motion_ego_intent.json \
  --intent_labels create_data/intent_labels_k64.json \
  --split val

python tools/evaluation/evaluation.py \
  --method FSDrive-ego-intent \
  --metric uniad \
  --gt_folder tools/data/metrics \
  --result_file "$EVAL_TRAJ_PATH"
