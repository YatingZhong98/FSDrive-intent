#!/bin/bash
#SBATCH --job-name=fsdrive_vllm_infer_ego_intent
#SBATCH --partition=a100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/vllm_infer_ego_intent_%j.out
#SBATCH --error=logs/vllm_infer_ego_intent_%j.err
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

cd /anvme/workspace/b305bb10-zyt/FSDrive/LlamaFactory

MODEL_ROOT=saves/qwen2_vl-2b/sft_ego_intent
MODEL_PATH="$MODEL_ROOT"
if [[ ! -f "${MODEL_PATH}/config.json" || ! -f "${MODEL_PATH}/model.safetensors" ]]; then
  MODEL_PATH="$(find "$MODEL_ROOT" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)"
fi
if [[ -z "$MODEL_PATH" || ! -f "${MODEL_PATH}/config.json" || ! -f "${MODEL_PATH}/model.safetensors" ]]; then
  echo "Missing model or checkpoint under ${MODEL_ROOT}" >&2
  exit 1
fi
SAVE_NAME=eval_results/results_sft_ego_intent.jsonl

python ../tools/patch_vllm_rope_config.py "$MODEL_PATH"

python scripts/vllm_infer.py \
  --model_name_or_path "$MODEL_PATH" \
  --dataset val_cot_motion_ego_intent \
  --dataset_dir /anvme/workspace/b305bb10-zyt/FSDrive/LlamaFactory/data \
  --template qwen2_vl \
  --cutoff_len 30720 \
  --max_new_tokens 2048 \
  --max_samples 10000 \
  --image_max_pixels 524288 \
  --save_name "$SAVE_NAME" \
  --temperature 0.1 \
  --top_p 0.1 \
  --top_k 10 \
  --skip_special_tokens False
