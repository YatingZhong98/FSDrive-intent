#!/bin/bash
#SBATCH --job-name=fsdrive_sft_agent_intent_prompt
#SBATCH --partition=a100
#SBATCH --constraint=a100_40&el9
#SBATCH --exclude=a0701
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/sft_agent_intent_prompt_%j.out
#SBATCH --error=logs/sft_agent_intent_prompt_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=m15024431998@163.com
set -e

source /anvme/workspace/b305bb10-zyt/FSDrive/activate_fsdrive.sh

cd /anvme/workspace/b305bb10-zyt/FSDrive
mkdir -p logs

module load cuda/12.8.1
NVCC_PATH="$(command -v nvcc)"
export CUDA_HOME="$(dirname "$(dirname "$NVCC_PATH")")"
export CUDA_ROOT="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

export FORCE_TORCHRUN=1
export NNODES=1
export NPROC_PER_NODE=4

PRETRAIN_MODEL=/anvme/workspace/b305bb10-zyt/FSDrive/LlamaFactory/saves/qwen2_vl-2b/pretrain_three_datasets
TRAIN_DATA=/anvme/workspace/b305bb10-zyt/FSDrive/LlamaFactory/data/train_cot_motion_agent_intent_prompt.json

if [[ ! -f "$PRETRAIN_MODEL/trainer_state.json" ]]; then
  echo "Missing pretrain model at $PRETRAIN_MODEL" >&2
  exit 1
fi

if [[ ! -f "$TRAIN_DATA" ]]; then
  echo "Missing agent-intent prompt SFT data at $TRAIN_DATA" >&2
  exit 1
fi

cd /anvme/workspace/b305bb10-zyt/FSDrive/LlamaFactory

llamafactory-cli train ../configs/sft_agent_intent_prompt.yaml
