#!/bin/bash
#SBATCH --job-name=fsdrive_sft_drv50k_all
#SBATCH --partition=a100
#SBATCH --exclude=a0701
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/sft_drivelm50k_all_%j.out
#SBATCH --error=logs/sft_drivelm50k_all_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m15024431998@163.com
set -e

FSDRIVE_ROOT=/anvme/workspace/b305bb10-zyt/FSDrive

source ${FSDRIVE_ROOT}/activate_fsdrive.sh

cd ${FSDRIVE_ROOT}
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

PRETRAIN_MODEL=${FSDRIVE_ROOT}/LlamaFactory/saves/qwen2_vl-2b/pretrain_three_datasets
COT_DATA=${FSDRIVE_ROOT}/LlamaFactory/data/train_cot_motion.json
DRIVELM_DATA=${FSDRIVE_ROOT}/LlamaFactory/data/train_drivelm_gvqa_50k.json

if [[ ! -f "${PRETRAIN_MODEL}/config.json" || ! -f "${PRETRAIN_MODEL}/model.safetensors" ]]; then
  echo "Missing pretrain model at ${PRETRAIN_MODEL}" >&2
  exit 1
fi

if [[ ! -f "${COT_DATA}" ]]; then
  echo "Missing trajectory SFT data at ${COT_DATA}" >&2
  exit 1
fi

if [[ ! -f "${DRIVELM_DATA}" ]]; then
  echo "Missing DriveLM GVQA data at ${DRIVELM_DATA}" >&2
  echo "Build it with: python create_data/subsample_drivelm_gvqa.py --num-samples 50000" >&2
  exit 1
fi

cd ${FSDRIVE_ROOT}/LlamaFactory

llamafactory-cli train ../configs/sft_drivelm_50k.yaml
