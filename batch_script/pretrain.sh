#!/bin/bash
#SBATCH --job-name=fsdrive_pretrain
#SBATCH --partition=a100
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=1-00:00:00
#SBATCH --signal=B:USR1@7200
#SBATCH --output=logs/pretrain_%j.out
#SBATCH --error=logs/pretrain_%j.err
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

TRAIN_PID=""

resubmit_on_timeout() {
  echo "Received Slurm timeout warning. Submitting the next pretrain job to continue from the latest checkpoint."
  cd /anvme/workspace/b305bb10-zyt/FSDrive
  sbatch batch_script/pretrain.sh

  if [[ -n "$TRAIN_PID" ]]; then
    kill -TERM "$TRAIN_PID" 2>/dev/null || true
    wait "$TRAIN_PID" 2>/dev/null || true
  fi

  exit 0
}

trap resubmit_on_timeout USR1

cd /anvme/workspace/b305bb10-zyt/FSDrive/LlamaFactory

llamafactory-cli train ../configs/pretrain.yaml &
TRAIN_PID=$!
wait "$TRAIN_PID"
