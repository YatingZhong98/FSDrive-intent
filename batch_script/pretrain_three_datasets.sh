#!/bin/bash
#SBATCH --job-name=fsdrive_pretrain_three
#SBATCH --partition=a100
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=1-00:00:00
#SBATCH --signal=B:USR1@1800
#SBATCH --output=logs/pretrain_three_%j.out
#SBATCH --error=logs/pretrain_three_%j.err
#SBATCH --mail-type=ALL
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

resubmit_on_timeout() {
  echo "Received USR1 before time limit; submitting continuation job after ${SLURM_JOB_ID}."
  cd /anvme/workspace/b305bb10-zyt/FSDrive
  sbatch --dependency=afterany:${SLURM_JOB_ID} batch_script/pretrain_three_datasets.sh
}

trap resubmit_on_timeout USR1

cd /anvme/workspace/b305bb10-zyt/FSDrive/LlamaFactory

llamafactory-cli train ../configs/pretrain_three_datasets.yaml
