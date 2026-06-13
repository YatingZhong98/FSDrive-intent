#!/bin/bash
#SBATCH --job-name=fsdrive_sft
#SBATCH --partition=a100
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/sft_%j.out
#SBATCH --error=logs/sft_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=m15024431998@163.com
set -e

FSDRIVE_ROOT=/anvme/workspace/b305bb10-zyt/FSDrive

if [[ "${1:-}" == "submit" ]]; then
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "Submit mode should be run from the login shell, not inside sbatch." >&2
    exit 1
  fi

  CHAIN_LENGTH="${2:-1}"
  if ! [[ "$CHAIN_LENGTH" =~ ^[0-9]+$ ]] || [[ "$CHAIN_LENGTH" -lt 1 ]]; then
    echo "Usage: bash batch_script/sft.sh submit [num_jobs]" >&2
    exit 1
  fi
  if [[ "$CHAIN_LENGTH" -ne 1 ]]; then
    echo "SFT resume is disabled in this script; submit a single fresh training job." >&2
    exit 1
  fi

  cd "$FSDRIVE_ROOT"
  mkdir -p logs

  previous_job_id=""
  for chain_index in $(seq 1 "$CHAIN_LENGTH"); do
    dependency_args=()
    if [[ -n "$previous_job_id" ]]; then
      dependency_args=(--dependency=afterany:"$previous_job_id")
    fi

    job_id=$(
      sbatch --parsable \
        "${dependency_args[@]}" \
        --export=ALL,SFT_CHAIN_INDEX="$chain_index",SFT_CHAIN_LENGTH="$CHAIN_LENGTH" \
        batch_script/sft.sh
    )
    echo "Submitted SFT chain job ${chain_index}/${CHAIN_LENGTH}: ${job_id}"
    previous_job_id="$job_id"
  done

  exit 0
fi

source /anvme/workspace/b305bb10-zyt/FSDrive/activate_fsdrive.sh

cd "$FSDRIVE_ROOT"
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

echo "SFT chain index: ${SFT_CHAIN_INDEX:-1}/${SFT_CHAIN_LENGTH:-1}"
echo "Starting fresh SFT training; resume is disabled."

cd "$FSDRIVE_ROOT/LlamaFactory"

llamafactory-cli train ../configs/sft.yaml overwrite_output_dir=true
