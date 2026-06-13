#!/bin/bash
#SBATCH --job-name=fsdrive_vis_cot
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/vis_cot_%j.out
#SBATCH --error=logs/vis_cot_%j.err
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

INPUT_JSON=${INPUT_JSON:-LlamaFactory/eval_results/eval_traj_sft_val_cot_motion.json}
OUTPUT_DIR=${OUTPUT_DIR:-vis_cot}

python MoVQGAN/vis.py \
  --input_json "$INPUT_JSON" \
  --output_dir "$OUTPUT_DIR"
