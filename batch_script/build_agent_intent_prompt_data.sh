#!/bin/bash
#SBATCH --job-name=fsdrive_build_agent_intent_prompt_data
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=logs/build_agent_intent_prompt_data_%j.out
#SBATCH --error=logs/build_agent_intent_prompt_data_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=m15024431998@163.com
set -e

source /anvme/workspace/b305bb10-zyt/FSDrive/activate_fsdrive.sh

cd /anvme/workspace/b305bb10-zyt/FSDrive
mkdir -p logs

python create_data/sft_data_agent_intent_prompt.py --split train
python create_data/sft_data_agent_intent_prompt.py --split val
