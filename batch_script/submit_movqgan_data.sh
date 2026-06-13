#!/bin/bash
set -euo pipefail

ROOT="${ROOT:-/anvme/workspace/b305bb10-zyt/FSDrive}"
PARTITION="${PARTITION:-a100}"
GPUS="${GPUS:-8}"
CPUS="${CPUS:-64}"
TIME="${TIME:-1-00:00:00}"
LOG_DIR="${ROOT}/logs"

mkdir -p "$LOG_DIR"

pretrain_job_id=$(
  sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=movqgan_pretrain_data
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:${GPUS}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --time=${TIME}
#SBATCH --output=${LOG_DIR}/movqgan_pretrain_data_%j.out
#SBATCH --error=${LOG_DIR}/movqgan_pretrain_data_%j.err

set -e

source ${ROOT}/activate_fsdrive.sh

cd ${ROOT}
python MoVQGAN/pretrain_data.py
EOF
)

echo "Submitted MoVQGAN pretrain data job: ${pretrain_job_id}"

sft_job_id=$(
  sbatch --parsable --dependency=afterok:${pretrain_job_id} <<EOF
#!/bin/bash
#SBATCH --job-name=movqgan_sft_data
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:${GPUS}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --time=${TIME}
#SBATCH --output=${LOG_DIR}/movqgan_sft_data_%j.out
#SBATCH --error=${LOG_DIR}/movqgan_sft_data_%j.err

set -e

source ${ROOT}/activate_fsdrive.sh

cd ${ROOT}
python MoVQGAN/sft_data.py
EOF
)

echo "Submitted MoVQGAN SFT data job: ${sft_job_id}"
echo "SFT job will start after pretrain data job finishes successfully."
echo "Check queue: squeue -u \$USER"
echo "Check logs:  ls -lh ${LOG_DIR}/movqgan_*"
