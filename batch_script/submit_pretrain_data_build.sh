#!/bin/bash
set -euo pipefail

ROOT="${ROOT:-/anvme/workspace/b305bb10-zyt/FSDrive}"
PARTITION="${PARTITION:-a100}"
GPU_PARTITION="${GPU_PARTITION:-${PARTITION}}"
CPU_PARTITION="${CPU_PARTITION:-${PARTITION}}"
GPUS="${GPUS:-2}"
GPU_CPUS="${GPU_CPUS:-16}"
CPU_CPUS="${CPU_CPUS:-8}"
CPU_GPUS="${CPU_GPUS:-1}"
RENDER_TIME="${RENDER_TIME:-08:00:00}"
OMNI_TIME="${OMNI_TIME:-04:00:00}"
GPU_TIME="${GPU_TIME:-1-00:00:00}"
BUILD_TIME="${BUILD_TIME:-02:00:00}"
LOG_DIR="${ROOT}/logs"
MAX_SAMPLES="${MAX_SAMPLES:-}"

mkdir -p "${LOG_DIR}"

max_samples_arg=""
if [[ -n "${MAX_SAMPLES}" ]]; then
  max_samples_arg="--max-samples ${MAX_SAMPLES}"
fi

omni_job_id=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=pretrain_omnidrive_data
#SBATCH --partition=${CPU_PARTITION}
#SBATCH --gres=gpu:${CPU_GPUS}
#SBATCH --cpus-per-task=${CPU_CPUS}
#SBATCH --time=${OMNI_TIME}
#SBATCH --output=${LOG_DIR}/pretrain_omnidrive_data_%j.out
#SBATCH --error=${LOG_DIR}/pretrain_omnidrive_data_%j.err

set -e
source ${ROOT}/activate_fsdrive.sh
cd ${ROOT}
python create_data/pretrain_omnidrive_data.py
EOF
)

echo "Submitted OmniDrive ShareGPT build job: ${omni_job_id}"

lane_render_job_id=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=nusc_lane_render
#SBATCH --partition=${CPU_PARTITION}
#SBATCH --gres=gpu:${CPU_GPUS}
#SBATCH --cpus-per-task=${CPU_CPUS}
#SBATCH --time=${RENDER_TIME}
#SBATCH --output=${LOG_DIR}/nusc_lane_render_%j.out
#SBATCH --error=${LOG_DIR}/nusc_lane_render_%j.err

set -e
source ${ROOT}/activate_fsdrive.sh
cd ${ROOT}
python create_data/pretrain_nuscenes_annotated_data.py --task lane ${max_samples_arg}
EOF
)

echo "Submitted nuScenes lane target render job: ${lane_render_job_id}"

bbox_render_job_id=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=nusc_bbox_render
#SBATCH --partition=${CPU_PARTITION}
#SBATCH --gres=gpu:${CPU_GPUS}
#SBATCH --cpus-per-task=${CPU_CPUS}
#SBATCH --time=${RENDER_TIME}
#SBATCH --output=${LOG_DIR}/nusc_bbox_render_%j.out
#SBATCH --error=${LOG_DIR}/nusc_bbox_render_%j.err

set -e
source ${ROOT}/activate_fsdrive.sh
cd ${ROOT}
python create_data/pretrain_nuscenes_annotated_data.py --task bbox ${max_samples_arg}
EOF
)

echo "Submitted nuScenes 3D detection target render job: ${bbox_render_job_id}"

lane_token_job_id=$(sbatch --parsable --dependency=afterok:${lane_render_job_id} <<EOF
#!/bin/bash
#SBATCH --job-name=nusc_lane_tokens
#SBATCH --partition=${GPU_PARTITION}
#SBATCH --gres=gpu:${GPUS}
#SBATCH --cpus-per-task=${GPU_CPUS}
#SBATCH --time=${GPU_TIME}
#SBATCH --output=${LOG_DIR}/nusc_lane_tokens_%j.out
#SBATCH --error=${LOG_DIR}/nusc_lane_tokens_%j.err

set -e
source ${ROOT}/activate_fsdrive.sh
cd ${ROOT}
python MoVQGAN/nuscenes_annotated_data.py \
  --image-dir ./LlamaFactory/data/nuscenes_annotated_targets/lane/train \
  --output-name gt_indices_nuscenes_lane_train.json
EOF
)

echo "Submitted nuScenes lane MoVQGAN token job: ${lane_token_job_id}"

bbox_token_job_id=$(sbatch --parsable --dependency=afterok:${bbox_render_job_id} <<EOF
#!/bin/bash
#SBATCH --job-name=nusc_bbox_tokens
#SBATCH --partition=${GPU_PARTITION}
#SBATCH --gres=gpu:${GPUS}
#SBATCH --cpus-per-task=${GPU_CPUS}
#SBATCH --time=${GPU_TIME}
#SBATCH --output=${LOG_DIR}/nusc_bbox_tokens_%j.out
#SBATCH --error=${LOG_DIR}/nusc_bbox_tokens_%j.err

set -e
source ${ROOT}/activate_fsdrive.sh
cd ${ROOT}
python MoVQGAN/nuscenes_annotated_data.py \
  --image-dir ./LlamaFactory/data/nuscenes_annotated_targets/bbox/train \
  --output-name gt_indices_nuscenes_3d_detection_train.json
EOF
)

echo "Submitted nuScenes 3D detection MoVQGAN token job: ${bbox_token_job_id}"

lane_build_job_id=$(sbatch --parsable --dependency=afterok:${lane_token_job_id} <<EOF
#!/bin/bash
#SBATCH --job-name=nusc_lane_json
#SBATCH --partition=${CPU_PARTITION}
#SBATCH --gres=gpu:${CPU_GPUS}
#SBATCH --cpus-per-task=${CPU_CPUS}
#SBATCH --time=${BUILD_TIME}
#SBATCH --output=${LOG_DIR}/nusc_lane_json_%j.out
#SBATCH --error=${LOG_DIR}/nusc_lane_json_%j.err

set -e
source ${ROOT}/activate_fsdrive.sh
cd ${ROOT}
python create_data/pretrain_nuscenes_annotated_data.py \
  --task lane \
  --skip-render \
  --target-token-path ./MoVQGAN/gt_indices_nuscenes_lane_train.json \
  ${max_samples_arg}
EOF
)

echo "Submitted nuScenes lane ShareGPT build job: ${lane_build_job_id}"

bbox_build_job_id=$(sbatch --parsable --dependency=afterok:${bbox_token_job_id} <<EOF
#!/bin/bash
#SBATCH --job-name=nusc_bbox_json
#SBATCH --partition=${CPU_PARTITION}
#SBATCH --gres=gpu:${CPU_GPUS}
#SBATCH --cpus-per-task=${CPU_CPUS}
#SBATCH --time=${BUILD_TIME}
#SBATCH --output=${LOG_DIR}/nusc_bbox_json_%j.out
#SBATCH --error=${LOG_DIR}/nusc_bbox_json_%j.err

set -e
source ${ROOT}/activate_fsdrive.sh
cd ${ROOT}
python create_data/pretrain_nuscenes_annotated_data.py \
  --task bbox \
  --skip-render \
  --target-token-path ./MoVQGAN/gt_indices_nuscenes_3d_detection_train.json \
  ${max_samples_arg}
EOF
)

echo "Submitted nuScenes 3D detection ShareGPT build job: ${bbox_build_job_id}"
echo ""
echo "Dependency chain:"
echo "  OmniDrive ShareGPT: ${omni_job_id}"
echo "  lane render -> tokens -> json: ${lane_render_job_id} -> ${lane_token_job_id} -> ${lane_build_job_id}"
echo "  bbox render -> tokens -> json: ${bbox_render_job_id} -> ${bbox_token_job_id} -> ${bbox_build_job_id}"
echo ""
echo "Expected outputs:"
echo "  ${ROOT}/LlamaFactory/data/pretrain_omnidrive_data.json"
echo "  ${ROOT}/LlamaFactory/data/pretrain_nuscenes_lane_data.json"
echo "  ${ROOT}/LlamaFactory/data/pretrain_nuscenes_3d_detection_data.json"
echo ""
echo "Check queue: squeue -u \$USER"
echo "Check logs:  ls -lh ${LOG_DIR}/pretrain_omnidrive_data_* ${LOG_DIR}/nusc_lane_* ${LOG_DIR}/nusc_bbox_*"
