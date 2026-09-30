#!/bin/bash
#SBATCH --account=YOUR_ACCOUNT        # edit
#SBATCH --partition=YOUR_GPU_PARTITION # edit (one node, 4 GPUs)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=360G
#SBATCH --time=12:00:00
#SBATCH --job-name=grpo
#SBATCH --output=logs/%x_%j.out
# piRL GRPO (Flow-SDE) control arm. CONFIG: libero_spatial_grpo_openpi_pi05
# (RLinf's own), libero_10_grpo_openpi_pi05, or *_grpo_openpi_pi05 from this repo.
source jobs/_common.sh
source jobs/_ray_start.sh
echo "config: ${CONFIG:-libero_spatial_grpo_openpi_pi05}  overrides: $*"
cd $REPO_PATH
python $EMBODIED_PATH/train_embodied_agent.py \
    --config-path $EMBODIED_PATH/config/ \
    --config-name ${CONFIG:-libero_spatial_grpo_openpi_pi05} \
    runner.logger.log_path=$LOG_DIR runner.save_interval=10 \
    actor.model.model_path=$CKPT rollout.model.model_path=$CKPT "$@"
rc=$?; echo "---- rc=$rc"; exit $rc
