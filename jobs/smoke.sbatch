#!/bin/bash
#SBATCH --account=YOUR_ACCOUNT        # edit
#SBATCH --partition=YOUR_GPU_PARTITION # edit (one node, 4 GPUs)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=360G
#SBATCH --time=01:30:00
#SBATCH --job-name=smoke
#SBATCH --output=logs/%x_%j.out
# Plumbing check for a new benchmark: 2 DGPO steps, 32 envs (one group of 8 per rank).
source jobs/_common.sh
source jobs/_ray_start.sh
echo "config: ${CONFIG}  overrides: $*"
cd $REPO_PATH
python $EMBODIED_PATH/train_embodied_agent.py \
    --config-path $EMBODIED_PATH/config/ \
    --config-name ${CONFIG} \
    runner.logger.log_path=$LOG_DIR \
    runner.max_epochs=2 runner.save_interval=-1 runner.val_check_interval=-1 \
    env.train.total_num_envs=32 env.train.rollout_epoch=1 \
    actor.model.model_path=$CKPT rollout.model.model_path=$CKPT "$@"
rc=$?; echo "---- rc=$rc"; exit $rc
