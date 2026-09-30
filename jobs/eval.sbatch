#!/bin/bash
#SBATCH --account=YOUR_ACCOUNT        # edit
#SBATCH --partition=YOUR_GPU_PARTITION # edit (one node, 4 GPUs)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=360G
#SBATCH --time=05:00:00
#SBATCH --job-name=eval
#SBATCH --output=logs/%x_%j.out
# Held-out, noise-free eval of SFT and/or checkpoints, one after another.
#   sbatch jobs/eval.sbatch sft /path/to/global_step_20/actor/model_state_dict/full_weights.pt
# EVAL_CONFIG_DIR / EVAL_CONFIG pick the eval config (default: LIBERO-Spatial);
# EVAL_ENVS x EVAL_EPOCHS episodes (default 128 x 4 = 512); EVAL_EXTRA = more overrides.
source jobs/_common.sh
source jobs/_ray_start.sh
cd $REPO_PATH
for ck in "$@"; do
    echo "################ EVAL $ck ################"
    extra=(); [ "$ck" != sft ] && extra=(runner.ckpt_path=$ck)
    python $REPO_PATH/evaluations/eval_embodied_agent.py \
        --config-path ${EVAL_CONFIG_DIR:-$REPO_PATH/evaluations/libero} \
        --config-name ${EVAL_CONFIG:-libero_spatial_openpi_pi05_eval} \
        runner.logger.log_path=$LOG_DIR/$(echo $ck | md5sum | cut -c1-8) \
        rollout.model.model_path=$CKPT \
        env.eval.total_num_envs=${EVAL_ENVS:-128} env.eval.rollout_epoch=${EVAL_EPOCHS:-4} \
        env.eval.video_cfg.save_video=False ${EVAL_EXTRA:-} "${extra[@]}"
    echo "---- EVAL_DONE rc=$? ckpt=$ck"
done
