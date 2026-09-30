# Site configuration: the only file with machine-specific paths. Every job
# sources it through _common.sh. Edit the three roots below for your cluster.

# Root for everything large: venv, checkpoints, assets, run outputs.
export L=${PIRL_DATA:-/path/to/large/storage}
# Python env with RLinf + openpi + LIBERO + ManiSkill (see README "Setup").
export RLINF_VENV=${RLINF_VENV:-$L/rlinf_venv}
# Run outputs (tensorboard, checkpoints) go to $CKPT_ROOT/pirl_dgpo/<jobid>.
export CKPT_ROOT=${CKPT_ROOT:-$L/checkpoints}

export HF_HOME=${HF_HOME:-$L/hf_cache}
export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}   # everything is pre-downloaded by scripts/fetch_assets.sh
# LIBERO asks an interactive question on first import unless this config exists
# (scripts/fetch_assets.sh writes it).
export LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH:-$L/libero_config}
# ManiSkill: RLinf scene assets (scripts/fetch_assets.sh maniskill). ManiSkill's
# own assets default to ~/.maniskill; set MS_ASSET_DIR to move them.
export MANISKILL_ASSET_DIR=${MANISKILL_ASSET_DIR:-$L/maniskill_assets}
export PYTHONUNBUFFERED=1
mkdir -p $CKPT_ROOT
