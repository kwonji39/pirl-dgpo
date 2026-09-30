# Shared job preamble: environment, checkpoint, preflight. Sourced by every sbatch.
# Inputs (env vars): CONFIG (RLinf config name), CKPT (SFT checkpoint dir),
# EXTRA_PYTHONPATH (side installs), MS_ASSET_DIR / MANISKILL_ASSET_DIR (ManiSkill).
set -uo pipefail
W=$SLURM_SUBMIT_DIR; cd $W
source $W/jobs/env.sh

CKPT=${CKPT:-$L/checkpoints/RLinf-Pi05-LIBERO-SFT}
[ -f "$CKPT/model.safetensors" ] || { echo "FATAL: no SFT checkpoint at $CKPT"; exit 1; }
[ -f "$RLINF_VENV/bin/activate" ] || { echo "FATAL: no venv at $RLINF_VENV"; exit 1; }
source $RLINF_VENV/bin/activate

export REPO_PATH=$W/RLinf            # created by scripts/setup_rlinf.sh
export EMBODIED_PATH=$REPO_PATH/examples/embodiment
export PYTHONPATH=${EXTRA_PYTHONPATH:+$EXTRA_PYTHONPATH:}$REPO_PATH:${PYTHONPATH:-}
export ROBOT_PLATFORM=LIBERO LIBERO_TYPE=standard
# osmesa, not egl: egl SIGABRTs under GPU contention on our nodes.
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export TOKENIZERS_PARALLELISM=false
export USE_TF=0 TRANSFORMERS_NO_TF=1 USE_TORCH=1   # torch path only
# Compute nodes may reject GitHub/HF TLS with the system CA store.
export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")
[ -n "${MS_ASSET_DIR:-}" ] && export MS_ASSET_DIR

LOG_DIR=$CKPT_ROOT/pirl_dgpo/${SLURM_JOB_ID}; mkdir -p $LOG_DIR

echo "################ preflight ################"
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not available"
print("torch", torch.__version__, "| GPUs", torch.cuda.device_count())
import mujoco
m = mujoco.MjModel.from_xml_string(
    "<mujoco><visual><global offwidth='64' offheight='64'/></visual><worldbody>"
    "<light pos='0 0 3'/><camera name='c' pos='0 -1 0.5' xyaxes='1 0 0 0 0.5 1'/>"
    "<geom type='box' size='.1 .1 .1' rgba='1 0 0 1'/></worldbody></mujoco>")
d = mujoco.MjData(m); mujoco.mj_forward(m, d)   # needed before rendering
r = mujoco.Renderer(m, 64, 64); r.update_scene(d, camera="c")
assert r.render().std() > 1.0, "blank frame: GL backend broken"
from rlinf.workers.actor.fsdp_dgpo_policy_worker import EmbodiedDGPOFSDPPolicy  # noqa
print("preflight ok: CUDA, osmesa rendering, RLinf + DGPO worker import")
PY
[ $? -eq 0 ] || { echo "FATAL: preflight failed"; exit 1; }
