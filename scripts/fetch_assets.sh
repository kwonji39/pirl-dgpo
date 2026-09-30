#!/bin/bash
# Download the SFT checkpoints and simulator assets into $L (jobs/env.sh).
# Needs internet; compute nodes on some clusters have none, so run this on a
# login/transfer node.
#   bash scripts/fetch_assets.sh [libero] [maniskill] [metaworld] [calvin]
set -euo pipefail
cd "$(dirname "$0")/.."; source jobs/env.sh
export HF_HUB_OFFLINE=0
mkdir -p $L/checkpoints

hf_get() {  # repo -> $L/checkpoints/<name>
    huggingface-cli download "RLinf/$1" --local-dir "$L/checkpoints/$1"
}

# LIBERO asks for its dataset root interactively on first import unless this exists.
mkdir -p $LIBERO_CONFIG_PATH
[ -f $LIBERO_CONFIG_PATH/config.yaml ] || python - <<'PY'
import os, importlib.util
spec = importlib.util.find_spec("libero")
if spec is None:
    raise SystemExit(0)                    # LIBERO not in this venv; skip
root = os.path.join(os.path.dirname(spec.origin), "libero")
cfg = {k: os.path.join(root, v) for k, v in dict(
    benchmark_root="", bddl_files="bddl_files", init_states="init_files",
    datasets="../datasets", assets="assets").items()}
with open(os.path.join(os.environ["LIBERO_CONFIG_PATH"], "config.yaml"), "w") as f:
    f.writelines(f"{k}: {v}\n" for k, v in cfg.items())
PY

# PaliGemma tokenizer used by openpi (~/.cache/openpi), needed by every benchmark
bash RLinf/requirements/embodied/download_assets.sh --assets openpi

for b in "${@:-libero}"; do
  case $b in
    libero)    hf_get RLinf-Pi05-LIBERO-SFT ;;
    metaworld) hf_get RLinf-Pi05-MetaWorld-SFT ;;
    calvin)    hf_get RLinf-Pi05-CALVIN-ABC-D-SFT ;;
    maniskill)
        hf_get RLinf-Pi05-ManiSkill-25Main-SFT
        # scene assets used by RLinf's ManiSkill tasks
        huggingface-cli download RLinf/maniskill_assets --repo-type dataset \
            --local-dir $L/maniskill_assets
        # bridge/WidowX assets (~/.maniskill) and the PhysX GPU library (~/.sapien/physx)
        bash RLinf/requirements/embodied/download_assets.sh --assets maniskill
        ;;
    *) echo "unknown benchmark $b"; exit 1 ;;
  esac
done
