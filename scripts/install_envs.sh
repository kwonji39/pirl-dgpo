#!/bin/bash
# Build one venv per simulator with RLinf's own installer (uv-based). CALVIN
# pins setuptools/hydra/transformers and MetaWorld 3 moves gymnasium, so they
# must not share a venv with LIBERO. Run on a node with internet access.
#   bash scripts/install_envs.sh libero|maniskill|metaworld|calvin
set -euo pipefail
cd "$(dirname "$0")/.."; source jobs/env.sh
case ${1:?benchmark} in
    libero|maniskill) env=maniskill_libero; venv=$L/venvs/rlinf_maniskill_libero ;;
    metaworld)        env=metaworld;        venv=$L/venvs/rlinf_metaworld ;;
    calvin)           env=calvin;           venv=$L/venvs/rlinf_calvin ;;
    *) echo "unknown benchmark $1"; exit 1 ;;
esac
cd RLinf
bash requirements/install.sh embodied --model openpi --env $env --venv $venv
echo "export RLINF_VENV=$venv   # before sbatch, for $1 jobs"
