"""Standalone eval configs for benchmarks RLinf ships no eval config for.

In eval-only mode RLinf builds the model from cfg.rollout.model, while the
training configs keep it under actor.model. Compose each training config with
Hydra, merge actor.model into rollout.model, and write the resolved result as
a flat config next to it (<name>_eval.yaml). No torch needed.
"""
import os
import sys

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

cfg_dir = sys.argv[1]
os.environ.setdefault("EMBODIED_PATH", os.path.dirname(cfg_dir))
for name in sys.argv[2:]:
    with initialize_config_dir(config_dir=cfg_dir, version_base="1.1"):
        cfg = compose(name, overrides=["actor.model.model_path=/set/at/runtime",
                                       "rollout.model.model_path=/set/at/runtime"])
    OmegaConf.set_struct(cfg, False)
    merged = OmegaConf.merge(cfg.actor.model, cfg.rollout.model)
    cfg.rollout.model = merged
    cfg.runner.only_eval = True
    out = OmegaConf.to_container(cfg, resolve=True)
    path = os.path.join(cfg_dir, f"{name}_eval.yaml")
    OmegaConf.save(OmegaConf.create(out), path)
    print("wrote", path, "| rollout.model.model_type =", out["rollout"]["model"]["model_type"])
