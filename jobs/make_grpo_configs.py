"""πRL GRPO (Flow-SDE) configs for ManiSkill / MetaWorld / CALVIN, the control
arm for the DGPO configs from make_dgpo_configs.py.

Starting from πRL's PPO configs, only the algorithm (GRPO groups of 8, critic-free
advantage, PPO-clip actor loss, no value head) and the sampler (Flow-SDE at one
random denoise step, noise 0.5, as in libero_spatial_grpo_openpi_pi05) change.
Env counts and rollout epochs are set at submit time to match each DGPO run.
"""
import re
import sys
from pathlib import Path

C = Path(sys.argv[1])
SRC = {
    "maniskill_grpo_openpi_pi05": "maniskill_ppo_openpi_pi05.yaml",
    "metaworld_50_grpo_openpi_pi05": "metaworld_50_ppo_openpi_pi05.yaml",
    "calvin_abc_d_grpo_openpi_pi05": "calvin_abc_d_ppo_openpi_pi05.yaml",
}


def sub1(pat, rep, s):
    s2, n = re.subn(pat, rep, s, count=1, flags=re.M)
    assert n == 1, f"pattern not found: {pat}"
    return s2


for name, src in SRC.items():
    s = (C / src).read_text()
    s = sub1(r'experiment_name: "[^"]*"', f'experiment_name: "{name}"', s)
    s = sub1(r"^  save_interval: .*$", "  save_interval: 10", s)
    s = sub1(r"^  group_size: 1\b.*$", "  group_size: 8", s)
    s = sub1(r"^  update_epoch: .*$", "  update_epoch: 1          # πRL GRPO (libero_spatial_grpo_openpi_pi05)", s)
    s = sub1(r"^  adv_type: gae.*$", "  adv_type: grpo", s)
    s = sub1(r"^  loss_type: actor_critic.*$", "  loss_type: actor", s)
    s = sub1(r"^  filter_rewards: .*$", "  filter_rewards: True", s)
    s = sub1(r"^  rewards_lower_bound: .*$", "  rewards_lower_bound: 0.1", s)
    s = sub1(r"^  rewards_upper_bound: .*$", "  rewards_upper_bound: 0.9", s)
    s = re.sub(r"^  entropy_bonus: .*$", "  entropy_bonus: 0", s, flags=re.M)
    tr = s.index("\n  train:", s.index("\nenv:"))
    head, tail = s[:tr], s[tr:]
    end = tail.index("\n  eval:")
    block = tail[:end]
    for k in ("auto_reset", "ignore_terminations", "group_size"):
        block = re.sub(rf"\n    {k}: .*", "", block)
    block = block.replace(
        "\n  train:",
        "\n  train:\n    auto_reset: False            # one episode per trajectory, as in the DGPO runs\n"
        "    ignore_terminations: False\n    group_size: ${algorithm.group_size}",
        1,
    )
    s = head + block + tail[end:]
    if "add_value_head: True" in s:
        s = sub1(r"^    add_value_head: True.*$", "    add_value_head: False", s)
    for k in ("noise_method", "noise_level", "joint_logprob", "noise_params", "is_nft"):
        s = re.sub(rf"\n      {k}: .*", "", s)
    s = sub1(
        r"^    openpi:\n",
        '    openpi:\n      noise_method: "flow_sde"   # πRL Flow-SDE: noise at one random denoise step\n'
        "      noise_level: 0.5\n      joint_logprob: False\n",
        s,
    )
    (C / f"{name}.yaml").write_text(s)
    print("wrote", name)
