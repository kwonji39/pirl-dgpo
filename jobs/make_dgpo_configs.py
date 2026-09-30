"""Turn piRL's PPO configs for ManiSkill / MetaWorld / CALVIN into DGPO configs.

Only the algorithm, the sampler and the training env's reset behaviour
change; model, env, rollout and optimiser settings stay piRL's. The DGPO block
is copied verbatim from libero_spatial_dgpo_openpi_pi05.yaml, so every
benchmark runs the same DGPO. Prism is switched on at submit time with
overrides (algorithm.filter_rewards=False +algorithm.dgpo_reward=prism).
"""
import re
import sys
from pathlib import Path

C = Path(sys.argv[1])
SRC = {
    "maniskill_dgpo_openpi_pi05": "maniskill_ppo_openpi_pi05.yaml",
    "metaworld_50_dgpo_openpi_pi05": "metaworld_50_ppo_openpi_pi05.yaml",
    "calvin_abc_d_dgpo_openpi_pi05": "calvin_abc_d_ppo_openpi_pi05.yaml",
}
lib = (C / "libero_spatial_dgpo_openpi_pi05.yaml").read_text()
DGPO = lib[lib.index("  # DGPO (arXiv:2510.08425)") : lib.index("  nft_tau:")]
DGPO += lib[lib.index("  nft_tau:") :].split("\n", 1)[0] + "\n"


def sub1(pat, rep, s, flags=re.M):
    s2, n = re.subn(pat, rep, s, count=1, flags=flags)
    assert n == 1, f"pattern not found: {pat}"
    return s2


for name, src in SRC.items():
    s = (C / src).read_text()
    s = sub1(r'experiment_name: "[^"]*"', f'experiment_name: "{name}"', s)
    s = sub1(r"^  save_interval: .*$", "  save_interval: 10", s)
    # algorithm: GRPO-style groups and advantages, DGPO loss, no critic
    s = sub1(r"^  group_size: 1\b.*$", "  group_size: 8", s)
    s = sub1(r"^  update_epoch: .*$", "  update_epoch: 2          # best DGPO setting on LIBERO-Spatial", s)
    s = sub1(r"^  adv_type: gae.*$", "  adv_type: grpo", s)
    s = sub1(r"^  loss_type: actor_critic.*$", "  loss_type: embodied_dgpo", s)
    s = sub1(r"^  filter_rewards: .*$", "  filter_rewards: True", s)
    s = sub1(r"^  rewards_lower_bound: .*$", "  rewards_lower_bound: 0.1", s)
    s = sub1(r"^  rewards_upper_bound: .*$", "  rewards_upper_bound: 0.9\n\n" + DGPO.rstrip("\n"), s)
    # the training env must end episodes (per-trajectory masks) and share
    # one reset state per group
    tr = s.index("\n  train:", s.index("\nenv:"))
    head, tail = s[:tr], s[tr:]
    end = tail.index("\n  eval:")
    block = tail[:end]
    block = re.sub(r"\n    auto_reset: .*", "", block)
    block = re.sub(r"\n    ignore_terminations: .*", "", block)
    block = re.sub(r"\n    group_size: .*", "", block)
    block = block.replace(
        "\n  train:",
        "\n  train:\n    auto_reset: False            # DGPO needs per-trajectory masks\n"
        "    ignore_terminations: False\n    group_size: ${algorithm.group_size}",
        1,
    )
    s = head + block + tail[end:]
    # actor: no value head, SDE rollouts that store the clean chunk
    s = sub1(r"^    add_value_head: True.*$", "    add_value_head: False", s) if "add_value_head: True" in s else s
    s = re.sub(r"\n      noise_method: .*", "", s)
    s = re.sub(r"\n      noise_level: .*", "", s)
    s = re.sub(r"\n      joint_logprob: .*", "", s)
    s = re.sub(r"\n      noise_params: .*", "", s)
    s = sub1(
        r"^    openpi:\n",
        "    openpi:\n      is_nft: True             # stores the clean chunk nft_x0 that DGPO trains on\n"
        '      noise_method: "flow_sde"   # DGPO w/ SDE: ODE rollouts gave no signal on LIBERO\n'
        "      noise_level: 0.3           # 0.5/sqrt(3): noise on all 3-4 steps ~ piRL's one step at 0.5\n"
        "      joint_logprob: False\n",
        s,
    )
    s = sub1(r"^  micro_batch_size: .*$", "  micro_batch_size: 128", s)
    (C / f"{name}.yaml").write_text(s)
    print("wrote", name)
