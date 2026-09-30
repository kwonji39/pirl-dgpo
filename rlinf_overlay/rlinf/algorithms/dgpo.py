# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Direct Group Preference Optimization (DGPO, arXiv:2510.08425) for flow VLAs.

DGPO's Eq. (17) for a group G of samples sharing a condition c:

    L = -log sigmoid( -beta * sum_{x in G} A(x) * [L_theta(x) - L_ref(x)] )

where A is the GRPO-normalised advantage (so sum_{G+}|A| = sum_{G-}|A| and the
partition function Z(c) cancels) and L_* is the denoising (DSM) loss on the
forward-noised sample.

In an embodied MDP a "sample" is a trajectory and the shared condition is the
reset state, which RLinf's GRPO groups already share. The trajectory
log-ratio is the sum over its action chunks (the dynamics cancel), so each
trajectory contributes

    ell_i = sum_k valid_ik * (L_theta_ik - L_ref_ik) * r_i

with r_i = 1/T ("sum": a constant, so Z still cancels exactly) or 1/n_i
("mean": length-normalised, heuristic).

The sigmoid wraps a sum over every chunk of every trajectory in the group, so
the loss does not decompose over micro-batches. Its gradient does:

    d L_g / d L_theta_ik = sigmoid(-z_g) * beta * A_i * r_i / G

so we compute z_g with detached losses first and then backprop the surrogate
sum_ik coef_ik * L_theta_ik with coef_ik = sigmoid(-z_g) * A_i * r_i * T. The
constant beta/(G*T) is dropped from the gradient, as in the official code
(``compute_group_dgpo_loss_allreduce``), so beta only sets the sigmoid's
operating point and not the step size.
"""

from dataclasses import dataclass

import torch


@dataclass
class DGPOGroupStats:
    coef: torch.Tensor  # [T, B] per-chunk surrogate coefficient (detached)
    z: torch.Tensor  # [num_groups] group logit, positive = preference satisfied
    gate: torch.Tensor  # [num_groups] sigmoid(-z)
    ell: torch.Tensor  # [B] per-trajectory implicit log-ratio estimate


def dsm_loss(
    v_pred: torch.Tensor, v_target: torch.Tensor, action_chunk: int, action_dim: int
) -> torch.Tensor:
    """Per-sample velocity MSE over the executed slice [:action_chunk, :action_dim]."""
    diff = (
        v_pred[:, :action_chunk, :action_dim].float()
        - v_target[:, :action_chunk, :action_dim].float()
    )
    return diff.square().flatten(1).mean(dim=1)


def dgpo_group_coefficients(
    delta: torch.Tensor,
    adv: torch.Tensor,
    valid: torch.Tensor,
    group_size: int,
    beta: float,
    traj_reduction: str = "sum",
) -> DGPOGroupStats:
    """Group logits and per-chunk surrogate coefficients.

    Args:
        delta: [T, B] detached L_theta - L_ref per chunk (ignored where not valid).
        adv: [B] per-trajectory advantage (zero-sum within each group).
        valid: [T, B] bool, chunks that belong to the trajectory (before done).
        group_size: trajectories per group; groups are contiguous along B.
        beta: DGPO temperature.
        traj_reduction: "sum" (divide by the fixed horizon T) or "mean" (divide
            by each trajectory's own valid length).
    """
    T, B = delta.shape
    assert B % group_size == 0, f"batch {B} not divisible by group_size {group_size}"
    valid_f = valid.to(delta.dtype)
    if traj_reduction == "sum":
        r = torch.full((B,), 1.0 / T, dtype=delta.dtype, device=delta.device)
    elif traj_reduction == "mean":
        r = 1.0 / valid_f.sum(dim=0).clamp_min(1.0)
    else:
        raise ValueError(f"Unsupported dgpo_traj_reduction: {traj_reduction}")

    ell = (delta * valid_f).sum(dim=0) * r  # [B]
    per_traj = adv.to(delta.dtype) * ell  # [B]
    z = -beta * per_traj.view(-1, group_size).sum(dim=1) / group_size  # [num_groups]
    gate = torch.sigmoid(-z)
    gate_b = gate.repeat_interleave(group_size)  # [B]
    coef = (gate_b * adv.to(delta.dtype) * r * T)[None, :] * valid_f  # [T, B]
    return DGPOGroupStats(
        coef=coef.detach(), z=z.detach(), gate=gate.detach(), ell=ell.detach()
    )


def dgpo_exact_loss(
    l_theta: torch.Tensor,
    l_ref: torch.Tensor,
    adv: torch.Tensor,
    valid: torch.Tensor,
    group_size: int,
    beta: float,
    traj_reduction: str = "sum",
) -> torch.Tensor:
    """The undecomposed objective, mean over groups. Used by the tests only."""
    T, B = l_theta.shape
    valid_f = valid.to(l_theta.dtype)
    if traj_reduction == "sum":
        r = torch.full((B,), 1.0 / T, dtype=l_theta.dtype)
    else:
        r = 1.0 / valid_f.sum(dim=0).clamp_min(1.0)
    ell = ((l_theta - l_ref) * valid_f).sum(dim=0) * r
    z = -beta * (adv * ell).view(-1, group_size).sum(dim=1) / group_size
    return torch.nn.functional.softplus(-z).mean()


def trajectory_jerk(
    actions: torch.Tensor, valid: torch.Tensor, num_action_chunks: int, dims: int = 6
) -> torch.Tensor:
    """Mean squared second difference of the executed actions, per trajectory.

    Args:
        actions: [T, B, num_action_chunks * action_dim] executed (env-space)
            actions, one row per chunk step.
        valid: [T, B] bool, chunk steps that belong to the trajectory.
        num_action_chunks: actions executed per chunk step.
        dims: leading action dims to score. LIBERO's first 6 are the
            continuous delta pose; the gripper (dim 6) is a binary switch.

    Returns:
        [B] jerk J >= 0. Trajectories shorter than 3 actions get 0.
    """
    T, B, _ = actions.shape
    a = actions.float().view(T, B, num_action_chunks, -1)[..., :dims]
    a = a.permute(1, 0, 2, 3).reshape(B, T * num_action_chunks, dims)
    m = valid.t().repeat_interleave(num_action_chunks, dim=1).to(a.dtype)  # [B, T*c]
    d2 = a[:, 2:] - 2 * a[:, 1:-1] + a[:, :-2]
    m2 = m[:, 2:] * m[:, 1:-1] * m[:, :-2]
    return (d2.square().sum(-1) * m2).sum(1) / m2.sum(1).clamp_min(1.0)


def prism_rloo_advantages(
    success: torch.Tensor, quality: torch.Tensor, group_size: int, lam: float
) -> torch.Tensor:
    """Prism-GRPO (arXiv:2608.17423): R = success + lam * q, RLOO advantage.

    A_i = R_i - mean_{j != i} R_j, with no division by the group std, so a
    same-outcome group keeps the raw scale of lam * delta_q instead of being
    blown up to unit variance. The advantages sum to zero within each group,
    which is what DGPO needs for Z(c) to cancel.
    """
    assert 0.0 < lam < 1.0, "lam < 1 keeps every success above every failure"
    r = (success.float() + lam * quality.float()).view(-1, group_size)
    loo_mean = (r.sum(dim=1, keepdim=True) - r) / (group_size - 1)
    return (r - loo_mean).view(-1)
