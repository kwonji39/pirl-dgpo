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

"""DGPO math: the micro-batch surrogate must carry the group objective's gradient."""

import pytest
import torch

from rlinf.algorithms.dgpo import dgpo_exact_loss, dgpo_group_coefficients, dsm_loss
from rlinf.workers.actor.fsdp_dgpo_policy_worker import _gather_chunks


def _grpo_adv(rewards, G):
    r = rewards.view(-1, G)
    return ((r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-6)).view(-1)


def _setup(T=12, G=4, n_groups=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    B = G * n_groups
    lengths = torch.randint(3, T + 1, (B,), generator=g)
    valid = torch.arange(T)[:, None] < lengths[None, :]
    rewards = (torch.rand(B, generator=g) > 0.5).float()
    rewards[:G] = torch.tensor([1.0, 0, 0, 0])[:G]  # at least one mixed group
    adv = _grpo_adv(rewards.double(), G)  # float32 leaves sum(A) ~ 1e-7 off zero
    l_ref = torch.rand(T, B, generator=g, dtype=torch.float64)
    l_theta = l_ref + 0.1 * torch.randn(T, B, generator=g, dtype=torch.float64)
    return l_theta, l_ref, adv, valid, G


@pytest.mark.parametrize("reduction", ["sum", "mean"])
@pytest.mark.parametrize("beta", [1.0, 100.0])
def test_surrogate_gradient_matches_exact(reduction, beta):
    l_theta, l_ref, adv, valid, G = _setup()
    T, B = l_theta.shape
    x = l_theta.clone().requires_grad_(True)
    n_groups = B // G
    exact = dgpo_exact_loss(x, l_ref, adv, valid, G, beta, reduction)
    (g_exact,) = torch.autograd.grad(exact, x)

    stats = dgpo_group_coefficients((l_theta - l_ref), adv, valid, G, beta, reduction)
    y = l_theta.clone().requires_grad_(True)
    (stats.coef * y).sum().backward()
    # surrogate drops the constant beta / (G * T) and the 1/n_groups of the mean
    scale = beta / (G * T) / n_groups
    torch.testing.assert_close(y.grad * scale, g_exact)


def test_partition_function_cancels():
    """Adding beta*log Z(c) to every trajectory of a group must not move z_g."""
    l_theta, l_ref, adv, valid, G = _setup()
    T, B = l_theta.shape
    base = dgpo_group_coefficients(l_theta - l_ref, adv, valid, G, 10.0)
    # a per-group shift of every trajectory's implicit reward ell_i by c_g:
    # under "sum" ell_i = sum_k delta_ik / T, so put T*c_g on one valid chunk
    shift = torch.zeros(T, B, dtype=l_theta.dtype)
    c = torch.randn(B // G, dtype=l_theta.dtype).repeat_interleave(G)
    shift[0] = T * c
    shifted = dgpo_group_coefficients(l_theta - l_ref + shift, adv, valid, G, 10.0)
    torch.testing.assert_close(shifted.z, base.z)


def test_zero_start_gate_is_half_and_matches_weighted_regression():
    """At theta == ref every gate is 0.5: DGPO starts as advantage-weighted DSM."""
    _, l_ref, adv, valid, G = _setup()
    stats = dgpo_group_coefficients(torch.zeros_like(l_ref), adv, valid, G, 100.0)
    torch.testing.assert_close(stats.gate, torch.full_like(stats.gate, 0.5))
    expected = 0.5 * adv[None, :] * valid  # r*T == 1 under "sum"
    torch.testing.assert_close(stats.coef, expected.to(stats.coef.dtype))


def test_invalid_and_filtered_chunks_get_no_gradient():
    l_theta, l_ref, adv, valid, G = _setup()
    adv[G : 2 * G] = 0.0  # a filtered (all-same-reward) group
    stats = dgpo_group_coefficients(l_theta - l_ref, adv, valid, G, 100.0)
    assert (stats.coef[~valid] == 0).all()
    assert (stats.coef[:, G : 2 * G] == 0).all()


def test_preferring_positives_saturates_gate():
    """If theta already fits positives better and negatives worse, z > 0 and gate -> 0."""
    _, l_ref, adv, valid, G = _setup()
    delta = -adv[None, :].expand_as(l_ref).clone() * valid  # L_theta < L_ref iff A > 0
    stats = dgpo_group_coefficients(delta, adv, valid, G, 1e3)
    mixed = adv.view(-1, G).abs().sum(1) > 0
    assert (stats.z[mixed] > 0).all()
    assert (stats.gate[mixed] < 1e-3).all()


def test_dsm_loss_slices_executed_part():
    v = torch.zeros(2, 5, 32)
    tgt = torch.zeros(2, 5, 32)
    tgt[:, :, 7:] = 100.0  # padded dims must not count
    tgt[0, :, :7] = 1.0
    out = dsm_loss(v, tgt, action_chunk=5, action_dim=7)
    torch.testing.assert_close(out, torch.tensor([1.0, 0.0]))


def test_gather_chunks_nested():
    fi = {
        "a": torch.arange(24).view(4, 6),
        "sub": {"b": torch.arange(48).view(4, 6, 2)},
    }
    t_idx = torch.tensor([0, 3, 2])
    b_idx = torch.tensor([5, 0, 1])
    out = _gather_chunks(fi, t_idx, b_idx)
    assert out["a"].tolist() == [5, 18, 13]
    assert out["sub"]["b"][:, 0].tolist() == [10, 36, 26]


def test_pooled_metrics_ignore_inactive_ranks():
    """Rank A: one group with gate 0.5; rank B: nothing active. Pooled gate is 0.5."""
    from rlinf.workers.actor.fsdp_dgpo_policy_worker import EmbodiedDGPOFSDPPolicy

    rank_a = {"_s/gate": 0.5, "_n/group": 1.0, "_n/chunk": 4.0, "_s/l_theta": 2.0}
    rank_b = {"_s/gate": 0.0, "_n/group": 0.0, "_n/chunk": 0.0, "_s/l_theta": 0.0}
    avg = {k: (rank_a[k] + rank_b[k]) / 2 for k in rank_a}  # what all_reduce AVG gives
    out = EmbodiedDGPOFSDPPolicy._pool_metrics(avg)
    assert out["dgpo/gate"] == pytest.approx(0.5)
    assert out["dgpo/l_theta"] == pytest.approx(0.5)
    assert not any(k.startswith(("_s/", "_n/")) for k in out)


def test_mean_reduction_balances_gradient_mass():
    """Run 54407201 collapsed under "sum": long failures outweigh short successes.

    Under "mean" every trajectory's total coefficient is gate * A_i * T whatever
    its length, so it sums to zero over a group, as it does per image in DGPO.
    """
    T, G = 48, 8
    valid = torch.zeros(T, G, dtype=torch.bool)
    valid[:20, :7] = True  # seven successes stop after 20 chunks
    valid[:, 7] = True  # one failure runs the whole horizon
    rewards = torch.tensor([1.0] * 7 + [0.0], dtype=torch.float64)
    adv = _grpo_adv(rewards, G)
    delta = torch.zeros(T, G, dtype=torch.float64)
    mass = {
        r: dgpo_group_coefficients(delta, adv, valid, G, 1e4, r).coef.sum().item()
        for r in ("sum", "mean")
    }
    assert mass["sum"] < -30  # net ascent on the DSM loss: the collapse
    assert mass["mean"] == pytest.approx(0.0, abs=1e-9)


def test_ref_ema_moves_reference_toward_policy():
    """dgpo_ref_ema_tau: ref <- ref + tau * (theta - ref); None leaves it frozen."""
    from types import SimpleNamespace

    from rlinf.workers.actor.fsdp_dgpo_policy_worker import EmbodiedDGPOFSDPPolicy

    ref = torch.nn.Linear(3, 2)
    torch.nn.init.zeros_(ref.weight)
    theta = {"weight": torch.ones(2, 3), "bias": torch.full((2,), 4.0)}
    worker = SimpleNamespace(
        dgpo_ref_ema_tau=0.25,
        _get_ref_model=lambda: ref,
        get_model_state_dict=lambda cpu_offload, full_state_dict: theta,
    )
    bias0 = ref.bias.detach().clone()
    EmbodiedDGPOFSDPPolicy._update_ref_ema(worker)
    torch.testing.assert_close(ref.weight, torch.full((2, 3), 0.25))
    torch.testing.assert_close(ref.bias, bias0 + 0.25 * (4.0 - bias0))

    worker.dgpo_ref_ema_tau = None
    before = ref.weight.detach().clone()
    EmbodiedDGPOFSDPPolicy._update_ref_ema(worker)
    torch.testing.assert_close(ref.weight, before)


def test_prism_rloo_zero_sum_and_band_separation():
    from rlinf.algorithms.dgpo import prism_rloo_advantages

    success = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1.0])
    quality = torch.rand(16, generator=torch.Generator().manual_seed(0))
    adv = prism_rloo_advantages(success, quality, 8, lam=0.5)
    torch.testing.assert_close(adv.view(2, 8).sum(1), torch.zeros(2))
    # every success outranks every failure in a mixed group
    assert adv[:4].min() > adv[4:8].max()
    # an all-success group still has signal, at the scale of lam * delta_q
    assert adv[8:].abs().max() > 0
    assert adv[8:].abs().max() <= 0.5 * 8 / 7 + 1e-6


def test_trajectory_jerk_masks_and_measures_curvature():
    from rlinf.algorithms.dgpo import trajectory_jerk

    T, B, c, d = 4, 2, 5, 7
    t = torch.arange(T * c, dtype=torch.float32)
    smooth = t[:, None].expand(-1, d) * 0.1  # linear in time -> zero jerk
    wiggly = smooth.clone()
    wiggly[::2, :6] += 1.0  # alternating offset -> large jerk
    acts = torch.stack([smooth, wiggly], dim=1).view(T, c, B, d)
    acts = acts.permute(0, 2, 1, 3).reshape(T, B, c * d)
    valid = torch.ones(T, B, dtype=torch.bool)
    j = trajectory_jerk(acts, valid, c)
    assert j[0].item() == pytest.approx(0.0, abs=1e-6)
    assert j[1].item() > 1.0
    # garbage after the episode ends must not count
    acts2 = acts.clone()
    acts2[2:, 0] = 1e3
    valid2 = valid.clone()
    valid2[2:, 0] = False
    assert trajectory_jerk(acts2, valid2, c)[0].item() == pytest.approx(0.0, abs=1e-6)


def _load_official_dgpo_loss():
    """compute_group_dgpo_loss_allreduce, read verbatim from the official repo.

    Luo-Yihong/DGPO ships no license file, so its source is not copied here; the
    test executes it from a local clone (DGPO_OFFICIAL_SRC) and skips without one.
    torch.distributed is replaced by a single-process stand-in.
    """
    import ast
    import os
    import types

    # e.g. DGPO_OFFICIAL_SRC=/path/to/DGPO/scripts/train_sd3_dgpo.py
    path = os.environ.get("DGPO_OFFICIAL_SRC", "")
    if not path or not os.path.exists(path):
        pytest.skip(f"official DGPO source not found at {path}")
    src = open(path).read()
    fn = next(
        n
        for n in ast.parse(src).body
        if isinstance(n, ast.FunctionDef)
        and n.name == "compute_group_dgpo_loss_allreduce"
    )
    fake_dist = types.SimpleNamespace(
        all_reduce=lambda t, op=None: t, ReduceOp=types.SimpleNamespace(SUM=None)
    )
    ns = {"torch": torch, "dist": fake_dist}
    exec(compile(ast.Module([fn], []), path, "exec"), ns)
    return ns["compute_group_dgpo_loss_allreduce"]


@pytest.mark.parametrize("beta", [1.0, 100.0, 1e4])
def test_matches_official_dgpo_gradient(beta):
    """With one chunk per trajectory, our gradient is the official DGPO gradient.

    The official loss is mean_i(w_g * A_i * L_theta_i) with w_g =
    sigmoid(beta/G * sum_g A * (L_theta - L_ref)); our surrogate is
    sum_i coef_i * L_theta_i. They must agree up to the 1/B of the mean.
    """
    official = _load_official_dgpo_loss()
    g = torch.Generator().manual_seed(1)
    G, n_groups, D = 8, 3, 35
    B = G * n_groups
    target = torch.randn(B, D, generator=g, dtype=torch.float64)
    ref_v = target + 0.3 * torch.randn(B, D, generator=g, dtype=torch.float64)
    rewards = (torch.rand(B, generator=g) > 0.5).double()
    rewards[:G] = torch.tensor([1.0, 0, 0, 0, 1, 0, 0, 0])
    adv = _grpo_adv(rewards, G)
    info = {"local_group_indices": torch.arange(B) // G, "num_groups": n_groups}

    v_off = ref_v + 0.05 * torch.randn(B, D, generator=g, dtype=torch.float64)
    v_off.requires_grad_(True)
    loss = official(v_off, ref_v, target, adv, info, None, beta, group_size=G)
    (grad_off,) = torch.autograd.grad(loss, v_off)

    v_ours = v_off.detach().clone().requires_grad_(True)
    l_th = dsm_loss(v_ours[:, None, :], target[:, None, :], 1, D)
    l_ref = dsm_loss(ref_v[:, None, :], target[:, None, :], 1, D)
    valid = torch.ones(1, B, dtype=torch.bool)
    stats = dgpo_group_coefficients(
        (l_th - l_ref).detach()[None, :].double(), adv, valid, G, beta, "sum"
    )
    (stats.coef[0] * l_th).sum().backward()
    # our dsm_loss computes in float32, the official one in the input dtype
    torch.testing.assert_close(v_ours.grad / B, grad_off, rtol=1e-4, atol=1e-9)
