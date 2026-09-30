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

"""DGPO actor for flow VLAs (see ``rlinf/algorithms/dgpo.py`` for the math).

Rollouts are the NFT ones (``openpi.is_nft=True``, ``noise_method=flow_ode``),
which store the clean chunk ``nft_x0`` the sampler produced. Training never
touches the sampling chain: each valid chunk is re-noised by the forward
process, x_t = (1-t) x0 + t eps with t ~ U[t_min, t_max], and scored by the
velocity DSM loss of the policy and of a frozen reference (the SFT model).

One optimizer step, for a group-aligned minibatch:
  1. no-grad forward of theta on every active chunk -> delta = L_theta - L_ref
  2. per-group logit z_g -> detached coefficient sigmoid(-z_g) * A_i
  3. micro-batched backward of sum(coef * L_theta), which has exactly the
     gradient of the group objective (the sigmoid does not decompose, its
     gradient does).
Groups never straddle ranks here (RLinf's GRPO advantages already assume it),
so z_g needs no collective; only the micro-batch counts are aligned across
ranks so the FSDP gradient sync happens on the same call everywhere.
"""

import math
import os

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from rlinf.algorithms.dgpo import (
    dgpo_group_coefficients,
    dsm_loss,
    prism_rloo_advantages,
    trajectory_jerk,
)
from rlinf.models import get_model
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.scheduler.worker.worker import Worker
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import append_to_dict
from rlinf.utils.nested_dict_process import put_tensor_device
from rlinf.utils.utils import clear_memory
from rlinf.workers.actor.fsdp_nft_policy_worker import EmbodiedNFTFSDPPolicy


def _gather_chunks(nested: dict, t_idx: torch.Tensor, b_idx: torch.Tensor) -> dict:
    """Pick chunks (t_idx[n], b_idx[n]) out of [T, B, ...] tensors."""
    out = {}
    for key, value in nested.items():
        if isinstance(value, torch.Tensor):
            out[key] = value[t_idx, b_idx]
        elif isinstance(value, dict):
            out[key] = _gather_chunks(value, t_idx, b_idx)
        else:
            out[key] = value
    return out


def _trim_time(nested: dict, T: int) -> dict:
    out = {}
    for key, value in nested.items():
        if isinstance(value, torch.Tensor):
            assert value.shape[0] in (T, T + 1), (
                f"forward_inputs[{key!r}] has time dim {value.shape[0]}, expected {T}"
            )
            out[key] = value[:T]
        elif isinstance(value, dict):
            out[key] = _trim_time(value, T)
        else:
            out[key] = value
    return out


class EmbodiedDGPOFSDPPolicy(EmbodiedNFTFSDPPolicy):
    """Embodied FSDP policy worker for DGPO.

    Inherits the NFT worker only for its rollout-policy EMA (``nft_tau``),
    which is DGPO's theta^- (identity for a while, then EMA).
    """

    def init_worker(self) -> None:
        super().init_worker()
        self._ref_model = None
        cfg = self.cfg.algorithm
        self.dgpo_beta = float(cfg.get("dgpo_beta", 100.0))
        self.dgpo_t_min = float(cfg.get("dgpo_t_min", 0.3))
        self.dgpo_t_max = float(cfg.get("dgpo_t_max", 1.0))
        self.dgpo_share_t = bool(cfg.get("dgpo_share_t_in_group", True))
        self.dgpo_traj_reduction = str(cfg.get("dgpo_traj_reduction", "sum"))
        self.dgpo_adv_clip = float(cfg.get("dgpo_adv_clip", 5.0))
        self.dgpo_kl_beta = float(cfg.get("dgpo_kl_beta", 0.0))
        self.dgpo_num_minibatches = int(cfg.get("dgpo_num_minibatches", 1))
        self.dgpo_loss_dims = str(cfg.get("dgpo_loss_dims", "executed"))
        self.dgpo_ref_offload = bool(cfg.get("dgpo_ref_offload", True))
        # Stage 2: "prism" replaces the GRPO advantage with Prism-GRPO's
        # R = success + lambda * q and its RLOO estimator. Needs
        # algorithm.filter_rewards=False, or same-outcome groups are masked out
        # before they reach this worker.
        self.dgpo_reward = str(cfg.get("dgpo_reward", "binary"))
        self.prism_lambda = float(cfg.get("prism_lambda", 0.5))
        # Continuous action dims scored for smoothness. Default: all but the
        # last (the gripper): 6 for LIBERO / ManiSkill / CALVIN, 3 for MetaWorld.
        jd = cfg.get("prism_jerk_dims", None)
        self.prism_jerk_dims = (
            int(jd) if jd is not None else self.model.config.action_env_dim - 1
        )
        if self.dgpo_reward == "prism":
            assert not cfg.get("filter_rewards", False), (
                "dgpo_reward=prism needs algorithm.filter_rewards=False"
            )
        # Dynamic reference (the official code's use_ema_ref, from TDM-R1):
        # after each update, ref <- ref + tau * (theta - ref). None keeps the
        # frozen SFT reference. With a frozen reference the gate closes once
        # the policy prefers successes more than SFT did (run 54420584: gate
        # 0.5 -> 0.2, eval flat at 0.838 / 0.840 at steps 10 / 20); a
        # trailing reference measures progress against the recent policy.
        tau = cfg.get("dgpo_ref_ema_tau", None)
        self.dgpo_ref_ema_tau = float(tau) if tau is not None else None
        # The surrogate's scale is arbitrary (beta/(G*T) is already dropped).
        # Under Adam it only matters against eps and grad clipping: DSM losses
        # on the policy's own ODE samples are ~1e-3, giving grad norms ~5e-3.
        self.dgpo_loss_scale = float(cfg.get("dgpo_loss_scale", 100.0))
        assert 0.0 <= self.dgpo_t_min < self.dgpo_t_max <= 1.0
        assert self.model.config.is_nft, (
            "DGPO needs the clean rollout chunk; set actor.model.openpi.is_nft=True"
        )

    # ------------------------------------------------------------------ ref

    def _get_ref_model(self):
        """Frozen copy of the initial (SFT) policy, built from model_path.

        Built with get_model rather than model_provider_func: the latter would
        materialise it on the GPU first and apply runner.ckpt_path, and the
        reference must stay the SFT policy.
        """
        if self._ref_model is None:
            ref_cfg = OmegaConf.create(
                OmegaConf.to_container(self.cfg.actor.model, resolve=True)
            )
            ref_cfg.load_to_device = False
            ref = get_model(ref_cfg)
            ref_config = getattr(ref, "config", None)
            if hasattr(ref_config, "compile_transformer_forward"):
                ref_config.compile_transformer_forward = False
            ref.eval()
            ref.requires_grad_(False)
            ref.to("cpu" if self.dgpo_ref_offload else self.device)
            self._ref_model = ref
        return self._ref_model

    @torch.no_grad()
    def _update_ref_ema(self) -> None:
        """Move the reference toward the current policy (no-op if frozen).

        get_model_state_dict gathers under FSDP, so every rank calls this.
        """
        tau = self.dgpo_ref_ema_tau
        if tau is None:
            return
        ref = self._get_ref_model()
        current = self.get_model_state_dict(cpu_offload=True, full_state_dict=True)
        ref_state = ref.state_dict()
        for name, ref_tensor in ref_state.items():
            src = current.get(name)
            if src is None or not ref_tensor.is_floating_point():
                continue
            ref_tensor.lerp_(src.to(ref_tensor.device, ref_tensor.dtype), tau)
        del current

    def _prism_advantages(self, batch, fi, valid, T, G, metrics) -> torch.Tensor:
        """Per-trajectory Prism-GRPO advantages; q = exp(-jerk / batch median)."""
        rewards = batch["rewards"][:T].float().cpu()  # [T, B, num_action_chunks]
        success = ((rewards.sum(-1) * valid).sum(0) > 0.5).float()
        jerk = trajectory_jerk(
            fi["action"].cpu(),
            valid,
            self.model.config.action_chunk,
            dims=self.prism_jerk_dims,
        )
        quality = torch.exp(-jerk / jerk.median().clamp_min(1e-8))
        adv = prism_rloo_advantages(success, quality, G, self.prism_lambda)
        same = success.view(-1, G).std(dim=1) == 0
        append_to_dict(
            metrics,
            {
                "prism/success": success.mean().item(),
                "prism/quality": quality.mean().item(),
                "prism/jerk_median": jerk.median().item(),
                "prism/same_outcome_group_frac": same.float().mean().item(),
            },
        )
        return adv

    def _loss_slice(self) -> tuple[int, int]:
        cfg = self.model.config
        if self.dgpo_loss_dims == "executed":
            return cfg.action_chunk, cfg.action_env_dim
        if self.dgpo_loss_dims == "full":
            # nft_forward already truncates the horizon to action_chunk
            return cfg.action_chunk, cfg.action_dim
        raise ValueError(f"Unsupported dgpo_loss_dims: {self.dgpo_loss_dims}")

    # ------------------------------------------------------------- forward

    def _velocity(self, model, forward_inputs, x0, eps, t):
        tb = t.view(-1, *([1] * (x0.ndim - 1))).to(x0.dtype)
        x_t = (1.0 - tb) * x0 + tb * eps
        with self.amp_context:
            out = model(
                forward_type=ForwardType.NFT,
                forward_inputs=forward_inputs,
                nft_inputs={"x_t": x_t, "timesteps": t.to(x0.dtype)},
                compute_values=False,
            )
        return out["v_theta"], eps - x0

    def _micro(self, fi, x0, eps, t, ids):
        """Slice one micro-batch of chunks and move it to the device."""
        t_idx, b_idx = ids[:, 0], ids[:, 1]
        dev = f"{Worker.torch_device_type}:{int(os.environ['LOCAL_RANK'])}"
        fi_mb = put_tensor_device(_gather_chunks(fi, t_idx, b_idx), dev)
        return (
            fi_mb,
            x0[t_idx, b_idx].to(dev),
            eps[t_idx, b_idx].to(dev),
            t[t_idx, b_idx].to(dev),
        )

    @torch.no_grad()
    def _nograd_losses(self, model, fi, x0, eps, t, ids, want_v=False):
        chunk, dim = self._loss_slice()
        mbs = self.cfg.actor.micro_batch_size
        losses, vs = [], []
        for s in range(0, ids.shape[0], mbs):
            fi_mb, x0_mb, eps_mb, t_mb = self._micro(fi, x0, eps, t, ids[s : s + mbs])
            v, target = self._velocity(model, fi_mb, x0_mb, eps_mb, t_mb)
            losses.append(dsm_loss(v, target, chunk, dim).cpu())
            if want_v:
                vs.append(v[:, :chunk, :dim].float().cpu())
        empty = torch.zeros(0)
        return (
            torch.cat(losses) if losses else empty,
            torch.cat(vs) if (want_v and vs) else None,
        )

    # ------------------------------------------------------------ training

    def _pad_ids(self, ids: torch.Tensor, filler: torch.Tensor):
        """Pad to the same micro-batch count on every rank; returns (ids, n_micro, n_real)."""
        n_real = ids.shape[0]
        n = torch.tensor([n_real], device=self.device)
        if dist.is_initialized():
            dist.all_reduce(n, op=dist.ReduceOp.MAX)
        n_max = int(n.item())
        mbs = self.cfg.actor.micro_batch_size
        n_micro = math.ceil(n_max / mbs) if n_max > 0 else 0
        n_pad = n_micro * mbs - n_real
        if n_pad > 0:
            ids = torch.cat([ids, filler.expand(n_pad, 2)], dim=0)
        return ids, n_micro, n_real

    @Worker.timer("run_training")
    def run_training(self) -> None:
        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
        if self.is_optimizer_offloaded:
            self.load_optimizer(self.device)
        self.model.train()

        batch = self.rollout_batch
        assert batch.get("loss_mask", None) is not None, (
            "DGPO needs per-trajectory masks: env.train.auto_reset=False and "
            "ignore_terminations=False"
        )
        adv_tb = batch["advantages"][..., 0].float().cpu()  # [T, B]
        valid = batch["loss_mask"][..., 0].bool().cpu()  # [T, B]
        T, B = adv_tb.shape
        fi = _trim_time(batch["forward_inputs"], T)
        x0 = fi["nft_x0"].float()  # [T, B, H, D]
        G = int(self.cfg.algorithm.group_size)
        assert B % G == 0, f"per-rank batch {B} not divisible by group_size {G}"
        n_groups = B // G

        # A_i is constant over a trajectory's valid chunks (zero where filtered)
        adv = (adv_tb * valid).sum(0) / valid.sum(0).clamp_min(1)
        metrics = {}
        if self.dgpo_reward == "prism":
            adv = self._prism_advantages(batch, fi, valid, T, G, metrics)
        adv = adv.clamp(-self.dgpo_adv_clip, self.dgpo_adv_clip)
        active = valid & (adv != 0)[None, :]
        group_of_b = torch.arange(B) // G

        # group-aligned minibatches; every rank has the same B so the same split
        n_mb = self.dgpo_num_minibatches
        assert n_groups % n_mb == 0, f"{n_groups} groups not divisible into {n_mb}"
        group_perm = torch.randperm(n_groups)
        mb_groups = group_perm.view(n_mb, -1)

        chunk, dim = self._loss_slice()
        mbs = self.cfg.actor.micro_batch_size
        update_epoch = int(self.cfg.algorithm.get("update_epoch", 1))
        for _ in range(update_epoch):
            # fresh forward-process noise for this pass (DGPO Alg. 1, line 11)
            if self.dgpo_share_t:
                t_g = torch.empty(n_groups).uniform_(self.dgpo_t_min, self.dgpo_t_max)
                t = t_g[group_of_b][None, :].expand(T, B).contiguous()
            else:
                t = torch.empty(T, B).uniform_(self.dgpo_t_min, self.dgpo_t_max)
            eps = torch.randn_like(x0)

            all_ids = active.nonzero()  # [N, 2] (t_idx, b_idx)
            filler = torch.zeros(1, 2, dtype=all_ids.dtype)

            # reference losses once per pass (the reference does not move)
            ref = self._get_ref_model()
            if self.dgpo_ref_offload:
                ref.to(self.device)
            l_ref_flat, v_ref_flat = self._nograd_losses(
                ref, fi, x0, eps, t, all_ids, want_v=self.dgpo_kl_beta > 0
            )
            if self.dgpo_ref_offload:
                ref.to("cpu")
                clear_memory()
            l_ref = torch.zeros(T, B)
            l_ref[all_ids[:, 0], all_ids[:, 1]] = l_ref_flat
            v_ref = None
            if v_ref_flat is not None:
                v_ref = torch.zeros(T, B, chunk, dim)
                v_ref[all_ids[:, 0], all_ids[:, 1]] = v_ref_flat

            for groups in mb_groups:
                in_mb = torch.zeros(B, dtype=torch.bool)
                for g in groups.tolist():
                    in_mb[g * G : (g + 1) * G] = True
                ids = all_ids[in_mb[all_ids[:, 1]]]
                ids, n_micro, n_real = self._pad_ids(ids, filler)
                if n_micro == 0:
                    continue  # nothing to learn from on any rank

                # 1. no-grad theta losses -> group logits (exact for this step)
                l_th_flat, _ = self._nograd_losses(self.model, fi, x0, eps, t, ids)
                real = ids[:n_real]
                delta = torch.zeros(T, B)
                delta[real[:, 0], real[:, 1]] = (
                    l_th_flat[:n_real] - l_ref[real[:, 0], real[:, 1]]
                )
                mb_active = active & in_mb[None, :]
                stats = dgpo_group_coefficients(
                    delta, adv, mb_active, G, self.dgpo_beta, self.dgpo_traj_reduction
                )
                coef = torch.zeros(ids.shape[0])
                coef[:n_real] = stats.coef[real[:, 0], real[:, 1]]

                # 2. surrogate backward
                self.optimizer.zero_grad()
                norm = float(max(n_real, 1))
                for m in range(n_micro):
                    sl = slice(m * mbs, (m + 1) * mbs)
                    fi_mb, x0_mb, eps_mb, t_mb = self._micro(fi, x0, eps, t, ids[sl])
                    backward_ctx = self.before_micro_batch(
                        self.model, is_last_micro_batch=(m + 1) == n_micro
                    )
                    v, target = self._velocity(self.model, fi_mb, x0_mb, eps_mb, t_mb)
                    l_th = dsm_loss(v, target, chunk, dim)
                    c = coef[sl].to(l_th.device)
                    loss = (c * l_th).sum() / norm
                    if v_ref is not None:
                        vr = v_ref[ids[sl, 0], ids[sl, 1]].to(v.device)
                        kl = (
                            (v[:, :chunk, :dim].float() - vr)
                            .square()
                            .flatten(1)
                            .mean(1)
                        )
                        real_mask = (torch.arange(m * mbs, (m + 1) * mbs) < n_real).to(
                            kl.device
                        )
                        loss = loss + self.dgpo_kl_beta * (kl * real_mask).sum() / norm
                    loss = loss * self.dgpo_loss_scale
                    with backward_ctx:
                        self.grad_scaler.scale(loss).backward()
                    del fi_mb, loss, v
                self.torch_platform.empty_cache()
                grad_norm, lr_list = self.optimizer_step()

                # metrics. all_reduce_dict packs values by sorted key, so every
                # rank must emit the same keys even when it had nothing active.
                with torch.no_grad():
                    # Sums and counts, not per-rank means: a rank with no active
                    # group would otherwise pull every average toward 0. The
                    # ratio of the reduced values is the exact pooled mean.
                    g_active = mb_active.view(T, -1, G).any(dim=(0, 2))
                    pos = mb_active & (adv > 0)[None, :]
                    neg = mb_active & (adv < 0)[None, :]
                    zg = stats.z[g_active]
                    append_to_dict(
                        metrics,
                        {
                            "actor/grad_norm": grad_norm,
                            "actor/lr": lr_list[0],
                            "dgpo/active_chunks": float(n_real),
                            "dgpo/active_group_frac": g_active.float().mean().item()
                            * n_mb,
                            "_n/chunk": float(n_real),
                            "_n/pos": pos.sum().item(),
                            "_n/neg": neg.sum().item(),
                            "_n/group": float(zg.numel()),
                            "_s/l_theta": l_th_flat[:n_real].sum().item(),
                            "_s/l_ref": l_ref[mb_active].sum().item(),
                            "_s/delta_pos": delta[pos].sum().item(),
                            "_s/delta_neg": delta[neg].sum().item(),
                            "_s/z": zg.sum().item(),
                            "_s/gate": stats.gate[g_active].sum().item(),
                            "_s/pref_acc": (zg > 0).float().sum().item(),
                        },
                    )

        self.lr_scheduler.step()
        self.soft_update_rollout_model()
        self._update_ref_ema()
        self.optimizer.zero_grad()
        clear_memory()
        mean_metric_dict = {key: np.mean(value) for key, value in metrics.items()}
        mean_metric_dict = all_reduce_dict(
            mean_metric_dict, op=torch.distributed.ReduceOp.AVG
        )
        return self._pool_metrics(mean_metric_dict)

    @staticmethod
    def _pool_metrics(m: dict) -> dict:
        """Turn reduced (sum, count) pairs into pooled means; drop the helpers."""
        denom = {
            "l_theta": "chunk",
            "l_ref": "chunk",
            "delta_pos": "pos",
            "delta_neg": "neg",
            "z": "group",
            "gate": "group",
            "pref_acc": "group",
        }
        out = {k: v for k, v in m.items() if not k.startswith(("_s/", "_n/"))}
        for name, count in denom.items():
            if f"_s/{name}" in m:
                n = m[f"_n/{count}"]
                out[f"dgpo/{name}"] = m[f"_s/{name}"] / n if n > 0 else 0.0
        return out
