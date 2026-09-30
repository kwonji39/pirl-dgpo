# Experiment log (raw lab notebook)

This is the running notebook kept during the experiments (2026-09-22 to
2026-09-30), unedited apart from this header. Numbers such as `54460468` are
Slurm job IDs. Script names are the old ones:
`00_dgpo_smoke`/`05_smoke` → `smoke.sbatch`, `01_dgpo_libero_spatial` →
`train_dgpo.sbatch`, `02_pirl_grpo_baseline` → `train_grpo.sbatch`,
`03_eval_ckpts` → `eval.sbatch`. The curated summary is in the top-level README.

---

## pirl-dgpo: DGPO in place of πRL's Flow-SDE GRPO

This directory holds a πRL training setup (RLinf, π0.5 on LIBERO-Spatial) in which
the policy-gradient update is replaced by **DGPO**, *Reinforcing Diffusion Models
by Direct Group Preference Optimization* (arXiv:2510.08425). Everything except
the update rule and the rollout sampler matches the πRL reproduction in
`~/manifold-vla` (`jobs/18_pirl_repro.sbatch`, job 54314933), so the two runs can
be compared step for step.

- `RLinf/`: a clean clone of `manifold-vla/repos/RLinf` at `09a74045`, branch
  `dgpo`. It does **not** include the SO(3) hooks, which stay uncommitted in
  manifold-vla's checkout.
- `jobs/`: `00_dgpo_smoke.sbatch` (2 steps, plumbing only) and
  `01_dgpo_libero_spatial.sbatch` (12 h, the same budget as the πRL run). Both
  reuse `manifold-vla/jobs/env.sh` and the same venv and SFT checkpoint.

## What changes relative to πRL

| | πRL (GRPO + Flow-SDE) | this (DGPO) |
|---|---|---|
| rollout sampler | SDE at one random denoise step (`noise_level` 0.5) | deterministic ODE, 3 steps (same budget) |
| what is stored | the denoising chain + Gaussian log-prob of one transition | the clean chunk `nft_x0` |
| training signal | clipped ratio of transition densities × A | DSM loss on a **fresh forward-noised** chunk, x_t = (1−t)x0 + tε |
| needs a tractable policy density | yes (this is why πRL injects noise) | no |
| reference model | none (`kl_beta: 0`) | frozen SFT policy, required by the implicit reward |

The sampler is the part that makes DGPO attractive here. πRL's Flow-SDE is
Flow-GRPO's trick for making the policy stochastic *enough to have a
log-likelihood*. DGPO never evaluates a likelihood, so rollouts can use the
same ODE sampler that evaluation uses.

## How DGPO maps onto an MDP

DGPO's objective (Eq. 17) is a Bradley–Terry likelihood between the positive
and negative halves of a group of samples that **share a condition c**. The
partition function Z(c) cancels because the |A| weights balance. In an image
model, c is the prompt.

In πRL each action chunk sees a different observation, so no two chunks share
a condition. What the samples in a group *do* share is the **reset state**:
RLinf's GRPO groups are 8 envs reset to the same initial state. The derivation
therefore holds at the trajectory level:

- sample = one trajectory, condition = its reset state (shared, so Z cancels exactly)
- log π_θ(τ)/π_ref(τ) = Σ_k log π_θ(a_k|o_k)/π_ref(a_k|o_k), because the dynamics cancel
- each chunk's log-ratio is replaced by the DSM-loss difference, as in Diffusion-DPO and DGPO

This gives z_g = −β/G · Σ_{i∈g} A_i · (1/T) Σ_k [L_θ − L_ref]_{ik}.

**Status of the theory: a principled trajectory-level extension, not the
original theorem.** Two separate facts are involved:

- *The ratio decomposition holds for any dynamics.* The transition terms
  P(s'|s,a) are the same under π_θ and π_ref, so they cancel.
- *The DPO closed form* π* = π_ref·exp(r(τ)/β)/Z(s_0) at the trajectory level
  needs deterministic dynamics. With stochastic transitions the KL-regularised
  optimum is a soft-Bellman solution instead (the same caveat as token-level
  DPO in "From r to Q*"). LIBERO/MuJoCo is deterministic given the state, so
  it holds here, but it is an assumption to state.

On top of that come DGPO's own approximations: the forward process stands in
for the inversion chain, and the Jensen bound moves the expectation over t
outside the sigmoid.

**Implementation.** The sigmoid spans every chunk of every trajectory in a
group (~8 × 48 forward passes of a 3B model), so the loss cannot be
micro-batched. Its gradient can be. Each optimizer step does:

1. a no-grad pass to get z_g;
2. a backward pass on the surrogate Σ σ(−z_g)·A_i·L_θ.

This gives exactly the gradient of the group objective
(`tests/unit_tests/test_dgpo.py` checks it against autograd). The official
DGPO code does the same thing (`compute_group_dgpo_loss_allreduce`). One
consequence: at θ = ref every gate is 0.5, so DGPO starts out as
advantage-weighted DSM regression. The gate σ(−z_g) then shuts off groups
whose preference is already satisfied, which acts as DGPO's trust region.

## Files

| | |
|---|---|
| `RLinf/rlinf/algorithms/dgpo.py` | the math: group logits, surrogate coefficients, DSM on the executed slice |
| `RLinf/rlinf/workers/actor/fsdp_dgpo_policy_worker.py` | the actor: forward-noising, frozen reference, group-aligned exact steps, cross-rank micro-batch alignment |
| `RLinf/examples/embodiment/config/libero_spatial_dgpo_openpi_pi05.yaml` | the πRL GRPO config with only `algorithm` and the sampler changed |
| `RLinf/tests/unit_tests/test_dgpo.py` | surrogate gradient = exact gradient (sum/mean, two β values); Z cancels; masking; gate saturation |

## Knobs, and why the defaults are what they are

| key | default | source / reason |
|---|---|---|
| `dgpo_beta` | 100 | paper default. It only sets the sigmoid's operating point, since the step size does not depend on β (as in the official code). **First thing to sweep**, because action-chunk DSM gaps are not image-latent DSM gaps |
| `dgpo_t_min` | 0.3 | the paper's timestep clip. Few-step (3) samples make it matter more here than in the paper |
| `update_epoch` | 4 | the paper's M = 4 timesteps per sample. Each pass draws fresh (t, ε) |
| `dgpo_traj_reduction` | `sum` | the exact trajectory log-ratio (÷T, a constant). `mean` normalises by length, which is heuristic because failures run the full 48 chunks and successes stop early |
| `nft_tau` | 1.0 | on-policy θ⁻. The paper uses θ⁻ = θ for 200 steps and then EMA 0.3 (`[1.0, 0.7, 200, 200]`), but a 12 h run is only ~34 steps |
| `noise_method` | `flow_ode` | DGPO's premise. `flow_sde` gives the paper's "DGPO w/ SDE" ablation with no code change |
| `dgpo_loss_dims` | `executed` | first 5 steps × 7 dims, the same slice πRL's log-probs use. The padded dims 7–31 carry no reward information |

## Open questions a run will answer

1. **Is ODE exploration enough?** Diversity now comes only from the initial
   noise of each chunk. If the SFT policy is near-deterministic, most groups
   will be all-success or all-fail, get filtered, and give no signal. Watch
   `dgpo/active_group_frac`. If it is low, use SDE rollouts (the paper's
   "w/ SDE" variant).
2. **β scale.** Watch `dgpo/z`, `dgpo/gate` and `dgpo/pref_acc`. If the gate
   stays at ~0.5 the whole run, β is too small for these Δ magnitudes. If it
   collapses to 0 within a few steps, β is too large.
3. **Speed.** DGPO's headline 20× comes from ODE sampling and from not training
   along the chain. Here the rollout is sim-bound (~135 s per rollout epoch in
   54314933), so expect a smaller gain in wall-clock and compare per env step.

## Smoke run 54402772 (2 steps, 32 envs, 20 min)

rc=0. The end-to-end path works on 4× H100: ODE rollouts, the frozen
reference, group-aligned exact steps, the FSDP sync, and the EMA weight sync.
There was no OOM. A step took 152 s, of which 26 s was training. What it
showed:

- **Δ is tiny at the start.** L_θ ≈ L_ref ≈ 1e-3 and |Δ| ≈ 1e-5, so z ≈ 1e-4 at
  β = 100 and every gate sits at 0.5. This is expected at θ = ref. Whether β
  needs to be 10–100× larger depends on how fast Δ grows, which only a longer
  run shows.
- **Gradients are small.** grad_norm was ≈ 5e-3, against ≈ 3 (clipped to 1) for
  πRL's GRPO. Adam makes this mostly irrelevant, but per-parameter gradients
  were only ~30× above ε, so `dgpo_loss_scale: 100` was added. It is a pure
  constant under Adam.
- The first version's `dgpo/gate`, `z` and `pref_acc` were averaged over
  ranks, and ranks with no active group counted as 0: gate 0.125 was really
  0.5 × 1/4 ranks. They are now pooled as sum/count.
- Only 1–2 of 4 groups had mixed outcomes. The SFT policy was 97–100 % successful
  on those 32 envs, so exploration (open question 1) is the first thing to
  watch.

## Later: Prism-GRPO (arXiv:2608.17423)

Planned as a change to the reward, not to the update. DGPO consumes only
per-trajectory advantages A_i (`batch["advantages"]` → `adv` in the worker),
so a new reward or advantage should plug in through RLinf's
`register_advantage` / reward registry without touching `dgpo.py`. The one
constraint to preserve is that **A sums to zero within each group**, because
that is what cancels Z(c). A non-zero-sum advantage needs re-centring before
the DGPO step.

## Run

```bash
cd ~/pirl-dgpo
sbatch jobs/00_dgpo_smoke.sbatch            # ~30-60 min: plumbing, finite dgpo/* metrics
sbatch jobs/01_dgpo_libero_spatial.sbatch   # 12 h; extra hydra overrides pass through
sbatch jobs/01_dgpo_libero_spatial.sbatch algorithm.dgpo_beta=10
```

Logs are written to `logs/`, and tensorboard and checkpoints to
`$L/checkpoints/pirl_dgpo/<jobid>`.

## Runs

| job | what | status |
|---|---|---|
| 54402772 | smoke, 2 steps | passed |
| 54406749 | full, save_interval 40 | cancelled after step 1: 40 is never reached in 12 h (~26 steps at 1651 s/step: 1116 s rollout, 531 s training) |
| 54407201 | full, β=100, `sum` | **collapsed**: success 0.89 → 0.34 (step 10) → 0.0 (step 21); cancelled at step 25. Checkpoints for steps 10 and 20 exist |
| 54414981 | full, β=1e4, `mean`, KL 0 (cancelled at step 16 for the control) | no collapse at step 8 (0.77, vs 0.60 in 54407201); gate 0.35–0.44; but success drifts down 0.875 → 0.62 by step 10 while pref_acc rises 0.46 → 0.70 |
| 54417736 | as 54414981 with `update_epoch=1` (4× fewer Adam steps per rollout) | steps 1–5: 0.881, 0.805, 0.783, 0.779, 0.805, the same as 54414981 and 54407201 |
| 54420216 | DGPO **w/ SDE rollouts** (`flow_sde`, noise 0.5), update_epoch 1 | submitted 2026-09-24 after the eval showed GRPO > SFT and DGPO ≈ SFT. Noise goes on all 3 denoise steps (the NFT path), not on one random step as in πRL |
| 54431345 | DGPO SDE 0.3, update_epoch 2 | running |
| 54438201 | DGPO SDE 0.3, update_epoch 1, **EMA reference τ=0.1** | running |
| 54440450 | **Stage 2**: DGPO SDE 0.3, update_epoch 2 + **Prism reward** (success + 0.5·q, RLOO, filter off) | started 2026-09-24 |
| 54447394 | **LIBERO-Long** GRPO control (`CONFIG=libero_10_grpo_openpi_pi05`) | started 2026-09-25 |
| 54447395 | **LIBERO-Long** DGPO, ODE rollouts, update_epoch 2, mean, β=1e4 (`libero_10_dgpo_openpi_pi05`) | started 2026-09-25 |
| 54418480 | **SFT control**: the DGPO config with `actor.optim.lr=0`, so the model never changes | the per-step SFT baseline every DGPO run must be compared against |
| 54408415 | πRL GRPO control, clean clone, save_interval 10 | TIMEOUT at 12 h, 33 steps; train success 0.62 → ~0.77; checkpoints at steps 10, 20, 30 |
| 54408137 | piRL Flow-SDE GRPO control (`jobs/02_pirl_grpo_baseline.sbatch`): same clean clone, save_interval 10 | submitted 2026-09-23 |

Step 1 of 54406749: train success_once 0.887. This is **not** comparable to
the πRL baseline's 0.609, because DGPO's rollouts are noise-free (they behave
like the evaluation policy) and πRL's are SDE-noised. Compare with a held-out
eval of both runs' checkpoints. The baseline 54314933 saved none (same
save_interval 40 issue) and needs a re-run with `runner.save_interval=10`.

## Why 54407201 collapsed

Under `dgpo_traj_reduction: sum`, a chunk's gradient coefficient is gate·A_i,
so a trajectory's total weight is gate·A_i·n_i. Successful episodes stop early
(~20 chunks) and failures run all 48 chunks. So in a typical group of 7
successes and 1 failure (A = +0.35 / −2.47), Σ A_i·n_i ≈ 50 − 119 < 0. That
is net gradient **ascent** on the DSM loss: the policy is pushed away from its
own samples. DGPO's per-image balance (Σ_{G+}|A| = Σ_{G−}|A|) holds in z, but
not in gradient mass once samples have different lengths.

The gate could not stop it: at β = 100, z stayed ~1e-2, so the gate sat at 0.5
through step 13. L_θ and L_ref both rose ~20× (0.0035 → 0.05–0.08) as the
policy's actions left the reference's distribution. The review's
length-normalisation point was right, and I was wrong to call `sum` unbiased
in practice.

Fix, in the config: `mean` makes every trajectory's total weight gate·A_i·T,
which sums to zero per group (`test_mean_reduction_balances_gradient_mass`).
β = 1e4 puts the gate in its working range at z ~ 1e-2.

## The step-1 success rate is not the baseline

Steps 2–5 of three very different DGPO runs (update_epoch 4 vs 1, sum/β=100
vs mean/β=1e4) agree to about 0.02, and the GRPO control dips at step 2 as
well. The env seed fixes which reset states each step draws, so step k sees
the same reset states in every run. Step 1's set is easy (0.88) and step 2's is
harder (0.80). The "drop after the first update" was therefore mostly a
change of reset set, not an effect of training.

Every run must be compared step for step against the SFT policy on the same
reset states. That is job 54418480, the same config with lr = 0.

## Held-out eval (job 54419816): same ruler, noise-free, 512 episodes each

| checkpoint | success_once |
|---|---|
| SFT | 0.806 |
| GRPO step 10 (54408415) | **0.932** |
| GRPO step 30 (54408415) | **0.912** |
| DGPO ODE, update_epoch 1, step 10 (54417736) | 0.772 |
| DGPO ODE, update_epoch 4, step 10 (54414981) | 0.650 |
| DGPO ODE, update_epoch 1, step 20 (54417736) | 0.732 |
| **DGPO SDE (noise 0.3), update_epoch 1, step 10 (54420584)** | **0.838** |
| DGPO SDE (noise 0.3), update_epoch 1, step 20 (54420584) | 0.840 |
| **DGPO SDE (noise 0.3), update_epoch 2, step 10 (54431345)** | **0.862** |
| DGPO SDE (noise 0.3), update_epoch 2, step 20 (54431345) | 0.866 |
| DGPO SDE, update_epoch 1, EMA ref τ=0.1, step 10 (54438201) | 0.850 |
| Stage 2: DGPO SDE ep2 + Prism, step 10 (54440450) | 0.862 |
| Stage 2: DGPO SDE ep2 + Prism + EMA ref τ=0.5, step 10 (54443606) | **0.870** |
| Stage 2: DGPO SDE ep2 + Prism, step 20 (54440450) | 0.858 |
| Stage 2: DGPO SDE ep2 + Prism + EMA ref τ=0.5, step 20 (54443606) | 0.836 |

GRPO improves the deterministic policy by about 10.6 pp. DGPO with
update_epoch 1 tracks the SFT control step for step in training (mean
−0.014 over steps 1–10), so it learns nothing yet. The leading hypothesis is
exploration: ODE groups differ only through their initial noise. Hence the
"DGPO w/ SDE" run, 54420216.

**The ODE variant is concluded.** In training, update_epoch 1 minus the SFT
control averages −0.007 over steps 1–8 and −0.058 over steps 12–16 (−0.028
over steps 1–16): a slow, steady loss, consistent with −3.4 pp in eval at
step 10. Runs 54417736 and 54418480 were cancelled at steps 23 and 16.

**SDE, first try (54420216, noise 0.5 on all 3 steps):** training success at
step 1 was 0.29 (GRPO's one-step SDE gives 0.62), and L_ref was 0.15 against
0.004 under ODE. The noise was far too strong, so it was cancelled after one
step. Rerun as 54420584 with noise 0.3 ≈ 0.5/√3, which matches the variance
of πRL's single noisy step.

**SDE with noise 0.3 (54420584), steps 1–3:** training success (noisy)
0.467, 0.518, 0.537, rising while the SFT control on the same reset sets falls
(0.875, 0.807, 0.791). About 90% of groups are mixed (≈40% under ODE), and the
gate is at 0.39 by step 3. This is the first DGPO variant to trend up. The
step-10 checkpoint's noise-free eval decides.

**SDE step-10 eval: 0.838.** This is +3.2 pp over SFT (about 1.3 s.e. of the
difference) and +6.6 pp over ODE DGPO at the same step (about 2.7 s.e.). It is
the first DGPO checkpoint above SFT, but still far below GRPO's 0.932 at step
10. Exploration was the missing ingredient, and learning is now slow rather
than harmful. Next: evaluate 54420584 at step 20, plus a parallel run with
SDE and update_epoch 2 to test whether more learning per rollout closes the
gap now that the gate is active (0.25–0.32).

**SDE step-20 eval: 0.840, flat against step 10.** The gate fell from 0.32 to
0.19–0.24. With a frozen SFT reference, DGPO stops once the policy prefers
successes over failures more than SFT did in DSM terms, and that proxy is
satisfied long before behaviour improves much. 54420584 was cancelled at
step ~21.

**Dynamic reference (54438201):** `dgpo_ref_ema_tau: 0.1`, so after every
update ref ← ref + 0.1·(θ − ref). This is the official code's `use_ema_ref`
option, taken from TDM-R1. The gate then measures progress against the
recent policy, which keeps it open while still limiting big steps. It is
otherwise identical to 54420584 (SDE 0.3, update_epoch 1). The test is
`test_ref_ema_moves_reference_toward_policy`.

**SDE with update_epoch 2, step-10 eval: 0.862.** This is +5.6 pp over SFT
(about 2.3 s.e., the first clearly significant DGPO gain) and +2.4 pp over
update_epoch 1 at the same step. More learning per rollout helps once
exploration supplies the signal. The gate is already at 0.22, so it may
plateau like update_epoch 1 did. If the EMA reference (54438201) keeps the
gate open, the next run combines the two.

## Stage 2: Prism reward (run 54440450)

The code is `trajectory_jerk` and `prism_rloo_advantages` in
`rlinf/algorithms/dgpo.py`, switched on with `algorithm.dgpo_reward=prism`.

- R = success + λ·q with λ = 0.5.
- q = exp(−J / median J of the step's batch), where J is the mean squared
  second difference of the executed actions over the 6 continuous
  delta-pose dims. The gripper is excluded because it is a binary switch.
  Only valid (pre-done) steps count.
- The advantage is RLOO, not GRPO: no division by the group std, so
  same-outcome groups keep the scale λ·Δq. It is zero-sum per group, so
  Z(c) still cancels.
- `filter_rewards=False`, enforced by an assertion, because Prism exists to use
  the all-success and all-failure groups.

Base config: the best DGPO so far (SDE 0.3, update_epoch 2, frozen reference),
so the direct comparison is 54431345. Tests:
`test_prism_rloo_zero_sum_and_band_separation` and
`test_trajectory_jerk_masks_and_measures_curvature`.

Note on `prism/success` vs `success_once` (0.43 vs 0.53 at step 1): LIBERO
uses a relative reward (`use_rel_reward`), so a success that is lost again
inside the same 5-action chunk nets +1 − 1 = 0, while `success_once` still
counts it. `prism/success` uses the net reward, which is the same definition
as GRPO's own `calculate_scores`, so every run scores trajectories the same
way.

## DGPO+SDE is stuck at about +5–6 pp, so the work moves to Stage 2

- update_epoch 2: 0.862 at step 10 and 0.866 at step 20, with the gate at 0.17.
- EMA reference τ = 0.1: 0.850 at step 10, against 0.838 for the frozen
  reference with the same setup. That is +1.2 pp, within noise, and its
  gate still closes (0.26 at step 12). A reference that trails this slowly
  does not keep the gate open.

Both runs were cancelled on 2026-09-24 at 22:48. Stage 2 continues:
54440450 (Prism, frozen reference) plus a new Prism run with a much faster
reference, τ = 0.5 (the official code's EMA decay 0.3 corresponds to
τ = 0.7), since Prism alone does not address why the gate closes.

**Prism step-10 eval: 0.862**, identical to the same setup without Prism
(54431345 at step 10). SDE rollouts already make about 90% of groups mixed
(only 3–9% are same-outcome here), so Prism's tie-breaking has little to
act on. Prism is built for regimes with many same-outcome groups, such as
ODE rollouts.

Prism + EMA reference τ = 0.5 (54443606): the gate stays open at 0.39–0.42
through step 6, whereas the frozen-reference runs were below 0.3 by then, and
training success drifts up (0.49 → 0.57). Its step-10 eval is next.

**Prism + EMA τ = 0.5, step-10 eval: 0.870.** This is the best DGPO result
so far, but only +0.8 pp over Prism alone (within noise). Every SDE-based
DGPO variant lands between 0.84 and 0.87 at step 10: +3 to +6 pp over SFT,
against GRPO's +12.6 pp. The ceiling looks like it belongs to the method, not
to any single setting. Even with τ = 0.5 the gate drifts down (0.32 by step
12). No further variants are being launched; the step-20 evals of both Prism
runs are the last data points.

## LIBERO-Long (libero_10): a test of the "no signal" explanation

πRL Table 3 gives π0.5 few-shot SFT success rates (one unified checkpoint for
all suites, the same RLinf-Pi05-LIBERO-SFT used here): Spatial 84.6,
Object 95.4, Goal 84.6, **Long 43.9**. GRPO (Flow-SDE) reaches 99.6 / 100 /
98.8 / 93.0.

On Spatial, ODE DGPO failed plausibly because SFT is already ~0.88
noise-free, so the 8 trajectories in a group barely differ. On Long, with SFT
at ~0.44, almost every group is mixed even without noise (P(all same) ≈
0.44^8 + 0.56^8 ≈ 1%). If ODE DGPO still does not learn there, the problem is
the method, not a lack of exploration.

Runs: GRPO (54447394) and ODE DGPO (54447395), each 12 h. Episodes are 480
steps (Spatial: 240) and the sampler uses 4 steps (the RLinf Long config), so
expect roughly 12–15 steps per run. The job scripts now take `CONFIG=` (and
`EVAL_CONFIG=` for the eval job) as environment variables; the defaults are
unchanged.

**Long step 1:** GRPO (SDE) training success 0.268; ODE DGPO 0.396 (the SFT
policy itself, noise-free, in line with the paper's 43.9%). Only 59% of DGPO
groups are mixed, not the ~99% estimated above. Success varies hugely by
task, and a group shares one reset state (one task), so outcomes cluster
within groups. Steps take about 1900 s, so a run covers ~21 steps in 12 h.
SFT eval on Long: job 54449191. The step-10 and step-20 checkpoints are
auto-evaluated (`logs/autoeval_long.sh`).

## Extending to ManiSkill, MetaWorld and CALVIN (started 2026-09-25)

The postdoc's plan: (1) an RL method from Flow-GRPO / DGPO / Prism-GRPO, (2) in
πRL, (3) on π0 / π0.5. LIBERO is done, so the same DGPO goes on πRL's other
three benchmarks, each with and without the Prism reward: at least 6 runs.

**Setup state:** none of the three environments or SFT checkpoints were on the
cluster.

| | ManiSkill | MetaWorld | CALVIN |
|---|---|---|---|
| πRL π0.5 config | `maniskill_ppo_openpi_pi05` | `metaworld_50_ppo_openpi_pi05` | `calvin_abc_d_ppo_openpi_pi05` |
| env package | `mani_skill` v3.0.0b22 plus assets (sapien 3.0.1 already present) | `metaworld==3.0.0` | CALVIN repo + tacto + pyfasthash |
| SFT checkpoint | RLinf-Pi05-ManiSkill-25Main-SFT | RLinf-Pi05-MetaWorld-SFT | RLinf-Pi05-CALVIN-ABC-D-SFT |
| note | needs Vulkan; GPU nodes have `/etc/vulkan/icd.d/nvidia_icd.json` but no system `libvulkan` | action dim 4 → Prism jerk dims 3 | eval chains 5 subtasks (9600 steps) |

**Separate venvs.** CALVIN pins setuptools, hydra and transformers, and
MetaWorld 3 may move gymnasium, so each benchmark gets its own venv
(`$L/venvs/rlinf_<env>`). It is a hardlink copy of the working `rlinf_venv`
(`cp -al`, then the `bin/` shebangs are rewritten), so the openpi and
transformers-fork file layout is identical, and installs replace files
instead of editing them.

**Configs.** `jobs/make_dgpo_configs.py` turns each πRL PPO config into a DGPO
config. It changes only the algorithm (GRPO groups of 8, DGPO loss,
update_epoch 2, no value head), the sampler (SDE 0.3, `is_nft`) and the
training env (no auto-reset, one reset state per group). The DGPO block is
copied from the LIBERO config. Prism is switched on at submit time.
`prism_jerk_dims` now defaults to action_dim − 1 (all but the gripper).

### LIBERO-Long held-out eval (noise-free; the eval horizon was raised to 3000)

The first attempt failed with robosuite's `executing action in terminated
episode`. robosuite's horizon (1000) is not reset between auto-reset
episodes. Spatial 240 × 4 = 960 stays under it; Long 520 × 4 = 2080 does not.
Fix: `EVAL_EXTRA=+env.eval.init_params.horizon=3000`.

| Long | step 10 | step 20 |
|---|---|---|
| SFT | 0.274 | |
| GRPO | 0.316 | **0.402** |
| DGPO, ODE | 0.208 | **0.106** |

Even with far more learning signal than on Spatial, ODE DGPO loses ground
steadily (−16.8 pp by step 20), so a lack of exploration is not the cause.

### Setup progress for the other benchmarks
- MetaWorld: `metaworld==3.0.0` and `gymnasium==1.3.0` are installed with
  `--target` into `$L/envpkgs/metaworld` and prepended through
  `EXTRA_PYTHONPATH` (the official install upgrades gymnasium in place, which
  would break LIBERO). Smoke 54459317 passed. Full runs 54459740 (DGPO) and
  54459741 (DGPO + Prism) started.
- ManiSkill: `mani_skill` v3.0.0b22 went into `rlinf_venv` (only additions, per
  a dry run). The first smoke (54459320) failed because compute nodes reject
  GitHub TLS with the system CA store, so the PhysX GPU library and the
  bridge/WidowX assets are now fetched on the login node, and the jobs set
  `SSL_CERT_FILE` to certifi's bundle. The login node cannot import torch (8 GB
  virtual-memory hard limit), so the fetch replicates `download_asset`'s
  unzip rules with urllib. Smoke re-run: 54459773.
- CALVIN: the repo is cloned at `$L/envsrc/calvin`. RLinf imports
  `calvin_agent.evaluation.utils`, which pulls in the MCIL model
  (pytorch-lightning), plus pybullet, pyfasthash, numpy-quaternion and gym. To
  do.
- ManiSkill smoke 54459773 passed. SAPIEN falls back to its builtin libvulkan
  ("Failed to find system libvulkan") and renders fine on the GPU nodes. Full
  runs: 54459859 (DGPO) and 54459860 (DGPO + Prism). The config keeps πRL's
  `val_check_interval: 10`, so noise-free eval numbers appear in the training
  log every 10 steps.
- CALVIN: extra deps side-installed with `--target $L/envpkgs/calvin
  --no-deps`: pybullet, numpy-quaternion, hydra-colorlog, colorlog,
  pytorch-lightning 1.8.6, lightning-lite 1.8.6, lightning-utilities 0.3.0,
  torchmetrics 0.11.4, tensorboardX, and the RLinf pyfasthash fork (built
  with `--no-build-isolation`). `calvin_env`, `calvin_models` and `tacto` run
  from source through `EXTRA_PYTHONPATH`. RLinf finds `calvin_models/conf` by
  walking up from `calvin_env.__file__`, which the source layout satisfies.
  Smoke: 54459804.

### Scheduling constraints and the running matrix (2026-09-26)

`nairr-gpu-normal` allows 4 running jobs per user (8 submitted).
`nairr-gpu-shared-normal` allows 8 jobs and 8 GPUs per user, at most 3 GPUs and 54 CPUs per
job. RLinf's `component_placement: all` adapts to the GPU count, so CALVIN
runs on 2 GPUs (64 envs → 32 per rank) and evals on 1.

| benchmark | DGPO | DGPO + Prism | notes |
|---|---|---|---|
| MetaWorld | 54459740 | 54459741 | 4 GPUs, ~22 min/step |
| ManiSkill | 54460468 | 54460334 | 4 GPUs; 160 envs (320 = πRL's 8-GPU setting OOMed / Vulkan device-lost at 80 envs per GPU); micro-batch 64 |
| CALVIN | 54460480 | 54460481 | shared, 2 GPUs; `CALVIN_USE_EGL=0` (EGL SIGABRTed mid-rollout in smoke 54459804); `rollout_epoch=2` (8 would be >1.3 h/step) |

MetaWorld SFT eval (shared, 1 GPU): 54460482, using the DGPO config's
`env.eval` section through `EVAL_CONFIG_DIR`.

**MetaWorld SFT eval (54460702): 0.461** (noise-free, 512 episodes, with the
generated `metaworld_50_dgpo_openpi_pi05_eval.yaml`; eval mode builds the
model from `rollout.model`, so `jobs/make_eval_configs.py` merges
`actor.model` into it).

**ManiSkill `ErrorDeviceLost` explained, most likely.** With
`val_check_interval: 10` each env worker builds both the training and the
eval ManiSkill GPU simulator (two SAPIEN/PhysX GPU systems and camera groups)
in one process at init. It failed at init in 4 of 4 full runs (at 40 and 16
envs per GPU) and passed in the smoke test, which ran with
`val_check_interval=-1`. Relaunched without in-run eval (54461145,
54461146; 160 envs, rollout_epoch 2 = πRL's 320 trajectories per step).
Evals run as separate jobs.

**Ray fix works.** cv_dgpo (54460699) and eval_mw (54460702) shared node
exp-19-16 without a GCS clash (per-job port and temp dir).

**ManiSkill Vulkan `ErrorDeviceLost` is node-specific.** Of 13 attempts, the
only successes ran on exp-19-17 (smoke 54459773, and 54461671 pinned there).
Every attempt on exp-19-02/03/05/07/08 failed while creating the GPU
camera group of the first env, whatever the env count (8–80 per GPU) and with
or without in-run eval; the exact passing smoke command failed on exp-19-02
(54461305). Runs are now pinned: 54461734 (DGPO, exp-19-17) and 54461735
(Prism, exp-19-13, to test a second node in the 19-13..17 group where
LIBERO, MetaWorld and CALVIN run).

**CALVIN evals** (πRL's eval setting: 50 envs, 5-subtask chains, noise-free,
`success_once`; 50 episodes ≈ ±5 pp):

| | step 10 | step 20 |
|---|---|---|
| SFT | 0.128 | |
| DGPO | 0.133 | 0.137 |
| DGPO + Prism | 0.130 | |

**Quota check (2026-09-26 06:10):** 1,125 / 10,000 GPU-h charged (including
walltime held by running jobs; ~990 actually used per sacct). Lustre: 976 GB /
9.5 TB and 614k / 2M files for the user, 50 TB for the group.

## MetaWorld: DGPO works (step-20 eval, 2026-09-26)

| MetaWorld (noise-free, 512 episodes, ±2.2 pp) | success | vs SFT |
|---|---|---|
| SFT | 0.461 | |
| **DGPO (SDE 0.3, ep2), step 20** (54459740) | **0.570** | **+10.9 pp** |
| **DGPO + Prism, step 20** (54459741) | **0.563** | **+10.2 pp** |

This is the first benchmark where DGPO clearly improves on SFT (~5 s.e.).
Prism adds nothing on top. The step-10 evals (54461357/61) hung after
connecting to Ray on exp-19-13 (the same node that fails ManiSkill's Vulkan)
and timed out at 5 h. They were re-submitted as 54462635, excluding
exp-19-13.

CALVIN Prism step 20: 0.143 (SFT 0.128, ±5 pp), no change.

ManiSkill: DGPO runs on exp-19-17 (~290 s/step, 85–95% mixed groups).
Prism failed on exp-19-13 (Vulkan), so it is queued on exp-19-17 behind
DGPO (54462636, `--dependency=afterany`). ManiSkill evals also need a
Vulkan-capable node, so they run on exp-19-17 after training.

### Learning curves so far (noise-free evals)

| | SFT | step 10 | step 20 | step 30 |
|---|---|---|---|---|
| MetaWorld DGPO | 0.461 | 0.512 | **0.570** | 0.539 |
| MetaWorld DGPO + Prism | 0.461 | 0.553 | **0.563** | 0.529 |
| CALVIN DGPO | 0.128 | 0.133 | 0.137 | 0.142 |
| CALVIN DGPO + Prism | 0.128 | 0.130 | 0.143 | 0.148 |

CALVIN's 50-episode eval (±5 pp) is too coarse for these gaps. The final
checkpoint will get a larger eval.

MetaWorld peaks at step 20 (+10–11 pp) and eases back by step 30 (+7–8 pp; a
3 pp drop, ~1.5 s.e.). The gates were at 0.17–0.20 by then, the same
closing-gate plateau seen on LIBERO, but from a much larger gain. Both
MetaWorld runs completed 12 h (~32 steps).

ManiSkill node map (Vulkan `ErrorDeviceLost` at GPU camera creation): works on
exp-19-15 and exp-19-17; fails on exp-19-02/03/05/07/08/13/14. ManiSkill Prism
moved to exp-19-15 (the queued exp-19-17 job was cancelled).

### ManiSkill DGPO learning curve (noise-free, 512 episodes, eval on exp-19-16 with 1 GPU)

| | SFT | step 20 | step 40 | step 60 | step 80 |
|---|---|---|---|---|---|
| ManiSkill DGPO (54461734) | 0.447 | 0.463 | 0.477 | 0.471 | **0.500** |

This is a steady climb, +5.3 pp by step 80. exp-19-16 is a third
Vulkan-capable node. The step-60 eval died on a Ray port clash (client server
10002, two evals on one node), so `_ray_start.sh` now gives every Ray port,
workers included, a per-job 1000-port block.

CALVIN runs finished 12 h (DGPO 100 steps, Prism 40). Larger final evals,
200 episodes each: SFT 54466178, DGPO step 100 54466179, Prism step 40
54466180.

## Final results of the 3-benchmark × Prism matrix (2026-09-27)

All six runs hit the 12 h limit; all evals are noise-free held-out.

| benchmark (episodes, ≈s.e.) | SFT | DGPO (best / final) | DGPO + Prism (best / final) |
|---|---|---|---|
| MetaWorld (512, ±2.2 pp) | 0.461 | **0.570** @20 / 0.539 @30 | **0.563** @20 / 0.529 @30 |
| ManiSkill (512, ±2.2 pp) | 0.447 | **0.500** @80–100 / 0.480 @140 | **0.488** @120 / 0.488 @120 |
| CALVIN (200, ±2.4 pp) | 0.136 | 0.140 @100 | 0.138 @40 |
| LIBERO-Spatial (512) | 0.806 | 0.862–0.870 @10 (SDE) | 0.870 @10 (+EMA ref) |

ManiSkill curves. DGPO: 0.463, 0.477, 0.471, 0.500, 0.500, 0.479, 0.480 at
steps 20–140. Prism: 0.477, 0.449, 0.475, 0.488 at steps 20, 40, 80, 120.

Reading:
- DGPO with SDE rollouts improves on SFT on MetaWorld (+11 pp) and
  ManiSkill (+5 pp) and LIBERO-Spatial (+6 pp), and does nothing on CALVIN.
- Every benchmark peaks early and then plateaus or eases back, the same
  closing-gate pattern seen on LIBERO.
- Prism's reward adds nothing measurable on any benchmark.
- For reference, πRL's GRPO reached +12.6 pp on LIBERO-Spatial and +12.8 pp on
  LIBERO-Long with the same budget. GRPO was not run on the other three here.

## GRPO control arm on MetaWorld / ManiSkill / CALVIN (started 2026-09-27)

`jobs/make_grpo_configs.py` builds the πRL GRPO configs from the PPO ones:
GRPO groups of 8, critic-free advantage, `loss_type: actor` (PPO clip 0.2),
update_epoch 1, Flow-SDE noise 0.5 at one random denoise step (as
`libero_spatial_grpo_openpi_pi05`). Env counts and rollout epochs match the
DGPO runs. Global batch is chosen to divide each rollout:

| run | job | envs × epochs | global / micro batch | where |
|---|---|---|---|---|
| MetaWorld GRPO | 54480154 | 64 × 8 | 2048 / 128 | nairr-gpu |
| ManiSkill GRPO | 54480155 | 160 × 2, no in-run eval | 1280 / 64 | nairr-gpu, exp-19-15 |
| CALVIN GRPO | 54480156 | 64 × 2, CPU render | 2048 / 128 | shared, 2 GPUs |

User constraint from 2026-09-27: use at most 3 of the 4 `nairr-gpu` job slots.

### First GRPO evals (noise-free, same eval configs as DGPO)

| | SFT | DGPO | DGPO + Prism | **GRPO** |
|---|---|---|---|---|
| MetaWorld step 10 | 0.461 | 0.512 | 0.553 | **0.609** |
| ManiSkill step 20 | 0.447 | 0.463 | 0.477 | **0.545** |
| ManiSkill step 40 | 0.447 | 0.477 | 0.449 | **0.539** |

GRPO beats DGPO at equal steps by a wide margin, and also beats DGPO's best
checkpoint (MetaWorld 0.570, ManiSkill 0.500), with 15–35% faster steps. This
matches LIBERO (GRPO +12.6 pp vs DGPO +6 pp).

## Final DGPO vs GRPO comparison (2026-09-28)

Noise-free held-out evals. MetaWorld and ManiSkill: 512 episodes (±2.2 pp).
CALVIN: 50 episodes (±5 pp) per point unless marked.

**MetaWorld** (SFT 0.461)

| step | 10 | 20 | 30 |
|---|---|---|---|
| DGPO | 0.512 | **0.570** | 0.539 |
| DGPO + Prism | 0.553 | 0.563 | 0.529 |
| **GRPO** | **0.609** | 0.598 | **0.611** |

**ManiSkill** (SFT 0.447)

| step | 20 | 40 | 80 | 120 | 160 | 200 |
|---|---|---|---|---|---|---|
| DGPO | 0.463 | 0.477 | 0.500 | 0.479 | (140: 0.480) | |
| DGPO + Prism | 0.477 | 0.449 | 0.475 | 0.488 | | |
| **GRPO** | 0.545 | 0.539 | **0.621** | 0.592 | **0.621** | 0.551 |

**CALVIN** (SFT 0.128; 0.136 with 200 episodes)

| step | 10 | 20 | 30 | 40 | 80 | 100 |
|---|---|---|---|---|---|---|
| DGPO | 0.133 | 0.137 | 0.142 | | | 0.140 (200 ep) |
| DGPO + Prism | 0.130 | 0.143 | 0.148 | 0.138 (200 ep) | | |
| GRPO | | 0.161 | | 0.164 | 0.135 | |

**Summary across benchmarks (best checkpoint, gain over SFT)**

| | LIBERO-Spatial | LIBERO-Long | MetaWorld | ManiSkill | CALVIN |
|---|---|---|---|---|---|
| DGPO (best variant) | +6.4 | −6.6 (ODE) | +10.9 | +5.3 | ~0 |
| DGPO + Prism | +6.4 | – | +10.2 | +4.1 | ~0 |
| **GRPO (πRL)** | **+12.6** | **+12.8** | **+15.0** | **+17.4** | +3.6 (noisy) |

GRPO also runs 15–35% faster per step. One eval (CALVIN GRPO step 60,
54488635) failed and was not rerun.

## Verification against the official DGPO code (2026-09-28)

`test_matches_official_dgpo_gradient` executes `compute_group_dgpo_loss_allreduce`
verbatim from a local clone of Luo-Yihong/DGPO (commit 6a45077; the repo has
no license file, so the source is read at test time, not copied), with
`torch.distributed` swapped for a single-process stand-in. With one chunk per
trajectory, our surrogate's gradient equals the official loss's gradient × B,
for β ∈ {1, 100, 1e4}. **Passed.** The DGPO core is the paper's. What is ours
is only the trajectory-level extension (summing chunk deltas under the valid
mask, with length normalisation).

## MetaWorld hyperparameter sweep (started 2026-09-28)

Base: MetaWorld DGPO (SDE 0.3, update_epoch 2, β 1e4, lr 5e-6, t_min 0.3,
frozen reference). One change per run, 9 h each, ≤3 concurrent `nairr-gpu`
jobs (the last three are chained with `afterany`). Step-10 and step-20 evals
run through `logs/autoeval_mwsweep.sh`.

| run | change | job |
|---|---|---|
| lr3 | lr 1.5e-5 | 54508025 |
| lr10 | lr 5e-5 | 54508026 |
| b1e3 | β 1e3 | 54508027 |
| b1e5 | β 1e5 | 54508028 |
| ep4 | update_epoch 4 | 54508029 |
| tmin01 | t_min 0.1 | 54508030 |
- lr10 (54508026) cancelled at step 10 (2026-09-28 19:45). It collapsed:
  training success 0.36 → 0.11, and L_ref 0.04 → 0.92 as the policy left the
  reference's distribution; the gate briefly opened to 0.89, then shut
  (0.01). The dependent ep4 run takes its slot.
- Step-10 evals (512 ep): lr3 **0.539**, b1e3 **0.557**, lr10 0.115 (the
  collapse is confirmed). Base DGPO at step 10 was 0.512 and GRPO 0.609. β 1e3
  keeps the gate open (~0.42 at step 13), whereas lr3's is down to 0.14.
- Step-20 evals: lr3 0.559, b1e3 0.557, against base DGPO 0.570 and GRPO
  0.598. A faster lr or a softer β gets there sooner (step 10 +3–5 pp) but
  plateaus at the same ~0.56–0.57. b1e3's gate stayed open (>0.4) and it
  still plateaued, so the closing gate is not the whole story.

### MetaWorld sweep: final (noise-free, 512 episodes, ±2.2 pp)

| setting | step 10 | step 20 |
|---|---|---|
| base DGPO (lr 5e-6, β 1e4, ep 2, t_min 0.3) | 0.512 | **0.570** |
| lr 1.5e-5 | 0.539 | 0.559 |
| lr 5e-5 | 0.115 (collapsed, cancelled) | – |
| β 1e3 | **0.557** | 0.557 |
| β 1e5 | 0.490 | 0.539 |
| update_epoch 4 | 0.531 | 0.557 |
| t_min 0.1 | 0.520 | 0.523 |
| **GRPO (πRL)** | **0.609** | **0.598** (step 30: 0.611) |

No single-knob change beats the base DGPO at step 20. Every setting that
learns lands at 0.52–0.57, and GRPO stays at 0.60–0.61. The MetaWorld gap at
the peak (0.570 vs 0.611, ≈1.3 s.e. of the difference) is the smallest of any
benchmark. On ManiSkill (0.500 vs 0.621) and LIBERO (0.866 vs 0.932; Long 0.21
vs 0.40) it is clear.

## DGPO with πRL's exact exploration noise (started 2026-09-29)

`actor.model.openpi.nft_single_noise_step=True` (new option, default False)
makes the NFT rollout path pick one random denoise step for the SDE noise, as
the non-NFT Flow-SDE path does, instead of noising every step. With
`noise_level=0.5`, DGPO's rollouts are then drawn exactly like the GRPO
baseline's; only the update rule differs. Base DGPO otherwise (ep2, β 1e4,
mean). Runs: MetaWorld 54533225, ManiSkill 54533226 (exp-19-15). Sanity
check: the step-1 training success should match GRPO's step 1 (MetaWorld
0.422, ManiSkill 0.538).
- Sanity check passed. Step-1 training success: MetaWorld 0.438 (GRPO 0.422;
  old all-step DGPO 0.369); ManiSkill 0.525 (GRPO 0.538). The rollouts now
  match GRPO's.
- ManiSkill training success, step-matched: steps 5–9 GRPO ahead (0.59–0.65
  vs 0.51–0.58); steps 10–12 **DGPO-1step ahead**, 0.597 / 0.647 / 0.672 vs GRPO
  0.588 / 0.569 / 0.578. This is noisy training metrics only; evals decide.
- **ManiSkill DGPO-1step step-20 eval: 0.420**, below SFT (0.447), against
  old DGPO 0.463 and GRPO 0.545, even though its training success
  (single-step noise 0.5) ran ahead of GRPO's at steps 10–12. The likely
  mechanism: DGPO regresses onto the sampler's final x0, and a single large
  noise injection leaves that noise in successful samples, so the policy
  learns the noised actions. GRPO optimises the transition log-likelihood
  instead. Training success is not a proxy for noise-free success.
- Evals so far. ManiSkill DGPO-1step: 0.420 / 0.467 / 0.498 at steps
  20 / 40 / 80, against old DGPO 0.463 / 0.477 / 0.500 and GRPO 0.545 / 0.539 /
  0.621. MetaWorld DGPO-1step step 10: 0.535 (old DGPO 0.512, GRPO 0.609).
  **Matching GRPO's exploration noise exactly does not close the gap**: the
  difference is the update rule, not the noise.
- **Final (both runs hit 12 h).** MetaWorld DGPO-1step: 0.535 / 0.553 / 0.488
  at steps 10 / 20 / 30 (old DGPO 0.512 / 0.570 / 0.539; GRPO 0.609 / 0.598 /
  0.611). ManiSkill DGPO-1step: 0.420 / 0.467 / 0.498 / 0.459 at steps
  20 / 40 / 80 / 120 (old 0.463 / 0.477 / 0.500 / 0.479; GRPO 0.545 / 0.539 /
  0.621 / 0.592). With exploration noise identical to GRPO's, DGPO is no
  better than with its all-step noise, and it is less stable (below SFT at
  ManiSkill step 20; 0.488 at MetaWorld step 30). The gap to GRPO is the
  update rule.
