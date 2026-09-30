"""Figures for the DGPO-in-πRL study. All numbers are the noise-free held-out
evals recorded in ../README.md (success_once). Error bars are ±1 binomial
standard error, sqrt(p(1-p)/n), with n the number of eval episodes."""
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))
SURF, INK, INK2, GRID, SFT_C = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"
C = {"DGPO": "#2a78d6", "DGPO + Prism": "#eb6834", "GRPO (πRL)": "#1baf7a",
     "DGPO, ODE (paper)": "#eda100", "DGPO, 1-step noise 0.5": "#e87ba4"}
MK = {"DGPO": "o", "DGPO + Prism": "s", "GRPO (πRL)": "^", "DGPO, ODE (paper)": "D",
      "DGPO, 1-step noise 0.5": "v"}
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.facecolor": SURF,
    "figure.facecolor": SURF, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False,
})


def se(p, n):
    return math.sqrt(p * (1 - p) / n)


# ---------------------------------------------------------------- curves
# benchmark -> (SFT, n_sft, {method: [(step, p, n), ...]}, note)
CURVES = {
    "LIBERO-Spatial": (0.806, 512, {
        "DGPO": [(10, .862, 512), (20, .866, 512)],
        "DGPO + Prism": [(10, .862, 512), (20, .858, 512)],
        "GRPO (πRL)": [(10, .932, 512), (30, .912, 512)],
    }, "DGPO: SDE 0.3, 2 updates/rollout"),
    "LIBERO-Long": (0.274, 512, {
        "DGPO": [(10, .208, 512), (20, .106, 512)],
        "GRPO (πRL)": [(10, .316, 512), (20, .402, 512)],
    }, "DGPO here: paper setting (ODE)"),
    "MetaWorld": (0.461, 512, {
        "DGPO": [(10, .512, 512), (20, .570, 512), (30, .539, 512)],
        "DGPO + Prism": [(10, .553, 512), (20, .563, 512), (30, .529, 512)],
        "GRPO (πRL)": [(10, .609, 512), (20, .598, 512), (30, .611, 512)],
    }, ""),
    "ManiSkill": (0.447, 512, {
        "DGPO": [(20, .463, 512), (40, .477, 512), (60, .471, 512), (80, .500, 512),
                 (100, .500, 512), (120, .479, 512), (140, .480, 512)],
        "DGPO + Prism": [(20, .477, 512), (40, .449, 512), (80, .475, 512), (120, .488, 512)],
        "GRPO (πRL)": [(20, .545, 512), (40, .539, 512), (80, .621, 512), (120, .592, 512),
                       (160, .621, 512), (200, .551, 512)],
    }, ""),
    "CALVIN": (0.1355, 200, {
        "DGPO": [(10, .133, 50), (20, .137, 50), (30, .142, 50), (100, .140, 200)],
        "DGPO + Prism": [(10, .130, 50), (20, .143, 50), (30, .148, 50), (40, .138, 200)],
        "GRPO (πRL)": [(20, .161, 50), (40, .164, 50), (80, .135, 50)],
    }, "50-episode evals (±5 pp)"),
}

fig, axes = plt.subplots(2, 3, figsize=(13, 7.2))
axes = axes.ravel()
for ax, (bench, (sft, n_sft, methods, note)) in zip(axes, CURVES.items()):
    xmax = max(s for m in methods.values() for s, _, _ in m)
    s_se = se(sft, n_sft)
    ax.axhspan(sft - s_se, sft + s_se, color=SFT_C, alpha=0.15, lw=0)
    ax.axhline(sft, color=SFT_C, lw=1.5)
    ax.text(xmax * 1.02, sft, f"SFT {sft:.3f}", color=INK2, va="center", fontsize=8.5)
    for m, pts in methods.items():
        xs = [0] + [s for s, _, _ in pts]
        ys = [sft] + [p for _, p, _ in pts]
        es = [s_se] + [se(p, n) for _, p, n in pts]
        ax.errorbar(xs, ys, yerr=es, color=C[m], marker=MK[m], ms=6, lw=2,
                    capsize=0, elinewidth=1, mec=SURF, mew=1.5, label=m)
    ax.set_title(bench, loc="left", color=INK, fontsize=11.5, fontweight="bold")
    ax.set_xlim(-xmax * 0.03, xmax * 1.22)
    ax.set_xlabel("training step" + (f"\n{note}" if note else ""))
    ax.set_ylabel("success rate (noise-free eval)")
axes[-1].axis("off")
handles, labels = axes[2].get_legend_handles_labels()
handles.append(plt.Line2D([], [], color=SFT_C, lw=1.5)); labels.append("SFT start (band = ±1 s.e.)")
axes[-1].legend(handles, labels, loc="center", fontsize=11)
fig.suptitle("DGPO vs πRL's GRPO on π0.5, held-out success rate by training step",
             x=0.01, ha="left", color=INK, fontsize=13)
fig.text(0.01, 0.005, "Error bars: ±1 binomial s.e. 512 episodes per point unless noted; step 0 = SFT.",
         color=INK2, fontsize=8.5)
fig.tight_layout(rect=(0, 0.02, 1, 0.95))
fig.savefig(os.path.join(OUT, "fig2_learning_curves.png"), dpi=180)
fig.savefig(os.path.join(OUT, "fig2_learning_curves.pdf"))
plt.close(fig)

# ---------------------------------------------------------------- summary bars
# best checkpoint gain over SFT, in percentage points; None = not run
BENCH = ["LIBERO-Spatial", "LIBERO-Long", "MetaWorld", "ManiSkill", "CALVIN"]
GAIN = {
    "DGPO": [6.0, -6.6, 10.9, 5.3, 0.65],
    "DGPO + Prism": [6.4, None, 10.2, 4.1, 1.25],
    "GRPO (πRL)": [12.6, 12.8, 15.0, 17.4, 2.85],
}
fig, ax = plt.subplots(figsize=(10, 4.6))
w = 0.26
for i, (m, vals) in enumerate(GAIN.items()):
    for j, v in enumerate(vals):
        x = j + (i - 1) * w
        if v is None:
            ax.text(x, 0.4, "not run", rotation=90, ha="center", va="bottom", color=INK2, fontsize=8)
            continue
        ax.bar(x, v, width=w * 0.88, color=C[m], label=m if j == 0 else None, zorder=2)
        ax.text(x, v + (0.5 if v >= 0 else -0.5), f"{v:+.1f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=8.5, color=INK)
ax.axhline(0, color=INK2, lw=1)
ax.set_xticks(range(len(BENCH)), BENCH)
ax.set_ylabel("best gain over SFT (pp)")
ax.set_title("Best-checkpoint improvement over SFT", loc="left", color=INK, fontsize=12.5, fontweight="bold")
ax.legend(loc="upper left", ncol=3)
ax.grid(axis="x", visible=False)
ax.text(1 - 0.02, 0.02, "LIBERO-Long DGPO = paper setting (ODE); CALVIN gaps are within ±5 pp noise",
        transform=ax.transAxes, ha="right", color=INK2, fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_summary_gain.png"), dpi=180)
fig.savefig(os.path.join(OUT, "fig1_summary_gain.pdf"))
plt.close(fig)

# ---------------------------------------------------------------- LIBERO-Spatial ablation
ROWS = [  # label, step-10 eval, family
    ("SFT (start)", .806, None),
    ("DGPO, ODE, 4 updates/rollout (paper setting)", .650, "DGPO"),
    ("DGPO, ODE, 1 update/rollout", .772, "DGPO"),
    ("DGPO, SDE 0.3, 1 update", .838, "DGPO"),
    ("  + moving reference (τ=0.1)", .850, "DGPO"),
    ("DGPO, SDE 0.3, 2 updates", .862, "DGPO"),
    ("  + Prism reward", .862, "DGPO + Prism"),
    ("  + Prism + moving reference (τ=0.5)", .870, "DGPO + Prism"),
    ("GRPO (πRL, Flow-SDE)", .932, "GRPO (πRL)"),
]
fig, ax = plt.subplots(figsize=(9, 4.8))
ys = list(range(len(ROWS)))[::-1]
for y, (lab, p, fam) in zip(ys, ROWS):
    # a dot plot: the x-axis does not start at zero, so bars would misstate ratios
    col = SFT_C if fam is None else C[fam]
    ax.errorbar(p, y, xerr=se(p, 512), color=col, lw=1.5, capsize=0, zorder=3,
                marker="o" if fam is None else MK[fam], ms=9, mec=SURF, mew=1.5)
    ax.text(p + se(p, 512) + 0.006, y, f"{p:.3f}", va="center", fontsize=9, color=INK)
ax.axvline(.806, color=SFT_C, lw=1.2, zorder=1)
ax.set_yticks(ys, [r[0] for r in ROWS])
ax.set_xlim(0.6, 0.97)
ax.set_xlabel("success rate at step 10 (noise-free eval, 512 episodes)")
ax.set_title("LIBERO-Spatial: what it took to make DGPO learn", loc="left", color=INK,
             fontsize=12.5, fontweight="bold")
ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_libero_ablation.png"), dpi=180)
fig.savefig(os.path.join(OUT, "fig3_libero_ablation.pdf"))
plt.close(fig)
# ---------------------------------------------------------------- MetaWorld sweep
SWEEP = [  # label, step 10, step 20 (None = not reached)
    ("base DGPO (lr 5e-6, β 1e4)", .512, .570),
    ("lr 1.5e-5", .539, .559),
    ("lr 5e-5 (collapsed)", .115, None),
    ("β 1e3", .557, .557),
    ("β 1e5", .490, .539),
    ("4 updates / rollout", .531, .557),
    ("t_min 0.1", .520, .523),
    ("GRPO (πRL)", .609, .598),
]
fig, ax = plt.subplots(figsize=(9, 4.6))
ys = list(range(len(SWEEP)))[::-1]
for y, (lab, p10, p20) in zip(ys, SWEEP):
    col = C["GRPO (πRL)"] if lab.startswith("GRPO") else C["DGPO"]
    ax.errorbar(p10, y + 0.14, xerr=se(p10, 512), color=col, marker="o", ms=7, mfc=SURF,
                mec=col, mew=1.8, lw=1.2, capsize=0, zorder=3, label="step 10" if y == ys[0] else None)
    if p20 is not None:
        ax.errorbar(p20, y - 0.14, xerr=se(p20, 512), color=col, marker="o", ms=7, mec=SURF,
                    mew=1.5, lw=1.2, capsize=0, zorder=3, label="step 20" if y == ys[0] else None)
        ax.text(p20 + se(p20, 512) + 0.006, y - 0.14, f"{p20:.3f}", va="center", fontsize=8.5, color=INK)
ax.axvline(.461, color=SFT_C, lw=1.2, zorder=1)
ax.set_yticks(ys, [r[0] for r in SWEEP])
ax.set_xlim(0.42, 0.66)
ax.text(0.463, ys[-1] - 0.45, "SFT 0.461", color=INK2, fontsize=8.5)
ax.set_xlabel("success rate (noise-free eval, 512 episodes)   ○ step 10   ● step 20")
ax.set_title("MetaWorld: one-knob DGPO sweep vs GRPO", loc="left", color=INK, fontsize=12.5, fontweight="bold")
ax.grid(axis="y", visible=False)
ax.text(0.425, ys[2] - 0.14, "step 10: 0.115 →", color=INK2, fontsize=8.5, va="center")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig4_metaworld_sweep.png"), dpi=180)
fig.savefig(os.path.join(OUT, "fig4_metaworld_sweep.pdf"))
plt.close(fig)


def curve_panel(ax, title, sft, series, note=""):
    """series: {label: (color_key, [(step, p, n)], linestyle)}"""
    xmax = max(st for _, pts, _ in series.values() for st, _, _ in pts)
    s_se = se(sft, 512)
    ax.axhspan(sft - s_se, sft + s_se, color=SFT_C, alpha=0.15, lw=0)
    ax.axhline(sft, color=SFT_C, lw=1.5)
    ax.text(xmax * 1.02, sft, f"SFT {sft:.3f}", color=INK2, va="center", fontsize=8.5)
    for lab, (key, pts, ls) in series.items():
        xs = [0] + [a for a, _, _ in pts]
        ys = [sft] + [b for _, b, _ in pts]
        es = [s_se] + [se(b, n) for _, b, n in pts]
        ax.errorbar(xs, ys, yerr=es, color=C[key], marker=MK[key], ms=6, lw=2, ls=ls,
                    capsize=0, elinewidth=1, mec=SURF, mew=1.5, label=lab)
    ax.set_title(title, loc="left", color=INK, fontsize=11.5, fontweight="bold")
    ax.set_xlim(-xmax * 0.03, xmax * 1.22)
    ax.set_xlabel("training step" + (f"\n{note}" if note else ""))
    ax.set_ylabel("success rate (noise-free eval)")


# ---------------------------------------------------------------- before / after exploration noise
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
curve_panel(axes[0], "LIBERO-Spatial", 0.806, {
    "DGPO, ODE (paper setting)": ("DGPO, ODE (paper)", [(10, .772, 512), (20, .732, 512)], "--"),
    "DGPO + SDE noise": ("DGPO", [(10, .862, 512), (20, .866, 512)], "-"),
    "GRPO (πRL, Flow-SDE)": ("GRPO (πRL)", [(10, .932, 512), (30, .912, 512)], "-"),
})
curve_panel(axes[1], "LIBERO-Long", 0.274, {
    "DGPO, ODE (paper setting)": ("DGPO, ODE (paper)", [(10, .208, 512), (20, .106, 512)], "--"),
    "GRPO (πRL, Flow-SDE)": ("GRPO (πRL)", [(10, .316, 512), (20, .402, 512)], "-"),
}, note="(DGPO + SDE was not run on Long)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper left", ncol=3, fontsize=9.5, bbox_to_anchor=(0.005, 0.93))
fig.suptitle("Before vs after exploration noise", x=0.01, ha="left", color=INK, fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.86))
fig.savefig(os.path.join(OUT, "fig5_noise_before_after.png"), dpi=180)
fig.savefig(os.path.join(OUT, "fig5_noise_before_after.pdf"))
plt.close(fig)

# ---------------------------------------------------------------- noise placement
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
curve_panel(axes[0], "MetaWorld", 0.461, {
    "DGPO, SDE 0.3 on every step": ("DGPO", [(10, .512, 512), (20, .570, 512), (30, .539, 512)], "-"),
    "DGPO, SDE 0.5 on one step (= GRPO's noise)": ("DGPO, 1-step noise 0.5", [(10, .535, 512), (20, .553, 512), (30, .488, 512)], "--"),
    "GRPO (πRL, Flow-SDE)": ("GRPO (πRL)", [(10, .609, 512), (20, .598, 512), (30, .611, 512)], "-"),
})
curve_panel(axes[1], "ManiSkill", 0.447, {
    "DGPO, SDE 0.3 on every step": ("DGPO", [(20, .463, 512), (40, .477, 512), (80, .500, 512), (120, .479, 512)], "-"),
    "DGPO, SDE 0.5 on one step (= GRPO's noise)": ("DGPO, 1-step noise 0.5", [(20, .420, 512), (40, .467, 512), (80, .498, 512), (120, .459, 512)], "--"),
    "GRPO (πRL, Flow-SDE)": ("GRPO (πRL)", [(20, .545, 512), (40, .539, 512), (80, .621, 512), (120, .592, 512)], "-"),
})
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper left", ncol=3, fontsize=9.5, bbox_to_anchor=(0.005, 0.93))
fig.suptitle("Where the noise goes: DGPO with GRPO's exact exploration noise", x=0.01, ha="left",
             color=INK, fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.86))
fig.savefig(os.path.join(OUT, "fig6_noise_placement.png"), dpi=180)
fig.savefig(os.path.join(OUT, "fig6_noise_placement.pdf"))
plt.close(fig)
print("ok")
