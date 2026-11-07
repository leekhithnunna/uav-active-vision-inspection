"""Phase 6 - all plots (PNG, 300 dpi) from the result CSV/NPZ files. No number is typed by hand.
Writes plots/NN_<name>.png and plots/plot_index.csv (file, title, source files) used by the report."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, PLOTS  # noqa: E402

# ------------------------------------------------------------------ style (validated reference palette, light)
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                                                         "#e87ba4", "#008300", "#6250d6", "#e34948")
SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.size": 9,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 10,
    "axes.titleweight": "bold", "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 2, "axes.axisbelow": True})

COLOR = {"DQN (Agent A)": BLUE, "PPO (Agent B)": BLUE, "A2C (Agent C)": BLUE, "A2C (C_withA)": BLUE,
         "ALL LEARNED (C + A + B)": BLUE,
         "DQN (Agent A, real-image trained)": VIOLET, "PPO (Agent B, real-image trained)": VIOLET,
         "A2C (Agent C, real-image trained)": VIOLET, "ALL LEARNED (real-image trained models)": VIOLET,
         "raster_low": ORANGE, "scripted_approach (upper bound)": ORANGE, "rule_supervisor": ORANGE,
         "rule_supervisor (Agent A inside)": ORANGE, "ALL SCRIPTED (rule + raster + approach)": ORANGE,
         "raster_mid": AQUA, "fixed_orbit": AQUA, "C_withA + A + B (extra)": AQUA,
         "random": YELLOW, "ablation: C learned, A/B scripted": MAGENTA, "ablation: rule C, A + B learned": GREEN}
SHORT = {"DQN (Agent A)": "DQN (A)", "PPO (Agent B)": "PPO (B)", "A2C (Agent C)": "A2C (C)",
         "A2C (C_withA)": "A2C (C_withA)", "scripted_approach (upper bound)": "scripted approach\n(upper bound)",
         "rule_supervisor (Agent A inside)": "rule supervisor\n(A inside)", "rule_supervisor": "rule supervisor",
         "ALL LEARNED (C + A + B)": "all learned\nC+A+B", "ALL SCRIPTED (rule + raster + approach)": "all scripted",
         "ablation: C learned, A/B scripted": "C learned\nA/B scripted", "ablation: rule C, A + B learned":
         "rule C\nA+B learned", "C_withA + A + B (extra)": "C_withA\n+A+B", "raster_low": "raster low\n(dense)",
         "raster_mid": "raster mid", "fixed_orbit": "fixed orbit",
         "DQN (Agent A, real-image trained)": "DQN real-img", "PPO (Agent B, real-image trained)": "PPO real-img",
         "A2C (Agent C, real-image trained)": "A2C real-img"}
GROUP_NAME = {"A": "Agent A - DQN (navigate)", "B": "Agent B - PPO (refine)", "C": "Agent C - A2C (supervise)",
              "C_withA": "Agent C trained with A inside", "full": "Integration (full mode)"}
INDEX = []


def short(p):
    return SHORT.get(p, p)


def kfmt(*axes):
    """Environment-step axes as 0, 50k, 100k ... (avoids overlapping tick labels)."""
    from matplotlib.ticker import FuncFormatter
    for ax in axes:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:.0f}k" if v else "0"))


def bar_labels(ax, x, v, err, fmt="{:.2f}", fs=7.5):
    """Value labels placed beyond the error bar so they never collide with the cap."""
    for xi, vi, ei in zip(x, v, err):
        ei = 0 if np.isnan(ei) else ei
        ax.annotate(fmt.format(vi), (xi, vi + ei if vi >= 0 else vi - ei), xytext=(0, 3 if vi >= 0 else -11),
                    textcoords="offset points", ha="center", fontsize=fs, color=INK2)


def save(fig, name, title, sources):
    PLOTS.mkdir(exist_ok=True)
    p = PLOTS / f"{name}.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    INDEX.append({"file": f"plots/{p.name}", "title": title, "sources": "; ".join(sources)})
    print("saved", p.name)


def rd(name):
    return pd.read_csv(RESULTS / name)


def nsub(df):
    return f"n={int(df['n_episodes'].iloc[0])} episodes/policy, seeds {df['seeds'].iloc[0]}"


# ------------------------------------------------------------------ 1 learning curves
def p01_learning_curves():
    tc, ev = rd("training_curves.csv"), rd("eval_curves.csv")
    fig, axs = plt.subplots(3, 3, figsize=(13, 9))
    for j, (run, title) in enumerate([("A", "Agent A - DQN"), ("B", "Agent B - PPO"), ("C", "Agent C - A2C")]):
        t, e = tc[tc.run == run], ev[ev.run == run]
        ax = axs[0, j]
        ax.plot(t.timestep, t["return"], color=AXIS, lw=0.5, alpha=0.6, label="episode return (raw)")
        ax.fill_between(t.timestep, t.return_ma50 - t.return_std50, t.return_ma50 + t.return_std50, color=BLUE,
                        alpha=0.18, lw=0, label="±1 std (50 ep.)")
        ax.plot(t.timestep, t.return_ma50, color=BLUE, label="50-episode moving average")
        ax.errorbar(e.timestep, e.eval_return_mean, yerr=e.eval_return_std, color=ORANGE, marker="o", ms=4, lw=1.5,
                    capsize=2, label=f"eval return (n={int(e.n_eval_episodes.iloc[0])}, deterministic)")
        ax.set_title(f"{title}: return")
        ax.set_ylabel("return")
        ax = axs[1, j]
        ax.plot(t.timestep, t.success_ma50, color=BLUE, label="training success (50-ep. MA)")
        ax.plot(e.timestep, e.eval_success_rate, color=ORANGE, marker="o", ms=4, lw=1.5, label="eval success rate")
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("success rate")
        ax.set_title(f"{title}: success rate")
        ax = axs[2, j]
        ax.plot(t.timestep, t.length_ma50, color=BLUE, label="training (50-ep. MA)")
        ax.plot(e.timestep, e.eval_length_mean, color=ORANGE, marker="o", ms=4, lw=1.5, label="eval")
        ax.set_ylabel("episode length (decisions)" if run == "C" else "episode length (steps)")
        ax.set_xlabel("environment steps" if run != "C" else "supervisor decisions (env steps)")
        ax.set_title(f"{title}: episode length")
    for i in range(3):
        axs[i, 0].legend(loc="lower right" if i < 2 else "upper right")
    kfmt(*axs.ravel())
    fig.suptitle("Learning curves of the three main runs (training seed 0)", fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "01_learning_curves", "Learning curves: raw return, 50-episode moving average, ±1 std band, eval return, "
         "success rate and episode length", ["training_curves.csv", "eval_curves.csv"])


def p01b_multiseed_curves():
    ev = rd("eval_curves.csv")
    ms = ev[ev.run.str.startswith("ms_")]
    if ms.empty:
        return
    agents = sorted(ms.agent.unique())
    fig, axs = plt.subplots(1, len(agents), figsize=(4.5 * len(agents), 3.6), squeeze=False)
    for ax, ag in zip(axs[0], agents):
        runs = [ag] + sorted(ms[ms.agent == ag].run.unique())
        piv = ev[ev.run.isin(runs)].pivot_table(index="timestep", columns="run", values="eval_return_mean")
        for r in piv.columns:
            ax.plot(piv.index, piv[r], color=AXIS, lw=0.8)
        m, s = piv.mean(1), piv.std(1)
        ax.fill_between(piv.index, m - s, m + s, color=BLUE, alpha=0.2, lw=0, label="±1 std across seeds")
        ax.plot(piv.index, m, color=BLUE, label=f"mean of {piv.shape[1]} training seeds")
        ax.set_title(f"Agent {ag}: eval return across training seeds")
        ax.set_xlabel("environment steps")
        ax.set_ylabel("eval return")
        ax.legend(loc="lower right")
        kfmt(ax)
    fig.tight_layout()
    save(fig, "01b_learning_curves_multiseed", "Eval-return learning curves across training seeds (mean ± std)",
         ["eval_curves.csv"])


# ------------------------------------------------------------------ 2 vs baselines with 95% CI
def p02_vs_baselines():
    s = rd("summary_S0_all.csv")
    groups = ["A", "B", "C", "C_withA"]
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.8), gridspec_kw={"width_ratios": [4, 4, 3, 2]})
    for ax, g in zip(axs, groups):
        d = s[s.group == g]
        x = np.arange(len(d))
        ax.bar(x, d.return_mean, yerr=d.return_ci95, color=[COLOR.get(p, MUTED) for p in d.policy], width=0.65,
               capsize=3, ecolor=INK2, edgecolor=SURF, linewidth=2)
        bar_labels(ax, x, d.return_mean.values, d.return_ci95.values)
        ax.set_xticks(x, [short(p) for p in d.policy], fontsize=7.5)
        ax.margins(y=0.12)
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.set_title(GROUP_NAME[g])
        ax.set_ylabel("mean held-out return")
    fig.suptitle(f"Learned agents vs baselines on held-out bridges (error bars = 95% CI, {nsub(s)})",
                 fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "02_vs_baselines", "Mean held-out return with 95% CI: learned agents vs baselines",
         ["summary_S0_all.csv"])


# ------------------------------------------------------------------ 3 success + episode length
def p03_success_length():
    s = rd("summary_S0_all.csv")
    groups = ["A", "B", "C"]
    fig, axs = plt.subplots(2, 3, figsize=(13, 6.5))
    for j, g in enumerate(groups):
        d = s[s.group == g]
        x = np.arange(len(d))
        ax = axs[0, j]
        ax.bar(x, d.success_rate * 100, yerr=d.success_ci95 * 100, color=[COLOR.get(p, MUTED) for p in d.policy],
               width=0.65, capsize=3, ecolor=INK2, edgecolor=SURF, linewidth=2)
        bar_labels(ax, x, d.success_rate.values * 100, d.success_ci95.values * 100, fmt="{:.0f}%")
        ax.set_ylim(0, 120)
        ax.set_xticks(x, [short(p) for p in d.policy], fontsize=7.5)
        ax.set_ylabel("success rate (%)")
        ax.set_title(f"{GROUP_NAME[g]}: success rate")
        ax = axs[1, j]
        w = 0.36
        ax.bar(x - w / 2, d.steps_success_mean.fillna(0), w, color=GREEN, label="successful episodes",
               edgecolor=SURF, linewidth=2)
        ax.bar(x + w / 2, d.steps_fail_mean.fillna(0), w, color=RED, label="failed episodes", edgecolor=SURF,
               linewidth=2)
        for xi, (a, b) in enumerate(zip(d.steps_success_mean, d.steps_fail_mean)):
            if np.isnan(a):
                ax.annotate("no success", (xi - w / 2, 0), xytext=(0, 2), textcoords="offset points", ha="center",
                            fontsize=6.5, color=MUTED, rotation=90, va="bottom")
            if np.isnan(b):
                ax.annotate("no failure", (xi + w / 2, 0), xytext=(0, 2), textcoords="offset points", ha="center",
                            fontsize=6.5, color=MUTED, rotation=90, va="bottom")
        ax.set_xticks(x, [short(p) for p in d.policy], fontsize=7.5)
        ax.set_ylabel("decisions per episode" if g == "C" else "steps per episode")
        ax.set_title(f"{GROUP_NAME[g]}: episode length")
        ax.set_ylim(0, np.nanmax([d.steps_success_mean.max(), d.steps_fail_mean.max()]) * 1.3)
        ax.legend(loc="upper center", ncol=2)
    fig.suptitle(f"Success rate (95% CI) and episode length of successful vs failed episodes ({nsub(s)})",
                 fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "03_success_and_length", "Success rate and episode length (successful vs failed episodes separately)",
         ["summary_S0_all.csv"])


# ------------------------------------------------------------------ 4 exploration schedules
def p04_exploration():
    tb = rd("tb_scalars.csv")
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
    ax = axs[0]
    for run, c in [("A_eps0.1", BLUE), ("A_eps0.3", ORANGE), ("A_eps0.6", AQUA), ("A", MAGENTA)]:
        d = tb[(tb.run == run) & (tb.tag == "rollout/exploration_rate")]
        ax.plot(d.step, d.value, color=c, label=f"{run} ({'main, 200k' if run == 'A' else 'short run, 150k'})")
    ax.set_title("Agent A (DQN): ε-greedy schedule")
    ax.set_xlabel("environment steps")
    ax.set_ylabel("exploration rate ε")
    ax.legend()
    for ax, runs, title in [(axs[1], [("B", BLUE), ("B_ent0.0", ORANGE), ("B_ent0.01", AQUA), ("B_ent0.05", YELLOW),
                                      ("B_clip0.1", MAGENTA), ("B_clip0.3", GREEN)], "Agent B (PPO): policy entropy"),
                            (axs[2], [("C", BLUE), ("C_ent0.0", ORANGE), ("C_ent0.01", AQUA), ("C_ent0.05", YELLOW)],
                             "Agent C (A2C): policy entropy")]:
        for run, c in runs:
            d = tb[(tb.run == run) & (tb.tag == "train/entropy_loss")]
            ax.plot(d.step, -d.value, color=c, lw=1.5, label=run)
        ax.set_title(title)
        ax.set_xlabel("environment steps")
        ax.set_ylabel("entropy  (= −entropy_loss)")
        ax.legend(ncol=2)
    kfmt(*axs)
    fig.suptitle("Exploration schedules (TensorBoard logs)", fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "04_exploration_schedules", "Exploration: ε schedule (DQN) and policy entropy (PPO, A2C)",
         ["tb_scalars.csv"])


# ------------------------------------------------------------------ 5 box plots
def p05_boxplots():
    groups = ["A", "B", "C", "full"]
    fig, axs = plt.subplots(1, 4, figsize=(16, 4), gridspec_kw={"width_ratios": [4, 4, 3, 5]})
    for ax, g in zip(axs, groups):
        e = pd.read_csv(RESULTS / f"episodes_{g}_S0_nominal.csv")
        pols = list(dict.fromkeys(e.policy))
        data = [e[e.policy == p]["return"].values for p in pols]
        bp = ax.boxplot(data, patch_artist=True, widths=0.55, medianprops={"color": INK, "lw": 1.5},
                        flierprops={"marker": "o", "ms": 3, "mfc": MUTED, "mec": "none"},
                        whiskerprops={"color": INK2}, capprops={"color": INK2})
        for patch, p in zip(bp["boxes"], pols):
            patch.set_facecolor(COLOR.get(p, MUTED))
            patch.set_alpha(0.75)
            patch.set_edgecolor(INK2)
        ax.set_xticks(range(1, len(pols) + 1), [short(p) for p in pols], fontsize=7)
        ax.set_title(GROUP_NAME[g])
        ax.set_ylabel("episode return")
    fig.suptitle("Distribution of held-out episode returns (100 episodes per policy, seeds 2000-2099)",
                 fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "05_boxplots_return", "Box plots of held-out episode return per policy",
         [f"episodes_{g}_S0_nominal.csv" for g in groups])


def p05b_boxplot_seeds():
    files = sorted(RESULTS.glob("multiseed_episodes_*_seed*.csv"))
    if not files:
        return
    e = pd.concat([pd.read_csv(f) for f in files])
    agents = sorted(e.agent.unique())
    fig, axs = plt.subplots(1, len(agents), figsize=(4.5 * len(agents), 3.8), squeeze=False)
    for ax, ag in zip(axs[0], agents):
        d = e[e.agent == ag]
        seeds = sorted(d.train_seed.unique())
        data = [d[d.train_seed == s]["return"].values for s in seeds]
        bp = ax.boxplot(data, patch_artist=True, widths=0.55, medianprops={"color": INK, "lw": 1.5},
                        flierprops={"marker": "o", "ms": 3, "mfc": MUTED, "mec": "none"})
        for patch, s in zip(bp["boxes"], seeds):
            patch.set_facecolor(ORANGE if s == 0 else BLUE)
            patch.set_alpha(0.75)
        ax.plot(range(1, len(seeds) + 1), [x.mean() for x in data], "D", color=INK, ms=5, label="mean")
        ax.set_xticks(range(1, len(seeds) + 1), [f"seed {s}" + (" (orig.)" if s == 0 else "") for s in seeds],
                      fontsize=8)
        ax.set_title(f"Agent {ag}: held-out return per training seed")
        ax.set_ylabel("episode return")
        ax.legend(loc="lower right")
    fig.suptitle("Variability across training seeds (100 held-out episodes per seed)", fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "05b_boxplot_training_seeds", "Box plot of held-out return per training seed",
         ["multiseed_episodes_*.csv"])


# ------------------------------------------------------------------ 6 robustness gap
def p06_robustness():
    g = rd("robustness_gap.csv")
    pols = ["DQN (Agent A)", "raster_low", "PPO (Agent B)", "A2C (Agent C)", "rule_supervisor",
            "ALL LEARNED (C + A + B)"]
    g = g[g.policy.isin(pols) & g.group.isin(["A", "B", "C", "full"])]
    scen = list(dict.fromkeys(g.scenario))
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.2))
    w = 0.8 / len(pols)
    for k, p in enumerate(pols):
        d = g[g.policy == p].set_index("scenario").reindex(scen)
        col = {"raster_low": ORANGE, "rule_supervisor": YELLOW, "ALL LEARNED (C + A + B)": GREEN,
               "PPO (Agent B)": AQUA, "A2C (Agent C)": MAGENTA}.get(p, BLUE)
        x = np.arange(len(scen)) + (k - len(pols) / 2 + 0.5) * w
        axs[0].bar(x, 100 * d.return_gap / d.return_nominal.abs(), w, color=col, label=short(p).replace("\n", " "),
                   edgecolor=SURF, linewidth=1)
        axs[1].bar(x, 100 * d.success_gap, w, color=col, edgecolor=SURF, linewidth=1)
    for ax, yl in [(axs[0], "return gap  (nominal − perturbed) / |nominal|  (%)"),
                   (axs[1], "success gap  (nominal − perturbed)  (percentage points)")]:
        ax.set_xticks(range(len(scen)), [s.replace("_", "\n", 1) for s in scen], fontsize=8)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_ylabel(yl)
    axs[0].set_title("Return drop per scenario (positive = worse than nominal)")
    axs[1].set_title("Success-rate drop per scenario")
    axs[0].legend(ncol=2, fontsize=7)
    fig.suptitle("Robustness gap: nominal S0 vs perturbed scenarios (100 held-out episodes each, seeds 2000-2099)",
                 fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "06_robustness_gap", "Robustness gap (return and success) per scenario", ["robustness_gap.csv"])


# ------------------------------------------------------------------ 7 visitation heatmaps
def p07_visitation():
    from matplotlib.colors import LinearSegmentedColormap
    z = np.load(RESULTS / "visitation_A_S0.npz")
    names, visits = list(z["names"]), z["visits"]
    s = rd("summary_A.csv").set_index("policy")
    show = ["DQN (Agent A)", "raster_low", "raster_mid", "random"]
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    fig, axs = plt.subplots(2, 2, figsize=(12, 5.6))
    vmax = max(visits[names.index(p)].sum(0).max() for p in show)
    for ax, p in zip(axs.ravel(), show):
        m = visits[names.index(p)].sum(0)
        im = ax.imshow(m, cmap=cmap, vmin=0, vmax=vmax, aspect="equal")
        ax.grid(False)
        ax.set_xticks(range(0, 16, 3))
        ax.set_yticks(range(8))
        ax.set_xlabel("column")
        ax.set_ylabel("row")
        r = s.loc[p]
        ax.set_title(f"{short(p).replace(chr(10), ' ')}\nposition coverage {r.pos_coverage_mean * 100:.0f}%, "
                     f"pose coverage {r.pose_coverage_mean * 100:.0f}%, visit entropy {r.visit_entropy_mean:.2f}",
                     fontsize=9)
    fig.colorbar(im, ax=axs, shrink=0.8, label="mean visits per episode (summed over altitudes)")
    fig.suptitle("Agent A state visitation on the 8×16 bridge (mean over 100 held-out episodes)", fontweight="bold",
                 color=INK)
    save(fig, "07_visitation_heatmaps", "Agent A state-visitation heatmaps: DQN vs raster vs random",
         ["visitation_A_S0.npz", "summary_A.csv"])


# ------------------------------------------------------------------ 8 option frequencies
def p08_options():
    s = rd("scenario_summary.csv")
    rows = s[(s.group.isin(["C", "C_withA", "full"])) & (s.scenario == "S0_nominal")]
    rows = pd.concat([rows, s[(s.scenario == "S1_learned") & (s.policy == "A2C (Agent C, real-image trained)")]])
    rows = rows.drop_duplicates("policy")
    lab = [short(p).replace("\n", " ") + (" [S1]" if sc == "S1_learned" else "")
           for p, sc in zip(rows.policy, rows.scenario)]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    y = np.arange(len(rows))
    left = np.zeros(len(rows))
    for col, name, c in [("freq_explore", "EXPLORE (→ Agent A)", BLUE), ("freq_refine", "REFINE (→ Agent B)", ORANGE),
                         ("freq_return", "RETURN home", AQUA)]:
        v = rows[col].values * 100
        ax.barh(y, v, left=left, color=c, label=name, edgecolor=SURF, linewidth=2, height=0.65)
        for yi, (l, vi) in enumerate(zip(left, v)):
            if vi >= 4:
                ax.text(l + vi / 2, yi, f"{vi:.0f}%", ha="center", va="center", fontsize=7.5,
                        color="#0b0b0b" if c == YELLOW else "#ffffff")
        left += v
    ax.set_yticks(y, lab, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of supervisor decisions (%)")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    ax.set_title("Option-selection frequency per supervisor (bandit-style arm frequency), 100 held-out episodes",
                 pad=28)
    fig.tight_layout()
    save(fig, "08_option_frequency", "Agent C option-selection frequency (EXPLORE / REFINE / RETURN) per supervisor",
         ["scenario_summary.csv"])


# ------------------------------------------------------------------ 9 diagnostics grid
def p09_diagnostics():
    tb = rd("tb_scalars.csv")
    panels = [("B", "train/policy_gradient_loss", "PPO policy-gradient loss"), ("B", "train/value_loss", "PPO value loss"),
              ("B", "train/entropy_loss", "PPO entropy (−entropy_loss)"), ("B", "train/approx_kl", "PPO approx. KL"),
              ("B", "train/clip_fraction", "PPO clip fraction"), ("B", "train/explained_variance", "PPO explained variance"),
              ("C", "train/policy_loss", "A2C policy loss"), ("C", "train/value_loss", "A2C value loss"),
              ("C", "train/entropy_loss", "A2C entropy (−entropy_loss)"), ("C", "train/explained_variance",
                                                                          "A2C explained variance"),
              ("A", "train/loss", "DQN loss (TD-error proxy, Huber)"), ("A", "rollout/exploration_rate", "DQN ε")]
    fig, axs = plt.subplots(3, 4, figsize=(15, 8.5))
    for ax, (run, tag, title) in zip(axs.ravel(), panels):
        d = tb[(tb.run == run) & (tb.tag == tag)]
        v = -d.value if "entropy_loss" in tag else d.value
        ax.plot(d.step, v, color=BLUE, lw=1, alpha=0.45 if len(d) > 30 else 1)
        if len(d) > 10 and "exploration" not in tag:
            ax.plot(d.step, pd.Series(v.values).rolling(10, min_periods=1, center=True).mean(), color=BLUE, lw=2,
                    label="10-point centred moving average")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("environment steps")
    axs[0, 0].legend()
    kfmt(*axs.ravel())
    fig.suptitle("Training diagnostics from TensorBoard (main runs) - training health, NOT task performance",
                 fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "09_training_diagnostics", "PPO / A2C / DQN training diagnostics (loss, entropy, KL, clip fraction, "
         "explained variance)", ["tb_scalars.csv"])


# ------------------------------------------------------------------ 10 integration ablation
def p10_integration():
    s = rd("summary_full.csv")
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for _, r in s.iterrows():
        ax.errorbar(r.energy_used_mean, r.defects_found_mean, xerr=r.energy_used_ci95, yerr=r.defects_found_ci95,
                    fmt="o", ms=9, color=COLOR.get(r.policy, MUTED), capsize=3, mec=SURF, mew=2,
                    label=short(r.policy).replace("\n", " ") + f"  (return {r.return_mean:.1f})")
    ax.set_xlabel("energy used per mission (fraction of battery)")
    ax.set_ylabel("defects found per mission (of 26)")
    ax.set_title(f"Integration ablation: energy vs defects found (95% CI, {nsub(s)})")
    ax.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    save(fig, "10_integration_ablation", "Integration ablation: energy used vs defects found per configuration",
         ["summary_full.csv"])


# ------------------------------------------------------------------ 11 regret
def p11_regret():
    s = rd("summary_S0_all.csv")
    learned = ["DQN (Agent A)", "PPO (Agent B)", "A2C (Agent C)", "A2C (C_withA)", "ALL LEARNED (C + A + B)"]
    d = s[s.policy.isin(learned)].drop_duplicates("policy")
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(d))
    ax.bar(x, d.regret_mean, yerr=d.regret_ci95, color=BLUE, capsize=4, ecolor=INK2, width=0.6)
    bar_labels(ax, x, d.regret_mean.values, d.regret_ci95.values, fs=8)
    ax.set_xticks(x, [f"{short(p).replace(chr(10), ' ')}\nvs {short(b).replace(chr(10), ' ')}"
                      for p, b in zip(d.policy, d.benchmark)], fontsize=7.5)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylabel("mean regret per episode\n(benchmark return − agent return)")
    ax.set_title(f"Regret vs the stated benchmark, paired on identical bridges (95% CI, {nsub(s)})")
    fig.tight_layout()
    save(fig, "11_regret", "Mean per-episode regret of each learned agent vs its benchmark", ["summary_S0_all.csv"])


# ------------------------------------------------------------------ 12 sensitivity
def p12_sensitivity():
    s = rd("sensitivity.csv")
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, ag, title in [(axs[0], "A", "Agent A: ε-decay fraction"), (axs[1], "B", "Agent B: entropy coef / clip range"),
                          (axs[2], "C", "Agent C: entropy coefficient")]:
        d = s[s.agent == ag]
        x = np.arange(len(d))
        for xi, (v, e, sr) in enumerate(zip(d.heldout_return, d.heldout_return_ci95, d.short_run)):
            ax.errorbar(xi, v, yerr=e, fmt="o", ms=9, capsize=4, color=BLUE, ecolor=INK2,
                        mfc=SURF if sr else BLUE, mec=BLUE, mew=2,
                        label=("short run" if sr else "main run") if xi in (0, len(d) - 1) else None)
        for xi, (v, e, sc) in enumerate(zip(d.heldout_return, d.heldout_return_ci95, d.heldout_success)):
            ax.annotate(f"{v:.2f}\n{sc * 100:.0f}% succ.", (xi, v + e), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=7, color=INK2)
        ax.set_xticks(x, [f"{r}\n({int(st / 1000)}k{' short' if sr else ' main'})"
                          for r, st, sr in zip(d.run, d.steps_trained, d.short_run)], fontsize=7)
        lo = min(d.heldout_return - d.heldout_return_ci95)
        hi = max(d.heldout_return + d.heldout_return_ci95)
        ax.set_ylim(lo - 0.3 * (hi - lo) - 0.05, hi + 0.45 * (hi - lo) + 0.1)
        ax.set_ylabel("mean held-out return (95% CI)")
        ax.set_xlim(-0.6, len(d) - 0.4)
        ax.set_title(title)
        ax.legend(loc="lower left")
    fig.suptitle("Hyperparameter sensitivity (CO5) on held-out seeds 2000-2099 (n=100 each; hollow = short run)",
                 fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "12_sensitivity", "Hyperparameter sensitivity of the three agents (held-out return, 95% CI)",
         ["sensitivity.csv"])


# ------------------------------------------------------------------ 13 real-image results
def p13_real_images():
    s = rd("scenario_summary.csv")
    s1 = s[s.scenario.isin(["S0_nominal", "S1_learned"])]
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, g in zip(axs, ["A", "B", "C"]):
        d = s1[s1.group == g]
        pols = list(dict.fromkeys(d.policy))
        piv = d.pivot_table(index="policy", columns="scenario", values="success_rate").reindex(pols)
        x = np.arange(len(pols))
        ax.bar(x - 0.2, piv["S0_nominal"].fillna(0) * 100, 0.4, color=BLUE, label="S0 analytic detector")
        ax.bar(x + 0.2, piv["S1_learned"] * 100, 0.4, color=VIOLET, label="S1 real-image detector")
        for xi, v in enumerate(piv["S1_learned"]):
            ax.annotate(f"{v * 100:.0f}%", (xi + 0.2, v * 100), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=7, color=INK2)
        ax.set_xticks(x, [short(p) for p in pols], fontsize=7)
        ax.set_ylim(0, 115)
        ax.set_ylabel("success rate (%)")
        ax.set_title(GROUP_NAME[g])
    axs[0].legend(loc="upper right")
    fig.suptitle("Analytic vs real-image (DeepCrack) detector: success rate on held-out seeds (n=100 each)",
                 fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "13_real_image_success", "Success rate with the analytic vs real-image detector", ["scenario_summary.csv"])


def main():
    for f in [p01_learning_curves, p01b_multiseed_curves, p02_vs_baselines, p03_success_length, p04_exploration,
              p05_boxplots, p05b_boxplot_seeds, p06_robustness, p07_visitation, p08_options, p09_diagnostics,
              p10_integration, p11_regret, p12_sensitivity, p13_real_images]:
        f()
    pd.DataFrame(INDEX).to_csv(PLOTS / "plot_index.csv", index=False)
    print(f"{len(INDEX)} plots")


if __name__ == "__main__":
    main()
