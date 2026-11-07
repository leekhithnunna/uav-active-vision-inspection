"""Phase 8 - builds the submission document from the result files.

Every number in the text is produced by num(), which reads it from a result file (or a module constant)
and records where it came from in report/report_numbers.csv; tests/test_eval.py re-reads every entry and
checks it. Derived values (percent changes etc.) are computed here first and written to
results/derived_metrics.csv, then cited from that file.

Outputs: report/Team08_Evaluation_Metrics.md, .docx, .pdf (if Word is available),
         report/report_numbers.csv, videos/video_index.md, videos/narration_script.md
The report builds with or without the optional multi-seed results.
"""
from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import EM, ROOT, RESULTS, REPORT, VIDEOS, PLOTS, SCENARIOS, CACHE, HELD_SEED0, HELD_N, METHOD, git_rev  # noqa: E402
import common  # noqa: E402,F401  (used by eval() of 'py:' sources)
import env.inspection_env as ie  # noqa: E402,F401

TITLE = "Evaluation Metrics: Uncertainty-Aware Active Vision for UAV-Based Infrastructure Inspection Using Deep RL"

# ====================================================================== number registry
REG: list[dict] = []
_cache: dict = {}


def _df(src):
    if src not in _cache:
        p = (EM / src) if not src.startswith("repo:") else ROOT / src[5:]
        _cache[src] = pd.read_csv(p)
    return _cache[src]


def lookup(src, col, agg=None, **key):
    """Value of column `col` in file `src` for the row(s) matching key (agg reduces several rows)."""
    if src.startswith("py:"):
        return float(eval(src[3:], {**vars(common), "ie": ie, "np": np, "json": json}))
    d = _df(src)
    for k, v in key.items():
        d = d[d[k].astype(str) == str(v)]
    x = d[col].astype(float)
    if agg is None:
        assert len(x) == 1, f"{src} {col} {key}: {len(x)} rows"
        return float(x.iloc[0])
    return float({"max": x.max, "min": x.min, "mean": x.mean, "sum": x.sum, "count": x.count,
                  "last": lambda: x.iloc[-1], "first": lambda: x.iloc[0],
                  "tail10mean": lambda: x.iloc[-10:].mean(), "head10mean": lambda: x.iloc[:10].mean()}[agg]())


def num(src, col, fmt=".2f", scale=1.0, *, agg=None, suffix="", **key):
    v = lookup(src, col, agg, **key) * scale
    s = format(v, fmt)
    REG.append({"text": s, "source": src, "column": col, "fmt": fmt, "scale": scale, "agg": agg or "",
                "key": json.dumps(key), "value": v})
    return s + suffix


def pct(src, col, fmt=".0f", **key):
    return num(src, col, fmt, 100.0, suffix="%", **key)


# shorthand sources
S0, SC, RG = "results/summary_S0_all.csv", "results/scenario_summary.csv", "results/robustness_gap.csv"
SE, SENS, MSZ, DER = ("results/sample_efficiency.csv", "results/sensitivity.csv", "results/model_size.csv",
                      "results/derived_metrics.csv")
TB, VAL, MSS, MSP = "results/tb_scalars.csv", "results/validation.csv", "results/multiseed_summary.csv", \
    "results/multiseed_per_seed.csv"
SA = "results/summary_A.csv"
HP_A = "py:json.load(open(RUN_DIRS['A'] / 'hyperparams.json'))['exploration_initial_eps']"
HP_AF = "py:json.load(open(RUN_DIRS['A'] / 'hyperparams.json'))['exploration_fraction']"
NOISE1 = "py:SCENARIOS['S3_noise0.05']['noise']"
NOISE2 = "py:SCENARIOS['S3_noise0.10']['noise']"


def cpu_name():
    try:
        import winreg
        k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
    except Exception:
        return platform.processor() or platform.machine()


def s0(group, policy, col, fmt=".2f"):
    return num(S0, col, fmt, group=group, policy=policy)


def s0p(group, policy, col, fmt=".0f"):
    return num(S0, col, fmt, 100.0, suffix="%", group=group, policy=policy)


def sc(scen, group, policy, col, fmt=".2f", scale=1.0, suffix=""):
    return num(SC, col, fmt, scale, suffix=suffix, scenario=scen, group=group, policy=policy)


def der(name, fmt=".0f", suffix=""):
    return num(DER, "value", fmt, suffix=suffix, name=name)


# <<NAME>> placeholders in the prose -> registered numbers (method settings, env constants, result columns)
PH = {
    "CI": ("py:METHOD['ci_level']", "", ".0f", {}), "MA": ("py:METHOD['ma_window']", "", ".0f", {}),
    "BAND": ("py:METHOD['band_std']", "", ".0f", {}), "DIAG": ("py:METHOD['diag_window']", "", ".0f", {}),
    "OOBPER": ("py:METHOD['oob_per']", "", ".0f", {}), "LATN": ("py:METHOD['latency_calls']", "", ".0f", {}),
    "HELDN": ("py:HELD_N", "", ".0f", {}), "HELD0": ("py:HELD_SEED0", "", ".0f", {}),
    "HELD1": ("py:HELD_SEED0 + HELD_N - 1", "", ".0f", {}), "ORIG0": ("py:ORIG_SEED0", "", ".0f", {}),
    "ORIGN": ("py:ORIG_N", "", ".0f", {}), "H": ("py:ie.H", "", ".0f", {}), "W": ("py:ie.W", "", ".0f", {}),
    "MAXNAV": ("py:ie.MAX_STEPS['navigate']", "", ".0f", {}), "MAXREF": ("py:ie.MAX_STEPS['refine']", "", ".0f", {}),
    "MAXSUP": ("py:ie.MAX_STEPS['supervise']", "", ".0f", {}),
    "BAT": ("py:SCENARIOS['S4_battery0.6']['start_battery'] * 100", "", ".0f", {}),
    "NDEF": ("results/episodes_full_S0_nominal.csv", "defects_total", ".0f",
             {"seed": 2000, "policy": "ALL LEARNED (C + A + B)"}),
    "NMET": ("results/derived_metrics.csv", "value", ".0f", {"name": "validation_metrics"}),
}


def resolve(text):
    """Replace every <<NAME>> placeholder by its registered value."""
    def sub(m):
        src, col, fmt, key = PH[m.group(1)]
        return num(src, col, fmt, **key)
    return re.sub(r"<<([A-Z0-9]+)>>", sub, text) if isinstance(text, str) else text


# ====================================================================== derived metrics (written to CSV first)
def derive():
    s = pd.read_csv(RESULTS / "summary_S0_all.csv").set_index(["group", "policy"])
    rows = []

    def add(name, value, formula):
        rows.append({"name": name, "value": float(value), "formula": formula})

    def pchg(a, b):
        return 100.0 * (a - b) / b

    A, R = s.loc[("A", "DQN (Agent A)")], s.loc[("A", "raster_low")]
    add("A_steps_vs_raster_pct", pchg(A.steps_success_mean, R.steps_success_mean),
        "100*(DQN steps_success_mean - raster_low steps_success_mean)/raster_low")
    add("A_energy_vs_raster_pct", pchg(A.energy_used_mean, R.energy_used_mean),
        "100*(DQN energy - raster_low energy)/raster_low energy")
    C, RU = s.loc[("C", "A2C (Agent C)")], s.loc[("C", "rule_supervisor")]
    add("C_energy_vs_rule_pct", pchg(C.energy_used_mean, RU.energy_used_mean), "100*(A2C energy - rule energy)/rule")
    add("C_decisions_vs_rule_pct", pchg(C.steps_mean, RU.steps_mean), "100*(A2C decisions - rule decisions)/rule")
    add("C_return_vs_rule_pct", pchg(C.return_mean, RU.return_mean), "100*(A2C return - rule return)/rule")
    L, SCR = s.loc[("full", "ALL LEARNED (C + A + B)")], s.loc[("full", "ALL SCRIPTED (rule + raster + approach)")]
    add("full_energy_learned_vs_scripted_pct", pchg(L.energy_used_mean, SCR.energy_used_mean),
        "100*(all-learned energy - all-scripted energy)/all-scripted")
    add("full_decisions_learned_vs_scripted_pct", pchg(L.steps_mean, SCR.steps_mean),
        "100*(all-learned decisions - all-scripted decisions)/all-scripted")
    add("full_defects_learned_minus_scripted", L.defects_found_mean - SCR.defects_found_mean,
        "all-learned defects - all-scripted defects")
    WA, RA = s.loc[("C_withA", "A2C (C_withA)")], s.loc[("C_withA", "rule_supervisor (Agent A inside)")]
    add("CwA_energy_vs_rule_pct", pchg(WA.energy_used_mean, RA.energy_used_mean), "100*(C_withA energy - rule energy)/rule")
    v = pd.read_csv(RESULTS / "validation.csv")
    add("validation_checks", len(v), "rows in validation.csv")
    add("validation_pass", int(v.within_1pct.sum()), "rows within 1%")
    add("validation_policies", v.groupby(["run", "policy"]).ngroups, "distinct (run, policy)")
    add("validation_metrics", v.metric.nunique(), "distinct metrics compared per policy")
    add("validation_max_rel_pct", 100 * v.rel_diff.max(), "max relative difference (%)")
    add("validation_files", v.source.nunique(), "distinct saved result files checked")
    tb = pd.read_csv(RESULTS / "tb_scalars.csv")
    add("tb_scalars", len(tb), "TensorBoard scalar points read")
    tc = pd.read_csv(RESULTS / "training_curves.csv")
    add("training_episodes", len(tc), "training episodes read from monitor files")
    sk = pd.read_csv(RESULTS / "scenario_summary.csv")
    add("scenario_rows", len(sk), "rows in scenario_summary.csv")
    add("n_scenarios", sk.scenario.nunique(), "scenarios evaluated (incl. S0)")
    eps = sum(len(pd.read_csv(p)) for p in RESULTS.glob("episodes_*.csv"))
    add("total_episodes", eps, "episodes in all episodes_*.csv files")
    ms = RESULTS / "multiseed_episodes_C_seed1.csv"
    if ms.exists():
        for f in sorted(RESULTS.glob("multiseed_episodes_C_seed*.csv")):
            e = pd.read_csv(f)
            n = e[["n_explore", "n_refine", "n_return"]].sum()
            add(f"C_seed{int(e.train_seed.iloc[0])}_refine_pct", 100 * n.n_refine / n.sum(),
                f"REFINE share of decisions, {f.name}")
    pd.DataFrame(rows).to_csv(RESULTS / "derived_metrics.csv", index=False)


# ====================================================================== document blocks
B: list[tuple] = []
FIG = {"n": 0}
TAB = {"n": 0}


def h(level, text):
    B.append(("h", level, resolve(text)))


def p(text):
    B.append(("p", resolve(text)))


def bullets(items):
    B.append(("ul", [resolve(x) for x in items]))


def formula(text):
    B.append(("formula", resolve(text)))


def table(header, rows, caption):
    TAB["n"] += 1
    B.append(("table", [resolve(x) for x in header], [[resolve(str(c)) for c in r] for r in rows],
              f"Table {TAB['n']}. {resolve(caption)}"))


def fig(name, caption):
    path = PLOTS / f"{name}.png"
    if not path.exists():
        return
    FIG["n"] += 1
    B.append(("img", f"plots/{name}.png", f"Figure {FIG['n']}. {resolve(caption)}"))


def qblock(question, metric, formula_text):
    p(f"**Question.** {question}")
    p(f"**Metric.** {metric}")
    formula(formula_text)


def videos_ref(ids):
    vf = VIDEOS / "video_files.csv"
    if not vf.exists():
        return
    v = pd.read_csv(vf).set_index("clip")
    have = [i for i in ids if i in v.index]
    if have:
        p("**Video evidence.** " + "; ".join(f"`videos/{i}.mp4` ({v.loc[i, 'scenario']})" for i in have) + ".")


def nfoot(df_src, **key):
    return (f"n = {num(df_src, 'n_episodes', '.0f', agg='first', **key)} episodes per policy, held-out seeds "
            "<<HELD0>>-<<HELD1>>")


# ====================================================================== content
GROUP_ORDER = ["A", "B", "C", "C_withA", "full"]
GTITLE = {"A": "Agent A (DQN, navigate)", "B": "Agent B (PPO, refine)", "C": "Agent C (A2C, supervise)",
          "C_withA": "Agent C trained with Agent A inside", "full": "Integration (full mode)"}


def policies(group, src=S0, scen=None):
    d = _df(src)
    d = d[d.group == group] if scen is None else d[(d.group == group) & (d.scenario == scen)]
    return list(dict.fromkeys(d.policy))


def sec_title():
    h(1, TITLE)
    p("**Team 08** - 22AIE401 Reinforcement Learning - Yeseswini (Agent A, DQN), Leekhith (Agent B, PPO; "
      "integration and metrics), Mukhesh (Agent C, A2C; DeepCrack detector).")
    p(f"Generated by `Evaluation_Metrics/scripts/build_report.py` from the result files at git revision "
      f"`{git_rev()}`. Every number in this document is read from a result file by code "
      f"(see `report/report_numbers.csv`).")
    h(2, "1. Project summary")
    p("A drone inspects an <<H>> x <<W>> grid of bridge-surface cells for cracks. Three cooperating agents share one "
      "Gymnasium environment: **Agent A (DQN)** decides where to fly and at which altitude, **Agent B (PPO)** "
      "refines the view of one suspicious cell with continuous distance / angle / offset actions, and "
      "**Agent C (A2C)** is a mission supervisor that chooses between the options EXPLORE (run Agent A), "
      "REFINE (run Agent B) and RETURN home. All decisions are driven by the per-cell uncertainty of a crack "
      "detector: an analytic formula, or a real-image U-Net with MC-dropout trained on DeepCrack. This document "
      "answers ten evaluation questions with metrics computed on held-out bridges, in several varying task "
      "environments, and points to recorded sessions (videos) for each insight.")


def sec_protocol():
    h(2, "2. Evaluation protocol")
    bullets([
        f"**Held-out bridges:** seeds <<HELD0>>-<<HELD1>> (<<HELDN>> episodes per policy and "
        f"scenario). Training used unseeded vectorised environments and the EvalCallback used seeds from <<ORIG0>>, so "
        f"these bridges were never seen during training or model selection. Every policy runs on the *same* "
        f"bridges, so comparisons are paired.",
        "**Policies:** the saved `best/best_model.zip` of every run, `deterministic=True`. Baselines are the "
        "scripted policies in `uav_inspection/env/baselines.py` (raster scans, fixed orbit, scripted approach "
        "upper bound, rule supervisor, random).",
        f"**Validation first:** before any new evaluation, every saved result was re-computed on the original "
        f"seeds with `baselines.evaluate()`: {der('validation_pass')} of {der('validation_checks')} checks "
        f"({der('validation_policies')} policies x <<NMET>> metrics, {der('validation_files')} saved files) matched "
        f"within 1%; the largest relative difference was {der('validation_max_rel_pct', '.2f', '%')} "
        "(`results/validation.csv`).",
        f"**Detectors:** analytic (default, success threshold mean uncertainty <= "
        f"{num('py:ie.InspectionEnv().success_unc', '', '.2f')}) and real-image DeepCrack cache (calibrated "
        f"threshold {num('py:np.load(CACHE)[\"success_unc\"]', '', '.3f')}).",
        "**Training seeds:** every original run used training seed 0. Extra seeds (1, 2, 3) were trained for the "
        "multi-seed analysis in Q5 where available, with identical hyperparameters (`multiseed_runs/`).",
        f"**Hardware:** {cpu_name()}, CPU only, Python `{platform.python_version()}`, "
        "PyTorch CPU build, Stable-Baselines3. Inference latency and the multi-seed training times are laptop "
        "measurements; the original training times were measured on Google Colab.",
        f"**Scale:** {der('total_episodes')} evaluation episodes in total over {der('n_scenarios')} scenarios.",
    ])
    p("**What counts as success** (design contract v1.1):")
    table(["Agent", "Episode limit", "Success"],
          [["A (navigate)", "<<MAXNAV>> steps", f"coverage >= {num('py:ie.InspectionEnv().success_cov', '', '.2f')} and "
            f"mean uncertainty <= threshold"],
           ["B (refine)", "<<MAXREF>> steps", "best view reached (view quality q = 4)"],
           ["C (supervise / full)", "<<MAXSUP>> decisions", "RETURN chosen with battery > 0 (safe return)"]],
          "Success definitions per agent.")


def sec_metrics():
    h(2, "3. Metric definitions")
    defs = [
        ("Episode return", "G = sum_t r_t  (undiscounted sum of rewards in one episode)"),
        ("Moving-average return", "MA_k(i) = (1/k) * sum_{j=i-k+1..i} G_j ,  k = <<MA>> training episodes; band = ±<<BAND>> std of the same window"),
        ("Success rate", "SR = (1/N) * sum_i 1[episode i successful]"),
        ("Episode length", "L = number of steps (A, B) or supervisor decisions (C); reported separately for successful and failed episodes"),
        ("Time to goal", "mean number of low-level environment steps until success, over successful episodes only"),
        ("Regret (per episode)", "regret_i = G_i(benchmark) - G_i(agent) on the same bridge i;  mean and cumulative sum over i"),
        ("Standard deviation", "s = sqrt( sum_i (x_i - mean)^2 / (N-1) )"),
        ("<<CI>>% confidence interval", "mean ± t_{0.975, N-1} * s / sqrt(N)   (Student t)"),
        ("Robustness gap", "gap = metric(S0 nominal) - metric(perturbed scenario); positive = worse when perturbed"),
        ("Constraint-violation rate", "battery-empty rate = (1/N) * sum_i 1[battery_i <= 0];  OOB rate = <<OOBPER>> * (out-of-bounds moves) / (low-level steps)"),
        ("Coverage", "fraction of the 128 cells seen at least once (task coverage);  position coverage = visited (row, col) / 128;  pose coverage = visited (altitude, row, col) / 384"),
        ("Visitation entropy", "H = -sum_c p_c log p_c / log(128), p_c = share of visits to cell c (1 = perfectly uniform)"),
        ("Sample efficiency", "environment steps until the evaluation success rate first reaches 80% / 100%"),
        ("Option frequency", "share of supervisor decisions given to each option (bandit-style arm-selection frequency)"),
    ]
    table(["Metric", "Definition / formula"], [[a, f"`{b}`"] for a, b in defs], "Metric definitions.")


def tab_success(groups):
    rows = []
    for g in groups:
        for pol in policies(g):
            rows.append([GTITLE[g], pol, s0p(g, pol, "success_rate"),
                         f"±{num(S0, 'success_ci95', '.0f', 100.0, group=g, policy=pol)} pp",
                         s0(g, pol, "n_success", ".0f")])
    return rows


def sec_q1():
    h(2, "Q1. Does each agent complete its task?")
    qblock("Does each agent complete its task on unseen bridges?", "Success rate (with <<CI>>% CI).",
           "SR = (1/N) * sum_i 1[success_i]")
    table(["Group", "Policy", "Success rate", "<<CI>>% CI", "Successful episodes"], tab_success(GROUP_ORDER),
          f"Success rate on held-out bridges ({nfoot(S0)}).")
    fig("03_success_and_length", "Success rate and episode length of successful vs failed episodes.")
    p("**Insight.** "
      f"Agent A (DQN) succeeds on {s0p('A', 'DQN (Agent A)', 'success_rate')} of the unseen bridges, against "
      f"{s0p('A', 'raster_low', 'success_rate')} for the dense raster. On the <<ORIGN>> original evaluation bridges the "
      f"saved result was {num(VAL, 'reproduced', '.0f', 100.0, suffix='%', run='A', policy='DQN (Agent A)', metric='success')}"
      f", so the agent generalises well but not perfectly: every failure is a battery-empty episode "
      f"(battery-empty rate {s0p('A', 'DQN (Agent A)', 'battery_empty_rate')}), meaning that on a few bridges the "
      "learned route revisits cells and runs out of energy before the uncertainty threshold is met. "
      f"Agent B (PPO) reaches the best view in {s0p('B', 'PPO (Agent B)', 'success_rate')} of episodes, the same as "
      f"the privileged scripted upper bound, while the fixed orbit never does ({s0p('B', 'fixed_orbit', 'success_rate')}). "
      f"For Agent C, success means a safe return; every supervisor, including random "
      f"({s0p('C', 'random', 'success_rate')}), satisfies it, because returning home early is always safe. "
      "Success rate alone therefore cannot separate the supervisors; Q2, Q3 and Q9 use return, defects found "
      "and energy instead.")
    videos_ref(["V2_agentA_vs_raster_seed1000", "V3_agentA_vs_raster_heldout2005", "V5_agentB_nominal_and_hardstart"])


def sec_q2():
    h(2, "Q2. How much reward does it collect, and is it learning?")
    qblock("How much return does each agent collect, and did training converge?",
           "Episode return (mean, std, median, <<CI>>% CI); learning curves with a <<MA>>-episode moving average and ±<<BAND>> std band.",
           "G = sum_t r_t ;  MA_50(i) = mean(G_{i-49..i})")
    rows = []
    for g in GROUP_ORDER:
        for pol in policies(g):
            rows.append([GTITLE[g], pol, s0(g, pol, "return_mean"), s0(g, pol, "return_std"),
                         s0(g, pol, "return_median"), f"±{s0(g, pol, 'return_ci95')}"])
    table(["Group", "Policy", "Mean return", "Std", "Median", "<<CI>>% CI"], rows, f"Held-out return ({nfoot(S0)}).")
    fig("01_learning_curves", "Learning curves (raw return, <<MA>>-episode moving average with ±<<BAND>> std band, eval return, "
        "success rate, episode length).")
    fig("02_vs_baselines", "Mean held-out return with <<CI>>% CI, learned agents vs baselines.")
    p("**Insight.** "
      f"All three learning curves rise and flatten. PPO converges fastest: its evaluation success first reaches 100% "
      f"after {num(SE, 'steps_to_100pct_success', ',.0f', run='B')} steps and its held-out return "
      f"({s0('B', 'PPO (Agent B)', 'return_mean')}) equals the upper bound ({s0('B', 'scripted_approach (upper bound)', 'return_mean')}). "
      f"A2C reaches a return of {s0('C', 'A2C (Agent C)', 'return_mean')}, below the rule supervisor "
      f"({s0('C', 'rule_supervisor', 'return_mean')}). The DQN curve is the slowest and noisiest: evaluation success first "
      f"reaches 100% only after {num(SE, 'steps_to_100pct_success', ',.0f', run='A')} steps and the eval return still "
      f"jumps between checkpoints, which is typical for vanilla DQN without Double/Dueling updates; the best checkpoint "
      f"(eval return {num(SE, 'best_eval_return', '.2f', run='A')}) is kept by the EvalCallback. On held-out bridges "
      f"the DQN return ({s0('A', 'DQN (Agent A)', 'return_mean')}) is slightly below the dense raster "
      f"({s0('A', 'raster_low', 'return_mean')}) because of the few battery-empty failures (Q1).")
    videos_ref(["V2_agentA_vs_raster_seed1000"])


def sec_q3():
    h(2, "Q3. How efficient is it?")
    qblock("How many steps, how much energy and how much training does each agent need?",
           "Episode length (successful / failed), time to goal, energy used, sample efficiency, training time, "
           "inference latency, model size.",
           "time-to-goal = mean low-level steps over successful episodes ; sample efficiency = steps to first 80% / 100% eval success")
    rows = []
    for g in ["A", "B", "C", "C_withA"]:
        for pol in policies(g):
            ss = s0(g, pol, "steps_success_mean", ".1f")
            sf = s0(g, pol, "steps_fail_mean", ".1f")
            rows.append([GTITLE[g], pol, ss if ss != "nan" else "-", sf if sf != "nan" else "-",
                         (lambda t: t if t != "nan" else "-")(s0(g, pol, "time_to_goal_mean", ".1f")),
                         s0(g, pol, "energy_used_mean", ".3f")])
    table(["Group", "Policy", "Length (successful)", "Length (failed)", "Time to goal (low-level steps)",
           "Energy used"], rows, f"Efficiency on held-out bridges ({nfoot(S0)}). Length is in steps for A/B and "
          "supervisor decisions for C; '-' = no such episodes.")
    se = _df(SE)
    rows = []
    for run in ["A", "A_learned", "B", "B_learned", "C", "C_learned", "C_withA"]:
        if run not in set(se.run):
            continue
        f80 = num(SE, "steps_to_80pct_success", ",.0f", run=run)
        f100 = num(SE, "steps_to_100pct_success", ",.0f", run=run)
        rows.append([run, num(SE, "steps_trained", ",.0f", run=run), f80 if f80 != "nan" else "never",
                     f100 if f100 != "nan" else "never", num(SE, "train_minutes", ".1f", run=run)])
    table(["Run", "Training steps", "Steps to 80% eval success", "Steps to 100% eval success",
           "Training time (min, Colab)"], rows, "Sample efficiency and training time (from training logs, training seed 0; each eval point is the "
          "EvalCallback mean over its eval episodes, see results/eval_curves.csv).")
    rows = [[a, _df(MSZ).set_index("agent").loc[a, "algorithm"],
             num(MSZ, "parameters", ",.0f", agent=a), num(MSZ, "file_kb", ",.0f", agent=a),
             num(MSZ, "latency_ms_median_1000", ".2f", agent=a), num(MSZ, "latency_ms_p95_1000", ".2f", agent=a)]
            for a in _df(MSZ).agent]
    table(["Agent", "Algorithm", "Parameters", "Model file (KB)", "Latency median (ms)", "Latency p95 (ms)"], rows,
          "Model size and inference latency (<<LATN>> deterministic predict() calls, single CPU thread).")
    p("**Insight.** "
      f"On the bridges that both solve, the DQN needs {s0('A', 'DQN (Agent A)', 'steps_success_mean', '.1f')} steps "
      f"against {s0('A', 'raster_low', 'steps_success_mean', '.1f')} for the dense raster "
      f"(change {der('A_steps_vs_raster_pct', '+.0f', '%')}) and changes energy use by {der('A_energy_vs_raster_pct', '+.0f', '%')}: it "
      "flies higher where one wide view is enough and stops as soon as the threshold is met. Its failed episodes are "
      f"long ({s0('A', 'DQN (Agent A)', 'steps_fail_mean', '.1f')} steps), i.e. they end by running out of battery. "
      f"PPO needs {s0('B', 'PPO (Agent B)', 'steps_success_mean', '.2f')} refinement steps, essentially the upper bound "
      f"({s0('B', 'scripted_approach (upper bound)', 'steps_success_mean', '.2f')}). Compared with the rule supervisor, A2C "
      f"changes the number of decisions by {der('C_decisions_vs_rule_pct', '+.0f', '%')}, energy by "
      f"{der('C_energy_vs_rule_pct', '+.0f', '%')} and return by {der('C_return_vs_rule_pct', '+.1f', '%')}. In sample efficiency "
      f"PPO and A2C learn within about {num(SE, 'steps_to_100pct_success', ',.0f', run='B')}-"
      f"{num(SE, 'steps_to_100pct_success', ',.0f', run='C')} steps, while the DQN needs "
      f"{num(SE, 'steps_to_100pct_success', ',.0f', run='A')}; the real-image DQN never reached 80% eval success in "
      f"{num(SE, 'steps_trained', ',.0f', run='A_learned')} steps. All policies are small: one decision takes a few "
      "milliseconds on a laptop CPU, far below any real-time limit.")


def sec_q4():
    h(2, "Q4. How close is it to the best possible / best baseline?")
    qblock("How much return does each learned agent lose compared with a strong benchmark on the same bridges?",
           "Per-episode regret against an explicitly stated benchmark (mean, <<CI>>% CI, cumulative).",
           "regret_i = G_i(benchmark) - G_i(agent) ;  cumulative regret = sum_i regret_i")
    rows = []
    bench_kind = {"A": "best baseline (dense raster)", "B": "privileged upper bound", "C": "rule supervisor",
                  "C_withA": "rule supervisor with the same Agent A", "full": "all-scripted system"}
    for g, pol in [("A", "DQN (Agent A)"), ("B", "PPO (Agent B)"), ("C", "A2C (Agent C)"), ("C_withA", "A2C (C_withA)"),
                   ("full", "ALL LEARNED (C + A + B)"), ("full", "ablation: rule C, A + B learned")]:
        b = _df(S0).set_index(["group", "policy"]).loc[(g, pol), "benchmark"]
        rows.append([pol, f"{b} ({bench_kind[g]})", s0(g, pol, "regret_mean"), f"±{s0(g, pol, 'regret_ci95')}",
                     s0(g, pol, "regret_cumulative", ".1f")])
    table(["Agent", "Benchmark", "Mean regret", "<<CI>>% CI", "Cumulative regret (<<HELDN>> bridges)"], rows,
          f"Regret on held-out bridges ({nfoot(S0)}). Positive = the benchmark collected more return.")
    fig("11_regret", "Mean regret of each learned agent against its benchmark.")
    p("**Insight.** "
      f"PPO's regret against the privileged upper bound is {s0('B', 'PPO (Agent B)', 'regret_mean', '.3f')} return per "
      "episode, so it has effectively learned the optimal approach without access to the privileged geometry. "
      f"The DQN loses {s0('A', 'DQN (Agent A)', 'regret_mean')} per episode to the dense raster, almost entirely from "
      f"its battery-empty failures. A2C trained with scripted sub-agents loses {s0('C', 'A2C (Agent C)', 'regret_mean')} "
      f"to the rule supervisor, but the A2C retrained with Agent A inside (C_withA) reduces this to "
      f"{s0('C_withA', 'A2C (C_withA)', 'regret_mean')} (<<CI>>% CI ±{s0('C_withA', 'A2C (C_withA)', 'regret_ci95')}, "
      "so not significantly different from zero). A negative regret means beating the benchmark: the rule supervisor "
      f"with learned A and B beats the all-scripted system by {num(S0, 'regret_mean', '.2f', -1.0, group='full', policy='ablation: rule C, A + B learned')} "
      "per mission, which shows that the learned navigator and refiner are better than their scripted versions.")


def sec_q5():
    h(2, "Q5. Is performance consistent?")
    qblock("How much does performance vary between bridges and between training seeds?",
           "Standard deviation, <<CI>>% confidence interval, box plots; mean ± std across training seeds.",
           "s = sqrt(sum (x - mean)^2 / (N-1)) ;  CI95 = mean ± t_{0.975,N-1} s / sqrt(N)")
    fig("05_boxplots_return", "Distribution of held-out episode returns per policy.")
    p("Per-bridge variability (std and <<CI>>% CI of the return) is listed in Table 3 (Q2). "
      f"The dense raster has almost no spread (std {s0('A', 'raster_low', 'return_std')}) because it flies the same "
      f"path on every bridge, while the DQN adapts its route (std {s0('A', 'DQN (Agent A)', 'return_std')}) and has a "
      "few low-return failure outliers. Supervisor returns vary mostly because the number of defects differs per "
      f"bridge (A2C std {s0('C', 'A2C (Agent C)', 'return_std')}, rule std {s0('C', 'rule_supervisor', 'return_std')}).")
    if (RESULTS / "multiseed_summary.csv").exists():
        m = _df(MSS)
        rows = [[a, num(MSS, "n_seeds", ".0f", agent=a), m.set_index("agent").loc[a, "seeds"],
                 num(MSS, "return_mean", ".3f", agent=a), num(MSS, "return_std", ".3f", agent=a),
                 f"±{num(MSS, 'return_ci95', '.3f', agent=a)}", pct(MSS, "success_mean", agent=a),
                 num(MSS, "steps_mean", ".2f", agent=a)] for a in m.agent]
        table(["Agent", "Training seeds", "Seed ids", "Mean held-out return", "Std across seeds",
               "<<CI>>% CI across seeds", "Mean success", "Mean length"], rows,
              "Across training seeds: each seed evaluated on the same <<HELDN>> held-out bridges; seed 0 is the original run.")
        mp = _df(MSP)
        rows = [[r.agent, f"{int(r.train_seed)}", num(MSP, "heldout_return", ".3f", agent=r.agent, train_seed=r.train_seed),
                 f"±{num(MSP, 'heldout_return_ci95', '.3f', agent=r.agent, train_seed=r.train_seed)}",
                 pct(MSP, "heldout_success", agent=r.agent, train_seed=r.train_seed)] for r in mp.itertuples()]
        table(["Agent", "Training seed", "Held-out return", "<<CI>>% CI (over bridges)", "Success"], rows,
              "Per-seed held-out results (n = <<HELDN>> bridges each, seeds <<HELD0>>-<<HELD1>>).")
        fig("05b_boxplot_training_seeds", "Held-out return per training seed.")
        fig("01b_learning_curves_multiseed", "Eval-return learning curves across training seeds (mean ± std).")
        txt = ("**Insight.** Across training seeds the final performance is very stable: PPO "
               f"{num(MSS, 'return_mean', '.3f', agent='B')} ± {num(MSS, 'return_std', '.3f', agent='B')} and A2C "
               f"{num(MSS, 'return_mean', '.2f', agent='C')} ± {num(MSS, 'return_std', '.2f', agent='C')}, i.e. the "
               "difference between seeds is much smaller than the difference between bridges. ")
        if "A" in set(m.agent):
            txt += (f"The DQN varies more between seeds ({num(MSS, 'return_mean', '.2f', agent='A')} ± "
                    f"{num(MSS, 'return_std', '.2f', agent='A')}), as expected from its noisier learning curve - and the "
                    f"original seed-0 model is the weakest of the four. The extra seeds reach "
                    f"{num(MSP, 'heldout_success', '.0f', 100.0, agg='min', suffix='%', agent='A', source='multiseed')}-"
                    f"{num(MSP, 'heldout_success', '.0f', 100.0, agg='max', suffix='%', agent='A', source='multiseed')} "
                    f"success and returns of {num(MSP, 'heldout_return', '.2f', agg='min', agent='A', source='multiseed')}-"
                    f"{num(MSP, 'heldout_return', '.2f', agg='max', agent='A', source='multiseed')}, *above* the dense "
                    f"raster ({s0('A', 'raster_low', 'return_mean')}), against "
                    f"{num(MSP, 'heldout_success', '.0f', 100.0, suffix='%', agent='A', train_seed=0)} success and "
                    f"{num(MSP, 'heldout_return', '.2f', agent='A', train_seed=0)} for seed 0. The single-seed conclusion "
                    "in Q1-Q4 that the DQN is slightly less reliable than the raster is therefore a property of that one "
                    "training run, not of the method. ")
        else:
            txt += "Agent A (DQN) was trained with a single seed only, so its seed-to-seed variability is not measured. "
        p(txt + "The seeds do differ in *how* Agent C reaches its return (Q6).")
    else:
        p("**Insight.** Only one training seed exists for every agent, so variability across training seeds could not be "
          "measured; all confidence intervals above are over bridges, not over training runs.")


def sec_q6():
    h(2, "Q6. How does it explore?")
    qblock("How do the agents explore during training, and how do the trained policies use the state space and the options?",
           "epsilon schedule (DQN), policy entropy (PPO, A2C), state-space coverage and visitation entropy (Agent A), "
           "option-selection frequency (Agent C).",
           "H = -sum_c p_c log p_c / log 128 ;  option frequency = n(option) / n(decisions)")
    fig("04_exploration_schedules", "epsilon schedule (DQN) and policy entropy (PPO, A2C) during training.")
    rows = [[pol, num(SA, "coverage_mean", ".3f", policy=pol), num(SA, "pos_coverage_mean", ".3f", policy=pol),
             num(SA, "pose_coverage_mean", ".3f", policy=pol), num(SA, "visit_entropy_mean", ".3f", policy=pol)]
            for pol in policies("A")]
    table(["Policy", "Task coverage (cells seen)", "Position coverage", "Pose coverage", "Visitation entropy"], rows,
          f"Agent A state-space coverage and visitation entropy, mean over held-out episodes ({nfoot(SA)}).")
    fig("07_visitation_heatmaps", "Agent A state-visitation heatmaps (DQN vs raster vs random).")
    rows = []
    for g in ["C", "C_withA", "full"]:
        for pol in policies(g):
            rows.append([GTITLE[g], pol, s0p(g, pol, "freq_explore", ".1f"), s0p(g, pol, "freq_refine", ".1f"),
                         s0p(g, pol, "freq_return", ".1f")])
    table(["Group", "Supervisor", "EXPLORE", "REFINE", "RETURN"], rows,
          f"Option-selection frequency (share of decisions, {nfoot(S0)}).")
    fig("08_option_frequency", "Option-selection frequency per supervisor (incl. extra training seeds if available).")
    txt = ("**Insight.** "
           f"The DQN's epsilon falls linearly from {num(HP_A, '', '.1f')} to {num(TB, 'value', '.2f', agg='last', run='A', tag='rollout/exploration_rate')} "
           f"over the first {num(HP_AF, '', '.0f', 100.0)}% of training; PPO's entropy falls steadily from {num(TB, 'value', '.2f', -1.0, agg='first', run='B', tag='train/entropy_loss')} "
           f"to {num(TB, 'value', '.2f', -1.0, agg='last', run='B', tag='train/entropy_loss')} as the Gaussian policy narrows. "
           f"The trained DQN sees {num(SA, 'coverage_mean', '.3f', policy='DQN (Agent A)')} of the cells while physically "
           f"visiting only {num(SA, 'pos_coverage_mean', '.2f', policy='DQN (Agent A)')} of the positions, against "
           f"{num(SA, 'pos_coverage_mean', '.2f', policy='raster_low')} for the dense raster: it uses higher altitudes "
           "(wider footprints) instead of flying over every cell, which is why it is more efficient. "
           f"The original A2C supervisor never chooses REFINE ({s0p('C', 'A2C (Agent C)', 'freq_refine', '.1f')} of "
           f"decisions; the rule supervisor uses it in {s0p('C', 'rule_supervisor', 'freq_refine', '.1f')}). Under the reward "
           f"+{num('py:ie.W_DEF_C', '', '.0f')} per new defect - {num('py:ie.W_ENERGY_C', '', '.1f')} x energy, refining one cell costs energy and only rarely confirms a *new* defect, so its "
           "expected value is close to zero. ")
    if (RESULTS / "multiseed_episodes_C_seed1.csv").exists():
        txt += ("The extra training seeds show that this is not a fixed property of A2C: with the same settings, seeds 1, 2 "
                f"and 3 choose REFINE in {der('C_seed1_refine_pct', '.1f', '%')}, {der('C_seed2_refine_pct', '.1f', '%')} "
                f"and {der('C_seed3_refine_pct', '.1f', '%')} of decisions with practically the same return (Q5). The "
                "option mix is therefore not determined by the reward - several mixes are almost equally good, and the "
                "seed decides which one A2C settles on.")
    p(txt)
    videos_ref(["V6_agentC_mission_seed2000", "V2_agentA_vs_raster_seed1000"])


def sec_q7():
    h(2, "Q7. Is it safe?")
    qblock("Does the agent violate the battery and flight-area constraints?",
           "Battery-empty rate (% of episodes) and out-of-bounds moves per <<OOBPER>> low-level steps.",
           "battery-empty rate = (1/N) sum 1[battery <= 0] ;  OOB rate = <<OOBPER>> * OOB moves / steps")
    rows = []
    for scen in ["S0_nominal", "S4_battery0.6"]:
        for g, pol in [("A", "DQN (Agent A)"), ("A", "raster_low"), ("A", "raster_mid"), ("A", "random"),
                       ("C", "A2C (Agent C)"), ("C", "rule_supervisor"), ("C_withA", "A2C (C_withA)"),
                       ("full", "ALL LEARNED (C + A + B)"), ("full", "ALL SCRIPTED (rule + raster + approach)")]:
            rows.append([scen, pol, sc(scen, g, pol, "battery_empty_rate", ".0f", 100.0, "%"),
                         sc(scen, g, pol, "oob_per_1000_steps", ".1f"), sc(scen, g, pol, "success_rate", ".0f", 100.0, "%")])
    table(["Scenario", "Policy", "Battery-empty rate", "OOB moves / <<OOBPER>> steps", "Success"], rows,
          f"Constraint violations, nominal battery vs starting battery 0.6 ({nfoot(SC)}).")
    p("**Insight.** "
      f"None of the learned policies ever leaves the flight area (DQN {sc('S0_nominal', 'A', 'DQN (Agent A)', 'oob_per_1000_steps', '.1f')} "
      f"out-of-bounds moves per <<OOBPER>> steps), while the random navigator makes "
      f"{sc('S0_nominal', 'A', 'random', 'oob_per_1000_steps', '.0f')} per <<OOBPER>> steps. **Key robustness finding:** when "
      f"the drone starts with only <<BAT>>% battery, the dense raster runs out of battery in "
      f"{sc('S4_battery0.6', 'A', 'raster_low', 'battery_empty_rate', '.0f', 100.0, '%')} of missions (success "
      f"{sc('S4_battery0.6', 'A', 'raster_low', 'success_rate', '.0f', 100.0, '%')}), because its fixed path needs about "
      f"{s0('A', 'raster_low', 'energy_used_mean', '.3f')} of a full battery. The DQN still succeeds in "
      f"{sc('S4_battery0.6', 'A', 'DQN (Agent A)', 'success_rate', '.0f', 100.0, '%')} of missions (battery-empty "
      f"{sc('S4_battery0.6', 'A', 'DQN (Agent A)', 'battery_empty_rate', '.0f', 100.0, '%')}): its battery level is part "
      "of the observation and its reward pays for battery left, so it finishes with a cheaper route. All supervisors with "
      "a battery rule stay safe; C_withA, which explores longer, empties the battery in "
      f"{sc('S4_battery0.6', 'C_withA', 'A2C (C_withA)', 'battery_empty_rate', '.0f', 100.0, '%')} of low-battery missions, "
      "because it was never trained with a low starting battery.")
    videos_ref(["V10_agentA_battery0.6_seed2000"])


KEY_POL = [("A", "DQN (Agent A)"), ("A", "raster_low"), ("B", "PPO (Agent B)"), ("B", "scripted_approach (upper bound)"),
           ("C", "A2C (Agent C)"), ("C", "rule_supervisor"), ("C_withA", "A2C (C_withA)"),
           ("full", "ALL LEARNED (C + A + B)"), ("full", "ALL SCRIPTED (rule + raster + approach)")]


def sec_q8():
    h(2, "Q8. Does it work in different environments?")
    qblock("How do the agents perform when the task environment changes?",
           "Success rate and return per scenario; robustness gap (nominal - perturbed).",
           "gap = metric(S0) - metric(scenario)")
    table(["Scenario", "Description"], [[k, v["desc"]] for k, v in SCENARIOS.items()],
          "Varying task environments (all on held-out seeds <<HELD0>>-<<HELD1>>, n = <<HELDN>> episodes per policy).")
    scen = list(SCENARIOS)
    d = _df(SC)
    for metric, label, fmt, scale, suf in [("success_rate", "Success rate", ".0f", 100.0, "%"),
                                          ("return_mean", "Mean return", ".2f", 1.0, "")]:
        rows = []
        for g, pol in KEY_POL:
            r = [pol]
            for s in scen:
                ok = ((d.scenario == s) & (d.group == g) & (d.policy == pol)).any()
                r.append(sc(s, g, pol, metric, fmt, scale, suf) if ok else "-")
            rows.append(r)
        table(["Policy"] + [s.replace("_", " ") for s in scen], rows, f"{label} per scenario (n = <<HELDN>> episodes per policy and scenario, held-out seeds <<HELD0>>-<<HELD1>>; "
              "'-' = not applicable).")
    rows = []
    for g, pol in [("A", "DQN (Agent A)"), ("B", "PPO (Agent B)"), ("C", "A2C (Agent C)"), ("full", "ALL LEARNED (C + A + B)")]:
        for s in scen[1:]:
            if not ((_df(RG).group == g) & (_df(RG).policy == pol) & (_df(RG).scenario == s)).any():
                continue
            rows.append([pol, s, num(RG, "return_gap", "+.2f", group=g, policy=pol, scenario=s),
                         num(RG, "success_gap", "+.0f", 100.0, suffix=" pp", group=g, policy=pol, scenario=s)])
    table(["Learned policy", "Scenario", "Return gap", "Success gap"], rows,
          "Robustness gap of the learned policies (nominal minus perturbed; positive = worse; n = <<HELDN>> episodes "
          "per scenario, held-out seeds <<HELD0>>-<<HELD1>>).")
    fig("06_robustness_gap", "Robustness gap per scenario (relative return drop and success drop).")
    rows = []
    for g in ["A", "B", "C"]:
        for pol in policies(g, SC, "S1_learned"):
            rows.append([GTITLE[g], pol, sc("S1_learned", g, pol, "success_rate", ".0f", 100.0, "%"),
                         sc("S1_learned", g, pol, "return_mean"), sc("S1_learned", g, pol, "steps_mean", ".1f"),
                         sc("S1_learned", g, pol, "energy_used_mean", ".3f")])
    table(["Group", "Policy", "Success", "Return", "Length", "Energy"], rows,
          "Real-image (DeepCrack) detector, analytic-trained and real-image-trained models (n = <<HELDN>> episodes each, seeds <<HELD0>>-<<HELD1>>).")
    fig("13_real_image_success", "Success rate with the analytic vs the real-image detector.")
    p("**Insight.** "
      f"**Real images are the hardest change for Agent A.** The analytic-trained DQN drops to "
      f"{sc('S1_learned', 'A', 'DQN (Agent A)', 'success_rate', '.0f', 100.0, '%')} success and even the DQN trained on "
      f"real-image uncertainty reaches only {sc('S1_learned', 'A', 'DQN (Agent A, real-image trained)', 'success_rate', '.0f', 100.0, '%')}, "
      f"while the dense raster keeps {sc('S1_learned', 'A', 'raster_low', 'success_rate', '.0f', 100.0, '%')}. Why: the "
      "U-Net's uncertainty is weaker and noisier than the analytic formula, the calibrated threshold is stricter, and the "
      "high and mid altitudes map to the two worst view qualities (q0, q1), so meeting the threshold requires flying low "
      "over almost every cell - exactly what the raster does by design. "
      f"Agents B and C transfer well: PPO keeps {sc('S1_learned', 'B', 'PPO (Agent B)', 'success_rate', '.0f', 100.0, '%')} "
      f"success and A2C a return of {sc('S1_learned', 'C', 'A2C (Agent C)', 'return_mean')} vs "
      f"{sc('S1_learned', 'C', 'rule_supervisor', 'return_mean')} for the rule supervisor. "
      f"With **35% defects** the DQN's success falls to {sc('S2_density35', 'A', 'DQN (Agent A)', 'success_rate', '.0f', 100.0, '%')}: "
      "more defective cells keep the mean uncertainty high (defects are more ambiguous in the detector model), so it "
      "needs more low passes than its learned budget allows. Returns of the supervisors rise or fall with defect density "
      "simply because the reward counts defects (the return gap is not a failure there). "
      f"**Observation noise** on the uncertainty features (sigma {num(NOISE1, '', '.2f')} and {num(NOISE2, '', '.2f')}) "
      f"does not hurt any agent (DQN success "
      f"{sc('S3_noise0.10', 'A', 'DQN (Agent A)', 'success_rate', '.0f', 100.0, '%')} at the larger sigma). "
      f"**Hard start geometry** leaves PPO at {sc('S5_hardstart', 'B', 'PPO (Agent B)', 'success_rate', '.0f', 100.0, '%')} "
      f"success in {sc('S5_hardstart', 'B', 'PPO (Agent B)', 'steps_mean', '.2f')} steps (upper bound "
      f"{sc('S5_hardstart', 'B', 'scripted_approach (upper bound)', 'steps_mean', '.2f')}). The low-battery result is "
      "discussed in Q7.")
    videos_ref(["V4_agentA_real_images_seed2000", "V8_detector_q0_to_q4", "V9_agentA_density35_seed2013",
                "V10_agentA_battery0.6_seed2000", "V5_agentB_nominal_and_hardstart"])


def sec_q9():
    h(2, "Q9. Do the agents work together?")
    qblock("Does the hierarchy of three learned agents work, and which learned parts help?",
           "Integration ablation in full mode: return, decisions, defects found, energy, mIoU, REFINE share.",
           "same <<HELDN>> bridges for every configuration (paired)")
    rows = [[pol, s0("full", pol, "return_mean"), s0("full", pol, "steps_mean", ".1f"),
             s0("full", pol, "low_level_steps_mean", ".1f"), s0("full", pol, "defects_found_mean"),
             s0("full", pol, "energy_used_mean", ".3f"), s0("full", pol, "miou_mean", ".3f"),
             s0p("full", pol, "freq_refine", ".1f")] for pol in policies("full")]
    table(["Configuration", "Return", "Decisions", "Low-level steps", "Defects (of <<NDEF>>)", "Energy", "mIoU",
           "REFINE share"], rows, f"Integration ablation in full mode ({nfoot(S0)}).")
    fig("10_integration_ablation", "Energy vs defects found per configuration.")
    p("**Insight.** "
      f"Compared with the all-scripted system, the fully learned hierarchy (C + A + B) completes every mission while changing "
      f"energy by {der('full_energy_learned_vs_scripted_pct', '+.0f', '%')} and the number of decisions by "
      f"{der('full_decisions_learned_vs_scripted_pct', '+.0f', '%')}, at a cost of "
      f"{der('full_defects_learned_minus_scripted', '+.2f')} defects found per mission. The best "
      f"configuration is the rule supervisor with learned A and B (return {s0('full', 'ablation: rule C, A + B learned', 'return_mean')}, "
      f"defects {s0('full', 'ablation: rule C, A + B learned', 'defects_found_mean')}), which shows that the learned "
      "navigator and refiner are individually better than the scripted ones. The weak link is the supervisor: the "
      f"original A2C never calls REFINE ({s0p('full', 'ALL LEARNED (C + A + B)', 'freq_refine', '.1f')}), so Agent B is "
      "never used inside the learned hierarchy and the A2C returns home early to save energy. The A2C retrained with "
      f"Agent A inside (C_withA) explores longer and gets a return of {s0('full', 'C_withA + A + B (extra)', 'return_mean')}, "
      "the same mean return as the best configuration, but it also does not use REFINE.")
    videos_ref(["V7_full_mission_learned_vs_rule"])


def sec_q10():
    h(2, "Q10. Is training healthy?")
    p("These are **training diagnostics**, not task performance: they show whether the optimisation behaved well.")
    qblock("Did the losses, value estimates and policy updates behave as expected, and are the results sensitive to hyperparameters?",
           "DQN loss (TD-error proxy); PPO policy-gradient loss, value loss, entropy, approximate KL, clip fraction, "
           "explained variance; A2C policy loss, value loss, entropy, explained variance; sensitivity studies.",
           "explained variance = 1 - Var(returns - V) / Var(returns) ;  approx KL ~ E[(r - 1) - log r], r = pi_new/pi_old")
    rows = []
    for run, tag, label in [("A", "train/loss", "DQN loss"), ("B", "train/value_loss", "PPO value loss"),
                            ("B", "train/approx_kl", "PPO approx. KL"), ("B", "train/clip_fraction", "PPO clip fraction"),
                            ("B", "train/explained_variance", "PPO explained variance"),
                            ("C", "train/value_loss", "A2C value loss"), ("C", "train/explained_variance", "A2C explained variance")]:
        rows.append([label, num(TB, "value", ".4f", agg="head10mean", run=run, tag=tag),
                     num(TB, "value", ".4f", agg="tail10mean", run=run, tag=tag),
                     num(TB, "value", ".4f", agg="max", run=run, tag=tag)])
    table(["Diagnostic", "Mean of first <<DIAG>> logs", "Mean of last <<DIAG>> logs", "Maximum"], rows,
          "Training diagnostics from TensorBoard (main runs, training seed 0; training logs, no evaluation episodes).")
    fig("09_training_diagnostics", "PPO / A2C / DQN training diagnostics.")
    rows = []
    for r in _df(SENS).itertuples():
        hp = (f"eps_fraction {num(SENS, 'eps_fraction', 'g', run=r.run)}" if r.agent == "A" else
              f"ent {num(SENS, 'ent_coef', 'g', run=r.run)}"
              + (f", clip {num(SENS, 'clip_range', 'g', run=r.run)}" if r.agent == "B" else ""))
        rows.append([r.run, hp, num(SENS, "steps_trained", ",.0f", run=r.run) + (" (short run)" if r.short_run else ""),
                     num(SENS, "heldout_return", ".2f", run=r.run), f"±{num(SENS, 'heldout_return_ci95', '.2f', run=r.run)}",
                     pct(SENS, "heldout_success", run=r.run), num(SENS, "heldout_steps", ".1f", run=r.run)])
    table(["Run", "Setting", "Training steps", "Held-out return", "<<CI>>% CI", "Success", "Length"], rows,
          "Hyperparameter sensitivity (CO5), every run evaluated on the <<HELDN>> held-out bridges (seeds <<HELD0>>-<<HELD1>>).")
    fig("12_sensitivity", "Hyperparameter sensitivity (held-out return with <<CI>>% CI).")
    p("**Insight.** "
      f"PPO is healthy: the explained variance rises to {num(TB, 'value', '.3f', agg='tail10mean', run='B', tag='train/explained_variance')}, "
      f"the approximate KL stays small ({num(TB, 'value', '.4f', agg='tail10mean', run='B', tag='train/approx_kl')} at the end) and "
      f"the clip fraction stays near {num(TB, 'value', '.2f', agg='tail10mean', run='B', tag='train/clip_fraction')}, so updates are "
      f"neither too large nor too timid. A2C's value function also fits well (explained variance "
      f"{num(TB, 'value', '.3f', agg='tail10mean', run='C', tag='train/explained_variance')}). The DQN loss *rises* after "
      f"the epsilon schedule ends (peak {num(TB, 'value', '.3f', agg='max', run='A', tag='train/loss')}) and falls again at the end, "
      "which is normal: once the agent reaches the large success bonus, the Q-targets grow and so do the TD errors; together with the noisy eval curve this is why the best checkpoint is used. "
      f"Sensitivity: a shorter epsilon decay gives the best DQN (eps_fraction {num(SENS, 'eps_fraction', '.1f', run='A_eps0.1')}: return "
      f"{num(SENS, 'heldout_return', '.2f', run='A_eps0.1')}, {num(SENS, 'eps_fraction', '.1f', run='A_eps0.6')}: "
      f"{num(SENS, 'heldout_return', '.2f', run='A_eps0.6')}), "
      "because a long exploration phase leaves too little time to exploit. PPO is insensitive to its entropy coefficient "
      f"and clip range (all returns between {num(SENS, 'heldout_return', '.2f', agg='min', agent='B')} and "
      f"{num(SENS, 'heldout_return', '.2f', agg='max', agent='B')}), and A2C to its entropy coefficient "
      f"({num(SENS, 'heldout_return', '.2f', agg='min', agent='C')}-{num(SENS, 'heldout_return', '.2f', agg='max', agent='C')}). "
      "The sensitivity runs are shorter than the main runs, as marked.")


def sec_findings():
    h(2, "5. Key findings")
    items = [
        f"**Agent B (PPO) is solved:** {s0p('B', 'PPO (Agent B)', 'success_rate')} success, regret "
        f"{s0('B', 'PPO (Agent B)', 'regret_mean', '.3f')} against the privileged upper bound, robust to a hard start, real "
        "images, hyperparameters and training seeds.",
        f"**Agent A (DQN) is more efficient than the dense raster; its reliability depends on the training seed:** "
        f"steps {der('A_steps_vs_raster_pct', '+.0f', '%')} and energy {der('A_energy_vs_raster_pct', '+.0f', '%')} on solved "
        f"bridges, but {s0p('A', 'DQN (Agent A)', 'success_rate')} vs {s0p('A', 'raster_low', 'success_rate')} success on "
        "unseen bridges for the original model." + (
            f" Three retrained seeds reach {num(MSP, 'heldout_success', '.0f', 100.0, agg='min', suffix='%', agent='A', source='multiseed')}"
            f"-{num(MSP, 'heldout_success', '.0f', 100.0, agg='max', suffix='%', agent='A', source='multiseed')} success "
            f"and a higher return than the raster, so the gap comes from the original training seed."
            if (RESULTS / "multiseed_per_seed.csv").exists() and "A" in set(pd.read_csv(RESULTS / "multiseed_per_seed.csv").agent)
            else ""),
        f"**Low battery favours the learned planner:** at <<BAT>>% starting battery the raster fails "
        f"{sc('S4_battery0.6', 'A', 'raster_low', 'battery_empty_rate', '.0f', 100.0, '%')} of missions, the DQN succeeds in "
        f"{sc('S4_battery0.6', 'A', 'DQN (Agent A)', 'success_rate', '.0f', 100.0, '%')}.",
        f"**Real images break Agent A:** {sc('S1_learned', 'A', 'DQN (Agent A, real-image trained)', 'success_rate', '.0f', 100.0, '%')} "
        "success even when trained on the real-image detector; B and C transfer without loss.",
        f"**The learned hierarchy saves energy:** energy {der('full_energy_learned_vs_scripted_pct', '+.0f', '%')} vs the "
        f"scripted system, defects found {der('full_defects_learned_minus_scripted', '+.2f')}; learned A + B under a rule "
        "supervisor is the best configuration overall.",
        f"**The original A2C never uses REFINE** ({s0p('C', 'A2C (Agent C)', 'freq_refine', '.1f')} of decisions), so Agent B is "
        "idle inside the learned hierarchy." + (
            f" Other training seeds do use it ({der('C_seed1_refine_pct', '.0f', '%')}-{der('C_seed3_refine_pct', '.0f', '%')}) "
            "with the same return: the reward does not make REFINE clearly worthwhile."
            if (RESULTS / "multiseed_episodes_C_seed3.csv").exists() else ""),
        "**Success rate is not enough for Agent C:** every supervisor, even random, returns safely; return, defects and "
        "energy are the informative metrics.",
    ]
    bullets(items)


def sec_limitations():
    h(2, "6. Limitations and future work")
    bullets([
        "**Analytic detector for most results.** The main numbers use a formula-based uncertainty; with the real-image "
        "detector Agent A fails most missions (Q8).",
        "**Observation noise (S3) only reaches Agent C in supervise/full mode.** The sub-agents A and B read their "
        "observations inside the environment, so in the hierarchical modes only the supervisor's observation is noised; "
        "the S3 numbers for C, C_withA and full therefore test the supervisor alone. In navigate and refine mode the noise "
        "does reach Agents A and B.",
        "**Training seeds.** All original runs used seed 0. Extra seeds exist only for the agents listed in Q5; the "
        "multi-seed training times were measured with several runs in parallel on a laptop and are not comparable with "
        "the Colab times.",
        "**Simplified world.** 2-D grid, no wind, collisions or 3-D geometry; energy is a fixed cost per action.",
        "**Vanilla DQN** (no Double/Dueling) gives noisy evaluation; the best checkpoint is used.",
        "**Reward design for Agent C** gives REFINE almost no net value, so whether the supervisor uses Agent B depends on "
        "the training seed. A reward for uncertainty reduction or confirmed defects would make B's role explicit.",
        "**Future work:** Double/Dueling DQN, a battery-randomised curriculum for Agent A and C_withA, a curriculum from the "
        "analytic to the real-image detector, more training seeds, and a 3-D simulator (AirSim).",
    ])


def sec_appendix():
    h(2, "7. Appendix")
    h(3, "7.1 Video index")
    vf = VIDEOS / "video_files.csv"
    if vf.exists():
        v = pd.read_csv(vf)
        table(["File", "Scenario", "Question", "Metric to watch", "Duration (s)"],
              [[r.file, r.scenario, r.question, r.metric,
                num("videos/video_files.csv", "duration_s", ".0f", clip=r.clip)] for r in v.itertuples()],
              "Recorded experimental sessions (see videos/video_index.md for results and insights).")
    h(3, "7.2 File map")
    table(["Path (in Evaluation_Metrics/)", "Content"], [
        ["scripts/common.py", "paths, ScenarioEnv subclass, UncertaintyNoise wrapper, instrumented episode runner, CI"],
        ["scripts/validate.py", "Phase 1: re-evaluation on the original seeds -> results/validation.csv"],
        ["scripts/evaluate_all.py, suites.py", "Phase 2: held-out evaluation -> episodes_*, summary_*, regret, violations"],
        ["scripts/evaluate_scenarios.py", "Phase 3: scenarios S1-S5 -> scenario_summary.csv, robustness_gap.csv"],
        ["scripts/training_diagnostics.py", "Phase 4: curves, TensorBoard scalars, sample efficiency, sensitivity, model size"],
        ["scripts/multiseed.py", "Phase 5: extra training seeds -> multiseed_runs/, multiseed_*.csv"],
        ["scripts/plots.py", "Phase 6: plots/*.png + plots/plot_index.csv"],
        ["scripts/make_videos.py", "Phase 7: videos/*.mp4, video_runs.csv"],
        ["scripts/build_report.py", "Phase 8: this document, report_numbers.csv, video index and narration"],
        ["report/known_issues.md", "mismatches and limitations found while evaluating"],
        ["tests/test_eval.py", "pytest checks (validation, numbers, files, no changes outside Evaluation_Metrics/)"],
    ], "Where everything is.")
    h(3, "7.3 How to reproduce")
    p("From the repository root, with the venv `%USERPROFILE%\\.venvs\\uav_demo` activated:")
    formula("python Evaluation_Metrics/run_all.py            # phases 1,2,3,4,6,7,8 (add --multiseed for phase 5)")
    formula("python -m pytest Evaluation_Metrics/tests -q     # verification")


# ====================================================================== video index + narration
NARR = {
    "V1_env_tour": "This is our bridge inspection task: a grid of eight by sixteen cells, and about one cell in five "
                   "has a crack. The drone starts at the base, top left, at high altitude. From high up it sees a five "
                   "by five patch, but blurry; from low down it sees one cell, but sharply. The colour of each cell shows "
                   "how uncertain the crack detector still is. Watch the counters: coverage, mean uncertainty, energy "
                   "and return. Here a random policy flies without a plan, so the uncertainty stays high and the battery "
                   "is wasted. {res} Every metric in our report is measured on this task.",
    "V2_agentA_vs_raster_seed1000": "Here Agent A, our DQN, races the dense raster scan on the same bridge. The raster "
                                    "flies low over every row. The DQN uses higher altitudes where one wide view is "
                                    "enough, and only goes down where the uncertainty stays high. Watch the step counter "
                                    "and the battery. {res} On the hundred unseen bridges this original model needs "
                                    "fewer steps and less energy than the raster, but it runs out of battery on a few "
                                    "of them. {seeds}",
    "V3_agentA_vs_raster_heldout2005": "This bridge, seed two thousand and five, was never used in training or model "
                                       "selection, so it tests generalisation. {res} The route is different from the "
                                       "previous bridge because the cracks are in different places, so the agent is not "
                                       "replaying a memorised path. Over all hundred held-out bridges the original model "
                                       "succeeds on most of them, and its failures are battery-empty episodes. {seeds}",
    "V4_agentA_real_images_seed2000": "Now the same comparison uses the real-image detector: the uncertainty comes from "
                                      "a U-Net that looks at DeepCrack photographs. Watch the mean uncertainty, because it "
                                      "has to fall below a stricter threshold. {res} Real-image uncertainty is noisier, "
                                      "and the high and mid altitudes give very poor images, so the drone has to fly low "
                                      "almost everywhere. The dense raster does exactly that, so it succeeds. Our DQN, "
                                      "even when trained on real images, succeeds on only a small share of bridges. This "
                                      "is our clearest weak result.",
    "V5_agentB_nominal_and_hardstart": "Agent B, our PPO refiner, must reach the best possible view of one suspicious cell "
                                       "by changing its distance, viewing angle and side offset. Watch the view quality "
                                       "climb to level four. The fixed orbit baseline keeps circling and never gets there. "
                                       "{res} The last part starts from a hard position: far away and at a large angle. "
                                       "PPO still reaches the best view in a few steps, almost exactly like the scripted "
                                       "upper bound that knows the true geometry.",
    "V6_agentC_mission_seed2000": "Agent C, our A2C supervisor, decides at every step whether to explore, refine one "
                                  "cell, or return home. Watch the timeline of options at the bottom. {res} Notice that "
                                  "it only ever chooses explore and then return: this original model never chooses "
                                  "refine. Under our reward, refining costs energy and rarely finds a new crack, so it "
                                  "is hardly worth it. When we retrained with other seeds, some versions did use refine, "
                                  "and they collected the same return.",
    "V7_full_mission_learned_vs_rule": "This is the complete system: the supervisor calls Agent A to explore and Agent B "
                                       "to refine. First the learned supervisor, then a rule supervisor driving the same "
                                       "learned Agents A and B. {res} The learned hierarchy saves energy but never calls "
                                       "Agent B. With the rule supervisor, Agent B works inside the hierarchy, and that "
                                       "combination finds the most cracks.",
    "V8_detector_q0_to_q4": "This clip shows why real images are harder. The same DeepCrack patch is shown at five "
                            "view qualities, from four pixels wide up to sixty-four. The middle row is the crack the "
                            "U-Net predicts; the bottom row is the spread of its dropout passes. Look at the crack patch: "
                            "at four pixels the network misses the crack completely, yet the spread and the raw entropy "
                            "printed below are both low, so it is confidently wrong. That is why the environment does not "
                            "use these raw numbers: it ranks them and forces uncertainty to fall as view quality improves, "
                            "so a blurry view always counts as uncertain. Since high and middle altitudes give the two "
                            "worst qualities, the drone must fly low almost everywhere. {res}",
    "V9_agentA_density35_seed2013": "Here the bridge has thirty-five percent cracked cells instead of twenty. Cracked "
                                    "cells are more ambiguous for the detector, so the mean uncertainty falls more slowly. "
                                    "{res} On this bridge the DQN runs out of battery before it meets the threshold, while "
                                    "the dense raster finishes. Over a hundred bridges at this density the DQN's success "
                                    "rate drops, which shows the limit of a policy trained only at twenty percent.",
    "V10_agentA_battery0.6_seed2000": "This is our key robustness result. The drone starts with only sixty percent "
                                      "battery. Watch the battery bar on each side. {res} The dense raster needs more "
                                      "than sixty percent of a full battery for its fixed path, so it always runs out. The "
                                      "DQN sees its battery level in its observation and is rewarded for battery left, so "
                                      "it chooses a cheaper route and succeeds on most bridges.",
}
INSIGHT = {
    "V1_env_tour": "random flight leaves uncertainty high and wastes energy - the reference point for all metrics",
    "V2_agentA_vs_raster_seed1000": "DQN reaches the goal with fewer steps and less energy than the dense raster",
    "V3_agentA_vs_raster_heldout2005": "the learned route adapts to an unseen bridge",
    "V4_agentA_real_images_seed2000": "real-image uncertainty makes Agent A fail; the dense raster still succeeds",
    "V5_agentB_nominal_and_hardstart": "PPO reaches q=4 in a few steps, even from a hard start",
    "V6_agentC_mission_seed2000": "the original A2C supervisor never chooses REFINE",
    "V7_full_mission_learned_vs_rule": "learned hierarchy saves energy; rule C + learned A/B uses Agent B and finds more defects",
    "V8_detector_q0_to_q4": "at low view quality the U-Net misses the crack while its raw uncertainty stays low (false confidence); the env therefore uses rank-normalised, quality-monotone uncertainty",
    "V9_agentA_density35_seed2013": "at 35% defects the DQN can run out of battery before meeting the threshold",
    "V10_agentA_battery0.6_seed2000": "with a low battery the raster fails and the DQN succeeds",
}
SPOKEN = {"Agent A - Viewpoint Planner (DQN)": "the DQN",
          "Agent A - Viewpoint Planner (DQN) [real-image detector]": "the real-image DQN",
          "BASELINE Dense raster, LOW altitude (1x1)": "the dense raster",
          "Agent B - View Refiner (PPO)": "PPO", "BASELINE FixedOrbit": "the fixed orbit",
          "Agent C - Mission Supervisor (A2C)": "the A2C supervisor", "BASELINE RuleSupervisor": "the rule supervisor",
          "ALL SCRIPTED": "the all-scripted system",
          "RULE supervisor (scripted)": "the rule supervisor with learned Agents A and B"}
WPM = 130                     # calm speaking pace used for the length check


def video_docs():
    vf, vr = VIDEOS / "video_files.csv", VIDEOS / "video_runs.csv"
    if not vf.exists():
        return
    files, runs = pd.read_csv(vf), pd.read_csv(vr) if vr.exists() else pd.DataFrame(columns=["clip"])

    def recs(cid):
        r = runs[runs["clip"].astype(str).str.startswith(cid + "_part")]
        r = r.drop_duplicates(subset=["label", "seed", "return", "steps"])   # same baseline shown in two parts
        out = []
        for e in r.to_dict("records"):
            e["ok"] = str(e["success"]).lower() == "true"
            e["empty"] = str(e["battery_empty"]).lower() == "true"
            e["who"] = SPOKEN.get(e["label"], e["label"])
            out.append(e)
        return out

    def table_text(cid):            # compact, for video_index.md
        return "; ".join(f"{e['who']} (seed {e['seed']}): {'success' if e['ok'] else ('battery empty' if e['empty'] else 'no success')}, "
                         f"return {float(e['return']):.1f}, {int(e['steps'])} steps, energy {float(e['energy_used']):.2f}"
                         for e in recs(cid))

    def spoken(cid):                # one or two short sentences for the narration
        rs = recs(cid)
        if not rs:
            rj = sorted((VIDEOS / "raw").glob(f"{cid}_part*/result.json"))
            if cid.startswith("V1") and rj:
                res = json.loads(rj[0].read_text())["result"]
                return f"In this run the random policy flew {int(res['steps'])} steps and collected a return of {res['return']:.1f}."
            return ""
        if cid.startswith("V5"):
            ppo = [e for e in rs if e["who"] == "PPO"]
            orb = [e for e in rs if e["who"] == "the fixed orbit"]
            return (f"In this run PPO reached the best view in {', '.join(str(int(e['steps'])) for e in ppo[:-1])} and "
                    f"{int(ppo[-1]['steps'])} steps, the last one from the hard start, while the fixed orbit "
                    f"{'never did' if not any(e['ok'] for e in orb) else 'sometimes did'} in {int(orb[0]['steps'])} steps.")
        unit = "decisions" if cid.startswith(("V6", "V7")) else "steps"
        parts = [f"{e['who']} {'succeeded' if e['ok'] else ('ran out of battery' if e['empty'] else 'did not succeed')} "
                 f"after {int(e['steps'])} {unit}, using {float(e['energy_used']) * 100:.0f} percent of the battery"
                 for e in rs]
        return "In this run, " + "; ".join(parts) + "."

    def seeds_sentence():
        p_ = RESULTS / "multiseed_per_seed.csv"
        if not p_.exists():
            return ""
        m = pd.read_csv(p_)
        m = m[(m.agent == "A") & (m.source == "multiseed")]
        if m.empty:
            return ""
        return (f"Three retrained copies of the same DQN do better: they succeed on {m.heldout_success.min() * 100:.0f} to "
                f"{m.heldout_success.max() * 100:.0f} percent of these bridges.")

    lines = ["# Video index", "", "Generated by `scripts/build_report.py` from `videos/video_files.csv` and "
             "`videos/video_runs.csv` (final values read from each recorded run).", "",
             "| File | Scenario | Question | Metric | Result (this run) | Insight |", "|---|---|---|---|---|---|"]
    for f in files.itertuples():
        lines.append(f"| `{f.file}` | {f.scenario} | {f.question} | {f.metric} | {table_text(f.clip) or 'see clip'} | "
                     f"{INSIGHT.get(f.clip, '')} |")
    (VIDEOS / "video_index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    nl = ["# Narration script (about 60 seconds per clip)", "",
          f"Simple English voice-over for each clip, written for a calm pace of about {WPM} words per minute. The "
          "sentence starting 'In this run' contains the values read from the recorded run. Each clip is about one "
          "minute long, so the voice-over fits the video.", ""]
    report = []
    for f in files.itertuples():
        txt = NARR.get(f.clip, "").replace("{res}", spoken(f.clip)).replace("{seeds}", seeds_sentence())
        txt = re.sub(r"\s+", " ", txt).strip()
        words = len(txt.split())
        secs = 60.0 * words / WPM
        report.append({"clip": f.clip, "words": words, "speech_s": round(secs), "video_s": f.duration_s})
        nl += [f"## {f.clip}  (`{f.file}`, video {f.duration_s:.0f} s, voice-over about {secs:.0f} s)", "",
               f"*Question:* {f.question}  ", f"*Metric:* {f.metric}", "", txt, ""]
    (VIDEOS / "narration_script.md").write_text("\n".join(nl), encoding="utf-8")
    rep = pd.DataFrame(report)
    rep.to_csv(VIDEOS / "narration_lengths.csv", index=False)
    print(rep.to_string(index=False))


# ====================================================================== renderers
def to_markdown():
    out = []
    for b in B:
        if b[0] == "h":
            out += ["#" * b[1] + " " + b[2], ""]
        elif b[0] == "p":
            out += [b[1], ""]
        elif b[0] == "ul":
            out += [f"- {x}" for x in b[1]] + [""]
        elif b[0] == "formula":
            out += ["```", b[1], "```", ""]
        elif b[0] == "table":
            _, hd, rows, cap = b
            out += [f"*{cap}*", "", "| " + " | ".join(hd) + " |", "|" + "---|" * len(hd)]
            out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows] + [""]
        elif b[0] == "img":
            out += [f"![{b[2]}](../{b[1]})", "", f"*{b[2]}*", ""]
    return "\n".join(out)


def _runs(par, text):
    """Add text to a docx paragraph, rendering **bold** and `code` spans."""
    import re
    for tok in re.split(r"(\*\*[^*]+\*\*|`[^`]+`)", text):
        if not tok:
            continue
        if tok.startswith("**"):
            par.add_run(tok[2:-2]).bold = True
        elif tok.startswith("`"):
            r = par.add_run(tok[1:-1])
            r.font.name = "Consolas"
        else:
            par.add_run(tok)


def to_docx(path):
    from docx import Document
    from docx.shared import Pt, Inches, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name, st.font.size = "Calibri", Pt(10.5)
    for s in doc.sections:
        s.left_margin = s.right_margin = Inches(0.8)
    for b in B:
        if b[0] == "h":
            doc.add_heading(b[2], level=min(b[1] - 1, 3) if b[1] > 1 else 0)
        elif b[0] == "p":
            _runs(doc.add_paragraph(), b[1])
        elif b[0] == "ul":
            for x in b[1]:
                _runs(doc.add_paragraph(style="List Bullet"), x)
        elif b[0] == "formula":
            par = doc.add_paragraph()
            r = par.add_run(b[1])
            r.font.name, r.font.size = "Consolas", Pt(9)
        elif b[0] == "table":
            _, hd, rows, cap = b
            c = doc.add_paragraph()
            rr = c.add_run(cap)
            rr.italic, rr.font.size = True, Pt(9)
            t = doc.add_table(rows=1, cols=len(hd))
            t.style = "Light Grid Accent 1"
            for i, x in enumerate(hd):
                t.rows[0].cells[i].text = ""
                _runs(t.rows[0].cells[i].paragraphs[0], f"**{x}**")
            for r in rows:
                cells = t.add_row().cells
                for i, x in enumerate(r):
                    _runs(cells[i].paragraphs[0], str(x).replace("`", ""))
            for row in t.rows:
                for cell in row.cells:
                    for par in cell.paragraphs:
                        for run in par.runs:
                            run.font.size = Pt(8)
            doc.add_paragraph()
        elif b[0] == "img":
            doc.add_picture(str(EM / b[1]), width=Inches(6.8))
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            c = doc.add_paragraph()
            r = c.add_run(b[2])
            r.italic, r.font.size = True, Pt(9)
            r.font.color.rgb = RGBColor(0x52, 0x51, 0x4E)
    doc.save(path)


def to_pdf(docx_path):
    pdf = docx_path.with_suffix(".pdf")
    code = ("import sys; from docx2pdf import convert; convert(sys.argv[1], sys.argv[2])")
    try:
        subprocess.run([sys.executable, "-c", code, str(docx_path), str(pdf)], timeout=300, check=True,
                       capture_output=True)
        return pdf if pdf.exists() else None
    except Exception as e:  # Word missing / blocked: the .md and .docx are still the deliverables
        print("PDF conversion skipped:", e)
        return None


def main(pdf=True):
    derive()
    for f in [sec_title, sec_protocol, sec_metrics]:
        f()
    h(2, "4. Evaluation questions")
    p("Each question follows the same pattern: Question -> Metric -> Formula -> Table -> Plot -> Insight -> Video.")
    for f in [sec_q1, sec_q2, sec_q3, sec_q4, sec_q5, sec_q6, sec_q7, sec_q8, sec_q9, sec_q10, sec_findings,
              sec_limitations, sec_appendix]:
        f()
    REPORT.mkdir(exist_ok=True)
    md = REPORT / "Team08_Evaluation_Metrics.md"
    md.write_text(to_markdown(), encoding="utf-8")
    pd.DataFrame(REG).to_csv(REPORT / "report_numbers.csv", index=False)
    dx = REPORT / "Team08_Evaluation_Metrics.docx"
    to_docx(dx)
    video_docs()
    out = [md, dx]
    if pdf:
        pp = to_pdf(dx)
        out += [pp] if pp else []
    print(f"report: {len(B)} blocks, {TAB['n']} tables, {FIG['n']} figures, {len(REG)} registered numbers")
    for o in out:
        print("  wrote", o.relative_to(ROOT))


if __name__ == "__main__":
    main(pdf="--no-pdf" not in sys.argv)
