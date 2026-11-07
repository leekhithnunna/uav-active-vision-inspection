"""Policy suites (which policies run in which scenario) + per-policy summary statistics.
Shared by evaluate_all.py (Phase 2, S0 nominal) and evaluate_scenarios.py (Phase 3)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (RESULTS, HELD_SEED0, HELD_N, make_env, run_policy, policy, load_model, sb3_policy,  # noqa: E402
                    RasterNav, ApproachRefiner, FixedOrbit, RuleSupervisor, RandomPolicy, ci95)

GROUPS = ["A", "B", "C", "C_withA", "full"]
GROUP_TITLE = {"A": "Agent A (DQN) - navigate", "B": "Agent B (PPO) - refine",
               "C": "Agent C (A2C) - supervise (scripted sub-agents)",
               "C_withA": "Agent C trained with Agent A inside - supervise",
               "full": "Integration - full mode (C -> A / B)"}
LEARNED = {"A": "DQN (Agent A)", "B": "PPO (Agent B)", "C": "A2C (Agent C)",
           "C_withA": "A2C (C_withA)", "full": "ALL LEARNED (C + A + B)"}
LEARNED_REAL = {"A": "DQN (Agent A, real-image trained)", "B": "PPO (Agent B, real-image trained)",
                "C": "A2C (Agent C, real-image trained)", "full": "ALL LEARNED (real-image trained models)"}
BENCHMARK = {"A": "raster_low", "B": "scripted_approach (upper bound)", "C": "rule_supervisor",
             "C_withA": "rule_supervisor (Agent A inside)", "full": "ALL SCRIPTED (rule + raster + approach)"}
BENCHMARK_WHY = {"A": "best baseline (dense raster)", "B": "privileged upper bound (scripted approach)",
                 "C": "hand-written rule supervisor", "C_withA": "rule supervisor with the same Agent A inside",
                 "full": "all-scripted system"}
MODE = {"A": "navigate", "B": "refine", "C": "supervise", "C_withA": "supervise", "full": "full"}


def suite(group, scenario):
    """List of (policy name, env factory, policy factory)."""
    S, mode = scenario, MODE[group]
    real = scenario == "S1_learned"
    A = lambda: sb3_policy(load_model("A"))
    B = lambda: sb3_policy(load_model("B"))
    env = lambda nav=None, ref=None: (lambda: make_env(mode, S, nav() if nav else None, ref() if ref else None))
    rnd = "random"
    if group == "A":
        out = [(LEARNED["A"], env(), lambda: policy("A"))]
        if real:
            out.append((LEARNED_REAL["A"], env(), lambda: policy("A_learned")))
        out += [("raster_low", env(), lambda: RasterNav(2)), ("raster_mid", env(), lambda: RasterNav(1)),
                ("random", env(), rnd)]
    elif group == "B":
        out = [(LEARNED["B"], env(), lambda: policy("B"))]
        if real:
            out.append((LEARNED_REAL["B"], env(), lambda: policy("B_learned")))
        out += [("fixed_orbit", env(), lambda: FixedOrbit()), ("random", env(), rnd),
                ("scripted_approach (upper bound)", env(), lambda: ApproachRefiner())]
    elif group == "C":
        out = [(LEARNED["C"], env(), lambda: policy("C"))]
        if real:
            out.append((LEARNED_REAL["C"], env(), lambda: policy("C_learned")))
        out += [("rule_supervisor", env(), lambda: RuleSupervisor()), ("random", env(), rnd)]
    elif group == "C_withA":
        out = [(LEARNED["C_withA"], env(A), lambda: policy("C_withA")),
               ("rule_supervisor (Agent A inside)", env(A), lambda: RuleSupervisor())]
    else:  # full
        out = [(LEARNED["full"], env(A, B), lambda: policy("C")),
               ("ALL SCRIPTED (rule + raster + approach)", env(lambda: RasterNav(1), ApproachRefiner),
                lambda: RuleSupervisor()),
               ("ablation: C learned, A/B scripted", env(lambda: RasterNav(1), ApproachRefiner), lambda: policy("C")),
               ("ablation: rule C, A + B learned", env(A, B), lambda: RuleSupervisor()),
               ("C_withA + A + B (extra)", env(A, B), lambda: policy("C_withA"))]
        if real:
            out.append((LEARNED_REAL["full"],
                        env(lambda: sb3_policy(load_model("A_learned")), lambda: sb3_policy(load_model("B_learned"))),
                        lambda: policy("C_learned")))
    return out


def run_group(group, scenario, record_cells=False, log=print):
    """Run every policy of a group on the held-out seeds; returns (episodes DataFrame, visitation dict)."""
    rows, visits = [], {}
    for name, ef, pf in suite(group, scenario):
        e = ef()
        pol = RandomPolicy(e.action_space) if pf == "random" else pf()
        eps = run_policy(e, pol, HELD_SEED0, HELD_N, record_cells)
        if record_cells:
            visits[name] = np.mean([r.pop("_visits") for r in eps], axis=0)
        for r in eps:
            r.update(scenario=scenario, group=group, policy=name)
        rows += eps
        d = pd.DataFrame(eps)
        log(f"  {scenario:14s} {group:8s} {name:42s} return {d['return'].mean():7.2f}  "
            f"success {d['success'].mean():.2f}  steps {d['steps'].mean():6.1f}")
    cols = ["scenario", "group", "policy", "seed", "return", "steps", "low_level_steps", "success", "coverage",
            "mean_uncertainty", "defects_found", "defects_total", "false_positives", "miou", "energy_used",
            "battery_left", "battery_empty", "out_of_bounds", "terminated", "truncated", "inference_ms",
            "n_explore", "n_refine", "n_return"]
    if record_cells:
        cols += ["pos_coverage", "pose_coverage", "visit_entropy"]
    return pd.DataFrame(rows)[cols], visits


def episodes_path(group, scenario):
    return RESULTS / f"episodes_{group}_{scenario}.csv"


STAT_METRICS = ["return", "steps", "low_level_steps", "energy_used", "coverage", "mean_uncertainty",
                "defects_found", "false_positives", "miou", "inference_ms",
                "pos_coverage", "pose_coverage", "visit_entropy"]


def summarize_group(df):
    """One row per policy: mean / std / median / 95% CI of each metric + success, efficiency, safety,
    regret vs the group's benchmark (paired on identical seeds)."""
    out = []
    group = df["group"].iloc[0]
    bench = df[df["policy"] == BENCHMARK[group]].set_index("seed")["return"] if BENCHMARK[group] in set(df["policy"]) else None
    for (scen, pol), d in df.groupby(["scenario", "policy"], sort=False):
        r = {"scenario": scen, "group": group, "policy": pol, "n_episodes": len(d),
             "seeds": f"{d['seed'].min()}-{d['seed'].max()}"}
        for m in STAT_METRICS:
            if m not in d:
                continue
            x = d[m].astype(float)
            r[f"{m}_mean"], r[f"{m}_std"] = x.mean(), x.std(ddof=1)
            r[f"{m}_median"], r[f"{m}_ci95"] = x.median(), ci95(x)
        succ, fail = d[d["success"] > 0], d[d["success"] == 0]
        r["success_rate"] = d["success"].mean()
        r["success_ci95"] = ci95(d["success"])
        r["n_success"] = len(succ)
        r["steps_success_mean"] = succ["steps"].mean() if len(succ) else np.nan
        r["steps_fail_mean"] = fail["steps"].mean() if len(fail) else np.nan
        r["time_to_goal_mean"] = succ["low_level_steps"].mean() if len(succ) else np.nan
        r["battery_empty_rate"] = d["battery_empty"].mean()
        r["oob_per_1000_steps"] = 1000.0 * d["out_of_bounds"].sum() / max(d["low_level_steps"].sum(), 1)
        n_dec = (d["n_explore"] + d["n_refine"] + d["n_return"]).sum()
        for o in ["explore", "refine", "return"]:
            r[f"freq_{o}"] = d[f"n_{o}"].sum() / n_dec if n_dec else np.nan
        if bench is not None:
            reg = bench.loc[d["seed"]].values - d["return"].values
            r["benchmark"] = BENCHMARK[group]
            r["regret_mean"], r["regret_std"], r["regret_ci95"] = reg.mean(), reg.std(ddof=1), ci95(reg)
            r["regret_cumulative"] = reg.sum()
        out.append(r)
    return pd.DataFrame(out)
