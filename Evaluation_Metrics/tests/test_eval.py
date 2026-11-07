"""Verification for Evaluation_Metrics/ (run: python -m pytest Evaluation_Metrics/tests -q)."""
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

EM = Path(__file__).resolve().parents[1]
ROOT = EM.parent
sys.path.insert(0, str(EM / "scripts"))
sys.dont_write_bytecode = True

from common import (InspectionEnv, ScenarioEnv, UncertaintyNoise, uncertainty_indices, make_env, run_policy,  # noqa: E402
                    evaluate, RasterNav, FixedOrbit, ApproachRefiner, RuleSupervisor, SCENARIOS)
import env.inspection_env as ie  # noqa: E402

REPORT_MD = EM / "report" / "Team08_Evaluation_Metrics.md"


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True)


# ------------------------------------------------------------------ Phase 1 validation
def test_validation_within_tolerance():
    v = pd.read_csv(EM / "results" / "validation.csv")
    assert len(v) > 0
    bad = v[~v.within_1pct.astype(bool)]
    assert bad.empty, bad[["run", "policy", "metric", "saved", "reproduced"]].to_string()


# ------------------------------------------------------------------ scenario subclass / wrapper
def _trajectory(env, pol, seed, n=60):
    obs, _ = env.reset(seed=seed)
    if hasattr(pol, "reset"):
        pol.reset()
    out, done, k = [obs.copy()], False, 0
    while not done and k < n:
        obs, r, te, tr, info = env.step(pol(obs))
        out.append(np.append(obs, r))
        done, k = te or tr, k + 1
    return np.concatenate(out)


@pytest.mark.parametrize("mode,pol", [("navigate", RasterNav(2)), ("refine", FixedOrbit()),
                                      ("supervise", RuleSupervisor())])
@pytest.mark.parametrize("kw", [{}, {"defect_ratio": 0.20}, {"start_battery": 1.0}, {"hard_start": False}])
def test_scenario_env_defaults_equal_base(mode, pol, kw):
    """S2 / S4 / S5 subclass at default values must reproduce the base env exactly."""
    for seed in (2000, 2001, 2002):
        a = _trajectory(InspectionEnv(mode), pol, seed)
        b = _trajectory(ScenarioEnv(mode, **kw), pol, seed)
        assert a.shape == b.shape and np.array_equal(a, b)


def test_scenario_knobs_take_effect():
    e = ScenarioEnv("navigate", defect_ratio=0.35)
    e.reset(seed=2000)
    assert e.defect.sum() == round(0.35 * ie.N_CELLS)
    assert ie.DEFECT_RATIO == 0.20                      # module constant restored
    e = ScenarioEnv("navigate", start_battery=0.6)
    obs, _ = e.reset(seed=2000)
    assert abs(obs[-1] - 0.6) < 1e-6
    e = ScenarioEnv("refine", hard_start=True)
    e.reset(seed=2000)
    assert e.d >= 0.9 and abs(e.th) >= 0.8 and abs(e.lat) >= 0.8


@pytest.mark.parametrize("mode", ["navigate", "refine", "supervise"])
def test_noise_only_on_uncertainty_features(mode):
    clean, noisy = InspectionEnv(mode), UncertaintyNoise(InspectionEnv(mode), 0.10)
    o1, _ = clean.reset(seed=2000)
    o2, _ = noisy.reset(seed=2000)
    idx = uncertainty_indices(mode)
    other = np.setdiff1d(np.arange(len(o1)), idx)
    assert np.array_equal(o1[other], o2[other])
    assert np.all((o2[idx] >= 0) & (o2[idx] <= 1))


def test_runner_matches_baselines_evaluate():
    a = evaluate(InspectionEnv("navigate"), RasterNav(1), 3, 2000)
    b = run_policy(make_env("navigate"), RasterNav(1), 2000, 3)
    assert [round(x["return"], 9) for x in a] == [round(x["return"], 9) for x in b]


# ------------------------------------------------------------------ report numbers
def test_every_registered_number_matches_its_source():
    import build_report
    reg = pd.read_csv(EM / "report" / "report_numbers.csv", dtype=str, keep_default_na=False)
    assert len(reg) > 100
    for r in reg.itertuples():
        import json
        key = json.loads(r.key)
        v = build_report.lookup(r.source, r.column, r.agg or None, **key) * float(r.scale)
        assert format(v, r.fmt) == r.text, (r.source, r.column, key, r.text, format(v, r.fmt))


def test_no_unregistered_decimal_in_report_text():
    md = REPORT_MD.read_text(encoding="utf-8")
    reg = set(pd.read_csv(EM / "report" / "report_numbers.csv", dtype=str)["text"].str.lstrip("+-"))
    txt = re.sub(r"```.*?```", "", md, flags=re.S)
    txt = re.sub(r"`[^`]*`", "", txt)
    txt = "\n".join(l for l in txt.splitlines() if not l.startswith("#") and not l.startswith("!["))
    allowed = set(re.findall(r"\d+\.\d+", " ".join(list(SCENARIOS) + [v["desc"] for v in SCENARIOS.values()])))
    bad = sorted({t for t in re.findall(r"(?<![\w.])\d+\.\d+(?![\w.])", txt) if t not in reg and t not in allowed})
    assert not bad, f"numbers in the report that do not come from a result file: {bad}"


# ------------------------------------------------------------------ referenced files exist
def test_referenced_plots_exist():
    md = REPORT_MD.read_text(encoding="utf-8")
    imgs = re.findall(r"!\[[^\]]*\]\(\.\./(plots/[^)]+)\)", md)
    assert len(imgs) >= 12
    for p in imgs:
        assert (EM / p).exists(), p
    for p in pd.read_csv(EM / "plots" / "plot_index.csv").file:
        assert (EM / p).exists(), p


def test_referenced_videos_exist_and_meet_limits():
    md = REPORT_MD.read_text(encoding="utf-8")
    vids = set(re.findall(r"videos/(V\d+_[\w.\-]+\.mp4)", md))
    assert len(vids) >= 9
    meta = pd.read_csv(EM / "videos" / "video_files.csv").set_index("file")
    for v in vids:
        p = EM / "videos" / v
        assert p.exists(), v
        assert p.stat().st_size <= 25e6, v
        assert meta.loc[f"videos/{v}", "duration_s"] <= 60, v
    for f in ["video_index.md", "narration_script.md"]:
        assert (EM / "videos" / f).exists()


def test_video_runs_match_evaluation_csvs():
    """The recorded sessions are the same episodes as in the evaluation tables (same seed -> same result)."""
    runs = pd.read_csv(EM / "videos" / "video_runs.csv")
    checks = {"V2_": ("S0_nominal", "1000"), "V3_": ("S0_nominal", None), "V4_": ("S1_learned", None),
              "V9_": ("S2_density35", None), "V10_": ("S4_battery0.6", None)}
    pol = {"Agent A - Viewpoint Planner (DQN)": "DQN (Agent A)",
           "Agent A - Viewpoint Planner (DQN) [real-image detector]": "DQN (Agent A, real-image trained)",
           "BASELINE Dense raster, LOW altitude (1x1)": "raster_low"}
    n = 0
    for prefix, (scen, skip) in checks.items():
        if skip:            # seed 1000 is not a held-out seed
            continue
        ep = pd.read_csv(EM / "results" / f"episodes_A_{scen}.csv")
        for r in runs[runs["clip"].str.startswith(prefix)].to_dict("records"):
            e = ep[(ep.seed == r["seed"]) & (ep.policy == pol[r["label"]])]
            assert len(e) == 1
            assert abs(e["return"].iloc[0] - r["return"]) < 1e-6, (prefix, r["label"], r["seed"])
            assert int(e["steps"].iloc[0]) == int(r["steps"])
            n += 1
    assert n >= 8


# ------------------------------------------------------------------ nothing changed outside Evaluation_Metrics/
def test_no_changes_outside_evaluation_metrics():
    dirty = [l[3:] for l in git("status", "--porcelain", "--untracked-files=all").splitlines()]
    outside = [p for p in dirty if not p.strip('"').startswith("Evaluation_Metrics/")]
    assert not outside, f"working-tree changes outside Evaluation_Metrics/: {outside}"
    base = git("merge-base", "HEAD", "main").strip()
    committed = git("diff", "--name-only", base, "HEAD").splitlines()
    outside = [p for p in committed if not p.startswith("Evaluation_Metrics/")]
    assert not outside, f"committed changes outside Evaluation_Metrics/: {outside}"


def test_live_demo_untouched():
    assert not [l for l in git("status", "--porcelain", "--", "Live_Demo").splitlines()]
