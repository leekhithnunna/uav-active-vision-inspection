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


# Identifiers that are names, not measurements (allowed with the reason shown in number_trace.csv)
IDENTIFIERS = {"08": "team number (Team 08)"}


def trace_report_numbers():
    """Every number (decimal or whole) in the report prose, with where it is traced from.
    Excluded: code / formula spans, headings, image links, and labels such as Table 3, Figure 2, Q1, S4, V7, Phase 5."""
    md = REPORT_MD.read_text(encoding="utf-8")
    reg = pd.read_csv(EM / "report" / "report_numbers.csv", dtype=str, keep_default_na=False)
    reg_texts = {t.lstrip("+-").replace(",", ""): f"{s}:{c}" for t, s, c in zip(reg.text, reg.source, reg.column)}
    scen_text = " ".join(list(SCENARIOS) + [v["desc"] for v in SCENARIOS.values()])
    vf = pd.read_csv(EM / "videos" / "video_files.csv")
    clip_text = " ".join(vf[["file", "scenario", "question", "metric"]].astype(str).values.ravel())
    num_re = r"(?<![\w.,\-])\d[\d,]*(?:\.\d+)?(?![\w.]|,\d)"
    defined = {"scenario definition (common.SCENARIOS)": set(re.findall(r"\d+(?:\.\d+)?", scen_text)),
               "video clip definition (make_videos.CLIPS)": set(re.findall(r"\d+(?:\.\d+)?", clip_text))}
    txt = re.sub(r"```.*?```", "", md, flags=re.S)
    txt = re.sub(r"`[^`]*`", "", txt)
    rows = []
    for ln, line in enumerate(txt.splitlines(), 1):
        if line.startswith("#") or line.startswith("!["):
            continue
        line = re.sub(r"\b(Table|Figure|Phase)\s+\d+|\b[QSV]\d+\b", " ", line)
        for tok in re.findall(num_re, line):
            t = tok.replace(",", "")
            src = (reg_texts.get(t) and f"result/constant registry ({reg_texts[t]})") or IDENTIFIERS.get(t)
            if not src:
                src = next((k for k, v in defined.items() if t in v), "")
            rows.append({"line": ln, "number": t, "traced_to": src or "UNTRACED", "context": line.strip()[:120]})
    df = pd.DataFrame(rows)
    df.to_csv(EM / "report" / "number_trace.csv", index=False)
    return df


def test_every_number_in_report_text_is_traced():
    """Whole numbers AND decimals: each must come from the number registry (result files, env constants,
    method settings), a scenario / clip definition, or a named identifier. Untraced ones are listed."""
    df = trace_report_numbers()
    assert len(df) > 200
    bad = df[df.traced_to == "UNTRACED"]
    assert bad.empty, "untraced numbers in the report:\n" + bad.to_string(index=False)


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


def test_voice_over_fits_each_clip():
    n = pd.read_csv(EM / "videos" / "narration_lengths.csv")
    assert len(n) == 10
    too_long = n[n.speech_s > n.video_s]
    assert too_long.empty, too_long.to_string(index=False)


def test_preview_frames_exist():
    for mp4 in (EM / "videos").glob("V*.mp4"):
        for tag in ("mid", "end"):
            assert (EM / "videos" / "preview_frames" / f"{mp4.stem}_{tag}.png").exists(), (mp4.stem, tag)


def test_submission_folder():
    sub = EM / "SUBMISSION"
    names = sorted(p.name for p in sub.iterdir())
    expected = sorted(["Team08_Evaluation_Metrics.pdf", "Team08_Evaluation_Metrics.docx", "video_index.md"]
                      + [p.name for p in (EM / "videos").glob("V*.mp4")])
    assert names == expected, names
    for p in sub.iterdir():
        assert p.stat().st_size <= 25e6, p.name
    for name in ["Team08_Evaluation_Metrics.pdf", "Team08_Evaluation_Metrics.docx"]:     # copies are current
        assert (sub / name).read_bytes() == (EM / "report" / name).read_bytes()
    for p in (EM / "videos").glob("V*.mp4"):
        assert (sub / p.name).read_bytes() == p.read_bytes()


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
