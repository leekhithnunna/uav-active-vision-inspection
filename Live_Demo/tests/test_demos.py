"""Smoke tests: every demo runs headless with a fixed seed in < 60 s, and the core library tells the truth.

    pytest Live_Demo/tests -q
"""
from __future__ import annotations

import importlib
import time

import numpy as np
import pytest

SEED = 1000
HEADLESS = ["--headless", "--fast", "--seed", str(SEED)]
DEMOS = [
    ("d1_env_tour", []),
    ("d2_agentA_step", ["--auto"]),
    ("d3_agentA_vs_raster", []),
    ("d4_agentB_refine", []),
    ("d5_detector_views", []),
    ("d6_agentC_mission", []),
    ("d7_full_mission", []),
    ("d7_full_mission", ["--supervisor", "rule"]),
    ("d8_learning_burst", ["--agent", "A"]),
    ("d8_learning_burst", ["--agent", "B"]),
    ("d8_learning_burst", ["--agent", "C"]),
    ("d9_results_dashboard", []),
]


@pytest.mark.parametrize("name,extra", DEMOS, ids=[f"{n}{'-' + '-'.join(e) if e else ''}" for n, e in DEMOS])
def test_demo_runs_headless(name, extra):
    t0 = time.time()
    out = importlib.import_module(f"demos.{name}").main(HEADLESS + extra)
    assert time.time() - t0 < 60, f"{name} too slow for the live slot"
    assert out is not None


# ---------------------------------------------------------------- core library


def test_models_load_as_trained_agents():
    from demo_core import loaders
    for key in ("A", "B", "C"):
        agent = loaders.load_agent(key, fallback=False)
        assert not agent.is_baseline and agent.model is not None


def test_missing_model_falls_back_to_labelled_baseline(monkeypatch):
    from demo_core import loaders, paths
    monkeypatch.setattr(paths, "model_path", lambda key, detector="analytic": paths.DEMO_ROOT / "nope.zip")
    loaders.load_model.cache_clear()
    agent = loaders.load_agent("B")
    loaders.load_model.cache_clear()
    assert agent.is_baseline and "BASELINE" in agent.label


def test_same_seed_same_episode():
    from demo_core import loaders, rollout
    agent = loaders.load_agent("A")
    runs = [rollout.run_episode(loaders.load_env("navigate"), agent, SEED, "A") for _ in range(2)]
    assert runs[0].total_return == runs[1].total_return
    assert [f.pos for f in runs[0].frames] == [f.pos for f in runs[1].frames]


def test_dqn_argmax_is_executed_action_and_rewards_add_up():
    from demo_core import introspect, loaders, rollout
    agent = loaders.load_agent("A")
    ep = rollout.run_episode(loaders.load_env("navigate"), agent, SEED, "A",
                             lambda o: introspect.q_values(agent.model, o))
    for f in ep.frames[1:]:
        assert f.intro["chosen"] == int(f.action)
        assert f.breakdown.matches, f.breakdown.parts


def test_ppo_gaussian_and_refine_breakdown():
    from demo_core import introspect, loaders, rollout
    agent = loaders.load_agent("B")
    ep = rollout.run_episode(loaders.load_env("refine"), agent, SEED, "B",
                             lambda o: introspect.gaussian(agent.model, o))
    for f in ep.frames[1:]:
        assert np.allclose(np.clip(f.intro["mu"], -1, 1), np.clip(f.action, -1, 1), atol=1e-5)
        assert (f.intro["sigma"] > 0).all()
        assert f.breakdown.matches, f.breakdown.parts


def test_a2c_probs_and_full_mission_matches_saved_demo():
    """d7's live mission on seed 1000 must reproduce For_Leekhith/full/demo_mission.txt."""
    from demo_core import introspect, loaders, paths, rollout
    A, B, C = (loaders.load_agent(k) for k in "ABC")
    ep = rollout.run_episode(loaders.load_env("full", nav_policy=A, refine_policy=B), C, SEED, "C",
                             lambda o: introspect.option_probs(C.model, o))
    for f in ep.decisions:
        assert abs(f.intro["values"].sum() - 1) < 1e-5
        assert f.breakdown.matches
    saved = paths.resolve(paths.config()["results"]["full"]["mission"]).read_text()
    assert f"MISSION END: return {ep.total_return:.2f}" in saved


def test_results_are_read_not_invented():
    from demo_core import metrics, paths
    df = metrics.vs_baselines(paths.results_entry("A"))
    assert df.iloc[0]["policy"].startswith("DQN")
    assert metrics.run_note(paths.results_entry("A", "sensitivity")[0], paths.results_entry("A")).startswith("short run")
    assert len(metrics.full_results()) == 4


def test_detector_unet_from_source():
    from demo_core import detector
    ns = detector.unet_namespace()
    assert ns["RES"] == [4, 8, 16, 32, 64]
    if detector.dataset_available():
        patches = detector.test_patches(SEED, 1, 1)
        views = detector.analyse(patches[0], mc=3)
        assert len(views) == 5 and patches[0].is_crack


def test_streamlit_app_renders():
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    from demo_core import paths
    at = AppTest.from_file(str(paths.DEMO_ROOT / "app" / "streamlit_app.py"), default_timeout=120).run()
    assert not at.exception
    at.button(key="next-A").click().run()
    assert not at.exception and at.slider(key="idx-A").value == 1
