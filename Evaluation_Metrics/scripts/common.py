"""Shared helpers for Evaluation_Metrics/ (Team 08).

Everything here only IMPORTS uav_inspection/ and READS model / result files. Scenario changes
(defect density, start battery, hard refine start, observation noise) are implemented as a subclass
and a gymnasium wrapper below, never by editing the source.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

import numpy as np
import gymnasium as gym

ROOT = Path(__file__).resolve().parents[2]          # files_coding/
EM = ROOT / "Evaluation_Metrics"
RESULTS, PLOTS, VIDEOS, REPORT = EM / "results", EM / "plots", EM / "videos", EM / "report"
MULTISEED = EM / "multiseed_runs"
UAV = ROOT / "uav_inspection"
if str(UAV) not in sys.path:
    sys.path.insert(0, str(UAV))

import env.inspection_env as ie                      # noqa: E402
from env.inspection_env import InspectionEnv, sb3_policy   # noqa: E402,F401
from env.baselines import (RasterNav, ApproachRefiner, FixedOrbit, RuleSupervisor,  # noqa: E402,F401
                           RandomPolicy, evaluate, summarize, KEYS)

CACHE = ROOT / "For_Mukhesh" / "detector_cache.npz"
ORIG_SEED0, ORIG_N = 1000, 20
HELD_SEED0, HELD_N = 2000, 100
# evaluation-method settings (used by the scripts AND quoted in the report, so they cannot drift apart)
METHOD = {
    "ci_level": 95,          # confidence level of all intervals (Student t)
    "ma_window": 50,         # moving-average window of the learning curves (episodes)
    "band_std": 1,           # learning-curve band = +-1 std of the same window
    "diag_window": 10,       # TensorBoard diagnostics: mean of first / last N logged values
    "oob_per": 1000,         # out-of-bounds rate is reported per this many low-level steps
    "latency_calls": 1000,   # predict() calls in the latency micro-benchmark
}
OPTION_NAMES = ("EXPLORE", "REFINE", "RETURN")

# ------------------------------------------------------------------ model registry (read-only)
MODELS = {
    "A": UAV / "agentA/best/best_model.zip",
    "A_learned": UAV / "agentA_learned/best/best_model.zip",
    "B": ROOT / "For_Leekhith/agentB/best/best_model.zip",
    "B_learned": ROOT / "For_Leekhith/agentB_learned/best/best_model.zip",
    "C": ROOT / "For_Mukhesh/agentC/best/best_model.zip",
    "C_withA": ROOT / "For_Mukhesh/agentC_withA/best/best_model.zip",
    "C_learned": ROOT / "For_Mukhesh/agentC_learned/best/best_model.zip",
}
RUN_DIRS = {
    "A": UAV / "agentA", "A_learned": UAV / "agentA_learned",
    "A_eps0.1": UAV / "agentA_eps0.1", "A_eps0.3": UAV / "agentA_eps0.3", "A_eps0.6": UAV / "agentA_eps0.6",
    "B": ROOT / "For_Leekhith/agentB", "B_learned": ROOT / "For_Leekhith/agentB_learned",
    "B_ent0.0": ROOT / "For_Leekhith/agentB_ent0.0", "B_ent0.01": ROOT / "For_Leekhith/agentB_ent0.01",
    "B_ent0.05": ROOT / "For_Leekhith/agentB_ent0.05", "B_clip0.1": ROOT / "For_Leekhith/agentB_clip0.1",
    "B_clip0.3": ROOT / "For_Leekhith/agentB_clip0.3",
    "C": ROOT / "For_Mukhesh/agentC", "C_learned": ROOT / "For_Mukhesh/agentC_learned",
    "C_withA": ROOT / "For_Mukhesh/agentC_withA",
    "C_ent0.0": ROOT / "For_Mukhesh/agentC_ent0.0", "C_ent0.01": ROOT / "For_Mukhesh/agentC_ent0.01",
    "C_ent0.05": ROOT / "For_Mukhesh/agentC_ent0.05",
}
ALGO = {"A": "DQN", "B": "PPO", "C": "A2C"}

_model_cache: dict = {}


def load_model(key_or_path):
    """Load an SB3 model (cached). key = registry key (e.g. 'A', 'B_learned') or a path."""
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import DQN, PPO, A2C
    path = Path(MODELS.get(key_or_path, key_or_path))
    if path not in _model_cache:
        cls = {"agentA": DQN, "agentB": PPO, "agentC": A2C}
        algo = next((c for k, c in cls.items() if k in str(path)), None)
        if algo is None:   # multiseed runs: folder name starts with A_/B_/C_
            algo = {"A": DQN, "B": PPO, "C": A2C}[path.parts[-3][0]]
        _model_cache[path] = algo.load(str(path), device="cpu")
    return _model_cache[path]


class SB3Policy:
    """Deterministic SB3 policy as a callable obs -> action."""
    def __init__(self, model, deterministic=True):
        self.model, self.det = model, deterministic

    def __call__(self, obs):
        return self.model.predict(obs, deterministic=self.det)[0]


def policy(key, deterministic=True):
    return SB3Policy(load_model(key), deterministic)


# ------------------------------------------------------------------ scenario environment
class ScenarioEnv(InspectionEnv):
    """InspectionEnv with scenario knobs. With default arguments it is IDENTICAL to InspectionEnv
    (same RNG call order); tests/test_eval.py checks this for every knob.

    defect_ratio : fraction of defective cells (contract default 0.20)
    start_battery: battery at reset (default 1.0)
    hard_start   : refine-mode start geometry far away and at a large angle / offset
    Also counts low-level out-of-bounds moves, navigation steps and refine steps, which the base env
    does not report inside supervise/full options.
    """

    def __init__(self, mode="navigate", defect_ratio=ie.DEFECT_RATIO, start_battery=1.0,
                 hard_start=False, **kw):
        self.defect_ratio, self.start_battery, self.hard_start = defect_ratio, start_battery, hard_start
        self.oob_count = self.nav_steps = self.refine_steps = 0
        super().__init__(mode, **kw)

    def _new_world(self):
        if self.defect_ratio == ie.DEFECT_RATIO:
            super()._new_world()                      # byte-identical path for the nominal scenario
        else:
            saved = ie.DEFECT_RATIO
            try:                                      # _new_world reads the module constant
                ie.DEFECT_RATIO = self.defect_ratio
                super()._new_world()
            finally:
                ie.DEFECT_RATIO = saved
        self.battery = float(self.start_battery)
        self.oob_count = self.nav_steps = self.refine_steps = 0

    def _nav_step(self, a):
        r, oob = super()._nav_step(a)
        self.nav_steps += 1
        self.oob_count += int(oob)
        return r, oob

    def _refine_step(self, a):
        self.refine_steps += 1
        return super()._refine_step(a)

    def _refine_start(self, target):
        if not (self.hard_start and self.mode == "refine"):
            return super()._refine_start(target)
        rng = self.np_random
        self.target = target
        self.d = float(rng.uniform(0.9, 1.0))
        self.th = float(rng.choice([-1, 1]) * rng.uniform(0.8, 1.0))
        self.lat = float(rng.choice([-1, 1]) * rng.uniform(0.8, 1.0))
        self.s = self._quality()
        self.r_t = 0


def uncertainty_indices(mode):
    """Observation indices that carry detector uncertainty (see InspectionEnv.nav_obs etc.)."""
    if mode == "navigate":
        w = (2 * ie.VIEW_R + 1) ** 2                          # 49
        nb = (ie.H // ie.BLOCK) * (ie.W // ie.BLOCK)          # 8
        return np.r_[w:2 * w, 2 * w + nb:2 * w + 2 * nb]      # 49..97 and 106..113
    if mode == "refine":
        return np.array([3])
    return np.array([2])                                      # supervise / full: mean uncertainty


class UncertaintyNoise(gym.ObservationWrapper):
    """Gaussian noise (sigma) on the uncertainty features only, clipped to [0, 1].
    The noise generator is re-seeded from the episode seed, so runs are reproducible.
    Limitation: in supervise/full modes only Agent C's observation is noised; the sub-agents read
    their observations inside the env and stay noise-free."""

    def __init__(self, env, sigma):
        super().__init__(env)
        self.sigma = float(sigma)
        self.idx = uncertainty_indices(env.unwrapped.mode)
        self.rng = np.random.default_rng(0)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(10_000_000 + seed)
        return super().reset(seed=seed, options=options)

    def observation(self, obs):
        obs = np.array(obs, dtype=np.float32, copy=True)
        obs[self.idx] = np.clip(obs[self.idx] + self.rng.normal(0, self.sigma, len(self.idx)), 0.0, 1.0)
        return obs


# ------------------------------------------------------------------ scenarios
SCENARIOS = {
    "S0_nominal": dict(desc="analytic detector, 20% defects, battery 1.0"),
    "S1_learned": dict(desc="real-image (DeepCrack) detector", detector="learned"),
    "S2_density10": dict(desc="defect density 10%", defect_ratio=0.10),
    "S2_density35": dict(desc="defect density 35%", defect_ratio=0.35),
    "S3_noise0.05": dict(desc="uncertainty-feature noise sigma=0.05", noise=0.05),
    "S3_noise0.10": dict(desc="uncertainty-feature noise sigma=0.10", noise=0.10),
    "S4_battery0.6": dict(desc="starting battery 0.6", start_battery=0.6),
    "S5_hardstart": dict(desc="hard refine start (far, large angle/offset); Agent B only", hard_start=True),
}


def make_env(mode, scenario="S0_nominal", nav_policy=None, refine_policy=None):
    sc = SCENARIOS[scenario]
    det = sc.get("detector", "analytic")
    env = ScenarioEnv(mode, defect_ratio=sc.get("defect_ratio", ie.DEFECT_RATIO),
                      start_battery=sc.get("start_battery", 1.0), hard_start=sc.get("hard_start", False),
                      detector=det, cache_path=str(CACHE) if det == "learned" else None,
                      nav_policy=nav_policy, refine_policy=refine_policy)
    if sc.get("noise"):
        env = UncertaintyNoise(env, sc["noise"])
    return env


# ------------------------------------------------------------------ instrumented episode runner
def run_episode(env, pol, seed, record_cells=False):
    """Same loop as baselines.evaluate() (reset(seed) -> policy.reset() -> step until done), plus
    extra per-episode measurements. Returns a dict (one CSV row)."""
    obs, info = env.reset(seed=seed)
    if hasattr(pol, "reset"):
        pol.reset()
    base = env.unwrapped
    ret, done, n_act, t_pol = 0.0, False, 0, 0.0
    opts = [0, 0, 0]
    oob_steps = 0
    visits = np.zeros((3, ie.H, ie.W), int) if record_cells else None   # (altitude, row, col)
    if record_cells:
        visits[(base.alt, *base.pos)] += 1
    while not done:
        t0 = time.perf_counter()
        a = pol(obs)
        t_pol += time.perf_counter() - t0
        n_act += 1
        if base.mode in ("supervise", "full"):
            opts[int(a)] += 1
        obs, r, term, trunc, info = env.step(a)
        ret += r
        oob_steps += int(info["out_of_bounds"])
        if record_cells:
            visits[(base.alt, *base.pos)] += 1
        done = term or trunc
    row = {k: info[k] for k in KEYS if k in info}
    row.update({
        "seed": seed, "return": ret, "success": float(info["success"]),
        "terminated": bool(term), "truncated": bool(trunc),
        "out_of_bounds": int(base.oob_count),       # low-level OOB moves (incl. inside options)
        "battery_empty": float(base.battery <= 0),   # env's own termination condition (battery <= 0)
        "battery_left": float(base.battery),
        "low_level_steps": int(base.nav_steps + base.refine_steps) if base.mode in ("supervise", "full")
        else int(info["steps"]),
        "inference_ms": 1000.0 * t_pol / max(n_act, 1),
        "n_explore": opts[0], "n_refine": opts[1], "n_return": opts[2],
    })
    if record_cells:
        row["_visits"] = visits
        row.update(visitation_stats(visits))
    return row


def visitation_stats(visits):
    """State-space coverage and visitation entropy of one episode.
    position coverage = visited (row, col) cells / 128; pose coverage = visited (alt, row, col) / 384;
    visitation entropy = -sum p log p over (row, col) visit frequencies, normalised by log(128)."""
    pos = visits.sum(axis=0).ravel().astype(float)
    p = pos[pos > 0] / pos.sum()
    return {"pos_coverage": float((pos > 0).mean()), "pose_coverage": float((visits > 0).mean()),
            "visit_entropy": float(-(p * np.log(p)).sum() / np.log(pos.size))}


def run_policy(env, pol, seed0=HELD_SEED0, n=HELD_N, record_cells=False):
    return [run_episode(env, pol, seed0 + i, record_cells) for i in range(n)]


# ------------------------------------------------------------------ statistics
def ci95(x):
    """Half-width of the Student-t 95% confidence interval of the mean."""
    from scipy import stats
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return float("nan")
    q = 1 - (1 - METHOD["ci_level"] / 100) / 2
    return float(stats.t.ppf(q, n - 1) * x.std(ddof=1) / np.sqrt(n))


def git_rev():
    import subprocess
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"
