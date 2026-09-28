"""Load environments and trained agents, with clear messages and clearly-labelled baseline fallbacks."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from . import paths

AGENT_INFO = {
    "A": dict(algo="DQN", mode="navigate", name="Agent A - Viewpoint Planner (DQN)", owner="Yeseswini"),
    "B": dict(algo="PPO", mode="refine", name="Agent B - View Refiner (PPO)", owner="Leekhith"),
    "C": dict(algo="A2C", mode="supervise", name="Agent C - Mission Supervisor (A2C)", owner="Mukhesh"),
}

# Models were pickled under Python 3.13; schedules are replaced so loading never depends on the
# pickled lambdas (they are only needed for training, never for inference).
CUSTOM_OBJECTS = {
    "learning_rate": 0.0,
    "lr_schedule": lambda _: 0.0,
    "clip_range": lambda _: 0.2,
    "exploration_schedule": lambda _: 0.0,
}


class ModelMissing(FileNotFoundError):
    """Raised when a configured model file does not exist."""


@dataclass
class Agent:
    """A policy callable obs -> action plus what the audience should be told about it."""
    key: str
    label: str
    policy: Callable[[Any], Any]
    model: Any | None
    path: Path | None
    is_baseline: bool

    def __call__(self, obs):
        return self.policy(obs)

    def reset(self) -> None:
        if hasattr(self.policy, "reset"):
            self.policy.reset()


def load_env(mode: str, detector: str = "analytic", **kwargs):
    """InspectionEnv for `mode`; the learned detector gets its cache path from config.yaml."""
    from env import InspectionEnv
    cache = None
    if detector == "learned":
        cache = paths.detector_path("cache")
        if not cache.exists():
            raise FileNotFoundError(f"detector cache not found at {cache} - run preflight_check.py")
    return InspectionEnv(mode, detector=detector, cache_path=str(cache) if cache else None, **kwargs)


def _algo_class(key: str):
    import stable_baselines3 as sb3
    return getattr(sb3, AGENT_INFO[key[0]]["algo"])


@lru_cache(maxsize=16)
def load_model(key: str, detector: str = "analytic"):
    """Load a trained SB3 model on CPU; key = 'A' | 'B' | 'C' | 'C_withA'."""
    path = paths.model_path(key, detector)
    name = AGENT_INFO[key[0]]["name"]
    if path is None or not path.exists():
        raise ModelMissing(f"{name} model ({key}, {detector}) not found at {path} - run preflight_check.py")
    return _algo_class(key).load(str(path), device="cpu", custom_objects=CUSTOM_OBJECTS)


def baseline_for(key: str) -> Agent:
    """Scripted stand-in for an agent - always labelled as a BASELINE."""
    from env.baselines import ApproachRefiner, RasterNav, RuleSupervisor
    pol, name = {"A": (RasterNav(1), "RasterNav(mid)"),
                 "B": (ApproachRefiner(), "ApproachRefiner"),
                 "C": (RuleSupervisor(), "RuleSupervisor")}[key[0]]
    return Agent(key, f"BASELINE {name} (not the trained agent)", pol, None, None, True)


def load_agent(key: str, detector: str = "analytic", fallback: bool = True) -> Agent:
    """Trained agent as a deterministic policy; if the model is missing, a labelled baseline (or raise)."""
    try:
        model = load_model(key, detector)
    except ModelMissing as e:
        if not fallback:
            raise
        from . import narrate
        narrate.warn(f"{e}\n-> falling back to the scripted BASELINE, clearly labelled on screen.")
        return baseline_for(key)
    label = AGENT_INFO[key[0]]["name"] + (" [agentC_withA]" if key == "C_withA" else "")
    if detector == "learned":
        label += " [real-image detector]"
    policy = lambda obs, m=model: m.predict(obs, deterministic=True)[0]
    return Agent(key, label, policy, model, paths.model_path(key, detector), False)


def sb_version_info(path: Path) -> dict:
    """Parse system_info.txt stored inside an SB3 .zip (versions used for training)."""
    import zipfile
    with zipfile.ZipFile(path) as z:
        text = z.read("system_info.txt").decode()
    out = {}
    for line in text.splitlines():
        if ":" in line:
            k, v = line.lstrip("- ").split(":", 1)
            out[k.strip()] = v.strip()
    return out
