"""Run episodes step by step and record frames (everything the renderer and narrator need).

One Stepper drives every mode. In supervise/full mode the sub-agents inside each option are wrapped so that
every low-level move is recorded too (smooth drone animation) - the env itself is not modified.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from env.inspection_env import H, W

from . import reward as rw
from .introspect import A_ACTIONS, C_SHORT

ALT_NAMES = ["HIGH", "MID", "LOW"]


@dataclass
class Frame:
    """The world as seen after a step (or sub-step)."""
    t: int
    pos: tuple[int, int]
    alt: int
    battery: float
    u: np.ndarray                    # (H, W) uncertainty, 1.0 where unseen
    seen: np.ndarray                 # (H, W) bool
    found: np.ndarray                # (H, W) bool - true defects the detector has reported
    metrics: dict
    refine: dict | None = None       # geometry during a refine (d, th, lat, s, q, target)
    phase: str = ""                  # e.g. "EXPLORE -> Agent A"
    action: Any = None
    action_label: str = ""
    reward: float | None = None
    cum_reward: float = 0.0
    intro: dict | None = None
    breakdown: rw.Breakdown | None = None
    obs: np.ndarray | None = None
    is_substep: bool = False
    decision: int = -1


@dataclass
class Episode:
    label: str
    frames: list[Frame]
    defect: np.ndarray               # (H, W) ground-truth defects
    info: dict = field(default_factory=dict)
    total_return: float = 0.0

    @property
    def decisions(self) -> list[Frame]:
        return [f for f in self.frames[1:] if not f.is_substep]


def snapshot(env, **kw) -> Frame:
    """Copy the env state into a Frame."""
    seen = env.best_q >= 0
    q = np.clip(env.best_q, 0, None)
    found = env.pdef_tab[np.arange(H * W), q] & seen & env.defect
    ref = None
    if getattr(env, "target", None) is not None and hasattr(env, "s"):
        ref = dict(d=env.d, th=env.th, lat=env.lat, s=env.s, q=int(round(4 * env.s)), target=int(env.target),
                   target_u=float(env.cell_u()[env.target]), best_q=int(env.best_q[env.target]))
    return Frame(t=int(env.t), pos=(int(env.pos[0]), int(env.pos[1])), alt=int(env.alt),
                 battery=float(max(env.battery, 0.0)), u=env.cell_u().reshape(H, W).copy(),
                 seen=seen.reshape(H, W), found=found.reshape(H, W), metrics=env.metrics(), refine=ref, **kw)


def action_label(mode: str, action) -> str:
    if mode == "navigate":
        return A_ACTIONS[int(action)]
    if mode == "refine":
        a = np.clip(np.asarray(action, float), -1, 1)
        return f"[{a[0]:+.2f}, {a[1]:+.2f}, {a[2]:+.2f}]"
    return C_SHORT[int(action)]


class _SubRecorder:
    """Wraps a sub-agent inside an option; records the world before each of its moves."""

    def __init__(self, policy, stepper: "Stepper", phase: str):
        self.policy, self.stepper, self.phase = policy, stepper, phase

    def __call__(self, obs):
        if self.stepper.record_substeps:
            self.stepper._sub.append(snapshot(self.stepper.env, phase=self.phase, is_substep=True,
                                              decision=self.stepper.n_decisions))
        return self.policy(obs)

    def reset(self):
        if hasattr(self.policy, "reset"):
            self.policy.reset()


class Stepper:
    """Step an env with a policy, returning a fully annotated Frame per step."""

    def __init__(self, env, policy: Callable, seed: int, introspect: Callable | None = None,
                 record_substeps: bool = True):
        self.env, self.policy, self.introspect = env, policy, introspect
        self.record_substeps = record_substeps
        if env.mode in ("supervise", "full"):
            env.nav_policy = _SubRecorder(getattr(env.nav_policy, "policy", env.nav_policy), self, "EXPLORE -> Agent A")
            env.refine_policy = _SubRecorder(getattr(env.refine_policy, "policy", env.refine_policy), self,
                                             "REFINE -> Agent B")
        if hasattr(policy, "reset"):
            policy.reset()
        self.obs, _ = env.reset(seed=seed)
        self.done, self.cum, self.n_decisions = False, 0.0, 0
        self._sub: list[Frame] = []
        self.first = snapshot(env, obs=self.obs.copy(), phase="START")

    def step(self) -> list[Frame]:
        """One agent decision; returns [sub-step frames..., frame after the decision]."""
        assert not self.done, "episode already finished"
        obs = self.obs
        intro = self.introspect(obs) if self.introspect else None
        action = self.policy(obs)
        b = rw.before(self.env)
        self._sub = []
        self.obs, r, term, trunc, info = self.env.step(action)
        bd = rw.breakdown(self.env, b, action, r, info, trunc)
        self.cum += r
        self.done = term or trunc
        mode = self.env.mode
        phase = f"{C_SHORT[int(action)]}" if mode in ("supervise", "full") else ""
        f = snapshot(self.env, action=action, action_label=action_label(mode, action), reward=float(r),
                     cum_reward=self.cum, intro=intro, breakdown=bd, obs=obs.copy(), phase=phase,
                     decision=self.n_decisions)
        f.metrics.update(out_of_bounds=info["out_of_bounds"], success=info["success"])
        self.n_decisions += 1
        subs, self._sub = self._sub, []
        return subs + [f]

    def run(self, label: str, max_decisions: int | None = None) -> Episode:
        """Run to the end of the episode."""
        frames = [self.first]
        while not self.done and (max_decisions is None or self.n_decisions < max_decisions):
            frames += self.step()
        return self.episode(label, frames)

    def episode(self, label: str, frames: list[Frame]) -> Episode:
        info = dict(frames[-1].metrics)
        return Episode(label, frames, self.env.defect.reshape(H, W).copy(), info, self.cum)


def run_episode(env, policy, seed: int, label: str, introspect=None, record_substeps: bool = True) -> Episode:
    """Convenience: a whole episode with one call."""
    return Stepper(env, policy, seed, introspect, record_substeps).run(label)
