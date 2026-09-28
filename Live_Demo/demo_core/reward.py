"""Split the env's scalar reward into its terms, using the constants from inspection_env (no copies).

Each breakdown re-derives the reward from state before/after the step and reports whether the parts
sum to the reward the env actually returned (`matches`), so nothing shown on screen is invented.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from env import inspection_env as E


@dataclass
class Breakdown:
    parts: dict[str, float]
    reward: float
    extra: dict = field(default_factory=dict)

    @property
    def total(self) -> float:
        return float(sum(self.parts.values()))

    @property
    def matches(self) -> bool:
        return abs(self.total - self.reward) < 1e-5


def before(env) -> dict:
    """Snapshot everything a breakdown needs, taken just before env.step()."""
    snap = {"u": env.cell_u().copy(), "seen": env.best_q >= 0, "battery": env.battery,
            "found": env.metrics()["defects_found"]}
    if env.mode == "refine":
        snap.update(s=env.s, u_t=float(env.cell_u()[env.target]))
    return snap


def _nav(env, b, action, r, info) -> Breakdown:
    du = float(np.sum(b["u"] - env.cell_u()))
    new = int(np.sum((env.best_q >= 0) & ~b["seen"]))
    revisit = new == 0 and du < 1e-6
    parts = {f"uncertainty reduced ({E.W_UNC} x {du:.2f})": E.W_UNC * du,
             "step cost": -E.STEP_PEN}
    if revisit:
        parts["revisit penalty"] = -E.REVISIT_PEN
    if info["out_of_bounds"]:
        parts["out-of-bounds penalty"] = -E.OOB_PEN
    if info["success"]:
        parts[f"success bonus (10 + 20 x battery {max(env.battery, 0):.2f})"] = \
            E.SUCCESS_BONUS_A + E.BATTERY_BONUS_A * max(env.battery, 0.0)
    return Breakdown(parts, r, {"new_cells": new, "du": du})


def _refine(env, b, action, r, info, truncated) -> Breakdown:
    a = np.clip(np.asarray(action, np.float32), -1, 1)
    u_new = float(env.cell_u()[env.target])
    parts = {f"target uncertainty reduced ({b['u_t']:.2f} -> {u_new:.2f})": E.W_UNC_B * (b["u_t"] - u_new),
             "action cost (0.01 x |a|)": -E.ACT_PEN_B * float(np.linalg.norm(a)),
             f"shaping 0.5 x (0.99 s' - s), s {b['s']:.2f} -> {env.s:.2f}": E.SHAPE_B * (E.GAMMA_SHAPE * env.s - b["s"])}
    if info["success"]:
        parts["success bonus (q = 4 reached)"] = E.SUCCESS_B
    if truncated:
        parts["timeout"] = E.TIMEOUT_B
    return Breakdown(parts, r, {"s": env.s, "q": int(round(4 * env.s))})


def _sup(env, b, action, r, info) -> Breakdown:
    d_found = info["defects_found"] - b["found"]
    d_bat = b["battery"] - env.battery
    parts = {f"new defects found ({E.W_DEF_C} x {d_found})": E.W_DEF_C * d_found,
             f"energy used ({E.W_ENERGY_C} x {d_bat:.3f})": -E.W_ENERGY_C * d_bat}
    if env.battery <= 0:
        parts["battery empty - mission failed"] = E.FAIL_C
    return Breakdown(parts, r, {"new_defects": d_found, "energy": d_bat})


def breakdown(env, b: dict, action, r: float, info: dict, truncated: bool) -> Breakdown:
    """Dispatch on env.mode."""
    if env.mode == "navigate":
        return _nav(env, b, action, r, info)
    if env.mode == "refine":
        return _refine(env, b, action, r, info, truncated)
    return _sup(env, b, action, r, info)
