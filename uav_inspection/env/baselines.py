"""
Baseline / stub policies + a shared evaluation helper.
Every policy is a callable: action = policy(obs). Stateful ones expose reset().
"""
import numpy as np
from .inspection_env import H, W


# ------------------------------------------------------------ helpers
def _nav_pose(obs):
    """Decode (row, col, altitude) from a navigate-mode observation (last 4 values)."""
    r, c, alt = obs[-4], obs[-3], obs[-2]
    return int(round(r * (H - 1))), int(round(c * (W - 1))), int(round(alt * 2))


def _serpentine(rows, c_lo, c_hi):
    wps = []
    for i, r in enumerate(rows):
        cols = range(c_lo, c_hi + 1) if i % 2 == 0 else range(c_hi, c_lo - 1, -1)
        wps += [(r, c) for c in cols]
    return wps


# ------------------------------------------------------------ Agent A baselines / stub
class RasterNav:
    """Passive lawnmower scan at a fixed altitude (0=high, 1=mid, 2=low)."""
    PLANS = {0: ([2, 6], 2, 13), 1: ([1, 4, 7], 1, 14), 2: (list(range(H)), 0, W - 1)}

    def __init__(self, altitude=1):
        self.altitude = altitude
        self.wps = _serpentine(*self.PLANS[altitude])
        self.reset()

    def reset(self):
        self.i = 0

    def __call__(self, obs):
        r, c, alt = _nav_pose(obs)
        if alt != self.altitude:
            return 5 if alt < self.altitude else 4          # 5 = descend, 4 = climb
        if (r, c) == self.wps[self.i]:
            self.i = (self.i + 1) % len(self.wps)
        tr, tc = self.wps[self.i]
        if tr != r:
            return 0 if tr < r else 1                        # N / S
        return 2 if tc > c else 3                            # E / W


# ------------------------------------------------------------ Agent B baselines / stub
class ApproachRefiner:
    """Scripted stub: move straight towards the ideal view (d=0, angle=0, lateral=0)."""
    def __call__(self, obs):
        d, th, lat = obs[0], obs[1], obs[2]
        return np.clip(-np.array([d, th, lat]) / 0.2, -1, 1).astype(np.float32)


class FixedOrbit:
    """Weak baseline: keep distance, sweep the viewing angle back and forth."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.dir = 1.0

    def __call__(self, obs):
        if obs[1] > 0.95:
            self.dir = -1.0
        elif obs[1] < -0.95:
            self.dir = 1.0
        return np.array([0.0, self.dir, 0.0], np.float32)


# ------------------------------------------------------------ Agent C baseline
class RuleSupervisor:
    """Explore twice, refine once; return home when battery < 15%."""
    def __init__(self, low_battery=0.15):
        self.low = low_battery
        self.reset()

    def reset(self):
        self.k = 0

    def __call__(self, obs):
        self.k += 1
        if obs[0] < self.low:
            return 2
        return 1 if self.k % 3 == 0 else 0


class RandomPolicy:
    def __init__(self, action_space, seed=0):
        self.space = action_space
        self.space.seed(seed)

    def __call__(self, obs):
        return self.space.sample()


# ------------------------------------------------------------ evaluation
KEYS = ["return", "steps", "success", "coverage", "mean_uncertainty", "defects_found",
        "defects_total", "false_positives", "miou", "energy_used"]


def evaluate(env, policy, n_episodes=20, seed0=1000):
    """Run n episodes with fixed seeds (same worlds for every policy -> fair comparison)."""
    rows = []
    for i in range(n_episodes):
        obs, info = env.reset(seed=seed0 + i)
        if hasattr(policy, "reset"):
            policy.reset()
        ret, done = 0.0, False
        while not done:
            obs, r, term, trunc, info = env.step(policy(obs))
            ret += r
            done = term or trunc
        row = {k: info[k] for k in KEYS if k in info}
        row["return"], row["success"] = ret, float(info["success"])
        rows.append(row)
    return rows


def summarize(rows):
    return {k: (float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows])))
            for k in KEYS}
