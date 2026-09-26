"""
Team 08 - Uncertainty-Aware Active Vision for UAV-Based Infrastructure Inspection
Shared Gymnasium environment (design contract v1.1, see README.md).

Modes
  navigate  -> Agent A (DQN)  action Discrete(6)  obs Box(118)
  refine    -> Agent B (PPO)  action Box(3)       obs Box(6)
  supervise -> Agent C (A2C)  action Discrete(3)  obs Box(6)   sub-agents = scripted stubs
  full      -> same as supervise, but you pass the TRAINED nav/refine policies

Detector
  analytic  -> formula-based uncertainty (no data needed, always works)
  learned   -> lookup table from detector_cache.npz (DeepCrack + U-Net + MC-dropout)
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ---------------- world constants (contract) ----------------
H, W = 8, 16                      # bridge surface grid
N_CELLS = H * W
N_Q = 5                           # view quality levels q = 0..4
ALT_RADIUS = (2, 1, 0)            # altitude 0=high,1=mid,2=low -> footprint 5x5, 3x3, 1x1
ALT_Q = (1, 2, 3)                 # quality reached from each altitude (q4 only via refine)
MOVES = {0: (-1, 0), 1: (1, 0), 2: (0, 1), 3: (0, -1)}   # N, S, E, W
BASE = (0, 0)                     # take-off / landing point

COST_MOVE, COST_ALT, COST_REFINE = 0.005, 0.01, 0.008
DEFECT_RATIO = 0.20

# ---------------- analytic detector tables (index = q) ----------------
U_DEFECT = np.array([0.90, 0.75, 0.55, 0.30, 0.08])   # defects are more ambiguous
U_CLEAN = np.array([0.80, 0.50, 0.30, 0.15, 0.05])
P_DETECT = np.array([0.20, 0.50, 0.80, 0.95, 0.99])   # true-positive prob
P_FALSE = np.array([0.10, 0.05, 0.02, 0.01, 0.00])    # false-positive prob
IOU_DEF = np.array([0.10, 0.30, 0.50, 0.65, 0.80])    # segmentation quality on defects

# ---------------- reward weights ----------------
# Agent A (navigate)
W_UNC, STEP_PEN, REVISIT_PEN, OOB_PEN = 0.2, 0.01, 0.05, 1.0   # W_UNC x total uncertainty reduced (unseen cell = 1.0)
SUCCESS_BONUS_A, BATTERY_BONUS_A = 10.0, 20.0   # success reward = 10 + 20 * battery left (finish fast & cheap)
# Agent B (refine)
W_UNC_B, ACT_PEN_B, SHAPE_B, SUCCESS_B, TIMEOUT_B, GAMMA_SHAPE = 1.0, 0.01, 0.5, 5.0, -2.0, 0.99
# Agent C (supervise)
W_DEF_C, W_ENERGY_C, FAIL_C = 2.0, 0.5, -10.0

VIEW_R = 3                        # Agent A sees a (2*3+1)^2 = 7x7 local window
BLOCK = 4                         # + coarse 4x4-cell block summary of the whole bridge
NAV_OBS_DIM = 2 * (2 * VIEW_R + 1) ** 2 + 2 * (H // BLOCK) * (W // BLOCK) + 4
MAX_STEPS = {"navigate": 200, "refine": 30, "supervise": 40, "full": 40}
EXPLORE_K, REFINE_K = 5, 10       # sub-steps per supervisor option


class InspectionEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, mode="navigate", detector="analytic", cache_path=None,
                 nav_policy=None, refine_policy=None,
                 success_cov=0.95, success_unc=None, render_mode=None):
        assert mode in MAX_STEPS, f"mode must be one of {list(MAX_STEPS)}"
        assert detector in ("analytic", "learned")
        self.mode, self.detector = mode, detector
        self.render_mode = render_mode
        self.max_steps = MAX_STEPS[mode]
        self.success_cov, self.success_unc = success_cov, 0.32     # analytic default
        self.alt_q = ALT_Q                                         # altitude -> view quality

        if detector == "learned":
            self._load_cache(cache_path)                           # may set a calibrated success_unc
        if success_unc is not None:                                # explicit value always wins
            self.success_unc = success_unc

        # observation / action spaces per mode
        if mode == "navigate":
            self.observation_space = spaces.Box(0.0, 1.0, (NAV_OBS_DIM,), np.float32)
            self.action_space = spaces.Discrete(6)
        elif mode == "refine":
            self.observation_space = spaces.Box(
                np.array([0, -1, -1, 0, 0, 0], np.float32), np.ones(6, np.float32))
            self.action_space = spaces.Box(-1.0, 1.0, (3,), np.float32)
        else:  # supervise / full
            self.observation_space = spaces.Box(0.0, 1.0, (6,), np.float32)
            self.action_space = spaces.Discrete(3)
            from .baselines import RasterNav, ApproachRefiner   # default stubs
            if mode == "full":
                assert nav_policy is not None and refine_policy is not None, \
                    "full mode needs trained nav_policy and refine_policy"
            self.nav_policy = nav_policy or RasterNav(altitude=1)
            self.refine_policy = refine_policy or ApproachRefiner()

    # ------------------------------------------------------------------ cache
    def _load_cache(self, path):
        """Cache contract: u (P,5) float in [0,1], pdef (P,5) bool, iou (P,5) float, is_defect (P,) bool."""
        assert path is not None, "detector='learned' needs cache_path"
        c = np.load(path)
        for k in ("u", "pdef", "iou", "is_defect"):
            assert k in c, f"cache missing key '{k}'"
        self.c_u = c["u"].astype(np.float32)
        self.c_pdef = c["pdef"].astype(bool)
        self.c_iou = c["iou"].astype(np.float32)
        isd = c["is_defect"].astype(bool)
        assert self.c_u.shape[1] == N_Q, "cache must have 5 quality levels"
        self.def_pool, self.clean_pool = np.where(isd)[0], np.where(~isd)[0]
        if "success_unc" in c:                                     # threshold calibrated by the cache builder
            self.success_unc = float(c["success_unc"])
        if "alt_q" in c:                                           # altitude->quality map calibrated by the builder
            self.alt_q = tuple(int(v) for v in c["alt_q"])
        assert len(self.def_pool) > 0 and len(self.clean_pool) > 0, "need both defect and clean patches"

    # ------------------------------------------------------------------ world
    def _new_world(self):
        rng = self.np_random
        n_def = int(round(DEFECT_RATIO * N_CELLS))
        self.defect = np.zeros(N_CELLS, bool)
        self.defect[rng.choice(N_CELLS, n_def, replace=False)] = True

        if self.detector == "analytic":
            base = np.where(self.defect[:, None], U_DEFECT, U_CLEAN) + rng.normal(0, 0.03, (N_CELLS, N_Q))
            self.u_tab = np.minimum.accumulate(np.clip(base, 0.01, 1.0), axis=1)  # better view -> lower u
            r = rng.random((N_CELLS, N_Q))
            self.pdef_tab = np.where(self.defect[:, None], r < P_DETECT, r < P_FALSE)
            self.iou_tab = np.where(self.defect[:, None],
                                    np.clip(IOU_DEF + rng.normal(0, 0.05, (N_CELLS, N_Q)), 0, 1), 0.0)
        else:  # learned: assign real image patches to cells
            pid = np.empty(N_CELLS, int)
            pid[self.defect] = rng.choice(self.def_pool, n_def, replace=len(self.def_pool) < n_def)
            pid[~self.defect] = rng.choice(self.clean_pool, N_CELLS - n_def,
                                           replace=len(self.clean_pool) < N_CELLS - n_def)
            self.patch_id = pid
            self.u_tab, self.pdef_tab, self.iou_tab = self.c_u[pid], self.c_pdef[pid], self.c_iou[pid]

        self.best_q = np.full(N_CELLS, -1, int)     # -1 = never seen
        self.pos, self.alt = list(BASE), 0          # start at base, high altitude
        self.battery, self.t = 1.0, 0

    def _idx(self, r, c):
        return r * W + c

    def cell_u(self):
        seen = self.best_q >= 0
        return np.where(seen, self.u_tab[np.arange(N_CELLS), np.clip(self.best_q, 0, None)], 1.0)

    def _footprint(self):
        rad = ALT_RADIUS[self.alt]
        r0, c0 = self.pos
        return [self._idx(r, c) for r in range(max(0, r0 - rad), min(H, r0 + rad + 1))
                for c in range(max(0, c0 - rad), min(W, c0 + rad + 1))]

    def _observe(self):
        """Camera view at current pose. Returns (#new cells, total uncertainty reduction; unseen cell counts as u=1)."""
        q = self.alt_q[self.alt]
        u_before = self.cell_u()
        cells = self._footprint()
        new = int(np.sum(self.best_q[cells] < 0))
        self.best_q[cells] = np.maximum(self.best_q[cells], q)
        du = float(np.sum(u_before[cells] - self.cell_u()[cells]))
        return new, du

    def metrics(self):
        seen = self.best_q >= 0
        q = np.clip(self.best_q, 0, None)
        det = self.pdef_tab[np.arange(N_CELLS), q] & seen
        iou = np.where(seen, self.iou_tab[np.arange(N_CELLS), q], 0.0)
        return {
            "coverage": float(seen.mean()),
            "mean_uncertainty": float(self.cell_u().mean()),
            "defects_found": int(np.sum(det & self.defect)),
            "false_positives": int(np.sum(det & ~self.defect)),
            "defects_total": int(self.defect.sum()),
            "miou": float(iou[self.defect].mean()),
            "energy_used": float(1.0 - self.battery),
            "steps": int(self.t),
            "out_of_bounds": False,
            "success": False,
        }

    def _nav_success(self):
        m = self.metrics()
        return m["coverage"] >= self.success_cov and m["mean_uncertainty"] <= self.success_unc

    # ------------------------------------------------------------------ Agent A core
    def nav_obs(self):
        """Egocentric observation (easier for an MLP than a global one-hot position):
        7x7 local window of [visited, uncertainty] centred on the UAV (outside the bridge = visited, u=0)
        + per-block [coverage, mean uncertainty] for 2x4 blocks of 4x4 cells
        + [row, col, altitude, battery] normalised."""
        visited = (self.best_q >= 0).reshape(H, W).astype(np.float32)
        u = self.cell_u().reshape(H, W).astype(np.float32)
        pv = np.pad(visited, VIEW_R, constant_values=1.0)
        pu = np.pad(u, VIEW_R, constant_values=0.0)
        r, c = self.pos
        win_v = pv[r:r + 2 * VIEW_R + 1, c:c + 2 * VIEW_R + 1].ravel()
        win_u = pu[r:r + 2 * VIEW_R + 1, c:c + 2 * VIEW_R + 1].ravel()
        blk_v = visited.reshape(H // BLOCK, BLOCK, W // BLOCK, BLOCK).mean(axis=(1, 3)).ravel()
        blk_u = u.reshape(H // BLOCK, BLOCK, W // BLOCK, BLOCK).mean(axis=(1, 3)).ravel()
        tail = [r / (H - 1), c / (W - 1), self.alt / 2.0, max(self.battery, 0.0)]
        return np.concatenate([win_v, win_u, blk_v, blk_u, tail]).astype(np.float32)

    def _nav_step(self, a):
        a, oob = int(a), False
        if a < 4:
            dr, dc = MOVES[a]
            nr, nc = self.pos[0] + dr, self.pos[1] + dc
            if 0 <= nr < H and 0 <= nc < W:
                self.pos = [nr, nc]
            else:
                oob = True
            self.battery -= COST_MOVE
        else:
            na = self.alt - 1 if a == 4 else self.alt + 1   # 4 = up (higher), 5 = down (lower)
            if 0 <= na <= 2:
                self.alt = na
            else:
                oob = True
            self.battery -= COST_ALT
        new, du = self._observe()
        revisit = new == 0 and du < 1e-6
        r = W_UNC * du - STEP_PEN - REVISIT_PEN * revisit - OOB_PEN * oob
        return r, oob

    # ------------------------------------------------------------------ Agent B core
    def _refine_start(self, target):
        rng = self.np_random
        self.target = target
        self.d = float(rng.uniform(0.5, 1.0))
        self.th = float(rng.uniform(-1, 1))
        self.lat = float(rng.uniform(-1, 1))
        self.s = self._quality()
        self.r_t = 0

    def _quality(self):
        return float(np.clip(1 - 0.6 * self.d - 0.4 * abs(self.th) - 0.3 * abs(self.lat), 0, 1))

    def refine_obs(self):
        u = self.cell_u()[self.target]
        horizon = MAX_STEPS["refine"] if self.mode == "refine" else REFINE_K
        return np.array([self.d, self.th, self.lat, u, max(self.battery, 0.0),
                         max(0.0, 1 - self.r_t / horizon)], np.float32)

    def _refine_step(self, a):
        a = np.clip(np.asarray(a, np.float32), -1, 1)
        self.d = float(np.clip(self.d + 0.2 * a[0], 0, 1))
        self.th = float(np.clip(self.th + 0.2 * a[1], -1, 1))
        self.lat = float(np.clip(self.lat + 0.2 * a[2], -1, 1))
        self.battery -= COST_REFINE
        self.r_t += 1
        u_prev = self.cell_u()[self.target]
        s_prev, self.s = self.s, self._quality()
        q = int(round(4 * self.s))
        self.best_q[self.target] = max(self.best_q[self.target], q)
        u_new = self.cell_u()[self.target]
        success = bool(self.best_q[self.target] == N_Q - 1)       # best possible view reached
        r = (W_UNC_B * (u_prev - u_new) - ACT_PEN_B * float(np.linalg.norm(a))
             + SHAPE_B * (GAMMA_SHAPE * self.s - s_prev)          # potential-based shaping
             + SUCCESS_B * success)
        return r, success

    # ------------------------------------------------------------------ Agent C helpers
    def sup_obs(self):
        m = self.metrics()
        dist = abs(self.pos[0] - BASE[0]) + abs(self.pos[1] - BASE[1])
        return np.array([max(self.battery, 0.0), m["coverage"], m["mean_uncertainty"],
                         m["defects_found"] / max(m["defects_total"], 1),
                         dist / (H + W - 2), min(1.0, self.t / self.max_steps)], np.float32)

    def _most_uncertain_in_view(self):
        cells = self._footprint()
        u = self.cell_u()
        return max(cells, key=lambda i: u[i])

    # ------------------------------------------------------------------ gym API
    def _obs(self):
        if self.mode == "navigate":
            return self.nav_obs()
        if self.mode == "refine":
            return self.refine_obs()
        return self.sup_obs()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._new_world()
        self._observe()                                   # first camera view from base
        if self.mode == "refine":
            tgt = int(self.np_random.integers(N_CELLS))
            self.best_q[tgt] = max(self.best_q[tgt], int(self.np_random.integers(1, 3)))
            self._refine_start(tgt)
        if self.mode in ("supervise", "full"):
            for p in (self.nav_policy, self.refine_policy):
                if hasattr(p, "reset"):
                    p.reset()
        return self._obs(), self.metrics()

    def step(self, action):
        terminated, oob, success = False, False, False

        if self.mode == "navigate":
            self.t += 1
            r, oob = self._nav_step(action)
            if self._nav_success():
                r += SUCCESS_BONUS_A + BATTERY_BONUS_A * max(self.battery, 0.0)
                terminated = success = True
            elif self.battery <= 0:
                terminated = True

        elif self.mode == "refine":
            self.t += 1
            r, success = self._refine_step(action)
            terminated = success
            if not terminated and self.t >= self.max_steps:
                r += TIMEOUT_B

        else:  # supervise / full
            self.t += 1
            found0, bat0 = self.metrics()["defects_found"], self.battery
            a = int(action)
            if a == 0:        # explore: K navigator steps
                for _ in range(EXPLORE_K):
                    self._nav_step(self.nav_policy(self.nav_obs()))
                    if self.battery <= 0:
                        break
            elif a == 1:      # refine the most uncertain cell in view
                self._refine_start(self._most_uncertain_in_view())
                for _ in range(REFINE_K):
                    _, ok = self._refine_step(self.refine_policy(self.refine_obs()))
                    if ok or self.battery <= 0:
                        break
            else:             # return to base, mission ends
                self.battery -= COST_MOVE * (abs(self.pos[0] - BASE[0]) + abs(self.pos[1] - BASE[1]))
                self.pos = list(BASE)
                terminated = True
            found1 = self.metrics()["defects_found"]
            r = W_DEF_C * (found1 - found0) - W_ENERGY_C * (bat0 - self.battery)
            if self.battery <= 0:
                r += FAIL_C
                terminated = True
            success = terminated and self.battery > 0 and a == 2

        truncated = (not terminated) and self.t >= self.max_steps
        info = self.metrics()
        info["out_of_bounds"], info["success"] = bool(oob), bool(success)
        return self._obs(), float(r), terminated, truncated, info

    def render(self):
        u = self.cell_u().reshape(H, W)
        rows = []
        for r in range(H):
            row = ""
            for c in range(W):
                if [r, c] == self.pos:
                    row += "U"
                elif self.best_q[self._idx(r, c)] < 0:
                    row += "."
                else:
                    row += str(min(9, int(u[r, c] * 10)))
            rows.append(row)
        return "\n".join(rows) + f"\nalt={['high','mid','low'][self.alt]} battery={self.battery:.2f}"


def sb3_policy(model):
    """Wrap a trained SB3 model as a callable obs -> action (for 'full' mode)."""
    return lambda obs: model.predict(obs, deterministic=True)[0]
