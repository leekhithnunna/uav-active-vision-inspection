# Team 08 - Uncertainty-Aware Active Vision for UAV Infrastructure Inspection

Shared Gymnasium environment + baselines + Agent A training.

## Ownership
| Member | Owns |
|---|---|
| Yeseswini | Shared env (`env/`), baselines, cache plug-in, **Agent A - DQN (navigate)** |
| Leekhith | **Agent B - PPO (refine)**, metrics/plotting, integration (`full` mode), report/poster assembly |
| Mukhesh | **Agent C - A2C (supervise)**, DeepCrack U-Net + `detector_cache.npz` |

## Layout
```
env/inspection_env.py   shared environment (4 modes, 2 detector modes)
env/baselines.py        raster / orbit / rule / random policies + evaluate()
scripts/test_env.py     smoke test (run first!)
scripts/run_baselines.py  baseline numbers -> results/*.csv
scripts/train_agent_a.py  Agent A (DQN) training + evaluation vs baselines
scripts/build_detector_cache.py  DeepCrack -> U-Net (MC-dropout) -> detector_cache.npz (run on GPU)
scripts/train_agent_b.py  Agent B (PPO) training + evaluation vs orbit/random
scripts/train_agent_c.py  Agent C (A2C) training + evaluation vs rule supervisor
scripts/run_full_mode.py  integration: all 3 agents together + ablation + narrated demo
Yeseswini_AgentA.ipynb     Colab: env, baselines, Agent A, Part II analysis (CPU)
Mukhesh_Cache_AgentC.ipynb Colab: detector cache (GPU) + Agent C
Leekhith_AgentB_Integration.ipynb Colab: Agent B + integration + team results
START_HERE.md              who does what, in which order
docs/                      guides, report sections, slide content
```

## Design contract v1.1 (frozen - change only after telling everyone)
**World:** 8x16 grid of bridge-surface cells, 20% defective (random per episode, seeded).
UAV starts at base (0,0), high altitude. Altitude high/mid/low -> footprint 5x5 / 3x3 / 1x1
-> view quality q = 1 / 2 / 3. q = 4 only via Agent B refinement. Each cell keeps its best view (best-view fusion).
Battery 1.0; move -0.005, altitude change -0.01, refine step -0.008.

| | Agent A (navigate) | Agent B (refine) | Agent C (supervise) |
|---|---|---|---|
| Obs (118) | egocentric 7x7 window of [visited, uncertainty] + 2x4 block [coverage, mean_u] + [row, col, alt, battery] | [d, angle, lateral, target_u, battery, steps_left] | [battery, coverage, mean_u, found_frac, dist_to_base, time_frac] |
| Action | Discrete(6): N,S,E,W,climb,descend | Box(3) in [-1,1]: d/angle/lateral (x0.2 per step) | Discrete(3): explore(5 A-steps), refine(<=10 B-steps), return |
| Reward | 0.2*total_uncertainty_reduced (unseen=1) - 0.01 - 0.05*revisit - 1*out_of_bounds; success: +10 + 20*battery_left | 1.0*du - 0.01*abs(a) + 0.5*(0.99*s' - s) shaping; +5 success; -2 timeout | 2*new_defects - 0.5*energy; -10 battery empty |
| Episode | <=200 steps; success = coverage>=0.95 AND mean_u<=0.32 | <=30 steps; success = best view (q=4) | <=40 decisions; success = safe return |

`info` (every mode): coverage, mean_uncertainty, defects_found, false_positives, defects_total, miou,
energy_used, steps, out_of_bounds, success.

## Detector cache contract (real images, Level 2)
Built by `scripts/build_detector_cache.py` from DeepCrack (Liu et al., Neurocomputing 2019):
U-Net trained on the TRAIN split only; cache patches are 64x64 crops from the TEST split only
(crack pixels >1% -> defect, 0 -> clean, in-between dropped as ambiguous).

| key | shape | type | meaning |
|---|---|---|---|
| `u` | (P,5) | float32 in [0,1] | uncertainty at view quality q=0..4 (effective res 4/8/16/32/64 px) |
| `pdef` | (P,5) | bool | detector says "defect" (predicted crack pixels > 0.5%) |
| `iou` | (P,5) | float32 | IoU of predicted mask vs ground truth |
| `is_defect` | (P,) | bool | ground truth |
| `success_unc` | scalar | float | calibrated navigate success threshold (env reads it automatically) |
| `alt_q` | (3,) | int | altitude -> quality map for real images: high=q0, mid=q1, low=q3 (env reads it) |

Uncertainty = mean per-pixel predictive entropy of the MC-dropout mean (10 passes), rank-normalised to [0,1],
then made non-increasing in q (best-view fusion: a better view never loses information).
Why entropy and not dropout std: on blurred views the network becomes falsely confident, so std alone
does not rise for bad views; predictive entropy (total uncertainty) does.
Why a different altitude map: the U-Net stays accurate down to ~16 px, so mid altitude must be coarser
(8 px) for descending to matter. Analytic mode is unchanged.

Use with: `InspectionEnv(mode, detector="learned", cache_path=".../detector_cache.npz")`.

## Using trained agents together (full mode)
```python
from stable_baselines3 import DQN, PPO, A2C
from env import InspectionEnv, sb3_policy
env = InspectionEnv("full", nav_policy=sb3_policy(DQN.load("agentA.zip")),
                    refine_policy=sb3_policy(PPO.load("agentB.zip")))
sup = A2C.load("agentC.zip")
```

## Notes for the viva
- SB3's `DQN` = DQN with experience replay + target network (Mnih et al., 2015). It does **not** include Double/Dueling.
- Detector in `analytic` mode is a documented abstraction; `learned` mode uses real DeepCrack images + U-Net MC-dropout.
- DeepCrack dataset: Liu et al., Neurocomputing 2019 (non-commercial research/educational use).

## Git workflow
- `main` = working, tested code. Never push broken code to main.
- Each member works on a branch: `yeseswini/agent-a`, `leekhith/agent-b`, `mukhesh/agent-c`, then opens a Pull Request.
- Models, caches, zips and TensorBoard logs are NOT in git (see `.gitignore`); they live in Google Drive `MyDrive/team08/`.
