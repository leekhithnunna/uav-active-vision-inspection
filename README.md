# Uncertainty-Aware Active Vision for UAV Infrastructure Inspection

**Team 08 · 22AIE401 Reinforcement Learning · 2026-27**
Yeseswini · Leekhith · Mukhesh

A drone inspects a bridge surface for cracks. Instead of flying a fixed lawnmower pattern, it uses
**three cooperating reinforcement-learning agents** that decide *where* to fly, *how* to look at a
suspicious spot, and *when* to stop and come home, all driven by how **uncertain** the crack
detector is about each part of the bridge.

```
                    ┌──────────────────────────────────────────┐
                    │  Agent C - Mission Supervisor (A2C)      │   "what should I do next?"
                    │  Explore  /  Refine  /  Return home      │
                    └─────────┬──────────────────┬─────────────┘
                     Explore  │                  │  Refine
                              ▼                  ▼
     ┌──────────────────────────────┐   ┌──────────────────────────────────┐
     │ Agent A - Viewpoint Planner  │   │ Agent B - Active View Refiner    │
     │ (DQN) where to fly + altitude│   │ (PPO) distance / angle / offset  │
     │ Discrete(6)                  │   │ continuous Box(3)                │
     └──────────────┬───────────────┘   └───────────────┬──────────────────┘
                    └──────────────┬────────────────────┘
                                   ▼
         Shared Gymnasium environment (8x16 bridge grid, battery, camera)
                                   │
                  Crack detector → per-cell uncertainty
          analytic (formula)  OR  learned (DeepCrack U-Net + MC-dropout)
```

---

## Table of contents
1. [Quick start](#1-quick-start)
2. [Repository map (every file and folder)](#2-repository-map)
3. [The environment](#3-the-environment)
4. [The three agents](#4-the-three-agents)
5. [The real-image detector (DeepCrack)](#5-the-real-image-detector-deepcrack)
6. [Results](#6-results)
7. [How to reproduce everything](#7-how-to-reproduce-everything)
8. [Output file formats](#8-output-file-formats)
9. [Team, ownership and workflow](#9-team-ownership-and-workflow)
10. [Limitations and future work](#10-limitations-and-future-work)
11. [References](#11-references)

---

## 1. Quick start

```bash
git clone https://github.com/leekhithnunna/uav-active-vision-inspection.git
cd uav-active-vision-inspection/uav_inspection
pip install -r requirements.txt          # gymnasium, stable-baselines3, numpy, matplotlib, pandas, tensorboard

python scripts/test_env.py               # smoke test: every mode x both detectors
python scripts/run_baselines.py --out results
```

Load the trained models that are already in this repo:

```python
import sys; sys.path.insert(0, "uav_inspection")
from stable_baselines3 import PPO, A2C
from env import InspectionEnv, sb3_policy

agentB = PPO.load("For_Leekhith/agentB/best/best_model.zip", device="cpu")
agentC = A2C.load("For_Mukhesh/agentC/best/best_model.zip", device="cpu")

env = InspectionEnv("refine")
obs, info = env.reset(seed=1000)
done = False
while not done:
    action, _ = agentB.predict(obs, deterministic=True)
    obs, r, term, trunc, info = env.step(action)
    done = term or trunc
print(info)   # success, steps, energy_used, ...
```

The notebooks (`*.ipynb`) are built for **Google Colab** with the project folder in Google Drive.
Scripts run anywhere with Python 3.10+ (developed on 3.12).

---

## 2. Repository map

```
.
├── README.md                     ← this file
├── YESESWINI_NEXT_STEPS.md       Yeseswini's remaining to-do list (copy of uav_inspection/docs/)
│
├── uav_inspection/               ★ THE SOURCE CODE (the actual project package)
│   ├── README.md                 design contract v1.1 (frozen spec of env, obs, actions, rewards)
│   ├── START_HERE.md             who does what, in which order
│   ├── requirements.txt          Python dependencies
│   ├── .gitignore                ignores runs/, *.zip, *.pt, *.npz, tb/ inside this folder
│   │
│   ├── env/                      shared Gymnasium environment
│   │   ├── __init__.py           exports InspectionEnv, sb3_policy
│   │   ├── inspection_env.py     InspectionEnv: 4 modes x 2 detectors (361 lines)
│   │   └── baselines.py          scripted/random baseline policies + evaluate()/summarize()
│   │
│   ├── scripts/                  command-line entry points
│   │   ├── _path.py              adds uav_inspection/ to sys.path so `from env import ...` works
│   │   ├── test_env.py           smoke test (gymnasium env_checker + random rollouts, fake cache)
│   │   ├── run_baselines.py      all baselines, 20 seeded episodes → results/*.csv
│   │   ├── build_detector_cache.py  DeepCrack → U-Net (MC-dropout) → detector_cache.npz  [GPU]
│   │   ├── train_agent_a.py      Agent A (DQN, navigate) train + eval vs raster baselines
│   │   ├── train_agent_b.py      Agent B (PPO, refine) train + eval vs orbit/random/scripted
│   │   ├── train_agent_c.py      Agent C (A2C, supervise) train + eval vs rule supervisor
│   │   └── run_full_mode.py      integration: A+B+C together, 4-way ablation, narrated demo mission
│   │
│   ├── results/                  baseline numbers, analytic detector (20 episodes each)
│   │   ├── baselines_summary_analytic.csv          one row per (mode, policy): mean + std of every metric
│   │   ├── baselines_summary_learned.csv           header only so far (real-image baselines not yet run)
│   │   └── baseline_<mode>_<policy>_analytic.csv   per-episode rows (8 files)
│   │
│   ├── Yeseswini_AgentA.ipynb    Colab: env, baselines, Agent A + Part II analysis (CPU)
│   ├── uav_inspection.zip        snapshot of this folder that the Colab notebooks unzip
│   │
│   ├── agentA/                   ★ main DQN run (200k steps, ε decay over 30 %)
│   │   ├── agentA_final.zip      final DQN model
│   │   ├── best/best_model.zip   best model by EvalCallback  ← use this one
│   │   ├── checkpoints/          agentA_<N>_steps.zip every 25k steps (25k … 200k)
│   │   ├── hyperparams.json, agentA_vs_baselines.csv
│   │   ├── monitor_train.monitor.csv, monitor_eval.monitor.csv, eval/evaluations.npz, tb/
│   │   ├── learning_curves.png   training return / success / episode length
│   │   ├── eps_sensitivity.png   CO5: ε-decay 10 % vs 30 % vs 60 %
│   │   ├── regret.png            CO4: cumulative regret vs best baseline
│   │   └── trajectories.png      poster figure: DQN flight path vs raster
│   ├── agentA_eps0.1/ agentA_eps0.3/ agentA_eps0.6/   CO5: ε-decay-fraction sensitivity (same layout)
│   ├── agentA_learned/           empty; reserved for Agent A on real images (notebook Step 9)
│   │
│   └── docs/
│       ├── LEEKHITH_GUIDE.md     Agent B MDP, integration plan, report/poster checklist
│       ├── MUKHESH_GUIDE.md      detector cache + Agent C MDP, GitHub branch workflow
│       ├── YESESWINI_NEXT_STEPS.md
│       └── yeseswini/
│           ├── AgentA_report_section.tex     LaTeX report section for Agent A
│           └── AgentA_slides_poster_viva.md  17-slide plan, results table, poster layout, viva answers
│
├── For_Leekhith/                 ★ Leekhith's hand-off package + Agent B TRAINING OUTPUTS
│   ├── START_HERE.md, LEEKHITH_GUIDE.md          copies of the docs above
│   ├── Leekhith_AgentB_Integration.ipynb         Colab: Agent B + integration + team results
│   ├── uav_inspection.zip                        snapshot of uav_inspection/ that Colab unzips
│   ├── results/                                  copy of the baseline CSVs
│   ├── agentB/                   main PPO run (100k steps, ent 0.0, clip 0.2)
│   │   ├── agentB_final.zip      final PPO model (Stable-Baselines3)
│   │   ├── best/best_model.zip   best model by EvalCallback  ← use this one
│   │   ├── hyperparams.json      exact settings of the run
│   │   ├── agentB_vs_baselines.csv  PPO vs fixed_orbit / random / scripted upper bound
│   │   ├── learning_curves.png, eval_curve.png
│   │   ├── monitor/0..3.monitor.csv  per-episode training logs (4 parallel envs)
│   │   ├── monitor_eval.monitor.csv  evaluation-env episode log
│   │   ├── eval/evaluations.npz      EvalCallback history (timesteps, rewards, lengths)
│   │   └── tb/PPO_1/events...        TensorBoard log
│   ├── agentB_ent0.0/  agentB_ent0.01/  agentB_ent0.05/   CO5: entropy-bonus sensitivity (same layout)
│   └── agentB_clip0.1/ agentB_clip0.3/                    CO5: PPO clip-range sensitivity (same layout)
│
└── For_Mukhesh/                  ★ Mukhesh's hand-off package + DETECTOR + Agent C OUTPUTS
    ├── Mukhesh_Cache_AgentC.ipynb   Colab: detector cache (GPU) + Agent C
    ├── START_HERE.md, MUKHESH_GUIDE.md, uav_inspection.zip, results/
    ├── unet_deepcrack.pt         trained U-Net weights (PyTorch)
    ├── detector_cache.npz        ★ real-image uncertainty lookup table used by detector="learned"
    ├── detector_report.json      per-quality uncertainty / detect rate / IoU + calibrated thresholds
    ├── detector_views.png        example patches at view qualities q0..q4
    ├── agentC/                   main A2C run (100k decisions, analytic detector) - same layout as agentB/
    ├── agentC_ent0.0/ agentC_ent0.01/ agentC_ent0.05/   CO5: entropy sensitivity
    └── agentC_learned/           A2C trained with the real-image (learned) detector
```

**Why are there copies?** `For_Leekhith/` and `For_Mukhesh/` began as the zip packages sent to each
teammate (code snapshot + guide + notebook). The training scripts then wrote each member's outputs
into their own folder on Google Drive. **`uav_inspection/` is the canonical source code** (plus
Agent A's outputs); the `For_*` folders hold Agent B's and Agent C's **experiment outputs** and each
member's notebook.

| Agent | Notebook | Trained models |
|---|---|---|
| A (DQN) | `uav_inspection/Yeseswini_AgentA.ipynb` | `uav_inspection/agentA*/` |
| B (PPO) | `For_Leekhith/Leekhith_AgentB_Integration.ipynb` | `For_Leekhith/agentB*/` |
| C (A2C) | `For_Mukhesh/Mukhesh_Cache_AgentC.ipynb` | `For_Mukhesh/agentC*/` |

---

## 3. The environment

`uav_inspection/env/inspection_env.py` → `InspectionEnv(mode, detector="analytic", cache_path=None, nav_policy=None, refine_policy=None)`

### World (design contract v1.1)
| Item | Value |
|---|---|
| Bridge surface | 8 × 16 grid of cells, **20 % defective** (26 of 128), re-drawn every episode from the seed |
| Start | base (0,0), high altitude, battery 1.0 |
| Altitude | high / mid / low → camera footprint 5×5 / 3×3 / 1×1 cells → view quality q = 1 / 2 / 3 |
| q = 4 | only reachable by Agent B's close-up refinement |
| Fusion | each cell keeps its **best** view so far (a better look never loses information) |
| Energy | move −0.005, altitude change −0.01, refine step −0.008 |

### Modes
| Mode | Used by | Observation | Action |
|---|---|---|---|
| `navigate` | Agent A | Box(118) | Discrete(6) |
| `refine` | Agent B | Box(6) | Box(3) in [−1, 1] |
| `supervise` | Agent C | Box(6) | Discrete(3); sub-agents are scripted stubs |
| `full` | integration | Box(6) | Discrete(3); sub-agents are the **trained** A and B you pass in |

### Detector modes
- **`analytic`**: uncertainty from a formula of view quality. Needs no data and is used for most results.
- **`learned`**: uncertainty, detection and IoU are looked up from `detector_cache.npz`, built from
  real DeepCrack images (Section 5). Usage: `InspectionEnv(mode, detector="learned", cache_path="For_Mukhesh/detector_cache.npz")`.

### `info` dict (returned every step, every mode)
`coverage, mean_uncertainty, defects_found, false_positives, defects_total, miou, energy_used, steps, out_of_bounds, success`

### Baselines (`env/baselines.py`)
| Class | Mode | Behaviour |
|---|---|---|
| `RasterNav(alt)` | navigate | lawnmower scan at fixed altitude (`raster_mid` = 1, `raster_low` = 2) |
| `FixedOrbit` | refine | circles the target without approaching |
| `ApproachRefiner` | refine | flies straight to the ideal view: an **upper bound** that uses privileged geometry |
| `RuleSupervisor(low_battery=0.15)` | supervise | hand-written explore/refine/return rules |
| `RandomPolicy` | any | uniform random actions |

`evaluate(env, policy, n_episodes=20, seed0=1000)` runs every policy on **the same 20 seeded bridges**,
so all comparisons in this repo are like-for-like. `summarize()` turns the rows into mean/std.

---

## 4. The three agents

All agents use Stable-Baselines3 with MLP [64, 64] policies (Agent A uses [256, 256]).

### Agent A: Viewpoint Planner (DQN), owner Yeseswini
| | |
|---|---|
| State (118) | egocentric 7×7 window of [visited, uncertainty] + 2×4 block summary [coverage, mean_u] + [row, col, alt, battery] |
| Actions | Discrete(6): N, S, E, W, climb, descend |
| Reward | 0.2 × total uncertainty reduced − 0.01/step − 0.05 revisit − 1 out-of-bounds; success +10 + 20 × battery left |
| Episode | ≤ 200 steps; success = coverage ≥ 95 % **and** mean uncertainty ≤ 0.32 |
| Settings | lr 5e-4, γ 0.995, buffer 100k, batch 128, target update 1000, ε 1.0 → 0.05 over 30 %, 200k steps |

Two lessons from experiments: a global one-hot position (386 inputs) **did not learn**, and switching
to an egocentric view fixed it. Separate coverage and uncertainty rewards led to **reward hacking**;
rewarding total uncertainty reduced plus a battery-scaled success bonus fixed that.

### Agent B: Active View Refiner (PPO), owner Leekhith
| | |
|---|---|
| State (6) | distance, viewing angle, lateral offset, target uncertainty, battery, steps left |
| Actions | continuous Box(3): Δdistance, Δangle, Δlateral (× 0.2 per step) |
| Dynamics | s = 1 − 0.6 d − 0.4 abs(angle) − 0.3 abs(lateral); q = round(4 s) |
| Reward | 1.0 × Δuncertainty − 0.01 × abs(a) + 0.5 × (0.99 s′ − s) **potential-based shaping**; +5 success; −2 timeout |
| Episode | ≤ 30 steps; success = best view (q = 4) |
| Settings | lr 3e-4, γ 0.99, clip 0.2, n_steps 256, batch 64, 10 epochs, GAE λ 0.95, 4 envs, 100k steps |

PPO fits here because the actions are continuous, which DQN cannot handle. The shaping term follows
Ng et al. (1999), so it speeds learning without changing the optimal policy.

### Agent C: Mission Supervisor (A2C), owner Mukhesh
| | |
|---|---|
| State (6) | battery, coverage, mean uncertainty, fraction of defects found, distance to base, time used |
| Actions | Discrete(3) **options**: Explore (5 Agent-A steps), Refine (≤ 10 Agent-B steps on the most uncertain cell in view), Return home |
| Reward | +2 per new defect − 0.5 × energy; −10 if battery empties |
| Episode | ≤ 40 decisions; success = safe return |
| Settings | lr 7e-4, γ 0.99, n_steps 8, ent 0.01, vf 0.5, 4 envs, 100k decisions |

This is a **semi-MDP / hierarchical RL** setup: each action is an option that runs several low-level steps.

---

## 5. The real-image detector (DeepCrack)

`uav_inspection/scripts/build_detector_cache.py` (run on a Colab T4 GPU, about 10 min):

1. Train a small **U-Net** on 64×64 crops from the DeepCrack **train** split (4000 iterations).
2. Cut 64×64 patches from the **test** split only, so there is no leakage (crack > 1 % → defect, 0 % → clean, in-between dropped).
3. For each patch and each view quality q = 0..4 (effective resolution 4 / 8 / 16 / 32 / 64 px), run **10 MC-dropout passes**:
   - `u` = mean per-pixel **predictive entropy** of the MC mean, rank-normalised to [0, 1] and made non-increasing in q
   - `pdef` = predicted crack pixels > 0.5 %
   - `iou` = predicted mask vs ground truth
4. Save `detector_cache.npz` (1200 defect + 1800 clean patches).

Entropy is used instead of dropout std because blurred views make the network *falsely confident*.
Std does not rise on those views, but predictive entropy does.

**Detector quality (`For_Mukhesh/detector_report.json`)**

| q | resolution | u (defect) | u (clean) | detect rate | false pos. | IoU (defect) |
|---|---|---|---|---|---|---|
| 0 | 4 px | 0.601 | 0.321 | 39.8 % | 0.9 % | 0.193 |
| 1 | 8 px | 0.563 | 0.234 | 82.2 % | 2.4 % | 0.501 |
| 2 | 16 px | 0.543 | 0.179 | 92.8 % | 4.8 % | 0.644 |
| 3 | 32 px | 0.532 | 0.153 | 96.9 % | 7.3 % | 0.696 |
| 4 | 64 px | 0.526 | 0.140 | 98.2 % | 12.5 % | 0.717 |

Calibrated `success_unc` = 0.282. Altitude → quality map for real images `alt_q` = [0, 1, 3]
(high = q0, mid = q1, low = q3), because the U-Net stays accurate down to about 16 px.

---

## 6. Results

All numbers are means over the **same 20 seeded bridges** with the analytic detector unless stated otherwise.

### Agent A: DQN vs raster scans (`uav_inspection/agentA/agentA_vs_baselines.csv`)
| Metric | Raster mid | Raster low (dense) | **DQN** |
|---|---|---|---|
| Success | 0 % | 100 % | **100 %** |
| Steps | 199 | 123 | **88.3 (−28 %)** |
| Energy | 1.00 | 0.625 | **0.50 (−20 %)** |
| Coverage | 100 % | 95.3 % | 99.9 % |
| Mean uncertainty | 0.351 | 0.219 | 0.319 |
| mIoU | 0.500 | 0.626 | 0.539 |
| Return | 6.20 | 35.43 | 35.06 |

The DQN reaches the same 100 % success as the dense raster with about 28 % fewer steps and 20 % less energy.
It does just enough to meet the uncertainty threshold, so the raster's over-inspection keeps a better mIoU.

**ε-decay sensitivity (CO5, `agentA_eps*/`)**: fraction of training over which ε decays 1.0 → 0.05.
| ε-decay fraction | Return | Steps | Success | Energy | mIoU |
|---|---|---|---|---|---|
| 10 % | **37.8** | **74.7** | 100 % | **0.40** | **0.559** |
| 30 % | 34.5 | 92.7 | 100 % | 0.52 | 0.552 |
| 60 % | 28.1 | 120.7 | 80 % | 0.64 | 0.539 |

Shorter exploration worked best here: a long ε schedule leaves too little time to exploit what was learned.

### Agent B: PPO (`For_Leekhith/agentB/agentB_vs_baselines.csv`)
| Policy | Return | Steps | Success | Energy |
|---|---|---|---|---|
| **PPO (Agent B)** | **5.67** | **3.66** | **100 %** | **0.029** |
| fixed_orbit | −2.28 | 30.0 | 0 % | 0.240 |
| random | −2.04 | 29.8 | 2 % | 0.239 |
| scripted_approach (upper bound) | 5.67 | 3.64 | 100 % | 0.029 |

PPO matches the privileged upper bound after about 2 min of training.
**Sensitivity (CO5):** entropy coefficient 0.0 / 0.01 / 0.05 and clip range 0.1 / 0.2 / 0.3 all reach
100 % success in 3.6–3.7 steps, so the task is robust to these hyperparameters.

### Agent C: A2C (`For_Mukhesh/agentC*/agentC_vs_baselines.csv`)
| Run | Return | Decisions | Defects /26 | mIoU | Energy |
|---|---|---|---|---|---|
| **A2C (ent 0.01, main)** | 39.8 | 19.3 | 20.95 | 0.500 | **0.51** |
| A2C ent 0.0 | 39.9 | 13.2 | 20.95 | 0.500 | **0.35** |
| A2C ent 0.05 | 40.5 | 16.5 | 21.25 | 0.515 | 0.42 |
| A2C, learned detector | 41.1 | 15.5 | 21.2 | 0.494 | 0.40 |
| rule_supervisor | **42.8** | 33.8 | **22.55** | **0.597** | 0.95 |
| random | 4.1 | 3.0 | 2.95 | 0.072 | 0.07 |

A2C gets within about 7 % of the rule-based return while using **about half the energy and half the decisions**.
It rarely chooses *Refine*, because under this reward refining costs energy and seldom confirms *new* defects.

### Integration: full mode (from `docs/LEEKHITH_GUIDE.md` test run)
| Config | Return | Coverage | Defects /26 | mIoU | Energy |
|---|---|---|---|---|---|
| All learned (C + A + B) | 39.3 | 0.96 | 20.6 | 0.504 | **0.49** |
| All scripted | 42.8 | 1.00 | 22.6 | 0.597 | 0.95 |
| C learned, A/B scripted | 41.1 | 1.00 | 21.6 | 0.520 | 0.45 |
| Rule C, A + B learned | **43.9** | 1.00 | **23.1** | **0.618** | 0.94 |

The learned hierarchy uses about half the energy. Learned A + B under a rule supervisor gives the best
defect count and mIoU. Agent C was trained with scripted sub-agents, so swapping in learned ones shifts
its input distribution. The fix is to retrain C with `--nav_model`.

### Baselines (`uav_inspection/results/baselines_summary_analytic.csv`)
| Mode | Policy | Return | Steps | Success | Coverage | Energy |
|---|---|---|---|---|---|---|
| navigate | random | −29.6 | 148.7 | 0 % | 0.51 | 1.00 |
| navigate | raster_mid | 6.2 | 199 | 0 % | 1.00 | 1.00 |
| navigate | raster_low | 35.4 | 123 | 100 % | 0.95 | 0.63 |
| refine | random | −2.24 | 30 | 0 % | n/a | 0.24 |
| refine | fixed_orbit | −2.29 | 30 | 0 % | n/a | 0.24 |
| refine | scripted_approach | 5.68 | 3.5 | 100 % | n/a | 0.03 |
| supervise | random | 4.1 | 3.0 | 100 % | 0.16 | 0.07 |
| supervise | rule_supervisor | 42.8 | 33.8 | 100 % | 1.00 | 0.95 |

---

## 7. How to reproduce everything

In Colab, set `DRIVE=/content/drive/MyDrive/<your folder>`. Run from inside `uav_inspection/`.

```bash
# 0. sanity + baselines (CPU, ~1 min)
python scripts/test_env.py
python scripts/run_baselines.py --out results

# 1. real-image detector (GPU, ~10 min)
git clone -q --depth 1 https://github.com/yhlleo/DeepCrack.git /content/DeepCrack
unzip -q -o /content/DeepCrack/dataset/DeepCrack.zip -d /content/deepcrack_data
python scripts/build_detector_cache.py --data /content/deepcrack_data --out $DRIVE

# 2. agents (CPU)
python scripts/train_agent_a.py --steps 200000 --out $DRIVE/agentA                 # ~10 min
python scripts/train_agent_a.py --eps_fraction 0.1 --out $DRIVE/agentA_eps0.1       # CO5 ε-decay study
python scripts/train_agent_b.py --steps 100000 --out $DRIVE/agentB                 # ~2 min
python scripts/train_agent_c.py --steps 100000 --out $DRIVE/agentC                 # ~4 min
#    real-image variants
python scripts/train_agent_c.py --detector learned --cache $DRIVE/detector_cache.npz --out $DRIVE/agentC_learned
#    sensitivity studies (CO5)
python scripts/train_agent_b.py --steps 60000 --ent_coef 0.01 --out $DRIVE/agentB_ent0.01
python scripts/train_agent_b.py --steps 60000 --clip 0.1      --out $DRIVE/agentB_clip0.1
python scripts/train_agent_c.py --steps 60000 --ent_coef 0.05 --out $DRIVE/agentC_ent0.05

# 3. integration + ablation + narrated demo mission
python scripts/run_full_mode.py --a $DRIVE/agentA/best/best_model.zip \
       --b $DRIVE/agentB/best/best_model.zip --c $DRIVE/agentC/best/best_model.zip --out $DRIVE/full
```

Or open the notebooks and run their numbered steps:
- **`Yeseswini_AgentA.ipynb`**: 1 setup → 3 smoke test → 4 baselines → 5 train → 7 curves → 8 table → 10/10b ε-decay sensitivity → 11 live demo → 12 convergence/regret → 13 flight-path figure → 9 real images
- **`Mukhesh_Cache_AgentC.ipynb`** (T4 GPU): 1 setup → 2 GPU check → 4 build cache → 5 detector report → 7 train C → 8 curves → 9 live demo → 10 entropy sensitivity → 11 real images / with Agent A
- **`Leekhith_AgentB_Integration.ipynb`**: 1 setup → 2 baselines → 3 train B → 4 curves → 5 live demo → 6 convergence → 7 entropy + clip sensitivity → 8 integration → 9 team summary → 10 real images

Viewing the training logs:
```bash
tensorboard --logdir For_Leekhith        # or For_Mukhesh, or uav_inspection/agentA
```

---

## 8. Output file formats

| File | Content |
|---|---|
| `*_final.zip`, `best/best_model.zip` | Stable-Baselines3 model; load with `PPO.load(...)` / `A2C.load(...)` / `DQN.load(...)` |
| `hyperparams.json` | every argument of the training run |
| `*_vs_baselines.csv` | one row per policy: `<metric>_mean`, `<metric>_std` for return, steps, success, coverage, mean_uncertainty, defects_found, defects_total, false_positives, miou, energy_used; plus `train_minutes` |
| `monitor/N.monitor.csv` | SB3 Monitor: one line per training episode (`r` return, `l` length, `t` time) |
| `eval/evaluations.npz` | `timesteps`, `results`, `ep_lengths` from EvalCallback |
| `tb/<ALGO>_1/events.*` | TensorBoard scalars |
| `results/baseline_<mode>_<policy>_<detector>.csv` | per-episode baseline metrics |
| `detector_cache.npz` | `u` (P,5) float, `pdef` (P,5) bool, `iou` (P,5) float, `is_defect` (P,) bool, `success_unc` scalar, `alt_q` (3,) int |

---

## 9. Team, ownership and workflow

| Member | Owns |
|---|---|
| **Yeseswini** | shared env, baselines, cache plug-in, **Agent A (DQN)** |
| **Leekhith** | **Agent B (PPO)**, metrics/plots, integration (`full` mode), report and poster assembly |
| **Mukhesh** | **Agent C (A2C)**, DeepCrack U-Net + `detector_cache.npz` |

Order of work: env + baselines + Agent A → (detector cache ‖ Agent B) → Agent C → Agent A on real
images → integration + team results. See `uav_inspection/START_HERE.md`.

Git: `main` holds working, tested code. Feature work goes on `<name>/agent-x` branches and is merged
by pull request.

---

## 10. Limitations and future work
- The main results use the **analytic** detector; the real-image version is harder to learn.
- The bridge is a 2-D grid with no wind, collisions or 3-D geometry.
- Vanilla DQN (SB3 has no Double/Dueling) gives unstable evaluation, which is handled by keeping the best checkpoint.
- Every run uses a single training seed; multiple seeds would give confidence intervals.
- Agent C was trained with scripted sub-agents; retraining it with the learned A and B should close the integration gap.
- Future work: Double/Dueling DQN, multi-seed runs, a 3-D simulator (AirSim), and a curriculum from the analytic to the real detector.

## 11. References
- Mnih et al., *Human-level control through deep reinforcement learning*, Nature 2015 (DQN)
- Schulman et al., *Proximal Policy Optimization Algorithms*, 2017 (PPO)
- Mnih et al., *Asynchronous Methods for Deep RL*, ICML 2016 (A2C/A3C)
- Ng, Harada & Russell, *Policy invariance under reward transformations*, ICML 1999 (reward shaping)
- Gal & Ghahramani, *Dropout as a Bayesian Approximation*, ICML 2016 (MC-dropout)
- Liu et al., *DeepCrack: A deep hierarchical feature learning architecture for crack segmentation*, Neurocomputing 2019 (dataset, non-commercial research/educational use)
- Sutton, Precup & Singh, *Between MDPs and semi-MDPs*, AIJ 1999 (options)
- Stable-Baselines3, Gymnasium

License: MIT (see `LICENSE`).
