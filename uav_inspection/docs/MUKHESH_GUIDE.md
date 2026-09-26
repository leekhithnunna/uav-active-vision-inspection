# Mukhesh - your part (Team 08)

You own two things:
1. **Detector cache** (support task): DeepCrack -> U-Net (MC-dropout) -> `detector_cache.npz` for the whole team.
2. **Agent C - Mission Supervisor (A2C)**: your individual viva component.

Everything is already coded and tested. Your job: run it, understand it, analyse it, present it.

## 0. One-time setup (5 min)
1. Accept the Google Drive share for `team08`, then: **drive.google.com -> Shared with me -> right-click `team08`
   -> Organize -> Add shortcut -> My Drive**. (Without this, Colab can't find `MyDrive/team08`.)
2. Accept the GitHub invite to the repo (Yeseswini sends it).
3. Open `Mukhesh_Cache_AgentC.ipynb` in Colab (File -> Upload notebook, or open from the repo).
4. **Runtime -> Change runtime type -> T4 GPU.**

## 1. Part 1 - detector cache (~15 min), notebook Steps 1-5
| Step | What happens | Check |
|---|---|---|
| 1 | Mount Drive, unzip project, install libs | `scripts` folder listed |
| 2 | GPU check | prints `Tesla T4` |
| 3 | Clone DeepCrack (original repo yhlleo/DeepCrack) | counts 300/300/237/237 |
| 4 | Train U-Net on TRAIN split, build cache from TEST split | ends with `Done ... detector_cache.npz` |
| 5 | Report table + figure | uncertainty goes DOWN and IoU goes UP from q0 to q4 |

**Then post in the team chat: "cache is in team08".** Yeseswini and Leekhith are waiting for it.

What the cache contains (see README): per image patch and per view quality q=0..4 -> uncertainty `u`,
detector decision `pdef`, segmentation `iou`, ground truth `is_defect`, plus a calibrated success threshold
and altitude->quality map that the env reads automatically.

Key design choices you must be able to explain:
- **No leakage**: U-Net sees only the TRAIN split; env patches come only from the TEST split.
- **Uncertainty = predictive entropy of the MC-dropout mean** (Gal & Ghahramani, 2016), not dropout std:
  on blurred views the net becomes falsely confident, so std alone doesn't rise; entropy does.
- **Best-view fusion**: uncertainty made non-increasing in view quality ("a better look never loses information").

## 2. Part 2 - Agent C (your viva component), notebook Steps 6-11
**MDP (mode `supervise`, a semi-MDP: each action is an option that runs several low-level steps)**
| | |
|---|---|
| State (6) | battery, coverage, mean uncertainty, fraction of defects found, distance to base, time used |
| Actions | Discrete(3): 0 Explore (5 navigator steps), 1 Refine (<=10 camera steps on the most uncertain cell in view), 2 Return home (ends mission) |
| Reward | +2 per newly confirmed defect - 0.5 x energy used; -10 if battery hits 0 |
| Episode | <= 40 decisions; success = returned home safely |
| Algorithm | A2C (advantage actor-critic), 4 parallel envs, n_steps=8, entropy bonus 0.01 |

Run order: Step 6 baselines -> Step 7 train (~3-5 min) -> Step 8 curves + table -> Step 9 live demo (CO3)
-> Step 10 entropy sensitivity (CO5) -> Step 11 optional (real images / with trained Agent A inside).

**Our test run (40k decisions):** A2C return 41.1 vs rule-based 42.8, but with **~half the energy (0.45 vs 0.95)**
and half the decisions. Interesting analysis point: the learned policy almost never chooses *Refine* - ask
yourself why (hint: refine costs energy and rarely confirms NEW defects under the reward). Step 10 shows
whether more entropy (exploration) changes that.

## 3. Pushing your work to GitHub (from your laptop, PowerShell)
```powershell
git clone https://github.com/<yeseswini-username>/uav-inspection.git
cd uav-inspection
git checkout -b mukhesh/agent-c                     # your own branch - never work on main
# ...edit / add files (e.g. download the executed notebook from Colab into this folder)...
git add .
git commit -m "Agent C: training results + analysis"
git push -u origin mukhesh/agent-c
```
Then on GitHub: **Compare & pull request -> Create pull request**. Yeseswini merges it.
Models, caches and zips are NOT pushed (see `.gitignore`) - they live in Google Drive.

## 4. Your deliverables
- Slides for Agent C (problem -> MDP -> A2C equation -> code -> demo -> results -> exploration analysis)
- Report section + individual poster (send to Leekhith for assembly)
- Detector slide material: `detector_report.json` table + `detector_views.png`
