# Live_Demo — Team 08 live demonstration

*Uncertainty-Aware Active Vision for UAV-Based Infrastructure Inspection Using Deep RL*

This folder runs the three trained agents live: A (DQN, viewpoint planner), B (PPO, view refiner) and
C (A2C, mission supervisor), both alone and together. It also runs the real-image crack detector live.
Every number on screen is either computed live from the environment or read from the saved result
CSV/NPZ files in the repo. Nothing is typed in by hand.

`uav_inspection/` is **imported, never modified**. Trained models and results are **read, never written**.

---

## 1. One-time setup (Windows, PowerShell)

The virtual environment lives **outside Google Drive** at `%USERPROFILE%\.venvs\uav_demo`. A ~1 GB venv
inside `L:\My Drive` would be synced by Drive and becomes slow or corrupted.

```powershell
# 1. Python 3.12 (3.10-3.12 work). Check:
py -3.12 --version

# 2. create the venv outside Google Drive and activate it
py -3.12 -m venv "$env:USERPROFILE\.venvs\uav_demo"
& "$env:USERPROFILE\.venvs\uav_demo\Scripts\Activate.ps1"
#   (if PowerShell blocks scripts: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned)

# 3. install the pinned, CPU-only packages (torch comes from the PyTorch CPU index, see requirements.txt)
cd "L:\My Drive\RL_Project\files_coding\Live_Demo"
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. (optional, for demo 5 live) DeepCrack dataset, also outside Google Drive
git clone --depth 1 https://github.com/yhlleo/DeepCrack.git "$env:USERPROFILE\data\DeepCrack"
Expand-Archive "$env:USERPROFILE\data\DeepCrack\dataset\DeepCrack.zip" "$env:USERPROFILE\data\DeepCrack\dataset"

# 5. build the backup GIFs/PNGs (~4 min) and run the checklist
python scripts\make_backups.py
python preflight_check.py
```

If the DeepCrack step is skipped, demo 5 shows the saved `detector_views.png` instead, labelled
"saved figure, not live". Nothing else depends on it.

**Paths.** Every path is in [`config.yaml`](config.yaml): models, results, detector cache and U-Net, the
DeepCrack folder, output folders and the temp folder for demo 8. If a teammate's copy of the repo sits
somewhere else, only `repo_root` (and maybe `deepcrack_dir`) needs to change.

## 2. Running the demos

Activate the venv first (`& "$env:USERPROFILE\.venvs\uav_demo\Scripts\Activate.ps1"`), then from `Live_Demo\`:

```powershell
python run_demo.py                 # menu: type 1-9 (+ flags), q to quit
python run_demo.py 3               # run one demo directly
python run_demo.py 8 --agent B     # flags go after the number
python demos\d4_agentB_refine.py   # every demo also runs on its own
```

| # | Demo | Presenter | Guideline steps |
|---|---|---|---|
| 1 | Environment tour: grid, altitudes and footprints, detector modes, random rollout | (optional opener) | 1, 2, 4 |
| 2 | Agent A step-through: map → Q-values → argmax → reward terms → next map (Enter per step) | Yeseswini | 1–5, 7, 8 |
| 3 | Agent A (DQN) vs dense raster, same bridge, animated with live counters | Yeseswini | 7, 8 |
| 4 | Agent B (PPO): geometry, μ and σ of the Gaussian, view quality s and level q, reward; vs FixedOrbit | Leekhith | 1–5, 7, 8 |
| 5 | Real-image detector: U-Net + MC-dropout on DeepCrack TEST patches at q0…q4 (live) | Mukhesh | 2, 4, 8 |
| 6 | Agent C (A2C): state vector, π(Explore/Refine/Return), timeline, vs RuleSupervisor | Mukhesh | 1–5, 7, 8 |
| 7 | Full mission: C → A / B, supervisor banner, vs all-scripted system | Leekhith | 1, 7, 8 |
| 8 | Live learning burst (`--agent A/B/C`): losses, ε or entropy, episode return, TD update | each presenter | 6 |
| 9 | Results dashboard (`--agent A/B/C/full`): learning curves + results-vs-baselines tables | each presenter | 8 |

Common flags for every demo: `--seed N` (default 1000; the same seed gives the same bridge),
`--detector analytic|learned`, `--fast` (fewer frames), `--save` (write GIF/PNG to `backup/`),
`--headless` (no windows). Demo-specific flags: d2 `--auto`, d3 `--raster low|mid`, d4 `--episodes N`,
d7 `--supervisor C|C_withA|rule`, d8 `--agent`, `--steps`, d9 `--agent`.

Controls: close a figure window to move on. In demo 2, press **Enter** for the next step, **r** to run to
the end, **q** to quit.

Optional web app (local, no internet): `streamlit run app\streamlit_app.py`.
Notebook version: `notebooks\Live_Demo.ipynb` (regenerate with `python scripts\build_notebook.py`).

## 3. Presentation day, in order

1. **T−30 min:** plug in the projector and set display scaling to 100 %.
2. **T−10 min:** activate the venv, `cd` into `Live_Demo`, and run `python preflight_check.py`. It should end with
   `ALL REQUIRED CHECKS GREEN` (exit code 0).
3. Open `backup\gifs\` in Explorer, in case a live demo fails.
4. Start `python run_demo.py` and run `1 --fast` once. This warms up the imports, so later demos start in about 2 s.
5. Presenters follow `presenter_notes\yeseswini.md` → `leekhith.md` → `mukhesh.md`.

If a demo crashes, the menu survives and prints which backup GIF to open. Every animated demo has
a GIF in `backup\gifs\`. Transcripts of the narrated runs are in `backup\snapshots\transcript_*.txt`.

## 4. What is verified

- **Rewards:** every reward shown is split into its terms using the constants in `inspection_env.py`.
  Each split is checked against the env's own reward, and the demo prints "terms sum to the env reward".
- **Agent A:** the DQN argmax always equals the executed action (tested).
- **Full mission:** d7 on seed 1000 exactly reproduces the saved `For_Leekhith/full/demo_mission.txt`
  (return 45.73, 24/26 defects). This is tested.
- **Missing models:** if a model is missing, the demo says so and runs the scripted **BASELINE**, labelled
  as a baseline on screen.
- **Short runs:** results from shorter runs (for example the 60k/150k sensitivity runs) are marked "short run" in demo 9.

## 5. Findings the audience may ask about (read from the repo)

- **Agent C never chooses REFINE.** This holds for both `agentC` and `agentC_withA`, on all 20 evaluation bridges.
  In the full mission Agent B is therefore never called, and d7 says so on screen. Use
  `--supervisor rule` to show Agent B running inside the hierarchy: this is the
  "ablation: rule C, A + B learned" row, which has the best saved return (44.84).
- **Agent B** matches the scripted upper bound (return 5.67 vs 5.67, `agentB_vs_baselines.csv`).
- **All learned vs all scripted** (`full_mode_results.csv`): 49 % fewer decisions and 46 % less energy,
  but a slightly lower return (40.94 vs 42.82) and slightly fewer defects found (21.5 vs 22.55).
- **Agent A with the real-image detector** (`agentA_learned`): 25 % success, below the dense raster.
- **Repo `README.md` vs CSV:** the row "A2C, learned detector | 41.1 | 15.5 | … | 0.40" does not match
  `For_Mukhesh/agentC_learned/agentC_vs_baselines.csv`, which has 13.40 decisions and energy 0.350. The demo shows the CSV values.

## 6. Folder map

```
Live_Demo/
├── run_demo.py  preflight_check.py  config.yaml  requirements.txt
├── demo_core/   paths, loaders, introspect, reward, rollout, render, animate, narrate, metrics,
│                mission (shared supervisor view), detector (U-Net read from build_detector_cache.py),
│                cli (common flags), style (colours/fonts)
├── demos/       d1 … d9  (each: main(argv) -> dict, runnable alone)
├── app/         streamlit_app.py
├── notebooks/   Live_Demo.ipynb
├── presenter_notes/  yeseswini.md  leekhith.md  mukhesh.md
├── backup/      gifs/  frames/  snapshots/      (made by scripts/make_backups.py)
├── scripts/     make_backups.py  build_notebook.py
└── tests/       test_demos.py  (pytest Live_Demo/tests -q)
```

**GIFs in git:** `backup/gifs/*.gif` are tracked because each one is under 5 MB (0.3–1.1 MB).
`make_backups.py` prints the sizes. If one ever grows past 5 MB, add it to the root `.gitignore`.
**No MP4:** MP4s are only written when `ffmpeg` is on PATH. It isn't on this machine, so there are GIFs only.
**Demo 8** trains into `%TEMP%\uav_demo_burst\`. It checks that the real model files' timestamps are unchanged.
