# Evaluation_Metrics — Team 08 evaluation-metrics submission

*Uncertainty-Aware Active Vision for UAV-Based Infrastructure Inspection Using Deep RL* (22AIE401)

This folder computes every evaluation metric of the mini project, from the trained models and saved
logs in the repo, and builds the submission document and videos from them. Every number in the
tables, plots and document is computed by code; nothing is typed by hand.

The rest of the repo is **read only**. `uav_inspection/`, `Live_Demo/`, `For_Leekhith/`, `For_Mukhesh/`
and `reports/` are imported or read, never written. Scenario changes are made with a subclass and a
wrapper inside this folder. `tests/test_eval.py` checks that nothing outside `Evaluation_Metrics/`
changed.

## What to upload

Everything to upload is copied into **`SUBMISSION/`** by `scripts/make_submission.py`, which also
prints the file sizes and flags anything over 25 MB:

| Deliverable | File |
|---|---|
| Document | `Team08_Evaluation_Metrics.pdf` and `.docx` |
| Videos | `V1_*.mp4` … `V10_*.mp4` (H.264, about 58 s each, about 1 MB or less) |
| Video index | `video_index.md` |

Not uploaded but useful: `videos/narration_script.md` (a voice-over per clip, checked to fit the clip
length, see `videos/narration_lengths.csv`) and `videos/preview_frames/` (the middle and end frame of
every clip, for a quick visual check).

## How to run

```powershell
& "$env:USERPROFILE\.venvs\uav_demo\Scripts\Activate.ps1"
pip install -r Evaluation_Metrics\requirements_extra.txt     # once
cd "L:\My Drive\RL_Project\files_coding"
python Evaluation_Metrics\run_all.py                          # phases 1-4, 6-8 + tests
python -m pytest Evaluation_Metrics\tests -q                  # verification only
```

Options: `--skip-videos`, `--from N` (start at phase N and reuse earlier results), `--multiseed`
(train the missing extra seeds as well, which takes about 60–90 min more), `--no-tests`.

Runtime on a laptop CPU (i5-1235U), approximate:

| Phase | Script | Time |
|---|---|---|
| 1 Validation (original seeds) | `scripts/validate.py` | 3 min |
| 2 Held-out evaluation | `scripts/evaluate_all.py` | 8 min |
| 3 Scenarios S1–S5 (parallel workers) | `scripts/evaluate_scenarios.py` | 19 min |
| 4 Training diagnostics | `scripts/training_diagnostics.py` | 3 min |
| 5 Extra training seeds (optional) | `scripts/multiseed.py` | B+C 16 min, A longer |
| 6 Plots | `scripts/plots.py` | 1 min |
| 7 Videos | `scripts/make_videos.py` | 6 min |
| 8 Report | `scripts/known_issues.py`, `scripts/build_report.py` | 1 min |
| 9 Tests | `tests/test_eval.py` | 1 min |

## Evaluation protocol in one paragraph

Every saved result in the repo was first re-computed on its original seeds (`results/validation.csv`;
all checks agree within 1%). All new numbers come from **100 held-out bridges (seeds 2000–2099)** that
were never used in training or model selection. Every policy runs on the same bridges, with
deterministic learned policies. The scenarios are: S0 nominal, S1 real-image (DeepCrack) detector,
S2 defect density 10% / 35%, S3 noise σ = 0.05 / 0.10 on the uncertainty features, S4 starting
battery 0.6, and S5 a hard start geometry (Agent B only).

## Folder map

```
Evaluation_Metrics/
├── run_all.py                one command for everything
├── requirements_extra.txt    pinned extra packages (scipy, tensorboard, imageio(-ffmpeg), python-docx, docx2pdf)
├── scripts/
│   ├── common.py             paths, ScenarioEnv (density / battery / hard start), UncertaintyNoise wrapper,
│   │                         instrumented episode runner, 95% CI
│   ├── suites.py             which policies run in which scenario + per-policy summary statistics
│   ├── validate.py           phase 1
│   ├── evaluate_all.py       phase 2
│   ├── evaluate_scenarios.py phase 3
│   ├── training_diagnostics.py phase 4
│   ├── multiseed.py          phase 5 (wraps the original training scripts with --seed / --out)
│   ├── plots.py              phase 6
│   ├── make_videos.py        phase 7 (Live_Demo used as a library; output folder redirected at runtime)
│   ├── known_issues.py       report/known_issues.md
│   └── build_report.py       phase 8 (md + docx + pdf, report_numbers.csv, video index, narration)
├── results/                  all CSV / NPZ outputs (episodes_*, summary_*, scenario_summary, robustness_gap, ...)
├── plots/                    PNG, 300 dpi + plot_index.csv
├── videos/                   MP4 clips + video_files.csv, video_runs.csv, video_index.md, narration_script.md
├── report/                   Team08_Evaluation_Metrics.{md,docx,pdf}, report_numbers.csv, known_issues.md
├── multiseed_runs/           extra training seeds (never written into the original run folders)
└── tests/test_eval.py        verification
```

## Key outputs

- `results/summary_S0_all.csv`: one row per policy with the mean, std, median and 95% CI of each
  metric, success rate, episode length for successful and failed episodes, time to goal,
  battery-empty rate, out-of-bounds moves per 1000 steps, option frequencies and regret.
- `results/scenario_summary.csv`, `results/robustness_gap.csv`: the same statistics for every
  scenario, and the gap between nominal and perturbed results.
- `results/training_curves.csv`, `eval_curves.csv`, `tb_scalars.csv`, `sample_efficiency.csv`,
  `sensitivity.csv`, `model_size.csv`: training diagnostics.
- `report/report_numbers.csv`: every number printed in the document, with its source file, column and
  row. The tests re-read and check each one.
- `report/known_issues.md`: mismatches found while evaluating, such as the README row for "A2C, learned
  detector" and the two training runs inside `agentC_learned/`.
