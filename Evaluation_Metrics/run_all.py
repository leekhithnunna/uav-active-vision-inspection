"""One-command reproduction of the evaluation-metrics submission.

  python Evaluation_Metrics/run_all.py                 # phases 1,2,3,4,6,7,8 (+ verification)
  python Evaluation_Metrics/run_all.py --multiseed     # also (re)train missing extra seeds (phase 5)
  python Evaluation_Metrics/run_all.py --skip-videos   # keep the existing videos
  python Evaluation_Metrics/run_all.py --from 6        # start at a phase (earlier results are reused)

Existing multi-seed runs in multiseed_runs/ are always evaluated and included in the report; training new
seeds only happens with --multiseed. Nothing outside Evaluation_Metrics/ is written.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

EM = Path(__file__).resolve().parent
S = EM / "scripts"
PY = sys.executable
ENV = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}


def run(phase, title, args):
    t0 = time.time()
    print(f"\n=== Phase {phase}: {title}", flush=True)
    subprocess.run([PY, "-B", *map(str, args)], check=True, env=ENV, cwd=str(EM.parent))
    print(f"=== Phase {phase} done in {(time.time() - t0) / 60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--multiseed", action="store_true", help="train missing extra seeds (phase 5)")
    ap.add_argument("--multiseed-agents", nargs="+", default=["B", "C", "A"])
    ap.add_argument("--skip-videos", action="store_true")
    ap.add_argument("--no-tests", action="store_true")
    ap.add_argument("--from", dest="start", type=int, default=1)
    a = ap.parse_args()
    t0 = time.time()
    steps = [
        (1, "validation on the original seeds", [S / "validate.py"]),
        (2, "held-out evaluation (S0 nominal)", [S / "evaluate_all.py"]),
        (3, "varying task environments (S1-S5)", [S / "evaluate_scenarios.py"]),
        (4, "training diagnostics from saved logs", [S / "training_diagnostics.py"]),
    ]
    for ph, title, args in steps:
        if ph >= a.start:
            run(ph, title, args)
    if a.start <= 5:
        if a.multiseed:
            run(5, "extra training seeds", [S / "multiseed.py", "train", "--agents", *a.multiseed_agents,
                                           "--seeds", "1", "2", "3"])
            run(4, "training diagnostics incl. new seeds", [S / "training_diagnostics.py", "--no-sensitivity"])
        if any((EM / "multiseed_runs").glob("*_seed*/best/best_model.zip")):
            run(5, "held-out evaluation of the multi-seed runs", [S / "multiseed.py", "eval"])
    if a.start <= 6:
        run(6, "plots", [S / "plots.py"])
    if a.start <= 7 and not a.skip_videos:
        run(7, "videos", [S / "make_videos.py"])
    if a.start <= 8:
        run(8, "known issues + report", [S / "known_issues.py"])
        run(8, "report", [S / "build_report.py"])
        run(8, "SUBMISSION folder", [S / "make_submission.py"])
    if not a.no_tests:
        run(9, "verification (pytest)", ["-m", "pytest", EM / "tests", "-q", "-p", "no:cacheprovider"])
    print(f"\nAll done in {(time.time() - t0) / 60:.1f} min. Deliverables: Evaluation_Metrics/report/ and "
          "Evaluation_Metrics/videos/")


if __name__ == "__main__":
    main()
