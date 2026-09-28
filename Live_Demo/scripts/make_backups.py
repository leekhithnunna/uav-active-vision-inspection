"""Regenerate everything in backup/ with one command (GIFs, key PNG frames, result tables, terminal transcripts).

    python scripts/make_backups.py          (~5-6 min on a laptop CPU)
"""
from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo_core import narrate, paths  # noqa: E402

# (demo module, argv, transcript name or None)
JOBS = [
    ("d1_env_tour", ["--save"], None),
    ("d2_agentA_step", ["--save", "--auto"], "transcript_d2_agentA_step"),
    ("d3_agentA_vs_raster", ["--save"], "transcript_d3_agentA_vs_raster"),
    ("d4_agentB_refine", ["--save"], "transcript_d4_agentB_refine"),
    ("d5_detector_views", ["--save"], "transcript_d5_detector"),
    ("d6_agentC_mission", ["--save"], "transcript_d6_agentC_mission"),
    ("d7_full_mission", ["--save"], "transcript_d7_full_mission"),
    ("d7_full_mission", ["--save", "--supervisor", "rule"], "transcript_d7_full_mission_rule"),
    ("d8_learning_burst", ["--save", "--agent", "A"], "transcript_d8_burst_A"),
    ("d8_learning_burst", ["--save", "--agent", "B"], "transcript_d8_burst_B"),
    ("d8_learning_burst", ["--save", "--agent", "C"], "transcript_d8_burst_C"),
    ("d9_results_dashboard", ["--headless"], "transcript_d9_results"),
]


def main() -> int:
    snaps = paths.out_dir("snapshots")
    seed = ["--seed", str(paths.default_seed())]
    failed, t_all = [], time.time()
    for name, argv, transcript in JOBS:
        t0 = time.time()
        narrate.start_recording()
        narrate.console.rule(f"[bold]{name} {' '.join(argv)}")
        try:
            importlib.import_module(f"demos.{name}").main(argv + seed)
            ok = True
        except Exception:  # noqa: BLE001 - keep generating the other backups
            narrate.console.print_exception(max_frames=4)
            ok = False
        if transcript:
            narrate.save_transcript(snaps / f"{transcript}.txt")
        status = "ok" if ok else "FAILED"
        print(f"[make_backups] {name:22s} {' '.join(argv):28s} {status:6s} {time.time() - t0:5.1f}s", flush=True)
        if not ok:
            failed.append(f"{name} {' '.join(argv)}")
    gifs = sorted(paths.out_dir("gifs").glob("*.gif"))
    print(f"\n[make_backups] done in {time.time() - t_all:.0f}s")
    for g in gifs:
        mb = g.stat().st_size / 1e6
        print(f"   {g.name:28s} {mb:5.2f} MB{'   (> 5 MB: git-ignored)' if mb > 5 else ''}")
    print(f"   frames: {len(list(paths.out_dir('frames').glob('*.png')))} PNG, "
          f"snapshots: {len(list(snaps.iterdir()))} files")
    if failed:
        print(f"[make_backups] FAILED: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
