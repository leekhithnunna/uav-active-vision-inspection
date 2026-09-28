"""Team 08 live demo - single entry point.

    python run_demo.py              interactive menu (choose 1-9, q to quit)
    python run_demo.py 3            run demo 3 directly
    python run_demo.py 3 --save     any extra flags are passed to the demo (--seed --detector --fast --save ...)
"""
from __future__ import annotations

import importlib
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from demo_core import narrate, paths  # noqa: E402

DEMOS = {
    "1": ("d1_env_tour", "Environment tour (grid, altitudes, detector modes, random rollout)", "all"),
    "2": ("d2_agentA_step", "Agent A step-through: Q-values -> action -> reward -> next state", "Yeseswini"),
    "3": ("d3_agentA_vs_raster", "Agent A (DQN) vs dense raster, same bridge, animated", "Yeseswini"),
    "4": ("d4_agentB_refine", "Agent B (PPO): Gaussian policy, view quality, vs FixedOrbit", "Leekhith"),
    "5": ("d5_detector_views", "Real-image detector: U-Net + MC-dropout at q0..q4 (live)", "Mukhesh"),
    "6": ("d6_agentC_mission", "Agent C (A2C): option probabilities, narrated mission", "Mukhesh"),
    "7": ("d7_full_mission", "Full mission: C -> A / B, vs all-scripted system", "Leekhith"),
    "8": ("d8_learning_burst", "Live learning burst (--agent A|B|C): the update step", "all"),
    "9": ("d9_results_dashboard", "Results dashboard: learning curves + tables (--agent A|B|C|full)", "all"),
}
BACKUP = {"1": "env_tour.gif", "3": "agentA_vs_raster.gif", "4": "agentB_refine.gif", "5": "detector.gif",
          "6": "agentC_mission.gif", "7": "full_mission.gif"}


def run(key: str, argv: list[str]) -> bool:
    """Run one demo; on failure point the presenter to the backup file instead of crashing the menu."""
    name = DEMOS[key][0]
    t0 = time.time()
    try:
        importlib.import_module(f"demos.{name}").main(argv)
        narrate.info(f"demo {key} finished in {time.time() - t0:.0f}s")
        return True
    except KeyboardInterrupt:
        narrate.warn("interrupted")
    except Exception:  # noqa: BLE001 - the menu must survive any demo failure on stage
        narrate.console.print_exception(max_frames=3)
        gif = BACKUP.get(key)
        hint = f"open backup/gifs/{gif}" if gif else "open backup/snapshots/"
        narrate.warn(f"demo {key} failed - {hint} (folder: {paths.DEMO_ROOT / 'backup'})")
    return False


def menu() -> None:
    from rich.table import Table
    while True:
        t = Table(title="Team 08 - UAV active vision - live demos", title_style="bold #991245")
        t.add_column("#", justify="right"); t.add_column("demo"); t.add_column("presenter")
        for k, (_, desc, who) in DEMOS.items():
            t.add_row(k, desc, who)
        narrate.console.print(t)
        narrate.say("[grey50]extra flags after the number, e.g. `3 --fast`, `8 --agent B`, `7 --supervisor rule`; q = quit")
        try:
            line = input("demo> ").strip().split()
        except EOFError:
            return
        if not line:
            continue
        if line[0].lower() in ("q", "quit", "exit"):
            return
        if line[0] not in DEMOS:
            narrate.warn(f"unknown choice {line[0]!r}")
            continue
        run(line[0], line[1:])


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in DEMOS:
        return 0 if run(argv[0], argv[1:]) else 1
    if argv:
        print(__doc__)
        return 2
    menu()
    return 0


if __name__ == "__main__":
    sys.exit(main())
