"""Run 10 minutes before the slot: green/red checklist. Exit code 0 only if everything the demo needs is green.

    python preflight_check.py            full check (~2-3 min, includes headless smoke runs of d3/d4/d6/d7)
    python preflight_check.py --quick    skip the smoke runs
"""
from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
REQUIRED_PKGS = ["stable_baselines3", "gymnasium", "torch", "numpy", "pandas", "matplotlib", "yaml", "rich", "PIL"]


@dataclass
class Result:
    name: str
    status: str
    detail: str
    seconds: float


def _versions() -> tuple[str, str]:
    import platform
    bad, parts = [], [f"Python {platform.python_version()}"]
    if not (3, 10) <= sys.version_info[:2] <= (3, 13):
        bad.append("Python must be 3.10-3.13")
    from importlib.metadata import version
    dist = {"yaml": "PyYAML", "PIL": "pillow", "stable_baselines3": "stable-baselines3"}
    for m in REQUIRED_PKGS:
        try:
            importlib.import_module(m)
            parts.append(f"{dist.get(m, m)} {version(dist.get(m, m))}")
        except ImportError:
            bad.append(f"{m} missing")
    return (FAIL, "; ".join(bad)) if bad else (PASS, ", ".join(parts))


def _sb3_matches() -> tuple[str, str]:
    import stable_baselines3 as sb3
    from demo_core import loaders, paths
    trained = loaders.sb_version_info(paths.model_path("A")).get("Stable-Baselines3", "?")
    same = trained.split(".")[:2] == sb3.__version__.split(".")[:2]
    return (PASS if same else WARN), f"installed {sb3.__version__}, models trained with {trained}"


def _uav_import() -> tuple[str, str]:
    import env  # noqa: F401  (uav_inspection/env)
    from env import InspectionEnv  # noqa: F401
    return PASS, f"env package from {Path(env.__file__).parent}"


def _model(key: str, detector: str = "analytic") -> Callable[[], tuple[str, str]]:
    def check():
        from demo_core import loaders
        agent = loaders.load_agent(key, detector, fallback=False)
        mode = loaders.AGENT_INFO[key[0]]["mode"]
        obs, _ = loaders.load_env(mode, detector).reset(seed=1000)
        a = agent(obs)
        return PASS, f"{agent.path.relative_to(agent.path.parents[3])} -> action {a}"
    return check


def _env(mode: str, detector: str) -> Callable[[], tuple[str, str]]:
    def check():
        from demo_core import loaders
        kw = {}
        if mode == "full":
            kw = dict(nav_policy=loaders.load_agent("A", detector), refine_policy=loaders.load_agent("B", detector))
        env = loaders.load_env(mode, detector, **kw)
        env.reset(seed=1000)
        _, r, *_ = env.step(env.action_space.sample())
        return PASS, f"reset + step ok (reward {r:+.3f})"
    return check


def _results(key: str) -> tuple[str, str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from demo_core import metrics, paths
    if key == "full":
        df = metrics.full_results()
        return PASS, f"full_mode_results.csv: {len(df)} configs"
    run = paths.results_entry(key)
    df = metrics.vs_baselines(run)
    fig, axs = plt.subplots(1, 3)
    metrics.plot_curves(axs, run, key)
    plt.close(fig)
    return PASS, f"{Path(run).name}: {len(df)} policies in table, curves drawable"


def _gifs() -> tuple[str, str]:
    from demo_core import paths
    names = ["agentA_vs_raster", "agentB_refine", "agentC_mission", "full_mission", "detector"]
    d = paths.out_dir("gifs")
    missing = [n for n in names if not (d / f"{n}.gif").exists()]
    if missing:
        return FAIL, f"missing {missing} - run: python scripts/make_backups.py"
    age_h = (time.time() - min((d / f"{n}.gif").stat().st_mtime for n in names)) / 3600
    return (PASS if age_h < 72 else WARN), f"all 5 present, oldest {age_h:.0f} h old"


def _dataset() -> tuple[str, str]:
    from demo_core import detector
    if detector.dataset_available():
        detector.load_unet()
        return PASS, f"DeepCrack TEST split at {detector.dataset_dir()}; U-Net loads"
    return WARN, f"no DeepCrack at {detector.dataset_dir()} - d5 will show the saved figure instead"


def _display() -> tuple[str, str]:
    import tkinter  # noqa: F401  (default interactive backend on Windows)
    return PASS, "tkinter available (windows can open)"


def _ffmpeg() -> tuple[str, str]:
    from demo_core import animate
    return (PASS, "ffmpeg found (MP4 too)") if animate.ffmpeg_available() else (PASS, "no ffmpeg: GIF only (fine)")


def _smoke(name: str, argv: list[str]) -> Callable[[], tuple[str, str]]:
    def check():
        from demo_core import narrate
        from rich.console import Console
        saved, narrate.console = narrate.console, Console(file=io.StringIO(), width=130)
        try:
            t0 = time.time()
            with contextlib.redirect_stdout(io.StringIO()):
                importlib.import_module(f"demos.{name}").main(argv)
            dt = time.time() - t0
        finally:
            narrate.console = saved
        return (PASS if dt < 60 else FAIL), f"{' '.join(argv)} ran in {dt:.1f}s"
    return check


def build(quick: bool) -> list[tuple[str, Callable, bool]]:
    """(name, check, required) - WARN-level items never fail the preflight."""
    from demo_core import paths
    cache = paths.detector_path("cache").exists()
    checks = [("python + packages", _versions, True), ("SB3 version vs models", _sb3_matches, True),
              ("uav_inspection importable", _uav_import, True)]
    checks += [(f"model {k} loads + acts", _model(k), True) for k in ("A", "B", "C")]
    checks += [("model C_withA loads + acts", _model("C_withA"), False)]
    checks += [(f"env {m} (analytic)", _env(m, "analytic"), True) for m in ("navigate", "refine", "supervise", "full")]
    if cache:
        checks += [(f"env {m} (learned)", _env(m, "learned"), False) for m in ("navigate", "refine", "supervise")]
        checks += [(f"model {k} (learned) loads", _model(k, "learned"), False) for k in ("A", "B", "C")]
    checks += [(f"results {k}", (lambda k=k: _results(k)), True) for k in ("A", "B", "C", "full")]
    checks += [("backup GIFs", _gifs, True), ("DeepCrack + U-Net (d5 live)", _dataset, False),
               ("display backend", _display, True), ("ffmpeg (optional)", _ffmpeg, False)]
    if not quick:
        common = ["--headless", "--fast", "--seed", str(paths.default_seed())]
        checks += [(f"smoke d{n[1]}", _smoke(n, common), True) for n in
                   ("d3_agentA_vs_raster", "d4_agentB_refine", "d6_agentC_mission", "d7_full_mission")]
    return checks


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--quick", action="store_true", help="skip the headless smoke runs")
    args = p.parse_args(argv)
    from rich.console import Console
    from rich.table import Table
    con = Console()
    con.rule("[bold #991245]Team 08 live demo - preflight check")
    results: list[tuple[Result, bool]] = []
    for name, fn, required in build(args.quick):
        t0 = time.time()
        try:
            status, detail = fn()
        except Exception as e:  # noqa: BLE001 - report every failure, never stop early
            status, detail = (FAIL if required else WARN), f"{type(e).__name__}: {e}"
        if status == FAIL and not required:
            status = WARN
        r = Result(name, status, detail, time.time() - t0)
        results.append((r, required))
        colour = {"PASS": "green", "WARN": "yellow", "FAIL": "red"}[status]
        con.print(f"[{colour}]{status:4s}[/] {name:32s} [grey50]{r.seconds:5.1f}s[/]  {detail}")
    t = Table(title="Summary")
    for c in ("check", "status", "required"):
        t.add_column(c)
    for r, req in results:
        colour = {"PASS": "green", "WARN": "yellow", "FAIL": "red"}[r.status]
        t.add_row(r.name, f"[{colour}]{r.status}", "yes" if req else "no")
    con.print(t)
    failed = [r.name for r, req in results if req and r.status == FAIL]
    if failed:
        con.print(f"[bold red]NOT READY - fix: {', '.join(failed)}")
        return 1
    con.print("[bold green]ALL REQUIRED CHECKS GREEN - ready to present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
