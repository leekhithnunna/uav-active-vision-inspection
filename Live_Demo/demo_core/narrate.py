"""Coloured terminal narration (rich): STATE -> ACTIONS -> SELECTED -> REWARD -> NEXT STATE."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .reward import Breakdown
from .rollout import ALT_NAMES, Frame

CRIMSON = "#991245"
console = Console(highlight=False)
if not console.is_terminal:                     # piped / captured output: do not squeeze tables into 80 columns
    console = Console(highlight=False, width=130)


def start_recording(width: int = 120) -> None:
    """Record everything printed from now on (used by make_backups for transcripts)."""
    global console
    console = Console(record=True, width=width, highlight=False, force_terminal=False)


def save_transcript(path: Path) -> None:
    path.write_text(console.export_text(), encoding="utf-8")


def header(title: str, subtitle: str = "") -> None:
    console.rule(f"[bold {CRIMSON}]{title}")
    if subtitle:
        console.print(f"[grey50]{subtitle}", justify="center")


def info(msg: str) -> None:
    console.print(f"[cyan]{msg}")


def warn(msg: str) -> None:
    console.print(Panel(msg, border_style="yellow", title="warning"))


def say(msg: str) -> None:
    console.print(msg)


def guideline(step: int, text: str) -> None:
    """Tag output with the course guideline step it demonstrates (1-8)."""
    console.print(f"[bold white on {CRIMSON}] GUIDELINE {step} [/] [bold]{text}")


def state(f: Frame, extra: dict | None = None) -> None:
    m = f.metrics
    t = Table(show_header=False, box=None, padding=(0, 2))
    rows = {"position [row, col]": f"{list(f.pos)}", "altitude": ALT_NAMES[f.alt], "battery": f"{f.battery:.2f}",
            "coverage": f"{m['coverage']:.1%}", "mean uncertainty": f"{m['mean_uncertainty']:.3f}",
            "defects found": f"{m['defects_found']}/{m['defects_total']}"}
    rows.update(extra or {})
    for k, v in rows.items():
        t.add_row(f"[grey50]{k}", f"[bold]{v}")
    console.print(Panel(t, title=f"STATE  (t = {f.t})", border_style="blue", expand=False))


def vector(names: Sequence[str], values: Sequence[float], title: str) -> None:
    t = Table(title=title, title_justify="left")
    for n in names:
        t.add_column(n, justify="right")
    t.add_row(*[f"{v:.3f}" for v in values])
    console.print(t)


def action_table(names: Sequence[str], values: Sequence[float], chosen: int, value_label: str,
                 title: str = "AVAILABLE ACTIONS") -> None:
    t = Table(title=title, title_justify="left")
    t.add_column("#", justify="right")
    t.add_column("action")
    t.add_column(value_label, justify="right")
    t.add_column("")
    for i, (n, v) in enumerate(zip(names, values)):
        mark = "<-- selected (argmax)" if i == chosen else ""
        style = f"bold white on {CRIMSON}" if i == chosen else ""
        t.add_row(str(i), n, f"{v:+.3f}", mark, style=style)
    console.print(t)


def gaussian_table(g: dict, executed) -> None:
    t = Table(title="POLICY OUTPUT  a ~ N(mu, sigma)  (deterministic demo: a = clip(mu))", title_justify="left")
    for c in ("dimension", "mu", "sigma", "executed a"):
        t.add_column(c, justify="right")
    for n, m, s, a in zip(g["names"], g["mu"], g["sigma"], np.clip(np.asarray(executed, float), -1, 1)):
        t.add_row(n, f"{m:+.3f}", f"{s:.3f}", f"[bold {CRIMSON}]{a:+.3f}")
    console.print(t)


def selected(label: str) -> None:
    console.print(f"[bold]SELECTED ACTION ->[/] [bold white on {CRIMSON}] {label} [/]")


def reward(bd: Breakdown) -> None:
    t = Table(title="REWARD", title_justify="left", show_footer=True)
    t.add_column("term", footer="[bold]total (env reward)")
    t.add_column("value", justify="right", footer=f"[bold]{bd.reward:+.3f}")
    for k, v in bd.parts.items():
        t.add_row(k, f"[{'green' if v >= 0 else 'red'}]{v:+.3f}")
    console.print(t)
    ok = "[green]terms sum to the env reward" if bd.matches else f"[red]terms sum {bd.total:+.3f} != env reward"
    console.print(f"   {ok}")


def compact(f: Frame) -> None:
    """One line per step (for 'run to end')."""
    m = f.metrics
    console.print(f"t={f.t:3d}  {f.action_label:22s} r={f.reward:+6.2f}  alt={ALT_NAMES[f.alt]:4s} "
                  f"cov={m['coverage']:5.1%}  u={m['mean_uncertainty']:.3f}  "
                  f"defects={m['defects_found']:2d}/{m['defects_total']}  battery={f.battery:.2f}")


def results(header_: Sequence[str], rows: Sequence[Sequence], title: str, highlight: int | None = None) -> None:
    t = Table(title=title, title_justify="left")
    for k, h in enumerate(header_):
        t.add_column(str(h), justify="left" if k == 0 else "right", no_wrap=k == 0)
    for i, r in enumerate(rows):
        t.add_row(*[str(c) for c in r], style=f"bold {CRIMSON}" if i == highlight else "")
    console.print(t)
