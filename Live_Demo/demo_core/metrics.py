"""Read saved results (never invent numbers): *_vs_baselines.csv, monitor CSVs, eval/evaluations.npz."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths, style

TABLE_COLS = [("return", "return"), ("success", "success"), ("steps", "steps"), ("coverage", "coverage"),
              ("defects_found", "defects"), ("miou", "mIoU"), ("energy_used", "energy")]
REFINE_COLS = [("return", "return"), ("success", "success"), ("steps", "steps"), ("energy_used", "energy")]


def vs_baselines(run_dir: Path) -> pd.DataFrame:
    """The *_vs_baselines.csv of a run folder (first match)."""
    files = sorted(Path(run_dir).glob("*_vs_baselines.csv"))
    if not files:
        raise FileNotFoundError(f"no *_vs_baselines.csv in {run_dir}")
    return pd.read_csv(files[0])


def full_results() -> pd.DataFrame:
    return pd.read_csv(paths.resolve(paths.config()["results"]["full"]["csv"]))


def hyperparams(run_dir: Path) -> dict:
    p = Path(run_dir) / "hyperparams.json"
    return json.loads(p.read_text()) if p.exists() else {}


def run_note(run_dir: Path, main_dir: Path | None = None) -> str:
    """'' for a full-length run, otherwise a visible label such as 'short run: 60k steps (main: 100k)'."""
    steps = hyperparams(run_dir).get("steps")
    main = hyperparams(main_dir).get("steps") if main_dir else None
    if steps and main and steps < main:
        return f"short run: {steps // 1000}k steps (main: {main // 1000}k)"
    return ""


def table_rows(df: pd.DataFrame, name_col: str = "policy", std: bool = True,
               cols: list[tuple[str, str]] | None = None, steps_label: str = "steps") -> tuple[list[str], list[list[str]]]:
    """Format mean (+- std) columns for display (steps_label='decisions' for supervisor results)."""
    cols = cols or TABLE_COLS
    header = [name_col] + [steps_label if k == "steps" else lbl for k, lbl in cols]
    rows = []
    for _, r in df.iterrows():
        row = [r[name_col]]
        for k, _ in cols:
            m, s = r[f"{k}_mean"], r.get(f"{k}_std", np.nan)
            fmt = "{:.0%}" if k in ("success", "coverage") else ("{:.1f}" if k in ("steps", "defects_found", "return") else "{:.3f}")
            row.append(fmt.format(m) + (f" +- {fmt.format(s)}" if std and k == "return" and not np.isnan(s) else ""))
        rows.append(row)
    return header, rows


def monitor(run_dir: Path) -> pd.DataFrame:
    """Training episodes from monitor CSVs (single or per-env), ordered by wall time, with cumulative steps."""
    run_dir = Path(run_dir)
    files = sorted(run_dir.glob("monitor/*.monitor.csv")) or sorted(run_dir.glob("monitor_train*.monitor.csv"))
    if not files:
        raise FileNotFoundError(f"no training monitor CSVs in {run_dir}")
    df = pd.concat([pd.read_csv(f, skiprows=1) for f in files]).sort_values("t").reset_index(drop=True)
    df["timesteps"] = df["l"].cumsum()
    if "success" in df:
        df["success"] = df["success"].astype(str).str.lower().eq("true").astype(float)
    return df


@dataclass
class EvalCurve:
    timesteps: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    ep_len: np.ndarray


def evaluations(run_dir: Path) -> EvalCurve:
    """EvalCallback log: deterministic evaluation return vs training steps."""
    z = np.load(Path(run_dir) / "eval" / "evaluations.npz")
    return EvalCurve(z["timesteps"], z["results"].mean(1), z["results"].std(1), z["ep_lengths"].mean(1))


def smooth(y: np.ndarray, frac: float = 0.03) -> np.ndarray:
    w = max(1, int(len(y) * frac))
    return pd.Series(y).rolling(w, min_periods=1).mean().to_numpy()


def plot_curves(axs, run_dir: Path, title: str, note: str = "") -> None:
    """Three panels: training + eval return, success rate, episode length (all from saved files)."""
    df, ev = monitor(run_dir), evaluations(run_dir)
    a0, a1, a2 = axs
    a0.plot(df["timesteps"], df["r"], color=style.LIGHT, lw=0.6, label="train episode")
    a0.plot(df["timesteps"], smooth(df["r"].to_numpy()), color=style.GREY, lw=2, label="train (smoothed)")
    a0.plot(ev.timesteps, ev.mean, "o-", color=style.CRIMSON, lw=2.5, ms=5, label="eval (deterministic)")
    a0.fill_between(ev.timesteps, ev.mean - ev.std, ev.mean + ev.std, color=style.CRIMSON, alpha=0.15)
    a0.set_title(f"{title}: return" + (f"\n[{note}]" if note else ""))
    a0.legend(loc="lower right")
    if "success" in df:
        a1.plot(df["timesteps"], smooth(df["success"].to_numpy(), 0.05), color=style.CRIMSON, lw=2.5)
        a1.set_ylim(-0.05, 1.05)
    a1.set_title("success rate (train, rolling)")
    a2.plot(df["timesteps"], smooth(df["l"].to_numpy()), color=style.GREY, lw=2, label="train")
    a2.plot(ev.timesteps, ev.ep_len, "o-", color=style.CRIMSON, lw=2, ms=5, label="eval")
    a2.set_title("episode length")
    a2.legend(loc="upper right")
    for a in axs:
        a.set_xlabel("environment steps")
        a.spines[["top", "right"]].set_visible(False)
        a.grid(alpha=0.3)
