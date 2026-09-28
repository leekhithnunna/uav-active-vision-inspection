"""Reads config.yaml, puts uav_inspection/ on sys.path and resolves every path the demo uses."""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

import yaml

DEMO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = DEMO_ROOT / "config.yaml"


@lru_cache(maxsize=1)
def config() -> dict:
    """The parsed config.yaml (cached)."""
    with open(CONFIG_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _expand(p: str | os.PathLike) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(p))))


def repo_root() -> Path:
    """files_coding/ (the folder that contains uav_inspection/ and Live_Demo/)."""
    return (DEMO_ROOT / _expand(config()["repo_root"])).resolve()


def resolve(p: str | os.PathLike | None) -> Path | None:
    """Resolve a config path relative to the repo root (absolute paths and env vars allowed)."""
    if p is None:
        return None
    path = _expand(p)
    return path if path.is_absolute() else (repo_root() / path).resolve()


def demo_path(p: str | os.PathLike) -> Path:
    """Resolve a path relative to Live_Demo/ (used for outputs)."""
    path = _expand(p)
    return path if path.is_absolute() else (DEMO_ROOT / path).resolve()


def uav_root() -> Path:
    return resolve(config()["uav_inspection"])


def ensure_uav_on_path() -> None:
    """Make `import env` work exactly like uav_inspection/scripts/_path.py does."""
    root = str(uav_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def default_seed() -> int:
    return int(config()["defaults"]["seed"])


def default_detector() -> str:
    return str(config()["defaults"]["detector"])


def model_path(key: str, detector: str = "analytic") -> Path | None:
    """Path of a trained model; key = 'A' | 'B' | 'C' | 'C_withA'."""
    return resolve(config()["models"].get(detector, {}).get(key))


def full_supervisor_key() -> str:
    return str(config().get("full_supervisor", "C"))


def results_entry(agent: str, which: str = "main"):
    """A results folder (Path) or list of folders from config['results'][agent][which]."""
    v = config()["results"][agent][which]
    return [resolve(x) for x in v] if isinstance(v, list) else resolve(v)


def detector_path(key: str) -> Path:
    return resolve(config()["detector"][key])


def out_dir(kind: str) -> Path:
    """backup/gifs | backup/frames | backup/snapshots | tmp_train - created on demand."""
    d = demo_path(config()["output"][kind])
    d.mkdir(parents=True, exist_ok=True)
    return d


ensure_uav_on_path()
