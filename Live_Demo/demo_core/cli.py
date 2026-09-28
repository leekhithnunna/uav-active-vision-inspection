"""Shared command-line flags for every demo: --seed --detector --save --headless --fast."""
from __future__ import annotations

import argparse

from . import paths, style


def parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--seed", type=int, default=paths.default_seed(), help="bridge seed (same seed -> same bridge)")
    p.add_argument("--detector", default=paths.default_detector(), choices=["analytic", "learned"])
    p.add_argument("--save", action="store_true", help="write GIF/PNG into backup/")
    p.add_argument("--headless", action="store_true", help="no windows (tests / backups)")
    p.add_argument("--fast", action="store_true", help="fewer frames / shorter runs")
    return p


def parse(p: argparse.ArgumentParser, argv=None) -> argparse.Namespace:
    """Parse flags and pick a matplotlib backend (Agg when headless or saving)."""
    args = p.parse_args(argv)
    args.headless = args.headless or args.save              # saving always renders off-screen (Agg)
    import matplotlib.pyplot as plt
    if args.headless:
        plt.switch_backend("Agg")
    style.apply()
    return args
