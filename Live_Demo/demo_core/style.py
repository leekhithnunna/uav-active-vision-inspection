"""One place for the slide-consistent look: Amrita crimson, grey, RdYlGn_r, serif, projector-size text."""
from __future__ import annotations

CRIMSON = "#991245"
GREY = "#595959"
LIGHT = "#d9d9d9"
UNSEEN = "#cccccc"
GOOD = "#1a7f3c"
BAD = "#c0392b"
UNC_CMAP = "RdYlGn_r"
ALT_COLORS = ["#1f77b4", "#6a3d9a", "#111111"]     # trail colour for HIGH / MID / LOW
OPTION_COLORS = ["#1f77b4", CRIMSON, GREY]          # Explore / Refine / Return

FONT = 15


def apply() -> None:
    """Set matplotlib rcParams (idempotent)."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": FONT,
        "axes.titlesize": FONT + 2,
        "axes.titleweight": "bold",
        "axes.labelsize": FONT,
        "axes.edgecolor": GREY,
        "axes.titlecolor": "#222222",
        "xtick.labelsize": FONT - 2,
        "ytick.labelsize": FONT - 2,
        "legend.fontsize": FONT - 2,
        "figure.facecolor": "white",
        "figure.titlesize": FONT + 6,
        "figure.titleweight": "bold",
        "savefig.facecolor": "white",
        "animation.embed_limit": 50,
    })
