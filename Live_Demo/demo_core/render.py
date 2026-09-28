"""Matplotlib views. Each view is created once on an Axes and then updated per frame (fast, GIF-friendly)."""
from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects as pe
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle, Wedge

from . import style
from .rollout import ALT_NAMES, Frame

H_OUTLINE = [pe.withStroke(linewidth=4, foreground="white")]
ALT_RADIUS = (2, 1, 0)
UNSEEN_RGBA = (0.80, 0.80, 0.80, 1.0)


def figure(w: float = 16, h: float = 8, title: str | None = None):
    """A new figure with the demo style applied."""
    style.apply()
    fig = plt.figure(figsize=(w, h))
    if title:
        fig.suptitle(title, color=style.CRIMSON)
    return fig


def _empty_xy() -> np.ndarray:
    return np.empty((0, 2))


def _xy(mask: np.ndarray) -> np.ndarray:
    r, c = np.nonzero(mask)
    return np.column_stack([c, r]) if len(r) else _empty_xy()


class BridgeView:
    """Bridge surface: uncertainty heatmap, unseen cells, defects, drone, footprint, altitude-coloured trail, HUD."""

    def __init__(self, ax, defect: np.ndarray, title: str = "", colorbar: bool = True, hud: bool = True):
        self.ax, self.defect, self.title = ax, defect, title
        h, w = defect.shape
        self.cmap, self.norm = plt.get_cmap(style.UNC_CMAP), plt.Normalize(0, 1)
        self.mesh = ax.pcolormesh(np.arange(-0.5, w), np.arange(-0.5, h), np.zeros((h, w)),
                                  edgecolors="white", linewidth=0.8)
        self.mesh.set_array(None)                               # colours are set per cell in update()
        ax.set_xticks([]); ax.set_yticks([])
        ax.add_patch(Rectangle((-0.5, -0.5), 1, 1, fill=False, ec=style.GREY, lw=2.5, ls="--", zorder=2))
        ax.text(-0.42, -0.42, "BASE", ha="left", va="top", fontsize=style.FONT - 6, color="black", zorder=2)
        self.pending = ax.scatter([], [], marker="x", s=55, c="#444444", alpha=0.35, linewidths=1.5, zorder=3)
        self.found = ax.scatter([], [], marker="X", s=140, c="black", edgecolors="white", linewidths=1.0, zorder=4)
        self.trail = LineCollection([], linewidths=3.2, zorder=5, path_effects=H_OUTLINE)
        ax.add_collection(self.trail)
        self.fp = Rectangle((0, 0), 1, 1, fill=False, ec=style.CRIMSON, lw=3.5, zorder=6)
        ax.add_patch(self.fp)
        (self.drone,) = ax.plot([], [], marker="^", ms=20, color=style.CRIMSON, mec="white", mew=2.2, zorder=7)
        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(h - 0.5, -0.5)
        ax.set_aspect("equal")
        self.ttl = ax.set_title(title)
        self.hud = ax.text(0.0, -0.05, "", transform=ax.transAxes, ha="left", va="top",
                           fontsize=style.FONT - 1, family="monospace") if hud else None
        self.bat_ax = ax.inset_axes([0.78, -0.19, 0.22, 0.07]) if hud else None
        if self.bat_ax is not None:
            self.bat_ax.set_xlim(0, 1)
            self.bat_ax.set_xticks([]); self.bat_ax.set_yticks([])
            self.bat_bar = self.bat_ax.barh([0], [1.0], color=style.GOOD, height=1.0)[0]
            self.bat_txt = self.bat_ax.text(0.5, 0, "", ha="center", va="center", fontsize=style.FONT - 3,
                                            fontweight="bold", path_effects=H_OUTLINE)
        if colorbar:
            from matplotlib.cm import ScalarMappable
            cb = plt.colorbar(ScalarMappable(self.norm, self.cmap), ax=ax, fraction=0.025, pad=0.01)
            cb.set_label("uncertainty")

    def update(self, frames: Sequence[Frame], i: int, title: str | None = None) -> None:
        f = frames[i]
        rgba = self.cmap(self.norm(f.u))
        rgba[~f.seen] = UNSEEN_RGBA
        self.mesh.set_facecolor(rgba.reshape(-1, 4))
        self.pending.set_offsets(_xy(self.defect & ~f.found & f.seen))
        self.found.set_offsets(_xy(f.found))
        pts = [(fr.pos[1], fr.pos[0], fr.alt) for fr in frames[: i + 1]]
        segs = [[(a[0], a[1]), (b[0], b[1])] for a, b in zip(pts[:-1], pts[1:]) if a[:2] != b[:2]]
        cols = [style.ALT_COLORS[b[2]] for a, b in zip(pts[:-1], pts[1:]) if a[:2] != b[:2]]
        self.trail.set_segments(segs)
        self.trail.set_color(cols)
        r, c = f.pos
        rad = ALT_RADIUS[f.alt]
        self.fp.set_bounds(c - rad - 0.5, r - rad - 0.5, 2 * rad + 1, 2 * rad + 1)
        self.drone.set_data([c], [r])
        if title is not None:
            self.ttl.set_text(title)
        if self.hud is not None:
            m = f.metrics
            self.hud.set_text(f"step {f.t:3d} | alt {ALT_NAMES[f.alt]:4s} | coverage {m['coverage']:4.0%} | "
                              f"mean u {m['mean_uncertainty']:.2f}\n"
                              f"defects {m['defects_found']:2d}/{m['defects_total']} | "
                              f"energy {m['energy_used']:.2f}")
            self.bat_bar.set_width(f.battery)
            self.bat_bar.set_color(style.GOOD if f.battery > 0.4 else ("#e6a100" if f.battery > 0.15 else style.BAD))
            self.bat_txt.set_text(f"battery {f.battery:.0%}")


def altitude_legend(ax) -> None:
    """Legend explaining trail colours, defect markers and fog."""
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=c, lw=3.5, label=f"path @ {n.lower()}") for c, n in zip(style.ALT_COLORS, ALT_NAMES)]
    handles += [Line2D([], [], marker="X", color="black", ls="", ms=11, label="defect found"),
                Line2D([], [], marker="x", color="#444444", alpha=0.5, ls="", ms=9, label="defect seen, not detected"),
                Rectangle((0, 0), 1, 1, fc=style.UNSEEN, label="not yet seen")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=6, frameon=False,
              fontsize=style.FONT - 3)


class BarView:
    """Bar chart of per-action values (Q-values, option probabilities, state vector); chosen bar in crimson."""

    def __init__(self, ax, names: Sequence[str], title: str, ylim: tuple | None = None, fmt: str = "{:+.2f}",
                 colors: Sequence[str] | None = None):
        self.ax, self.fmt, self.fixed = ax, fmt, ylim
        self.custom = colors is not None
        self.colors = list(colors) if colors else [style.GREY] * len(names)
        self.bars = ax.bar(range(len(names)), np.zeros(len(names)), color=self.colors, edgecolor="black")
        ax.set_xticks(range(len(names)), names, rotation=0, fontsize=style.FONT - 3)
        ax.set_title(title)
        ax.axhline(0, color="black", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        if ylim:
            ax.set_ylim(*ylim)
        self.labels = [ax.text(i, 0, "", ha="center", va="bottom", fontsize=style.FONT - 2, fontweight="bold")
                       for i in range(len(names))]

    def update(self, values: Sequence[float], chosen: int | None = None) -> None:
        values = np.asarray(values, float)
        if self.fixed is None:
            lo, hi = min(0.0, values.min()), max(0.0, values.max())
            pad = 0.18 * (hi - lo or 1.0)
            self.ax.set_ylim(lo - pad * (lo < 0), hi + pad)
        for i, (b, v) in enumerate(zip(self.bars, values)):
            b.set_height(v)
            is_chosen = chosen == i
            b.set_color(self.colors[i] if (self.custom or not is_chosen) else style.CRIMSON)
            b.set_alpha(1.0 if chosen is None or is_chosen else 0.45)
            b.set_edgecolor("black")
            b.set_linewidth(3 if is_chosen else 1)
            self.labels[i].set_text(self.fmt.format(v))
            self.labels[i].set_y(max(v, 0))
            self.labels[i].set_color("black" if self.custom else (style.CRIMSON if is_chosen else "black"))
            self.labels[i].set_fontsize(style.FONT if is_chosen else style.FONT - 3)


class GaussianView:
    """PPO policy per action dimension: mean (dot) +- one std (bar); the executed action is clip(mean)."""

    def __init__(self, ax, names: Sequence[str], title: str = "Gaussian policy  a ~ N(mu, sigma)"):
        self.ax = ax
        n = len(names)
        ax.axvspan(-1, 1, color=style.LIGHT, alpha=0.5, zorder=0)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlim(-3.3, 3.3)
        ax.set_ylim(-0.6, n - 0.4)
        ax.set_yticks(range(n), names)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel("action value  (grey = allowed range [-1, 1])")
        self.err = [ax.errorbar([0], [i], xerr=[[0], [0]], fmt="o", color=style.CRIMSON, ms=11, capsize=8,
                                elinewidth=4, capthick=3) for i in range(n)]
        self.txt = [ax.text(3.25, i, "", ha="right", va="center", fontsize=style.FONT - 2, family="monospace")
                    for i in range(n)]

    def update(self, mu: Sequence[float], sigma: Sequence[float]) -> None:
        for i, (m, s) in enumerate(zip(mu, sigma)):
            line, caps, (bars,) = self.err[i]
            line.set_data([m], [i])
            bars.set_segments([[(m - s, i), (m + s, i)]])
            caps[0].set_data([m - s], [i]); caps[1].set_data([m + s], [i])
            self.txt[i].set_text(f"mu {m:+.2f}\nsigma {s:.2f}")


class GeometryView:
    """Refine geometry (schematic side view): UAV relative to the target cell and the ideal viewpoint."""

    def __init__(self, ax, title: str = "View geometry"):
        self.ax = ax
        ax.set_xlim(-1.35, 1.35)
        ax.set_ylim(-0.55, 1.3)
        ax.set_aspect("equal")
        ax.set_xlabel("lateral offset")
        ax.set_ylabel("distance d")
        ax.add_patch(Rectangle((-0.12, -0.52), 0.24, 0.14, color=style.BAD, zorder=2))
        ax.text(0.17, -0.45, "uncertain cell", va="center", fontsize=style.FONT - 3, color=style.BAD)
        ax.plot([0], [0], marker="*", ms=26, color=style.GOOD, mec="black", zorder=3)
        ax.text(0.1, 0.05, "ideal view (s=1)", fontsize=style.FONT - 3, color=style.GOOD)
        ax.spines[["top", "right"]].set_visible(False)
        (self.path,) = ax.plot([], [], "--", color=style.GREY, lw=2)
        self.cone = Wedge((0, 0), 0.5, 0, 0, color=style.CRIMSON, alpha=0.18)
        ax.add_patch(self.cone)
        (self.uav,) = ax.plot([], [], marker="^", ms=22, color=style.CRIMSON, mec="white", mew=2, zorder=5)
        self.ttl = ax.set_title(title)
        self.txt = ax.text(0.01, 0.99, "", va="top", transform=ax.transAxes, fontsize=style.FONT - 4,
                           family="monospace")

    def update(self, refs: Sequence[dict], title: str | None = None) -> None:
        r = refs[-1]
        x, y = r["lat"], r["d"]
        self.path.set_data([q["lat"] for q in refs], [q["d"] for q in refs])
        self.uav.set_data([x], [y])
        to_target = np.degrees(np.arctan2(-0.45 - y, -x))
        heading = to_target + 60 * r["th"]
        self.cone.set_center((x, y))
        self.cone.set_theta1(heading - 14); self.cone.set_theta2(heading + 14)
        self.txt.set_text(f"d {r['d']:.2f}  angle {r['th']:+.2f}  lateral {r['lat']:+.2f}\n"
                          f"view quality s = {r['s']:.2f} -> q = {r['q']}\n"
                          f"target uncertainty u = {r['target_u']:.2f}")
        if title is not None:
            self.ttl.set_text(title)


class QualityGauge:
    """Vertical gauge of view quality s with the five q-level bands."""

    def __init__(self, ax):
        self.ax = ax
        for q in range(5):
            lo, hi = max(0, (q - 0.5) / 4), min(1, (q + 0.5) / 4)
            ax.axhspan(lo, hi, color=["#d73027", "#fc8d59", "#fee08b", "#91cf60", "#1a9850"][q], alpha=0.35)
            ax.text(1.05, (lo + hi) / 2, f"q{q}", va="center", fontsize=style.FONT - 3, transform=ax.get_yaxis_transform())
        self.bar = ax.bar([0], [0], width=0.6, color=style.CRIMSON)[0]
        ax.set_xlim(-0.5, 0.5); ax.set_ylim(0, 1)
        ax.set_xticks([]); ax.set_title("s", fontsize=style.FONT)

    def update(self, s: float) -> None:
        self.bar.set_height(s)


class Timeline:
    """Row of coloured blocks, one per supervisor decision, labelled with its reward (artists created once)."""

    def __init__(self, ax, decisions: Sequence[Frame], names: Sequence[str] = ("EXPLORE", "REFINE", "RETURN")):
        self.ax = ax
        n = max(len(decisions), 10)
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel("decision #")
        ax.set_title("Decision timeline (block = option, number = reward)")
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=c, label=n) for c, n in zip(style.OPTION_COLORS, names)],
                  loc="upper right", ncol=3, fontsize=style.FONT - 4, frameon=False, bbox_to_anchor=(1, 1.45))
        fs = style.FONT - 5 if len(decisions) <= 24 else style.FONT - 8
        self.blocks = []
        for k, f in enumerate(decisions):
            rect = ax.add_patch(Rectangle((k - 0.45, 0.1), 0.9, 0.8, color=style.OPTION_COLORS[int(f.action)], visible=False))
            txt = ax.text(k, 0.5, f"{f.reward:+.2f}", ha="center", va="center", color="white", fontsize=fs,
                          fontweight="bold", visible=False)
            self.blocks.append((rect, txt))

    def update(self, n_done: int) -> None:
        """Show the first n_done decisions."""
        for k, (rect, txt) in enumerate(self.blocks):
            rect.set_visible(k < n_done)
            txt.set_visible(k < n_done)


def text_panel(ax, text: str = "", size: int | None = None, **kw):
    """Axes turned into a text box; returns the Text artist."""
    ax.axis("off")
    return ax.text(0.0, 1.0, text, va="top", ha="left", family="monospace", fontsize=size or style.FONT, **kw)


def table(ax, header: Sequence[str], rows: Sequence[Sequence], title: str = "", highlight_row: int | None = None,
          size: int | None = None):
    """Render a small results table on an Axes (projector-sized)."""
    ax.axis("off")
    n = len(header)
    widths = [2.4 / (n + 1.4)] + [1.0 / (n + 1.4)] * (n - 1)          # wider first (name) column
    t = ax.table(cellText=[[str(c) for c in r] for r in rows], colLabels=list(header), loc="center",
                 cellLoc="center", colWidths=widths)
    t.auto_set_font_size(False)
    t.set_fontsize(size or style.FONT)
    t.scale(1, 1.8)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor(style.GREY)
        if r == 0:
            cell.set_facecolor(style.CRIMSON); cell.set_text_props(color="white", fontweight="bold")
        elif highlight_row is not None and r - 1 == highlight_row:
            cell.set_facecolor("#f6dbe5")
    if title:
        ax.set_title(title)
    return t
