"""FuncAnimation helpers: live window, GIF (PillowWriter, always works) and MP4 (only if ffmpeg exists)."""
from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
from matplotlib import animation

from . import narrate, paths


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def play(fig, update: Callable[[int], None], n_frames: int, args, name: str, fps: int = 6,
         frames_png: dict[str, int] | None = None) -> list[Path]:
    """Show and/or save an animation according to --headless / --save / --fast.

    update(i) draws frame i. Saving writes backup/gifs/<name>.gif (+ .mp4 if ffmpeg) and any requested
    key frames as backup/frames/<name>_<tag>.png. Headless without --save just renders every frame once
    (so tests exercise the drawing code) and returns quickly.
    """
    saved: list[Path] = []
    if args.save:
        saved += _save(fig, update, n_frames, name, fps)
        for tag, i in (frames_png or {}).items():
            update(min(i, n_frames - 1) if i >= 0 else n_frames + i)
            p = paths.out_dir("frames") / f"{name}_{tag}.png"
            fig.savefig(p, dpi=110, bbox_inches="tight")
            saved.append(p)
    elif args.headless:
        step = max(1, n_frames // 12) if args.fast else 1
        for i in list(range(0, n_frames, step)) + [n_frames - 1]:
            update(i)
        fig.canvas.draw()
    if not args.headless:
        anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, repeat=False)
        fig._demo_anim = anim                                   # keep a reference while the window is open
        plt.show()
    else:
        plt.close(fig)
    return saved


def write_gif(fig, update: Callable[[int], None], n_frames: int, path: Path, fps: int, dpi: int = 72) -> None:
    """Render each frame with Agg and write an animated GIF with Pillow (fast octree palette, loops forever)."""
    from PIL import Image
    old_dpi = fig.get_dpi()
    fig.set_dpi(dpi)
    images = []
    for i in range(n_frames):
        update(i)
        fig.canvas.draw()
        rgb = Image.frombuffer("RGBA", fig.canvas.get_width_height(), bytes(fig.canvas.buffer_rgba())).convert("RGB")
        images.append(rgb.quantize(colors=128, method=Image.Quantize.FASTOCTREE))
    fig.set_dpi(old_dpi)
    images[0].save(path, save_all=True, append_images=images[1:], duration=int(1000 / fps), loop=0)


def _save(fig, update, n_frames: int, name: str, fps: int) -> list[Path]:
    t0 = time.time()
    gif = paths.out_dir("gifs") / f"{name}.gif"
    write_gif(fig, update, n_frames, gif, fps)
    out = [gif]
    mb = gif.stat().st_size / 1e6
    narrate.info(f"saved {gif.name} ({mb:.1f} MB, {n_frames} frames, {time.time() - t0:.0f}s)")
    if mb > 5:
        narrate.warn(f"{gif.name} is {mb:.1f} MB (> 5 MB): it will be git-ignored, see README")
    if ffmpeg_available():
        mp4 = gif.with_suffix(".mp4")
        anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, repeat=False)
        anim.save(mp4, writer=animation.FFMpegWriter(fps=fps), dpi=100)
        out.append(mp4)
    return out


def save_png(fig, name: str, kind: str = "frames") -> Path:
    """Save a still figure into backup/<kind>/<name>.png."""
    p = paths.out_dir(kind) / f"{name}.png"
    fig.savefig(p, dpi=110, bbox_inches="tight")
    return p


def finish(fig, args, name: str | None = None, kind: str = "frames") -> Path | None:
    """For static figures: save if asked, show unless headless."""
    p = save_png(fig, name, kind) if (args.save and name) else None
    if args.headless:
        plt.close(fig)
    else:
        plt.show()
    return p
