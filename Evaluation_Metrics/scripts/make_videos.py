"""Phase 7 - videos of experimental sessions in varying task environments.

The Live_Demo demos are imported as a LIBRARY (no Live_Demo file is edited):
  * demo_core.paths.out_dir is redirected at runtime -> Evaluation_Metrics/videos/raw/<clip>/<kind>/
    (Live_Demo's own --save would overwrite the tracked backup GIFs, see known_issues #5)
  * for scenario clips, demo_core.loaders.load_env is replaced at runtime by a ScenarioEnv factory
  * demo_core.rollout.run_episode is wrapped to record each episode's final metrics for the title card
Each GIF is converted to MP4 (H.264, yuv420p) with the bundled imageio-ffmpeg, a 2.5 s title card is
prepended (clip, scenario, metric to watch, final values read from the run) and the clip is kept <= 60 s.
Outputs: videos/V*.mp4, videos/video_runs.csv (values read from each run), videos/video_index.md,
videos/narration_script.md
"""
from __future__ import annotations

import importlib
import os
import sys
import time
from pathlib import Path

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import ROOT, VIDEOS, RESULTS, ScenarioEnv, CACHE, SCENARIOS  # noqa: E402

LIVE = ROOT / "Live_Demo"
sys.path.insert(0, str(LIVE))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
from demo_core import paths as dpaths, loaders, rollout  # noqa: E402

RAW = VIDEOS / "raw"
FPS_OUT = 8
MAX_S, MAX_MB, CARD_S = 60.0, 25.0, 2.5

# ------------------------------------------------------------------ runtime redirections (no file edits)
_state = {"clip": "unset", "episodes": []}
_orig_load_env, _orig_run_episode = loaders.load_env, rollout.run_episode


def _out_dir(kind: str) -> Path:
    d = RAW / _state["clip"] / kind
    d.mkdir(parents=True, exist_ok=True)
    return d


def _recording_run_episode(env, policy, seed, label, *a, **k):
    ep = _orig_run_episode(env, policy, seed, label, *a, **k)
    m = ep.frames[-1].metrics
    _state["episodes"].append({"clip": _state["clip"], "label": label, "seed": seed, "return": ep.total_return,
                               "steps": m["steps"], "success": bool(m["success"]), "energy_used": m["energy_used"],
                               "coverage": m["coverage"], "mean_uncertainty": m["mean_uncertainty"],
                               "defects_found": m["defects_found"], "defects_total": m["defects_total"],
                               "battery_empty": bool(env.battery <= 0)})
    return ep


def _scenario_loader(scenario):
    sc = SCENARIOS[scenario]

    def load_env(mode, detector="analytic", **kw):
        det = sc.get("detector", detector)
        return ScenarioEnv(mode, defect_ratio=sc.get("defect_ratio", 0.20), start_battery=sc.get("start_battery", 1.0),
                           hard_start=sc.get("hard_start", False), detector=det,
                           cache_path=str(CACHE) if det == "learned" else None, **kw)
    return load_env


dpaths.out_dir = _out_dir
rollout.run_episode = _recording_run_episode

# ------------------------------------------------------------------ clip definitions
# (clip id, [(demo module, argv, scenario-or-None)], scenario label, question, metric to watch)
CLIPS = [
    ("V1_env_tour", [("demos.d1_env_tour", ["--seed", "1000"], None)], "S0 nominal (seed 1000)",
     "What is the task and what do we measure?", "coverage, mean uncertainty, energy, return"),
    ("V2_agentA_vs_raster_seed1000", [("demos.d3_agentA_vs_raster", ["--seed", "1000"], None)],
     "S0 nominal, training-eval bridge seed 1000", "Does Agent A finish the inspection with fewer steps?",
     "steps, energy, success"),
    ("V3_agentA_vs_raster_heldout2005", [("demos.d3_agentA_vs_raster", ["--seed", "2005"], None)],
     "S0 nominal, held-out bridge seed 2005", "Does Agent A generalise to an unseen bridge?",
     "success, steps, energy"),
    ("V4_agentA_real_images_seed2000", [("demos.d3_agentA_vs_raster", ["--seed", "2000", "--detector", "learned"], None)],
     "S1 real-image (DeepCrack) detector, held-out seed 2000", "Does Agent A work with real-image uncertainty?",
     f"success (mean uncertainty must reach the calibrated {float(np.load(CACHE)['success_unc']):.3f})"),
    ("V5_agentB_nominal_and_hardstart",
     [("demos.d4_agentB_refine", ["--seed", "2000", "--episodes", "2"], None),
      ("demos.d4_agentB_refine", ["--seed", "2000", "--episodes", "1"], "S5_hardstart")],
     "S0 nominal (seeds 2000-2001) + S5 hard start geometry (seed 2000)",
     "Does PPO reach the best view (q=4), even from a hard start?", "steps to q=4, return"),
    ("V6_agentC_mission_seed2000", [("demos.d6_agentC_mission", ["--seed", "2000"], None)],
     "S0 nominal, held-out seed 2000", "Which options does the A2C supervisor choose?",
     "option sequence (EXPLORE / REFINE / RETURN), defects, energy"),
    ("V7_full_mission_learned_vs_rule",
     [("demos.d7_full_mission", ["--seed", "2000", "--supervisor", "C"], None),
      ("demos.d7_full_mission", ["--seed", "2000", "--supervisor", "rule"], None)],
     "S0 nominal, held-out seed 2000, full mode", "Do the three agents work together?",
     "defects found, energy, whether Agent B is ever called"),
    ("V8_detector_q0_to_q4", [("demos.d5_detector_views", [], None)], "real DeepCrack test patches",
     "Why are real images harder?", "uncertainty and detection at view quality q0..q4"),
    ("V9_agentA_density35_seed2013", [("demos.d3_agentA_vs_raster", ["--seed", "2013"], "S2_density35")],
     "S2 defect density 35%, held-out seed 2013 (a DQN failure case)",
     "Does Agent A cope with many more defects?", "success, battery"),
    ("V10_agentA_battery0.6_seed2000", [("demos.d3_agentA_vs_raster", ["--seed", "2000"], "S4_battery0.6")],
     "S4 starting battery 0.6, held-out seed 2000", "Is Agent A safe when the battery is low?",
     "battery-empty, success"),
]


# ------------------------------------------------------------------ GIF -> MP4 with title card
def read_gif(path):
    from PIL import Image, ImageSequence
    im = Image.open(path)
    frames, durs = [], []
    for fr in ImageSequence.Iterator(im):
        frames.append(np.asarray(fr.convert("RGB")))
        durs.append(fr.info.get("duration", 125))
    return frames, durs


def to_fixed_fps(frames, durs, fps, stretch=1.0):
    """Repeat/drop frames so the GIF timing (slowed down by `stretch`) is kept at a constant output fps."""
    out, t, acc = [], 0.0, 0.0
    for f, d in zip(frames, durs):
        acc += stretch * d / 1000.0
        while t < acc:
            out.append(f)
            t += 1.0 / fps
    return out


def pad_to(img, h, w):
    canvas = np.full((h, w, 3), 252, np.uint8)
    y, x = (h - img.shape[0]) // 2, (w - img.shape[1]) // 2
    canvas[y:y + img.shape[0], x:x + img.shape[1]] = img
    return canvas


def title_card(h, w, lines):
    import matplotlib.pyplot as plt
    dpi = 100
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi, facecolor="#fcfcfb")
    fig.text(0.05, 0.85, lines[0], fontsize=max(14, h // 28), fontweight="bold", color="#0b0b0b", va="top")
    y = 0.70
    for ln in lines[1:]:
        fig.text(0.05, y, ln, fontsize=max(10, h // 48), color="#52514e", va="top", family="DejaVu Sans")
        y -= 0.075 * (1 + ln.count("\n"))
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img[:h, :w]


def write_mp4(path, card, frames):
    import imageio.v2 as imageio
    w = imageio.get_writer(str(path), fps=FPS_OUT, codec="libx264", quality=7, pixelformat="yuv420p",
                           macro_block_size=16, ffmpeg_log_level="error")
    for f in [card] * int(CARD_S * FPS_OUT) + frames:
        w.append_data(f)
    w.close()


def result_lines(eps):
    out = []
    for e in eps:
        tag = "SUCCESS" if e["success"] else ("BATTERY EMPTY" if e["battery_empty"] else "not successful")
        out.append(f"{e['label'][:46]:46s} seed {e['seed']}: return {e['return']:+.2f}, steps {e['steps']}, "
                   f"energy {e['energy_used']:.2f}, defects {e['defects_found']}/{e['defects_total']} -> {tag}")
    return out


MIN_S = 55.0          # short demos are slowed down to at least this length, so a ~1 min voice-over fits


def render_parts(cid, parts):
    """Run the demos (as a library) and return (GIF paths, episode records)."""
    gifs = []
    _state["episodes"] = []
    for k, (mod, argv, scen) in enumerate(parts):
        _state["clip"] = f"{cid}_part{k}"
        loaders.load_env = _scenario_loader(scen) if scen else _orig_load_env
        m = importlib.import_module(mod)
        res = m.main(argv + ["--save", "--headless"])
        gifs += [p for p in res.get("saved", []) if str(p).endswith(".gif")]
        import json
        scalars = {k: float(v) for k, v in res.items() if isinstance(v, (int, float, np.floating, np.integer))
                   and not isinstance(v, bool)}
        (RAW / _state["clip"] / "result.json").write_text(json.dumps({"demo": mod, "argv": argv, "scenario": scen,
                                                                      "result": scalars}, indent=1))
        import matplotlib.pyplot as plt
        plt.close("all")
    loaders.load_env = _orig_load_env
    return gifs, list(_state["episodes"])


def build_clip(cid, parts, scen_label, question, metric, convert_only=False):
    if convert_only:     # reuse the GIFs and run records of a previous render
        gifs = sorted(RAW.glob(f"{cid}_part*/gifs/*.gif"))
        runs = pd.read_csv(VIDEOS / "video_runs.csv")
        eps = runs[runs["clip"].astype(str).str.startswith(cid + "_part")].to_dict("records")
    else:
        gifs, eps = render_parts(cid, parts)
    raw = [read_gif(g) for g in gifs]
    total = sum(sum(du) for _, du in raw) / 1000.0
    stretch = max(1.0, MIN_S / max(total, 1e-6))
    seqs = [to_fixed_fps(fr, du, FPS_OUT, stretch) for fr, du in raw]
    h = max(s[0].shape[0] for s in seqs)
    w = max(s[0].shape[1] for s in seqs)
    h, w = h + (-h) % 16, w + (-w) % 16
    frames = [pad_to(f, h, w) for s in seqs for f in s]
    budget = int((MAX_S - CARD_S - 0.5) * FPS_OUT)
    if len(frames) > budget:                      # keep <= 60 s by uniform subsampling
        frames = [frames[int(i)] for i in np.linspace(0, len(frames) - 1, budget)]
    extra = []
    if not eps:          # demos without recorded episodes (env tour, detector): use the demo's own result dict
        import json
        for rj in sorted(RAW.glob(f"{cid}_part*/result.json")):
            res = json.loads(rj.read_text())["result"]
            extra.append(", ".join(f"{k} = {v:g}" for k, v in res.items()))
    lines = [cid.replace("_", " "), f"Scenario: {scen_label}", f"Question: {question}", f"Metric to watch: {metric}",
             "Final values read from this run:"] + (result_lines(eps) or extra or ["(see clip)"])
    card = title_card(h, w, lines)
    out = VIDEOS / f"{cid}.mp4"
    write_mp4(out, card, frames)
    mb = out.stat().st_size / 1e6
    dur = CARD_S + len(frames) / FPS_OUT
    print(f"{cid}: {dur:.1f} s (x{stretch:.1f} slow-down), {mb:.1f} MB, {len(eps)} episodes", flush=True)
    return eps, dur, mb, stretch


def main(only=None, convert_only=False):
    VIDEOS.mkdir(exist_ok=True)
    rows, meta = [], []
    old = VIDEOS / "video_runs.csv"
    keep_prev = (only or convert_only) and old.exists()
    prev = pd.read_csv(old) if keep_prev else None
    prev_meta = pd.read_csv(VIDEOS / "video_files.csv") if keep_prev and (VIDEOS / "video_files.csv").exists() else None
    for cid, parts, scen, q, metric in CLIPS:
        if only and cid not in only:
            continue
        t0 = time.time()
        eps, dur, mb, stretch = build_clip(cid, parts, scen, q, metric, convert_only)
        rows += eps
        meta.append({"clip": cid, "file": f"videos/{cid}.mp4", "scenario": scen, "question": q, "metric": metric,
                     "duration_s": round(dur, 1), "size_mb": round(mb, 2), "slow_down": round(stretch, 2),
                     "render_s": round(time.time() - t0)})
    runs, files = pd.DataFrame(rows), pd.DataFrame(meta)
    if prev is not None and prev_meta is not None:
        base = prev["clip"].astype(str).str.replace(r"_part\d+$", "", regex=True)
        runs = pd.concat([prev[~base.isin(files["clip"])], runs])
        files = pd.concat([prev_meta[~prev_meta["clip"].isin(files["clip"])], files])
    order = {c[0]: i for i, c in enumerate(CLIPS)}
    files = files.sort_values("clip", key=lambda s: s.map(order)).reset_index(drop=True)
    runs.to_csv(VIDEOS / "video_runs.csv", index=False)
    files.to_csv(VIDEOS / "video_files.csv", index=False)
    export_previews()


def export_previews():
    """One PNG from the middle and one from the end of every final MP4 -> videos/preview_frames/."""
    import imageio.v2 as imageio
    out = VIDEOS / "preview_frames"
    out.mkdir(exist_ok=True)
    for mp4 in sorted(VIDEOS.glob("V*.mp4")):
        rd = imageio.get_reader(str(mp4))
        frames = [f for f in rd]
        rd.close()
        n = len(frames)
        card = int(CARD_S * FPS_OUT)               # the middle frame is taken from the animation, not the card
        for tag, i in [("mid", card + (n - card) // 2), ("end", n - 1)]:
            imageio.imwrite(str(out / f"{mp4.stem}_{tag}.png"), frames[i])
    print(f"preview frames -> {out}")


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    main(a or None, convert_only="--convert-only" in sys.argv)
