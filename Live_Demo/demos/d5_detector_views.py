"""d5 - Real-image detector: U-Net + MC-dropout on DeepCrack TEST patches at view qualities q0..q4 (live)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo_core import cli, detector, narrate, paths  # noqa: E402


def _report_table() -> None:
    rep = detector.report()
    rows = [[r["q"], f"{r['res']} px", f"{r['u_defect']:.3f}", f"{r['u_clean']:.3f}", f"{r['detect_rate']:.1%}",
             f"{r['false_pos']:.1%}", f"{r['iou_defect']:.3f}"] for r in rep["report"]]
    narrate.results(["q", "resolution", "u (crack)", "u (clean)", "detect rate", "false pos", "IoU (crack)"], rows,
                    f"Saved detector report (detector_report.json: {rep['n_defect']} crack + {rep['n_clean']} clean "
                    f"TEST patches, {rep['mc_passes']} MC passes)")
    narrate.say(f"   env mapping (learned mode): altitude HIGH/MID/LOW -> q{rep['alt_q'][0]}/q{rep['alt_q'][1]}/"
                f"q{rep['alt_q'][2]};  q4 only through Agent B's refinement.")


def _live(args) -> dict:
    """Run the U-Net live on TEST patches and animate q0 -> q4."""
    from demo_core import animate, render, style
    cfg = paths.config()["detector"]
    patches = detector.test_patches(args.seed, n_crack=max(1, cfg["n_patches"] - 1), n_clean=1)
    if args.fast:
        patches = patches[:1] + patches[-1:]
    narrate.guideline(1, f"U-Net loaded from {paths.detector_path('unet').name}; "
                         f"{len(patches)} patches from the DeepCrack TEST split")
    results = []
    for p in patches:
        views = detector.analyse(p, cfg["mc_passes"], seed=args.seed)
        results.append((p, views))
        narrate.results(["q", "res", "entropy u (bits)", "MC std", "IoU", "detected"],
                        [[v.q, f"{v.res}px", f"{v.entropy:.3f}", f"{v.std.mean():.3f}", f"{v.iou:.2f}",
                          "yes" if v.detected else "no"] for v in views],
                        f"{'CRACK' if p.is_crack else 'CLEAN'} patch {p.source}")

    fig = render.figure(18, 10.5, "Demo 5 - Real-image crack detector (U-Net, live MC-dropout) at view qualities q0..q4")
    gs = fig.add_gridspec(3, 6, hspace=0.35, wspace=0.12)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(6)] for r in range(3)]
    for row in axes:
        for a in row:
            a.set_xticks([]); a.set_yticks([])
    for r, lbl in enumerate(["camera view", "crack probability\n(MC mean)", "uncertainty\n(MC std)"]):
        axes[r][0].set_ylabel(lbl, fontsize=style.FONT)
    frames = [(i, q) for i in range(len(results)) for q in range(5)]
    frames += [frames[-1]] * 2

    def update(k: int) -> None:
        i, qmax = frames[k]
        p, views = results[i]
        for q in range(5):
            v = views[q]
            show = q <= qmax
            for r, (img, kw) in enumerate([(v.view, {}), (v.mean, dict(cmap="gray", vmin=0, vmax=1)),
                                           (v.std, dict(cmap="magma", vmin=0, vmax=0.12))]):
                a = axes[r][q]
                a.clear(); a.set_xticks([]); a.set_yticks([])
                if show:
                    a.imshow(img, **kw)
            axes[0][q].set_title(f"q{q} ({v.res}px)" if show else "", fontsize=style.FONT)
            lbl = f"u={v.entropy:.2f}  IoU={v.iou:.2f}" if p.is_crack else f"u={v.entropy:.2f}  {'FP' if v.detected else 'no FP'}"
            axes[2][q].set_xlabel(lbl if show else "", fontsize=style.FONT - 2)
        for r in range(3):
            a = axes[r][5]
            a.clear(); a.set_xticks([]); a.set_yticks([])
        axes[0][5].imshow(p.image); axes[0][5].set_title("full-res patch", fontsize=style.FONT)
        axes[1][5].imshow(p.label, cmap="gray", vmin=0, vmax=1); axes[1][5].set_title("ground truth", fontsize=style.FONT)
        axes[2][5].axis("off")
        axes[2][5].text(0.05, 0.5, f"{'CRACK' if p.is_crack else 'CLEAN'}\npatch {i + 1}/{len(results)}",
                        fontsize=style.FONT + 2, fontweight="bold", color=style.CRIMSON, va="center")
        for r in range(3):
            axes[r][0].set_ylabel(["camera view", "crack prob.\n(MC mean)", "uncertainty\n(MC std)"][r], fontsize=style.FONT)

    narrate.guideline(4, "Same patch seen at q0 (far/high) ... q4 (Agent B's best view)")
    saved = animate.play(fig, update, len(frames), args, "detector", fps=2, frames_png={"crack": 4, "final": -1})
    crack = [v for p, vs in results if p.is_crack for v in vs]
    return {"patches": len(results), "saved": saved,
            "u_q0": sum(v.entropy for v in crack if v.q == 0) / max(1, sum(v.q == 0 for v in crack)),
            "u_q4": sum(v.entropy for v in crack if v.q == 4) / max(1, sum(v.q == 4 for v in crack))}


def _fallback(args) -> dict:
    """No dataset / weights: show the saved detector figure."""
    import matplotlib.image as mpimg
    from demo_core import animate, render
    fig_path = paths.detector_path("figure")
    narrate.warn(f"live detector unavailable - showing the SAVED figure {fig_path.name} (made by build_detector_cache.py)")
    fig = render.figure(14, 9, "Demo 5 - Detector views (saved figure, not live)")
    ax = fig.add_subplot(111)
    ax.imshow(mpimg.imread(fig_path)); ax.axis("off")
    animate.finish(fig, args, "detector_saved")
    return {"patches": 0, "saved": []}


def main(argv=None) -> dict:
    p = cli.parser(__doc__.splitlines()[0])
    args = cli.parse(p, argv)
    narrate.header("DEMO 5 - Real-image detector", "DeepCrack -> U-Net with MC-dropout -> per-cell uncertainty")
    _report_table()
    try:
        ok = detector.dataset_available() and paths.detector_path("unet").exists()
        if not ok:
            narrate.warn(f"DeepCrack TEST images not found at {detector.dataset_dir()} (see README) or U-Net missing")
        out = _live(args) if ok else _fallback(args)
    except ImportError as e:
        narrate.warn(f"torch / PIL import failed ({e})")
        out = _fallback(args)
    rep = detector.report()
    r0, r4 = rep["report"][0], rep["report"][-1]
    narrate.guideline(8, "What the saved evaluation says (detector_report.json)")
    narrate.say(f"   Per patch, raw MC entropy is NOT monotone in q ({rep['mono_violations']:.0%} of q-steps rise before "
                f"fusion; at q0 the net can be confidently wrong, IoU 0).\n"
                f"   The cache therefore rank-normalises u and applies best-view fusion (running min over q).\n"
                f"   Over {rep['n_defect']} crack patches: u {r0['u_defect']:.2f} -> {r4['u_defect']:.2f}, "
                f"detect rate {r0['detect_rate']:.0%} -> {r4['detect_rate']:.0%}, IoU {r0['iou_defect']:.2f} -> "
                f"{r4['iou_defect']:.2f} from q0 to q4 (false positives {r0['false_pos']:.1%} -> {r4['false_pos']:.1%}).")
    return out


if __name__ == "__main__":
    main()
