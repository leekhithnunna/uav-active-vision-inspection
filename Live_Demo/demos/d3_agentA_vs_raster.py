"""d3 - Agent A (DQN) vs dense raster scan on the SAME bridge, animated side by side with live counters."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo_core import cli, loaders, metrics, narrate, paths  # noqa: E402

RASTER = {"low": (2, "Dense raster, LOW altitude (1x1)", "Raster LOW (baseline)"),
          "mid": (1, "Raster, MID altitude (3x3)", "Raster MID (baseline)")}


def _counter(ep, name: str, i: int) -> str:
    f = ep.frames[min(i, len(ep.frames) - 1)]
    m = f.metrics
    done = i >= len(ep.frames) - 1
    status = ("DONE - success" if m["success"] else "DONE - not successful") if done else "flying"
    return (f"{name:22s} steps {f.t:3d} | energy {m['energy_used']:.2f} | coverage {m['coverage']:4.0%} | "
            f"mean u {m['mean_uncertainty']:.3f} | defects {m['defects_found']:2d}/{m['defects_total']} | {status}")


def _pct(a: float, b: float) -> float:
    return 100.0 * (a - b) / b if b else 0.0


def main(argv=None) -> dict:
    p = cli.parser(__doc__.splitlines()[0])
    p.add_argument("--raster", choices=list(RASTER), default="low", help="which raster baseline to race")
    args = cli.parse(p, argv)
    from demo_core import animate, render, rollout
    from env.baselines import RasterNav

    agent = loaders.load_agent("A", args.detector)
    alt, rlabel, rshort = RASTER[args.raster]
    narrate.header("DEMO 3 - Agent A vs raster", f"same seed {args.seed} -> same bridge for both")
    narrate.guideline(1, f"Two identical environments (navigate mode, seed {args.seed}, detector {args.detector})")
    ep_a = rollout.run_episode(loaders.load_env("navigate", args.detector), agent, args.seed, agent.label)
    ep_r = rollout.run_episode(loaders.load_env("navigate", args.detector), RasterNav(alt), args.seed,
                               f"BASELINE {rlabel}")
    n = max(len(ep_a.frames), len(ep_r.frames))
    stride = 3 if args.fast else 2
    idx = list(range(0, n, stride)) + [n - 1] * (8 if not args.fast else 2)      # hold the end screen

    fig = render.figure(18, 8.5, "Demo 3 - Learned viewpoint planning vs scripted raster (same bridge)")
    gs = fig.add_gridspec(3, 2, height_ratios=[2.5, 0.05, 1.0], hspace=0.28, wspace=0.12, top=0.95, bottom=0.02)
    va = render.BridgeView(fig.add_subplot(gs[0, 0]), ep_a.defect)
    vr = render.BridgeView(fig.add_subplot(gs[0, 1]), ep_r.defect)
    render.altitude_legend(va.ax)
    va.ax.get_legend().set_bbox_to_anchor((1.0, -0.22))
    bottom = fig.add_subplot(gs[2, :])
    txt = render.text_panel(bottom, "", size=15)
    vtxt = bottom.text(0.0, 0.45, "", va="top", fontsize=18, fontweight="bold", color=render.style.CRIMSON)
    foot = bottom.text(0.0, 0.12, "", va="top", fontsize=13, color=render.style.GREY)

    fa, fr = ep_a.frames[-1].metrics, ep_r.frames[-1].metrics
    ds, de = _pct(fa["steps"], fr["steps"]), _pct(fa["energy_used"], fr["energy_used"])
    rows = [[e.label[:34], e.frames[-1].metrics["steps"], f"{e.frames[-1].metrics['energy_used']:.3f}",
             f"{e.frames[-1].metrics['coverage']:.0%}", f"{e.frames[-1].metrics['mean_uncertainty']:.3f}",
             f"{e.frames[-1].metrics['defects_found']}/{e.frames[-1].metrics['defects_total']}",
             "yes" if e.frames[-1].metrics["success"] else "no", f"{e.total_return:+.2f}"]
            for e in (ep_a, ep_r)]
    header = ["policy", "steps", "energy", "coverage", "mean u", "defects", "success", "return"]
    verdict = (f"Agent A on this bridge: {ds:+.0f}% steps, {de:+.0f}% energy vs {rlabel.split(',')[0].lower()}  "
               f"(defects found {fa['defects_found']} vs {fr['defects_found']})")

    def update(k: int) -> None:
        i = idx[k]
        va.update(ep_a.frames, min(i, len(ep_a.frames) - 1), title=ep_a.label)
        vr.update(ep_r.frames, min(i, len(ep_r.frames) - 1), title=ep_r.label)
        end = i >= n - 1
        lines = [_counter(ep_a, "Agent A (DQN)", i), _counter(ep_r, rshort, i)]
        lines += ["", "", "", "computed live from this run - saved 20-bridge means: demo 9"] if end else []
        txt.set_text("\n".join(lines))
        vtxt.set_text(verdict if end else "")

    narrate.guideline(7, "Trained policy vs scripted baseline, animated")
    saved = animate.play(fig, update, len(idx), args, "agentA_vs_raster", fps=8, frames_png={"end": -1, "mid": len(idx) // 2})
    narrate.guideline(8, "Final performance on this bridge (live)")
    narrate.results(header, rows, f"Seed {args.seed}: live results", highlight=0)
    narrate.say(f"[bold]{verdict}")
    try:
        df = metrics.vs_baselines(paths.results_entry("A"))
        h, r = metrics.table_rows(df)
        narrate.results(h, r, "Saved evaluation (mean of 20 seeded bridges) - agentA_vs_baselines.csv")
    except FileNotFoundError as e:
        narrate.warn(str(e))
    return {"steps_pct": ds, "energy_pct": de, "agent_steps": fa["steps"], "raster_steps": fr["steps"], "saved": saved}


if __name__ == "__main__":
    main()
