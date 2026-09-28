"""d9 - Results dashboard: learning curves + results-vs-baselines tables, read from the saved files (no fake numbers)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo_core import cli, loaders, metrics, narrate, paths  # noqa: E402

XLABEL = {"A": "environment steps", "B": "environment steps", "C": "supervisor decisions"}


def _sensitivity(key: str) -> list[list]:
    """First (agent) row of every variant run's *_vs_baselines.csv, labelled with its training length."""
    main = paths.results_entry(key)
    runs = [("main", main)] + [(Path(d).name, d) for d in paths.results_entry(key, "sensitivity")]
    for extra in ("learned", "withA"):
        if extra in paths.config()["results"][key]:
            runs.append((Path(paths.results_entry(key, extra)).name, paths.results_entry(key, extra)))
    rows = []
    for name, d in runs:
        try:
            r = metrics.vs_baselines(d).iloc[0]
        except FileNotFoundError:
            rows.append([name, "missing", "", "", "", "", ""])
            continue
        hp = metrics.hyperparams(d)
        tag = hp.get("detector", "analytic") + (f", eps_fraction={hp['eps_fraction']}" if key == "A" and "eps_fraction" in hp else "")
        tag += f", ent={hp['ent_coef']}" if "ent_coef" in hp else ""
        tag += f", clip={hp['clip']}" if key == "B" and "clip" in hp else ""
        rows.append([name, f"{hp.get('steps', 0) // 1000}k", tag, f"{r['return_mean']:+.2f}", f"{r['success_mean']:.0%}",
                     f"{r['steps_mean']:.1f}", f"{r['energy_used_mean']:.3f}", metrics.run_note(d, main) or "full run"])
    return rows


def agent_page(key: str, args) -> dict:
    from demo_core import animate, render
    run = paths.results_entry(key)
    info = loaders.AGENT_INFO[key]
    fig = render.figure(19, 10.5, f"Demo 9 - {info['name']} ({info['owner']}): training + final evaluation")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1], hspace=0.45, wspace=0.25)
    axs = [fig.add_subplot(gs[0, i]) for i in range(3)]
    metrics.plot_curves(axs, run, info["algo"], metrics.run_note(run))
    for a in axs:
        a.set_xlabel(XLABEL[key])
    df = metrics.vs_baselines(run)
    cols = metrics.REFINE_COLS if key == "B" else metrics.TABLE_COLS
    h, rows = metrics.table_rows(df, cols=cols, steps_label="decisions" if key == "C" else "steps")
    n_eval = 50 if key == "B" else 20
    render.table(fig.add_subplot(gs[1, :]), h, rows, highlight_row=0, size=15,
                 title=f"{Path(run).name}/{next(Path(run).glob('*_vs_baselines.csv')).name}  "
                       f"(mean over {n_eval} seeded {'targets' if key == 'B' else 'bridges'})")
    narrate.results(h, rows, f"{info['name']} vs baselines - {Path(run).name}", highlight=0)
    sens = _sensitivity(key)
    narrate.results(["run", "trained", "setting", "return", "success", "steps", "energy", "note"], sens,
                    f"{info['name']}: all saved runs (agent row of each *_vs_baselines.csv)")
    out = paths.out_dir("snapshots")
    df.to_csv(out / f"results_{key}.csv", index=False)
    import pandas as pd
    pd.DataFrame(sens, columns=["run", "trained", "setting", "return", "success", "steps", "energy", "note"]).to_csv(
        out / f"runs_{key}.csv", index=False)
    animate.save_png(fig, f"results_{key}", "snapshots")
    animate.finish(fig, args)
    return {"rows": len(rows), "runs": len(sens)}


def full_page(args) -> dict:
    import numpy as np
    from demo_core import animate, render, style
    df = metrics.full_results()
    fig = render.figure(19, 10.5, "Demo 9 - Full hierarchy (A + B + C): learned vs scripted + ablations")
    gs = fig.add_gridspec(2, 4, height_ratios=[1.2, 1], hspace=0.5, wspace=0.3)
    short = ["ALL\nlearned", "ALL\nscripted", "C lrn.\nA,B scr.", "rule C\nA,B lrn."]
    colors = [style.CRIMSON, style.GREY, "#c0587e", "#8c8c8c"]
    for i, (k, lbl) in enumerate([("return", "return"), ("steps", "decisions"), ("energy_used", "energy used"),
                                  ("defects_found", "defects found (of 26)")]):
        ax = fig.add_subplot(gs[0, i])
        ax.bar(range(len(df)), df[f"{k}_mean"], yerr=df[f"{k}_std"], color=colors, capsize=6, edgecolor="black")
        ax.set_xticks(range(len(df)), short[: len(df)], fontsize=11)
        ax.set_title(lbl)
        ax.spines[["top", "right"]].set_visible(False)
        for j, v in enumerate(df[f"{k}_mean"]):
            ax.text(j, v / 2, f"{v:.2f}" if v < 5 else f"{v:.1f}", ha="center", va="center", fontsize=13,
                    fontweight="bold", color="white")
    h, rows = metrics.table_rows(df, name_col="config", steps_label="decisions")
    render.table(fig.add_subplot(gs[1, :]), h, rows, highlight_row=0, size=14,
                 title="full_mode_results.csv (mean over 20 seeded bridges)")
    narrate.results(h, rows, "Full hierarchy - full_mode_results.csv", highlight=0)
    a, s = df.iloc[0], df.iloc[1]
    pct = lambda x, y: 100 * (x - y) / y
    narrate.say(f"[bold]ALL LEARNED vs ALL SCRIPTED (saved means): decisions {pct(a['steps_mean'], s['steps_mean']):+.0f}%, "
                f"energy {pct(a['energy_used_mean'], s['energy_used_mean']):+.0f}%, return "
                f"{a['return_mean'] - s['return_mean']:+.2f}, defects {a['defects_found_mean'] - s['defects_found_mean']:+.2f}")
    df.to_csv(paths.out_dir("snapshots") / "results_full.csv", index=False)
    animate.save_png(fig, "results_full", "snapshots")
    animate.finish(fig, args)
    return {"rows": len(rows)}


def main(argv=None) -> dict:
    p = cli.parser(__doc__.splitlines()[0])
    p.add_argument("--agent", choices=["all", "A", "B", "C", "full"], default="all")
    args = cli.parse(p, argv)
    narrate.header("DEMO 9 - Results dashboard", "every number below is read from the saved result files")
    narrate.guideline(8, "Learning curves and final performance metrics")
    todo = ["A", "B", "C", "full"] if args.agent == "all" else [args.agent]
    out = {}
    for k in todo:
        try:
            out[k] = full_page(args) if k == "full" else agent_page(k, args)
        except FileNotFoundError as e:
            narrate.warn(f"{k}: {e}")
            out[k] = None
    narrate.info(f"PNG + CSV exported to {paths.out_dir('snapshots')}")
    return out


if __name__ == "__main__":
    main()
