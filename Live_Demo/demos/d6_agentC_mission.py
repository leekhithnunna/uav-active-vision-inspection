"""d6 - Agent C (A2C) mission supervisor: state vector, pi(Explore/Refine/Return), chosen option, reward, timeline."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo_core import cli, introspect, loaders, metrics, narrate, paths  # noqa: E402
from demo_core.mission import mission_figure, narrate_decision  # noqa: E402

def main(argv=None) -> dict:
    p = cli.parser(__doc__.splitlines()[0])
    p.add_argument("--detail", type=int, default=3, help="fully narrated decisions (rest are one line)")
    args = cli.parse(p, argv)
    from demo_core import animate, rollout
    from env.baselines import RuleSupervisor

    agent = loaders.load_agent("C", args.detector)
    intro = (lambda o: introspect.option_probs(agent.model, o)) if agent.model is not None else None
    narrate.header("DEMO 6 - Agent C mission supervisor", f"{agent.label}   seed {args.seed}")
    narrate.guideline(1, "Environment started: InspectionEnv(mode='supervise'); inside options: scripted "
                         "RasterNav (explore) + ApproachRefiner (refine) - exactly as Agent C was trained")
    ep = rollout.run_episode(loaders.load_env("supervise", args.detector), agent, args.seed, agent.label, intro)
    for f in ep.decisions:
        narrate_decision(f, detail=f.decision < args.detail)
    rule = rollout.run_episode(loaders.load_env("supervise", args.detector), RuleSupervisor(), args.seed,
                               "BASELINE RuleSupervisor", record_substeps=False)

    fig, update = mission_figure(ep, f"Demo 6 - {agent.label}: one semi-MDP decision = one option")
    stride = 3 if args.fast else 2
    idx = list(range(0, len(ep.frames), stride)) + [len(ep.frames) - 1] * 4
    narrate.guideline(7, "Trained supervisor, full mission animated")
    saved = animate.play(fig, lambda k: update(idx[k]), len(idx), args, "agentC_mission", fps=6, frames_png={"end": -1})

    narrate.guideline(8, "Final performance on this bridge (live) + saved 20-bridge evaluation")
    counts = {o: sum(f.action_label == o for f in ep.decisions) for o in introspect.C_SHORT}
    narrate.say(f"   option counts: {counts}")
    rows = [[e.label[:36], len(e.decisions), f"{e.total_return:+.2f}", e.info["defects_found"],
             f"{e.info['energy_used']:.3f}", f"{e.info['coverage']:.0%}", "yes" if e.info["success"] else "no"]
            for e in (ep, rule)]
    narrate.results(["supervisor", "decisions", "return", "defects", "energy", "coverage", "safe return"], rows,
                    f"Seed {args.seed}: live", highlight=0)
    try:
        h, r = metrics.table_rows(metrics.vs_baselines(paths.results_entry("C")), steps_label="decisions")
        narrate.results(h, r, "Saved evaluation (mean of 20 seeded bridges) - agentC_vs_baselines.csv")
    except FileNotFoundError as e:
        narrate.warn(str(e))
    return {"decisions": len(ep.decisions), "return": ep.total_return, "rule_return": rule.total_return,
            "counts": counts, "saved": saved}


if __name__ == "__main__":
    main()
