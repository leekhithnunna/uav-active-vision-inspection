"""d7 - Full mission: Agent C supervises; Explore runs Agent A (DQN), Refine runs Agent B (PPO); vs all-scripted."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo_core import cli, introspect, loaders, metrics, narrate, paths  # noqa: E402
from demo_core.mission import mission_figure, narrate_decision  # noqa: E402


def _supervisor(key: str, detector: str):
    """Learned supervisor (C / C_withA) or the scripted rule supervisor, with matching introspection."""
    if key == "rule":
        from env.baselines import RuleSupervisor
        return loaders.Agent("rule", "RULE supervisor (scripted)", RuleSupervisor(), None, None, True), None
    sup = loaders.load_agent(key, detector)
    return sup, ((lambda o: introspect.option_probs(sup.model, o)) if sup.model is not None else None)


def _moves(ep) -> int:
    """Low-level UAV moves (sub-steps inside the options); env 'steps' counts supervisor decisions."""
    return sum(f.is_substep for f in ep.frames)


def _row(name: str, ep) -> list:
    i = ep.info
    return [name, len(ep.decisions), _moves(ep), f"{ep.total_return:+.2f}", f"{i['defects_found']}/{i['defects_total']}",
            f"{i['coverage']:.0%}", f"{i['energy_used']:.3f}", f"{i['miou']:.3f}", "yes" if i["success"] else "no"]


def main(argv=None) -> dict:
    p = cli.parser(__doc__.splitlines()[0])
    p.add_argument("--supervisor", choices=["C", "C_withA", "rule"], default=paths.full_supervisor_key(),
                   help="C (default, matches full_mode_results.csv), C_withA, or rule (shows Agent B in the loop)")
    p.add_argument("--detail", type=int, default=2, help="fully narrated decisions")
    args = cli.parse(p, argv)
    from demo_core import animate, rollout
    from env.baselines import ApproachRefiner, RasterNav, RuleSupervisor

    A, B = loaders.load_agent("A", args.detector), loaders.load_agent("B", args.detector)
    sup, intro = _supervisor(args.supervisor, args.detector)
    narrate.header("DEMO 7 - Full hierarchical mission", f"supervisor: {sup.label} | explore: {A.label} | refine: {B.label}")
    narrate.guideline(1, f"InspectionEnv(mode='full', nav_policy=Agent A, refine_policy=Agent B), seed {args.seed}")
    env = loaders.load_env("full", args.detector, nav_policy=A, refine_policy=B)
    ep = rollout.run_episode(env, sup, args.seed, sup.label, intro)
    for f in ep.decisions:
        narrate_decision(f, detail=f.decision < args.detail)
    scripted = rollout.run_episode(loaders.load_env("full", args.detector, nav_policy=RasterNav(1),
                                                    refine_policy=ApproachRefiner()),
                                   RuleSupervisor(), args.seed, "ALL SCRIPTED")
    counts = {o: sum(f.action_label == o for f in ep.decisions) for o in introspect.C_SHORT}

    fig, update = mission_figure(ep, f"Demo 7 - Full mission: {sup.label} -> Agent A / Agent B", show_geometry=True)
    stride = 3 if args.fast else 2
    idx = list(range(0, len(ep.frames), stride)) + [len(ep.frames) - 1]
    n_end = 6
    ls, ss = ep.info, scripted.info
    end_text = (f"MISSION END (this bridge) - return {ep.total_return:+.1f} vs scripted {scripted.total_return:+.1f}\n"
                f"UAV moves {_moves(ep)} vs {_moves(scripted)} | energy {ls['energy_used']:.2f} vs {ss['energy_used']:.2f}")

    def frame(k: int) -> None:
        update(idx[min(k, len(idx) - 1)])
        if k >= len(idx) - 1:
            update.banner.set_text(end_text)
            update.banner.set_fontsize(15)

    narrate.guideline(7, "All three trained agents working together, animated")
    gif = "full_mission" if args.supervisor == paths.full_supervisor_key() else f"full_mission_{args.supervisor}"
    saved = animate.play(fig, frame, len(idx) + n_end, args, gif, fps=6, frames_png={"end": -1})

    narrate.guideline(8, "Final performance: learned hierarchy vs fully scripted system, same bridge")
    narrate.say(f"   supervisor option counts: {counts}")
    if counts["REFINE"] == 0:
        narrate.warn("The learned supervisor chose REFINE 0 times on this bridge, so Agent B was never called.\n"
                     "(It is the same on all 20 evaluation bridges: the trained A2C prefers EXPLORE -> RETURN.)\n"
                     "Agent B is shown in demo 4; `--supervisor rule` shows Agent B running inside the hierarchy.")
    header = ["system", "decisions", "UAV moves", "return", "defects", "coverage", "energy", "mIoU", "safe return"]
    narrate.results(header, [_row(f"{sup.label[:28]} + A + B", ep), _row("ALL SCRIPTED (rule+raster+approach)", scripted)],
                    f"Seed {args.seed}: live", highlight=0)
    try:
        h, r = metrics.table_rows(metrics.full_results(), name_col="config", steps_label="decisions")
        narrate.results(h, r, "Saved evaluation (mean of 20 seeded bridges) - full_mode_results.csv")
    except FileNotFoundError as e:
        narrate.warn(str(e))
    return {"counts": counts, "return": ep.total_return, "scripted_return": scripted.total_return,
            "moves": _moves(ep), "scripted_moves": _moves(scripted), "saved": saved}


if __name__ == "__main__":
    main()
