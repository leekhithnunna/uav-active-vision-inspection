"""d1 - Environment tour: start the env, grid, altitudes/footprints, detector modes, a random rollout."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo_core import cli, loaders, narrate, paths  # noqa: E402


def _footprint_frames(detector: str, seed: int) -> list:
    """Same bridge seen from HIGH / MID / LOW at the same spot (one camera view each)."""
    from demo_core import rollout
    from env.inspection_env import H, W
    frames = []
    for alt in range(3):
        env = loaders.load_env("navigate", detector)
        env.reset(seed=seed)
        env.best_q[:] = -1                       # show only this one camera view
        env.pos, env.alt = [4, 7], alt
        env._observe()
        frames.append(rollout.snapshot(env))
    return frames, env.defect.reshape(H, W)


def _detector_table(detector_ok: bool) -> None:
    from env import inspection_env as E
    rows = [[q, f"{E.U_DEFECT[q]:.2f}", f"{E.U_CLEAN[q]:.2f}", f"{E.P_DETECT[q]:.2f}"] for q in range(5)]
    header = ["q", "analytic u (defect)", "analytic u (clean)", "analytic P(detect)"]
    if detector_ok:
        from demo_core import detector
        rep = {r["q"]: r for r in detector.report()["report"]}
        for r in rows:
            q = r[0]
            r += [f"{rep[q]['u_defect']:.2f}", f"{rep[q]['u_clean']:.2f}", f"{rep[q]['detect_rate']:.2f}"]
        header += ["learned u (defect)", "learned u (clean)", "learned detect rate"]
    narrate.results(header, rows, "Detector modes: uncertainty u per view quality q (lower = better view)")


def main(argv=None) -> dict:
    p = cli.parser(__doc__.splitlines()[0])
    p.add_argument("--steps", type=int, default=60, help="random-rollout length")
    args = cli.parse(p, argv)
    from demo_core import animate, render, rollout, style
    from env.baselines import RandomPolicy
    from env.inspection_env import H, MAX_STEPS, N_CELLS, W

    narrate.header("DEMO 1 - Environment tour", "InspectionEnv: UAV inspecting a bridge surface")
    narrate.guideline(1, "Start the environment")
    for mode in ("navigate", "refine", "supervise"):
        env = loaders.load_env(mode, args.detector)
        narrate.say(f"  mode=[bold]{mode:9s}[/]  obs {env.observation_space}  action {env.action_space}  "
                    f"max steps {MAX_STEPS[mode]}")
    narrate.say(f"  bridge grid {H} x {W} = {N_CELLS} cells, ~20% defective; altitude HIGH/MID/LOW -> "
                f"footprint 5x5 / 3x3 / 1x1")
    cache_ok = paths.detector_path("cache").exists()
    _detector_table(cache_ok)

    fp_frames, defect = _footprint_frames(args.detector, args.seed)
    env = loaders.load_env("navigate", args.detector)
    n = 20 if args.fast else args.steps
    policy = RandomPolicy(env.action_space, seed=args.seed)
    st = rollout.Stepper(env, policy, args.seed)
    ep = st.run("random", max_decisions=n)
    narrate.guideline(2, "Observation after reset (ASCII render: U = UAV, . = unseen, digit = uncertainty x 10)")
    narrate.say(env.render())

    fig = render.figure(17, 11.5, "Demo 1 - Environment tour (uncertainty heatmap: green = certain, red = uncertain)")
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 2.2], hspace=0.3)
    for a in range(3):
        v = render.BridgeView(fig.add_subplot(gs[0, a]), defect, colorbar=False, hud=False,
                              title=f"{rollout.ALT_NAMES[a]} altitude: {5 - 2 * a}x{5 - 2 * a} footprint, "
                                    f"q={env.alt_q[a]}")
        v.update(fp_frames, a)
    big = render.BridgeView(fig.add_subplot(gs[1, :]), ep.defect)
    render.altitude_legend(big.ax)

    def update(i: int) -> None:
        f = ep.frames[i]
        big.update(ep.frames, i, title=f"Random policy (NOT an agent) - step {f.t}: {f.action_label or 'reset'}")

    saved = animate.play(fig, update, len(ep.frames), args, "env_tour", fps=6, frames_png={"final": -1})
    narrate.guideline(4, f"Random rollout: {len(ep.frames) - 1} steps, return {ep.total_return:+.2f}, "
                         f"coverage {ep.info['coverage']:.0%}")
    return {"steps": len(ep.frames) - 1, "return": ep.total_return, "saved": saved}


if __name__ == "__main__":
    main()
