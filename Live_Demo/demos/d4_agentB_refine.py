"""d4 - Agent B (PPO) view refinement: geometry, Gaussian policy (mu, sigma), view quality s / q, reward; vs FixedOrbit."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo_core import cli, introspect, loaders, metrics, narrate, paths  # noqa: E402

OBS_NAMES = ["distance d", "angle", "lateral", "target u", "battery", "time left"]


def _narrate_step(f, detail: bool) -> None:
    if not detail:
        narrate.say(f"   step {f.t:2d}: a={f.action_label}  s={f.refine['s']:.2f} (q{f.refine['q']})  "
                    f"target u={f.refine['target_u']:.2f}  r={f.reward:+.3f}")
        return
    narrate.guideline(2, "Observation (6 numbers)")
    narrate.vector(OBS_NAMES, f.obs, "OBSERVATION  s_t")
    if f.intro is not None:
        narrate.guideline(3, "Continuous action space [-1, 1]^3: policy outputs a Gaussian per dimension")
        narrate.gaussian_table(f.intro, f.action)
    narrate.selected(f.action_label)
    narrate.guideline(4, f"Environment response: s = {f.refine['s']:.2f} -> view level q{f.refine['q']}, "
                         f"target uncertainty {f.refine['target_u']:.2f}")
    narrate.guideline(5, "Reward")
    narrate.reward(f.breakdown)


def _run(env, policy, seed, label, intro=None):
    from demo_core import rollout
    return rollout.run_episode(env, policy, seed, label, intro)


def main(argv=None) -> dict:
    p = cli.parser(__doc__.splitlines()[0])
    p.add_argument("--episodes", type=int, default=3, help="number of targets (seeds seed, seed+1, ...)")
    args = cli.parse(p, argv)
    import numpy as np
    from demo_core import animate, render, style
    from env.baselines import FixedOrbit

    agent = loaders.load_agent("B", args.detector)
    intro = (lambda o: introspect.gaussian(agent.model, o)) if agent.model is not None else None
    n_ep = 1 if args.fast else args.episodes
    narrate.header("DEMO 4 - Agent B view refinement", f"{agent.label} vs FixedOrbit baseline, same seeds")
    narrate.guideline(1, "Environment started: InspectionEnv(mode='refine') - hover near ONE uncertain cell")

    pairs, rows = [], []
    for e in range(n_ep):
        seed = args.seed + e
        ppo = _run(loaders.load_env("refine", args.detector), agent, seed, agent.label, intro)
        orb = _run(loaders.load_env("refine", args.detector), FixedOrbit(), seed, "BASELINE FixedOrbit")
        pairs.append((ppo, orb))
        tgt = ppo.frames[0].refine
        narrate.say(f"\n[bold]Target {e + 1} (seed {seed}): cell {divmod(tgt['target'], 16)}, start d={tgt['d']:.2f} "
                    f"angle={tgt['th']:+.2f} lateral={tgt['lat']:+.2f}  s={tgt['s']:.2f}")
        for f in ppo.frames[1:]:
            _narrate_step(f, detail=(e == 0))
        for ep in (ppo, orb):
            m = ep.frames[-1]
            rows.append([f"seed {seed}", ep.label[:30], m.t, "yes" if m.metrics["success"] else "no",
                         f"q{m.refine['best_q']}", f"{m.metrics['energy_used']:.3f}", f"{ep.total_return:+.2f}"])

    # ------------------------------------------------------------ figure
    fig = render.figure(18, 10.5, "Demo 4 - Agent B (PPO): move the camera to the best view of one uncertain cell")
    gs = fig.add_gridspec(2, 4, width_ratios=[3, 0.3, 3, 0.3], height_ratios=[1.35, 1], hspace=0.38, wspace=0.45)
    g_p, q_p = render.GeometryView(fig.add_subplot(gs[0, 0])), render.QualityGauge(fig.add_subplot(gs[0, 1]))
    g_o, q_o = render.GeometryView(fig.add_subplot(gs[0, 2])), render.QualityGauge(fig.add_subplot(gs[0, 3]))
    gauss = render.GaussianView(fig.add_subplot(gs[1, :2]), introspect.B_DIMS)
    ax_s = fig.add_subplot(gs[1, 2:])
    (l_p,) = ax_s.plot([], [], "o-", color=style.CRIMSON, lw=3, label="PPO (Agent B)")
    (l_o,) = ax_s.plot([], [], "s-", color=style.GREY, lw=2, label="FixedOrbit (baseline)")
    for q in range(1, 5):
        ax_s.axhline((q - 0.5) / 4, color=style.LIGHT, ls="--", lw=1)
    ax_s.set_xlim(-0.5, 30.5); ax_s.set_ylim(0, 1.02)
    ax_s.set_xlabel("step"); ax_s.set_ylabel("view quality s"); ax_s.legend(loc="upper right")
    ax_s.spines[["top", "right"]].set_visible(False)
    s_ttl = ax_s.set_title("")

    index = []
    for e, (ppo, orb) in enumerate(pairs):
        n = max(len(ppo.frames), len(orb.frames))
        stride = 3 if args.fast else 2
        index += [(e, k) for k in list(range(0, n, stride)) + [n - 1] * 4]

    def update(i: int) -> None:
        e, k = index[i]
        ppo, orb = pairs[e]
        kp, ko = min(k, len(ppo.frames) - 1), min(k, len(orb.frames) - 1)
        g_p.update([f.refine for f in ppo.frames[: kp + 1]], title=f"{agent.label.split(' - ')[0]} (PPO)  step {kp}"
                   + (" - SUCCESS q4" if kp == len(ppo.frames) - 1 and ppo.info["success"] else ""))
        g_o.update([f.refine for f in orb.frames[: ko + 1]], title=f"FixedOrbit (BASELINE)  step {ko}"
                   + (" - timeout" if ko == len(orb.frames) - 1 and not orb.info["success"] else ""))
        q_p.update(ppo.frames[kp].refine["s"]); q_o.update(orb.frames[ko].refine["s"])
        g = ppo.frames[max(1, min(kp + 1, len(ppo.frames) - 1))].intro
        if g is not None:
            gauss.update(g["mu"], g["sigma"])
        l_p.set_data(range(kp + 1), [f.refine["s"] for f in ppo.frames[: kp + 1]])
        l_o.set_data(range(ko + 1), [f.refine["s"] for f in orb.frames[: ko + 1]])
        s_ttl.set_text(f"target {e + 1}/{len(pairs)}: return PPO {ppo.frames[kp].cum_reward:+.2f} | "
                       f"orbit {orb.frames[ko].cum_reward:+.2f}")

    narrate.guideline(7, "Trained policy on each target, animated next to the baseline")
    saved = animate.play(fig, update, len(index), args, "agentB_refine", fps=5, frames_png={"end": -1})
    narrate.guideline(8, "Final performance (live, same seeds)")
    narrate.results(["seed", "policy", "steps", "success", "best q", "energy", "return"], rows,
                    "Agent B vs FixedOrbit - live")
    try:
        h, r = metrics.table_rows(metrics.vs_baselines(paths.results_entry("B")), cols=metrics.REFINE_COLS)
        narrate.results(h, r, "Saved evaluation (mean of 50 seeded targets) - agentB_vs_baselines.csv")
    except FileNotFoundError as e:
        narrate.warn(str(e))
    ppo_ret = float(np.mean([p.total_return for p, _ in pairs]))
    orb_ret = float(np.mean([o.total_return for _, o in pairs]))
    return {"ppo_return": ppo_ret, "orbit_return": orb_ret, "saved": saved,
            "breakdowns_match": all(f.breakdown.matches for p, o in pairs for f in p.frames[1:] + o.frames[1:])}


if __name__ == "__main__":
    main()
