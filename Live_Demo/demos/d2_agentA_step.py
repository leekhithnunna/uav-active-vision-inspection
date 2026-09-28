"""d2 - Agent A (DQN) step-through: map -> Q-values -> argmax action -> reward breakdown -> next map."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo_core import cli, introspect, loaders, metrics, narrate, paths  # noqa: E402


class _Figure:
    """Live window: bridge map (left), Q-value bars (top right), reward terms (bottom right)."""

    def __init__(self, defect, label: str):
        from demo_core import render
        self.fig = render.figure(18, 8, f"Demo 2 - {label}: one decision at a time")
        gs = self.fig.add_gridspec(2, 3, width_ratios=[1.6, 1.6, 1.25], height_ratios=[1.25, 1], hspace=0.45)
        self.map = render.BridgeView(self.fig.add_subplot(gs[:, :2]), defect)
        render.altitude_legend(self.map.ax)
        self.q = render.BarView(self.fig.add_subplot(gs[0, 2]), ["N", "S", "E", "W", "Climb", "Desc."],
                                "Q(s, a) - chosen = argmax")
        self.txt = render.text_panel(self.fig.add_subplot(gs[1, 2]), "", size=13)

    def update(self, frames, i: int) -> None:
        f = frames[i]
        self.map.update(frames, i, title=f"after step {f.t}: {f.action_label or 'reset'}")
        if f.intro is not None:
            self.q.update(f.intro["values"], f.intro["chosen"])
        if f.breakdown is not None:
            lines = [f"action: {f.action_label}", ""] + [f"{v:+.3f}  {k[:34]}" for k, v in f.breakdown.parts.items()]
            lines += ["-" * 30, f"{f.reward:+.3f}  reward (env)", f"{f.cum_reward:+.3f}  return so far"]
            self.txt.set_text("\n".join(lines))


def _ask(auto: bool, k: int, detail_steps: int) -> str:
    if auto:
        return "" if k < detail_steps else "r"
    hint = "[Enter] next step   [r] run to end   [q] quit" if k >= detail_steps else "[Enter] next step   [r] run to end"
    narrate.say(f"[bold]{hint}[/]")
    try:
        return input("> ").strip().lower()
    except EOFError:
        return "r"


def _detail(f) -> None:
    """Narrate one decision covering guideline steps 3-5."""
    if f.intro is not None:
        narrate.guideline(3, "Available actions and the selected action (DQN picks argmax Q)")
        narrate.action_table(f.intro["names"], f.intro["values"], f.intro["chosen"], "Q(s, a)")
    narrate.selected(f.action_label)
    narrate.guideline(4, "Environment response")
    bd = f.breakdown
    oob = " (blocked: out of bounds!)" if f.metrics["out_of_bounds"] else ""
    narrate.say(f"   UAV now at {list(f.pos)}, altitude {['HIGH', 'MID', 'LOW'][f.alt]}{oob}; "
                f"{bd.extra['new_cells']} new cells seen, total uncertainty reduced by {bd.extra['du']:.3f}")
    narrate.guideline(5, "Reward and next state")
    narrate.reward(bd)
    narrate.state(f)


def main(argv=None) -> dict:
    p = cli.parser(__doc__.splitlines()[0])
    p.add_argument("--auto", action="store_true", help="do not wait for Enter")
    p.add_argument("--detail-steps", type=int, default=3, help="fully narrated steps before offering run-to-end")
    args = cli.parse(p, argv)
    import matplotlib.pyplot as plt
    from demo_core import rollout

    agent = loaders.load_agent("A", args.detector)
    env = loaders.load_env("navigate", args.detector)
    intro = (lambda o: introspect.q_values(agent.model, o)) if agent.model is not None else None
    st = rollout.Stepper(env, agent, args.seed, intro)
    auto = args.auto or args.headless

    narrate.header("DEMO 2 - Agent A step-through", f"{agent.label}   seed {args.seed}   detector {args.detector}")
    narrate.guideline(1, f"Environment started: InspectionEnv(mode='navigate'), bridge seed {args.seed}")
    narrate.say(f"   model: {agent.path}")
    narrate.guideline(2, "Current observation / state")
    narrate.say(f"   observation = {st.obs.shape[0]} numbers: 7x7 local window (visited, uncertainty) "
                f"+ 2x4 block summary + (row, col, altitude, battery)")
    narrate.state(st.first)
    narrate.say(env.render())

    frames = [st.first]
    defect = env.defect.reshape(st.first.u.shape)
    view = None
    if not args.headless:
        plt.ion()
        view = _Figure(defect, agent.label)
        view.update(frames, 0)
        plt.show(block=False)
        plt.pause(0.1)

    k, mode = 0, "detail"
    while not st.done:
        if mode == "detail":
            new = st.step()
            frames += new
            _detail(frames[-1])
            k += 1
            if args.save and k <= args.detail_steps:
                _save_step(frames, defect, k, agent.label)
        else:
            new = st.step()
            frames += new
            narrate.compact(frames[-1])
        if view is not None:
            view.update(frames, len(frames) - 1)
            plt.pause(0.05 if mode == "run" else 0.2)
        if mode == "detail" and not st.done:
            ans = _ask(auto, k, args.detail_steps)
            if ans == "q":
                break
            if ans == "r":
                narrate.guideline(7, "Trained policy runs to the end of the episode")
                mode = "run"

    f = frames[-1]
    narrate.guideline(8, "Final performance of this episode (live) vs saved 20-episode evaluation")
    live = [["this episode (live)", f"{st.cum:+.2f}", str(f.metrics["success"]), f.t, f"{f.metrics['coverage']:.0%}",
             f"{f.metrics['defects_found']}/{f.metrics['defects_total']}", f"{f.metrics['energy_used']:.3f}"]]
    try:
        df = metrics.vs_baselines(paths.results_entry("A"))
        for _, r in df.iterrows():
            live.append([f"saved: {r['policy']}", f"{r['return_mean']:+.2f}", f"{r['success_mean']:.0%}",
                         f"{r['steps_mean']:.1f}", f"{r['coverage_mean']:.0%}", f"{r['defects_found_mean']:.1f}",
                         f"{r['energy_used_mean']:.3f}"])
    except FileNotFoundError as e:
        narrate.warn(str(e))
    narrate.results(["run", "return", "success", "steps", "coverage", "defects", "energy"], live,
                    "Agent A - results (saved rows: mean of 20 seeded bridges, agentA_vs_baselines.csv)", highlight=0)
    narrate.say("[grey50]Learning / update step (guideline 6): run demo 8 (live training burst).")
    if view is not None:
        plt.ioff()
        plt.show()
    if args.save:
        _save_step(frames, defect, None, agent.label)
    return {"steps": f.t, "return": st.cum, "success": f.metrics["success"],
            "breakdowns_match": all(fr.breakdown.matches for fr in frames[1:])}


def _save_step(frames, defect, k, label) -> None:
    """Key PNG frame for slides: after step k (or the final state)."""
    import matplotlib.pyplot as plt
    from demo_core import animate
    v = _Figure(defect, label)
    v.update(frames, len(frames) - 1)
    animate.save_png(v.fig, f"agentA_step{k}" if k else "agentA_final")
    plt.close(v.fig)


if __name__ == "__main__":
    main()
