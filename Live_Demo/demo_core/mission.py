"""Shared view + narration for supervisor missions (d6 Agent C alone, d7 full hierarchy)."""
from __future__ import annotations

from . import introspect, narrate

SUP_OBS = ["battery", "coverage", "mean u", "found frac", "dist to base", "time used"]


def narrate_decision(f, detail: bool) -> None:
    """Terminal story of one supervisor decision (shared by d6 and d7)."""
    if not detail:
        narrate.say(f"   decision {f.decision:2d}: [bold]{f.action_label:8s}[/] r={f.reward:+6.2f}  "
                    f"battery {f.battery:.2f}  coverage {f.metrics['coverage']:.0%}  "
                    f"defects {f.metrics['defects_found']}/{f.metrics['defects_total']}")
        return
    narrate.guideline(2, f"Supervisor state before decision {f.decision}")
    narrate.vector(SUP_OBS, f.obs, "OBSERVATION  s_t (6 numbers in [0, 1])")
    if f.intro is not None:
        narrate.guideline(3, "Options and the selected option (A2C actor: pi(o | s))")
        narrate.action_table(f.intro["names"], f.intro["values"], int(f.action), "pi(o|s)", "AVAILABLE OPTIONS")
    narrate.selected(f.action_label)
    narrate.guideline(4, "Option executed (up to 5 navigator / 10 refiner sub-steps)")
    narrate.guideline(5, "Reward and next state")
    narrate.reward(f.breakdown)
    narrate.say(f"   battery {f.battery:.2f} | coverage {f.metrics['coverage']:.0%} | defects "
                f"{f.metrics['defects_found']}/{f.metrics['defects_total']}")


def mission_figure(ep, title: str, show_geometry: bool = False):
    """Bridge map + option probabilities + state bars (or refine geometry) + decision timeline. Returns (fig, update)."""
    from demo_core import render, style
    fig = render.figure(18, 11, title)
    gs = fig.add_gridspec(3, 3, width_ratios=[1.6, 1.6, 1.15], height_ratios=[1.2, 1.2, 0.5], hspace=0.55, wspace=0.38)
    view = render.BridgeView(fig.add_subplot(gs[:2, :2]), ep.defect)
    render.altitude_legend(view.ax)
    probs = render.BarView(fig.add_subplot(gs[0, 2]), introspect.C_SHORT, "pi(option | state)", ylim=(0, 1.15),
                           fmt="{:.2f}", colors=style.OPTION_COLORS)
    if show_geometry:
        geo = render.GeometryView(fig.add_subplot(gs[1, 2]), "Agent B view geometry")
        geo.txt.set_text("Agent B not called yet\n(no REFINE decision)")
        state = None
    else:
        geo = None
        state = render.BarView(fig.add_subplot(gs[1, 2]), ["bat", "cov", "u", "found", "dist", "time"],
                               "supervisor state s_t", ylim=(0, 1.15), fmt="{:.2f}")
    decisions = ep.decisions
    if all(d.intro is None for d in decisions):
        probs.ax.set_title("scripted rule: explore, explore, refine...\n(no learned probabilities)",
                           fontsize=style.FONT - 2)
    tl = render.Timeline(fig.add_subplot(gs[2, :]), decisions)
    banner = fig.text(0.39, 0.905, "", ha="center", fontsize=style.FONT + 5, fontweight="bold", color="white",
                      bbox=dict(boxstyle="round,pad=0.4", fc=style.GREY, ec="none"))
    by_idx = {f.decision: f for f in decisions}

    def update(i: int) -> None:
        f = ep.frames[i]
        d = by_idx.get(f.decision)
        view.update(ep.frames, i)
        if d is not None and d.intro is not None:
            probs.update(d.intro["values"], int(d.action))
        if state is not None and d is not None:
            state.update(d.obs, None)
        tl.update(f.decision if f.is_substep else f.decision + 1)
        a = int(d.action) if d is not None else 0
        text = {0: "EXPLORE  ->  Agent A (DQN) flies", 1: "REFINE  ->  Agent B (PPO) moves camera",
                2: "RETURN HOME  ->  mission ends"}[a] if d is not None else "START"
        banner.set_text(f"decision {max(f.decision, 0)}:  {text}")
        banner.get_bbox_patch().set_facecolor(style.OPTION_COLORS[a] if d is not None else style.GREY)
        if geo is not None and f.refine is not None and f.phase.startswith("REFINE"):
            refs = [x.refine for x in ep.frames[: i + 1]                      # this refine only
                    if x.decision == f.decision and x.phase.startswith("REFINE") and x.refine is not None]
            geo.update(refs, title=f"Agent B: s={f.refine['s']:.2f} (q{f.refine['q']})")

    update.banner = banner
    return fig, update
