"""Optional local web app (no internet needed):  streamlit run app/streamlit_app.py

Sidebar: seed + detector. Tabs per agent, full mission and results. Everything comes from demo_core.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from demo_core import introspect, loaders, metrics, paths, render, rollout, style  # noqa: E402

st.set_page_config(page_title="Team 08 - UAV active vision", layout="wide")
style.apply()


@st.cache_resource
def agent(key: str, detector: str):
    return loaders.load_agent(key, detector)


def episode(tab: str, seed: int, detector: str):
    """Run (and cache in the session) the episode shown in a tab."""
    k = f"{tab}-{seed}-{detector}"
    if k not in st.session_state:
        from env.baselines import ApproachRefiner, FixedOrbit, RasterNav, RuleSupervisor
        A, B, C = agent("A", detector), agent("B", detector), agent("C", detector)
        if tab == "A":
            st.session_state[k] = [rollout.run_episode(loaders.load_env("navigate", detector), A, seed, A.label,
                                                       lambda o: introspect.q_values(A.model, o)),
                                   rollout.run_episode(loaders.load_env("navigate", detector), RasterNav(2), seed,
                                                       "BASELINE raster LOW")]
        elif tab == "B":
            st.session_state[k] = [rollout.run_episode(loaders.load_env("refine", detector), B, seed, B.label,
                                                       lambda o: introspect.gaussian(B.model, o)),
                                   rollout.run_episode(loaders.load_env("refine", detector), FixedOrbit(), seed,
                                                       "BASELINE FixedOrbit")]
        elif tab == "C":
            st.session_state[k] = [rollout.run_episode(loaders.load_env("supervise", detector), C, seed, C.label,
                                                       lambda o: introspect.option_probs(C.model, o)),
                                   rollout.run_episode(loaders.load_env("supervise", detector), RuleSupervisor(), seed,
                                                       "BASELINE RuleSupervisor")]
        else:
            env = loaders.load_env("full", detector, nav_policy=A, refine_policy=B)
            scripted = loaders.load_env("full", detector, nav_policy=RasterNav(1), refine_policy=ApproachRefiner())
            st.session_state[k] = [rollout.run_episode(env, C, seed, "C + A + B (learned)",
                                                       lambda o: introspect.option_probs(C.model, o)),
                                   rollout.run_episode(scripted, RuleSupervisor(), seed, "ALL SCRIPTED")]
    return st.session_state[k]


def frame_slider(tab: str, n: int) -> int:
    """Slider plus step buttons sharing one index."""
    key = f"idx-{tab}"
    st.session_state.setdefault(key, 0)
    c1, c2, c3 = st.columns([1, 1, 6])
    if c1.button("< step", key=f"prev-{tab}"):
        st.session_state[key] = max(0, st.session_state[key] - 1)
    if c2.button("step >", key=f"next-{tab}"):
        st.session_state[key] = min(n - 1, st.session_state[key] + 1)
    st.session_state[key] = min(st.session_state[key], n - 1)
    return c3.slider("frame", 0, n - 1, key=key)


def bridge_fig(eps, i: int):
    fig = render.figure(16, 5.5)
    for j, ep in enumerate(eps):
        v = render.BridgeView(fig.add_subplot(1, len(eps), j + 1), ep.defect, colorbar=j == len(eps) - 1)
        v.update(ep.frames, min(i, len(ep.frames) - 1), title=ep.label)
    return fig


def show_breakdown(f) -> None:
    if f.breakdown is not None:
        st.table(pd.DataFrame({"term": list(f.breakdown.parts), "value": [round(v, 4) for v in f.breakdown.parts.values()]}))
        st.caption(f"env reward {f.reward:+.4f} - terms add up: {f.breakdown.matches}")


def tab_a(seed, det):
    ep, raster = episode("A", seed, det)
    i = frame_slider("A", len(ep.frames))
    compare = st.checkbox("compare with dense raster (same seed)", key="cmpA")
    st.pyplot(bridge_fig([ep, raster] if compare else [ep], i), clear_figure=True)
    f = ep.frames[min(i + 1, len(ep.frames) - 1)]
    c1, c2 = st.columns(2)
    if f.intro is not None:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        render.BarView(ax, ["N", "S", "E", "W", "Climb", "Desc."], f"Q(s,a) at step {f.t - 1} -> {f.action_label}").update(
            f.intro["values"], f.intro["chosen"])
        c1.pyplot(fig, clear_figure=True)
    with c2:
        show_breakdown(f)


def tab_b(seed, det):
    ppo, orb = episode("B", seed, det)
    n = max(len(ppo.frames), len(orb.frames))
    i = frame_slider("B", n)
    fig = render.figure(16, 5)
    for j, ep in enumerate((ppo, orb)):
        k = min(i, len(ep.frames) - 1)
        render.GeometryView(fig.add_subplot(1, 3, j + 1), ep.label[:28]).update([f.refine for f in ep.frames[: k + 1]])
    g = ppo.frames[max(1, min(i + 1, len(ppo.frames) - 1))].intro
    if g is not None:
        render.GaussianView(fig.add_subplot(1, 3, 3), introspect.B_DIMS).update(g["mu"], g["sigma"])
    st.pyplot(fig, clear_figure=True)
    show_breakdown(ppo.frames[min(max(i, 1), len(ppo.frames) - 1)])


def tab_supervisor(tab, seed, det):
    ep, base = episode(tab, seed, det)
    i = frame_slider(tab, len(ep.frames))
    compare = st.checkbox("show scripted baseline next to it (final state)", key=f"cmp{tab}")
    st.pyplot(bridge_fig([ep] + ([base] if compare else []), i if not compare else 10**6), clear_figure=True)
    f = ep.frames[i]
    d = next((x for x in ep.decisions if x.decision == f.decision), None)
    if d is not None and d.intro is not None:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        render.BarView(ax, introspect.C_SHORT, f"pi(option | state), decision {d.decision} -> {d.action_label}",
                       ylim=(0, 1.1), fmt="{:.2f}", colors=style.OPTION_COLORS).update(d.intro["values"], int(d.action))
        st.pyplot(fig, clear_figure=True)
        show_breakdown(d)
    rows = [[e.label, len(e.decisions), round(e.total_return, 2), e.info["defects_found"], round(e.info["energy_used"], 3)]
            for e in (ep, base)]
    st.table(pd.DataFrame(rows, columns=["policy", "decisions", "return", "defects", "energy"]))


def tab_results():
    which = st.radio("agent", ["A", "B", "C", "full"], horizontal=True)
    if which == "full":
        h, r = metrics.table_rows(metrics.full_results(), name_col="config", steps_label="decisions")
        st.table(pd.DataFrame(r, columns=h))
        return
    run = paths.results_entry(which)
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    metrics.plot_curves(axs, run, loaders.AGENT_INFO[which]["algo"], metrics.run_note(run))
    st.pyplot(fig, clear_figure=True)
    h, r = metrics.table_rows(metrics.vs_baselines(run))
    st.table(pd.DataFrame(r, columns=h))


def main() -> None:
    st.sidebar.title("Team 08 - UAV active vision")
    seed = int(st.sidebar.number_input("seed (same seed = same bridge)", value=paths.default_seed(), step=1))
    det = st.sidebar.radio("detector", ["analytic", "learned"])
    st.sidebar.caption("All numbers are computed live or read from the saved result files.")
    tabs = st.tabs(["Agent A (DQN)", "Agent B (PPO)", "Agent C (A2C)", "Full mission", "Results"])
    with tabs[0]:
        tab_a(seed, det)
    with tabs[1]:
        tab_b(seed, det)
    with tabs[2]:
        tab_supervisor("C", seed, det)
    with tabs[3]:
        tab_supervisor("full", seed, det)
    with tabs[4]:
        tab_results()


main()
