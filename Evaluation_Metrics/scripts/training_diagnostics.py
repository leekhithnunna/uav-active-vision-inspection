"""Phase 4 - training diagnostics from the SAVED logs (no retraining).

Reads monitor CSVs, eval/evaluations.npz, monitor_eval CSVs and TensorBoard events of every run and
writes plot-ready tables (results/):
  training_curves.csv     per training episode: run, timestep, return, length, success, 50-ep moving average/std
  eval_curves.csv         per EvalCallback call: timestep, mean/std eval return, eval success rate, eval length
  tb_scalars.csv          every TensorBoard scalar (long format: run, tag, step, value, wall_time)
  sample_efficiency.csv   env steps to first reach 80% / 100% eval success, train minutes, final values
  sensitivity.csv         CO5 sensitivity runs: saved-CSV result + held-out (seeds 2000-2099) result
  model_size.csv          parameter count, file size, mean inference latency per agent
These are TRAINING diagnostics (losses, entropy, KL ...), not task performance.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (ROOT, RESULTS, RUN_DIRS, MULTISEED, make_env, run_policy, SB3Policy, load_model,  # noqa: E402
                    ci95)

MA = 50
N_EVAL = {"A": 10, "B": 20, "C": 10}
MAIN_STEPS = {"A": 200_000, "B": 100_000, "C": 100_000}


def all_runs():
    """Original runs + multi-seed runs if present: {run name: (agent letter, folder)}."""
    runs = {k: (k[0], d) for k, d in RUN_DIRS.items()}
    for d in sorted(MULTISEED.glob("*_seed*")):
        if (d / "eval/evaluations.npz").exists():
            runs[f"ms_{d.name}"] = (d.name[0], d)
    return runs


def read_monitor(path):
    df = pd.read_csv(path, skiprows=1)
    t0 = json.loads(open(path).readline()[1:])["t_start"]
    df["abs_t"] = df["t"] + t0
    return df


def training_curve(run, agent, d):
    files = sorted(glob.glob(str(d / "monitor" / "*.monitor.csv"))) or [str(d / "monitor_train.monitor.csv")]
    df = pd.concat([read_monitor(f) for f in files]).sort_values("abs_t").reset_index(drop=True)
    df["success"] = df["success"].astype(str).str.lower().eq("true").astype(float)
    out = pd.DataFrame({"run": run, "agent": agent, "episode": np.arange(1, len(df) + 1),
                        "timestep": df["l"].cumsum(), "return": df["r"], "length": df["l"], "success": df["success"]})
    for c in ["return", "length", "success"]:
        out[f"{c}_ma{MA}"] = out[c].rolling(MA, min_periods=1).mean()
    out[f"return_std{MA}"] = out["return"].rolling(MA, min_periods=2).std()
    return out


def eval_curve(run, agent, d):
    e = np.load(d / "eval/evaluations.npz")
    ts, res, lens = e["timesteps"], e["results"], e["ep_lengths"]
    out = pd.DataFrame({"run": run, "agent": agent, "timestep": ts, "eval_return_mean": res.mean(1),
                        "eval_return_std": res.std(1), "eval_length_mean": lens.mean(1),
                        "n_eval_episodes": res.shape[1]})
    me = read_monitor(d / "monitor_eval.monitor.csv")
    s = me["success"].astype(str).str.lower().eq("true").astype(float).values
    n = res.shape[1]
    k = min(len(ts), len(s) // n)
    succ = np.full(len(ts), np.nan)
    succ[:k] = s[:k * n].reshape(k, n).mean(1)
    # sanity: monitor_eval returns must line up with evaluations.npz
    r = me["r"].values[:k * n].reshape(k, n).mean(1)
    out["eval_success_rate"] = succ
    out["monitor_aligned"] = bool(np.allclose(r, res.mean(1)[:k], atol=1e-3))
    return out


def tb_scalars(run, d):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    tbs = sorted((d / "tb").glob("*_*"), key=lambda p: int(p.name.split("_")[-1]))
    if not tbs:
        return pd.DataFrame()
    ea = EventAccumulator(str(tbs[-1]), size_guidance={"scalars": 0})   # latest run (see known_issues #2)
    ea.Reload()
    rows = [(run, tag, s.step, s.value, s.wall_time, tbs[-1].name) for tag in ea.Tags()["scalars"]
            for s in ea.Scalars(tag)]
    return pd.DataFrame(rows, columns=["run", "tag", "step", "value", "wall_time", "tb_dir"])


def saved_row(d, agent):
    f = d / f"agent{agent}_vs_baselines.csv"
    if not f.exists():
        return {}
    df = pd.read_csv(f)
    return df.iloc[0].to_dict()


def sensitivity(runs):
    rows = []
    groups = {"A": ["A_eps0.1", "A_eps0.3", "A_eps0.6", "A"], "B": ["B_ent0.0", "B_ent0.01", "B_ent0.05",
              "B_clip0.1", "B_clip0.3", "B"], "C": ["C_ent0.0", "C_ent0.01", "C_ent0.05", "C"]}
    mode = {"A": "navigate", "B": "refine", "C": "supervise"}
    for agent, names in groups.items():
        for name in names:
            d = RUN_DIRS[name]
            hp = json.load(open(d / "hyperparams.json"))
            eps = run_policy(make_env(mode[agent]), SB3Policy(load_model(d / "best/best_model.zip")))
            h = pd.DataFrame(eps)
            s = saved_row(d, agent)
            rows.append({"agent": agent, "run": name, "steps_trained": hp["steps"],
                         "short_run": hp["steps"] < MAIN_STEPS[agent],
                         "eps_fraction": hp.get("exploration_fraction"), "ent_coef": hp.get("ent_coef"),
                         "clip_range": hp.get("clip_range"),
                         "train_minutes": s.get("train_minutes"),
                         "saved_return": s.get("return_mean"), "saved_success": s.get("success_mean"),
                         "saved_steps": s.get("steps_mean"), "saved_energy": s.get("energy_used_mean"),
                         "heldout_n": len(h), "heldout_return": h["return"].mean(),
                         "heldout_return_ci95": ci95(h["return"]), "heldout_success": h["success"].mean(),
                         "heldout_steps": h["steps"].mean(), "heldout_energy": h["energy_used"].mean(),
                         "heldout_defects": h["defects_found"].mean(), "heldout_miou": h["miou"].mean()})
            print(f"  sensitivity {name:10s} held-out return {h['return'].mean():7.2f} success {h['success'].mean():.2f}",
                  flush=True)
    return pd.DataFrame(rows)


def sample_eff(ev, runs):
    rows = []
    for run, (agent, d) in runs.items():
        e = ev[ev["run"] == run]
        def first(th):
            hit = e[e["eval_success_rate"] >= th]
            return int(hit["timestep"].iloc[0]) if len(hit) else np.nan
        s = saved_row(d, agent)
        hp = json.load(open(d / "hyperparams.json"))
        rows.append({"run": run, "agent": agent, "steps_trained": hp["steps"], "train_seed": hp.get("seed"),
                     "steps_to_80pct_success": first(0.8), "steps_to_100pct_success": first(1.0),
                     "best_eval_return": e["eval_return_mean"].max(),
                     "steps_to_best_eval_return": int(e.loc[e["eval_return_mean"].idxmax(), "timestep"]),
                     "final_eval_return": e["eval_return_mean"].iloc[-1],
                     "final_eval_success": e["eval_success_rate"].iloc[-1],
                     "train_minutes": s.get("train_minutes")})
    return pd.DataFrame(rows)


def model_size():
    s0 = pd.read_csv(RESULTS / "summary_S0_all.csv")
    lat = s0.set_index("policy")["inference_ms_mean"]
    rows = []
    for key, pol in [("A", "DQN (Agent A)"), ("B", "PPO (Agent B)"), ("C", "A2C (Agent C)"),
                     ("C_withA", "A2C (C_withA)")]:
        p = RUN_DIRS[key] / "best/best_model.zip"
        m = load_model(p)
        n_policy = sum(t.numel() for t in m.policy.parameters())
        # micro-benchmark: median wall time of 1000 deterministic predict() calls on a real observation
        import time
        obs, _ = make_env({"A": "navigate", "B": "refine"}.get(key, "supervise")).reset(seed=2000)
        ts = []
        for _ in range(1000):
            t0 = time.perf_counter()
            m.predict(obs, deterministic=True)
            ts.append(time.perf_counter() - t0)
        rows.append({"agent": key, "algorithm": type(m).__name__, "model_file": str(p.relative_to(ROOT)).replace("\\", "/"),
                     "file_kb": p.stat().st_size / 1024, "parameters": n_policy,
                     "net_arch": str(m.policy_kwargs.get("net_arch")),
                     "latency_ms_median_1000": 1000 * float(np.median(ts)),
                     "latency_ms_p95_1000": 1000 * float(np.percentile(ts, 95)),
                     "inference_ms_per_action_heldout": lat.get(pol)})
    return pd.DataFrame(rows)


def main():
    runs = all_runs()
    tc = pd.concat([training_curve(r, a, d) for r, (a, d) in runs.items()], ignore_index=True)
    tc.to_csv(RESULTS / "training_curves.csv", index=False)
    ev = pd.concat([eval_curve(r, a, d) for r, (a, d) in runs.items()], ignore_index=True)
    ev.to_csv(RESULTS / "eval_curves.csv", index=False)
    bad = ev.groupby("run")["monitor_aligned"].first()
    print("eval/monitor alignment:", "all OK" if bad.all() else f"MISALIGNED {list(bad[~bad].index)}")
    tb = pd.concat([tb_scalars(r, d) for r, (a, d) in runs.items()], ignore_index=True)
    tb.to_csv(RESULTS / "tb_scalars.csv", index=False)
    sample_eff(ev, runs).to_csv(RESULTS / "sample_efficiency.csv", index=False)
    model_size().to_csv(RESULTS / "model_size.csv", index=False)
    if "--no-sensitivity" not in sys.argv:
        sensitivity(runs).to_csv(RESULTS / "sensitivity.csv", index=False)
    print(f"runs: {len(runs)}  training episodes: {len(tc)}  tb scalars: {len(tb)}")


if __name__ == "__main__":
    main()
