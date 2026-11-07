"""Phase 5 (optional) - extra training seeds with the SAME settings as the original runs.

The original training scripts (uav_inspection/scripts/train_agent_*.py) already accept --seed and --out,
so this wrapper only reads each original hyperparams.json, passes those values explicitly and writes to
Evaluation_Metrics/multiseed_runs/<A|B|C>_seed<k>/  (never into the original folders).

  python multiseed.py train --agents B C --seeds 1 2 3 --workers 6
  python multiseed.py eval                      # held-out evaluation of every finished run (+ original seed 0)
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import UAV, MULTISEED, RESULTS, RUN_DIRS, make_env, run_policy, SB3Policy, load_model, ci95  # noqa: E402

SCRIPT = {"A": "train_agent_a.py", "B": "train_agent_b.py", "C": "train_agent_c.py"}
MODE = {"A": "navigate", "B": "refine", "C": "supervise"}
LEARNED = {"A": "DQN (Agent A)", "B": "PPO (Agent B)", "C": "A2C (Agent C)"}


def train_args(agent):
    hp = json.load(open(RUN_DIRS[agent] / "hyperparams.json"))
    a = ["--steps", hp["steps"], "--lr", hp["lr"], "--gamma", hp["gamma"]]
    if agent == "A":
        a += ["--eps_fraction", hp["eps_fraction"], "--eps_final", hp["eps_final"], "--eps_start", hp["eps_start"]]
    if agent == "B":
        a += ["--clip", hp["clip"], "--ent_coef", hp["ent_coef"], "--n_envs", hp["n_envs"]]
    if agent == "C":
        a += ["--ent_coef", hp["ent_coef"], "--n_envs", hp["n_envs"]]
    return [str(x) for x in a]


def train(agents, seeds, workers):
    MULTISEED.mkdir(exist_ok=True)
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    todo = [(a, s) for a in agents for s in seeds
            if not (MULTISEED / f"{a}_seed{s}" / f"agent{a}_vs_baselines.csv").exists()]
    running, t0 = [], time.time()
    while todo or running:
        while todo and len(running) < workers:
            a, s = todo.pop(0)
            out = MULTISEED / f"{a}_seed{s}"
            out.mkdir(exist_ok=True)
            log = open(out / "train.log", "w")
            cmd = [sys.executable, "-B", str(UAV / "scripts" / SCRIPT[a]), "--seed", str(s), "--out", str(out)]
            cmd += train_args(a)
            running.append((a, s, log, subprocess.Popen(cmd, cwd=str(UAV / "scripts"), stdout=log,
                                                         stderr=subprocess.STDOUT, env=env)))
            print(f"started {a} seed {s}", flush=True)
        time.sleep(5)
        for item in list(running):
            if item[3].poll() is not None:
                item[2].close()
                running.remove(item)
                print(f"finished {item[0]} seed {item[1]} (exit {item[3].returncode}) at "
                      f"{(time.time() - t0) / 60:.1f} min", flush=True)


def evaluate_runs():
    rows = []
    for agent in "ABC":
        runs = [(0, RUN_DIRS[agent], "original")] + [
            (int(d.name.split("seed")[1]), d, "multiseed") for d in sorted(MULTISEED.glob(f"{agent}_seed*"))
            if (d / "best/best_model.zip").exists() and (d / f"agent{agent}_vs_baselines.csv").exists()]
        if len(runs) == 1:
            continue
        for seed, d, src in runs:
            eps = pd.DataFrame(run_policy(make_env(MODE[agent]), SB3Policy(load_model(d / "best/best_model.zip"))))
            saved = pd.read_csv(d / f"agent{agent}_vs_baselines.csv").iloc[0]
            rows.append({"agent": agent, "train_seed": seed, "source": src, "n_episodes": len(eps),
                         "seeds": f"{eps['seed'].min()}-{eps['seed'].max()}",
                         "heldout_return": eps["return"].mean(), "heldout_return_ci95": ci95(eps["return"]),
                         "heldout_success": eps["success"].mean(), "heldout_steps": eps["steps"].mean(),
                         "heldout_energy": eps["energy_used"].mean(), "train_minutes": saved.get("train_minutes")})
            eps.assign(agent=agent, train_seed=seed).to_csv(RESULTS / f"multiseed_episodes_{agent}_seed{seed}.csv",
                                                            index=False)
            print(f"  {agent} seed {seed}: held-out return {eps['return'].mean():.2f} success {eps['success'].mean():.2f}",
                  flush=True)
    if not rows:
        print("no multi-seed runs found")
        return
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "multiseed_per_seed.csv", index=False)
    agg = df.groupby("agent").agg(n_seeds=("train_seed", "count"), seeds=("train_seed", lambda s: " ".join(map(str, s))),
                                  return_mean=("heldout_return", "mean"), return_std=("heldout_return", "std"),
                                  return_ci95=("heldout_return", ci95), success_mean=("heldout_success", "mean"),
                                  success_std=("heldout_success", "std"), steps_mean=("heldout_steps", "mean"),
                                  energy_mean=("heldout_energy", "mean"), train_minutes_mean=("train_minutes", "mean"))
    agg.reset_index().to_csv(RESULTS / "multiseed_summary.csv", index=False)
    print(agg.round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval"])
    ap.add_argument("--agents", nargs="+", default=["B", "C"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    train(a.agents, a.seeds, a.workers) if a.cmd == "train" else evaluate_runs()


if __name__ == "__main__":
    main()
