"""Phase 3 - varying task environments (robustness). Same 100 held-out seeds (2000-2099) per scenario.

  python evaluate_scenarios.py            run S1..S5 (one worker process per scenario, in parallel), then aggregate
  python evaluate_scenarios.py --only S2_density10   run one scenario (used by the workers)
  python evaluate_scenarios.py --aggregate           only rebuild scenario_summary.csv / robustness_gap.csv

Outputs (results/): episodes_<group>_<scenario>.csv, scenario_summary.csv, robustness_gap.csv, scenario_skips.csv
"""
import argparse
import subprocess
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, SCENARIOS  # noqa: E402
from suites import GROUPS, LEARNED_REAL, run_group, episodes_path, summarize_group  # noqa: E402

PERTURBED = [s for s in SCENARIOS if s != "S0_nominal"]
LIMITS = [  # documented limitations (not failures)
    ("S3_noise0.05 / S3_noise0.10", "C, C_withA, full",
     "only Agent C's own observation is noised; sub-agents A/B read their observations inside the env"),
    ("S5_hardstart", "A, C, C_withA, full", "hard start geometry only exists in refine mode (Agent B)"),
    ("S1_learned", "C_withA", "only the analytic-trained C_withA model exists (no real-image C_withA)"),
]


def groups_for(scen):
    return ["B"] if scen == "S5_hardstart" else GROUPS


def run_one(scen):
    skips = []
    for g in groups_for(scen):
        try:
            df, _ = run_group(g, scen, log=lambda m: print(m, flush=True))
            df.to_csv(episodes_path(g, scen), index=False)
        except Exception as e:  # record and continue
            skips.append((scen, g, f"{type(e).__name__}: {e}"))
            traceback.print_exc()
    pd.DataFrame(skips, columns=["scenario", "policy", "reason"]).to_csv(
        RESULTS / f"_skips_{scen}.csv", index=False)


def aggregate():
    sums = []
    for scen in SCENARIOS:
        for g in GROUPS:
            p = episodes_path(g, scen)
            if p.exists():
                sums.append(summarize_group(pd.read_csv(p)))
    ss = pd.concat(sums, ignore_index=True)
    ss.insert(1, "scenario_desc", ss["scenario"].map(lambda s: SCENARIOS[s]["desc"]))
    ss.to_csv(RESULTS / "scenario_summary.csv", index=False)

    nom = ss[ss["scenario"] == "S0_nominal"].set_index(["group", "policy"])
    gap = []
    for _, r in ss[ss["scenario"] != "S0_nominal"].iterrows():
        key = (r["group"], r["policy"])
        if key not in nom.index:   # real-image-trained models have no nominal counterpart
            continue
        n = nom.loc[key]
        gap.append({"group": r["group"], "policy": r["policy"], "scenario": r["scenario"],
                    "scenario_desc": r["scenario_desc"], "n_episodes": r["n_episodes"], "seeds": r["seeds"],
                    "return_nominal": n["return_mean"], "return_perturbed": r["return_mean"],
                    "return_gap": n["return_mean"] - r["return_mean"],
                    "success_nominal": n["success_rate"], "success_perturbed": r["success_rate"],
                    "success_gap": n["success_rate"] - r["success_rate"],
                    "battery_empty_rate": r["battery_empty_rate"]})
    pd.DataFrame(gap).to_csv(RESULTS / "robustness_gap.csv", index=False)

    skips = [pd.read_csv(p) for p in sorted(RESULTS.glob("_skips_*.csv")) if p.stat().st_size > 30]
    sk = pd.concat(skips, ignore_index=True) if skips else pd.DataFrame(columns=["scenario", "policy", "reason"])
    sk = pd.concat([sk, pd.DataFrame(LIMITS, columns=["scenario", "policy", "reason"])], ignore_index=True)
    sk.to_csv(RESULTS / "scenario_skips.csv", index=False)
    print(f"aggregated {len(ss)} summary rows, {len(gap)} gap rows, {len(sk)} skip/limit rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.only:
        return run_one(a.only)
    if not a.aggregate:
        t0 = time.time()
        logs = RESULTS / "logs"
        logs.mkdir(exist_ok=True)
        todo, running = list(PERTURBED), []
        while todo or running:
            while todo and len(running) < a.workers:
                s = todo.pop(0)
                f = open(logs / f"scenario_{s}.log", "w")
                running.append((s, f, subprocess.Popen([sys.executable, "-B", __file__, "--only", s],
                                                       stdout=f, stderr=subprocess.STDOUT)))
            time.sleep(2)
            for item in list(running):
                if item[2].poll() is not None:
                    item[1].close()
                    running.remove(item)
                    print(f"finished {item[0]} (exit {item[2].returncode}) at {(time.time() - t0) / 60:.1f} min",
                          flush=True)
    aggregate()


if __name__ == "__main__":
    main()
