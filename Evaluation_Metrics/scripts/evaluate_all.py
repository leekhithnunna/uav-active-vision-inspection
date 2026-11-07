"""Phase 2 - held-out evaluation (core metrics), scenario S0 nominal.

Held-out bridges: seeds 2000-2099 (100 episodes, never used by training or EvalCallback, which used
seeds 1000-1049 / unseeded training envs). Learned policies are deterministic.
Outputs (results/):
  episodes_<group>_S0_nominal.csv   one row per episode
  summary_<group>.csv               one row per policy (mean, std, median, 95% CI, success, regret, ...)
  summary_S0_all.csv                all groups together
  regret_episodes.csv               per-episode regret vs the stated benchmark
  violations.csv                    battery-empty rate and out-of-bounds moves per 1000 steps
  visitation_A_S0.npz               Agent A mean cell-visitation counts per policy (8x16)
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS  # noqa: E402
from suites import GROUPS, BENCHMARK, BENCHMARK_WHY, run_group, episodes_path, summarize_group  # noqa: E402

SCEN = "S0_nominal"


def main(groups=GROUPS):
    RESULTS.mkdir(exist_ok=True)
    t0 = time.time()
    sums, regrets = [], []
    for g in groups:
        df, visits = run_group(g, SCEN, record_cells=(g == "A"))
        df.to_csv(episodes_path(g, SCEN), index=False)
        if visits:
            np.savez(RESULTS / "visitation_A_S0.npz", names=np.array(list(visits)),
                     visits=np.stack(list(visits.values())))
        s = summarize_group(df)
        s.to_csv(RESULTS / f"summary_{g}.csv", index=False)
        sums.append(s)
        b = df[df["policy"] == BENCHMARK[g]].set_index("seed")["return"]
        for pol, d in df.groupby("policy", sort=False):
            if pol == BENCHMARK[g]:
                continue
            regrets.append(pd.DataFrame({"group": g, "policy": pol, "benchmark": BENCHMARK[g],
                                         "benchmark_kind": BENCHMARK_WHY[g], "seed": d["seed"].values,
                                         "return": d["return"].values, "benchmark_return": b.loc[d["seed"]].values,
                                         "regret": b.loc[d["seed"]].values - d["return"].values}))
    allsum = pd.concat(sums, ignore_index=True)
    allsum.to_csv(RESULTS / "summary_S0_all.csv", index=False)
    pd.concat(regrets, ignore_index=True).to_csv(RESULTS / "regret_episodes.csv", index=False)
    allsum[["group", "policy", "n_episodes", "seeds", "battery_empty_rate", "oob_per_1000_steps",
            "low_level_steps_mean"]].to_csv(RESULTS / "violations.csv", index=False)
    print(f"Phase 2 done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main(sys.argv[1:] or GROUPS)
