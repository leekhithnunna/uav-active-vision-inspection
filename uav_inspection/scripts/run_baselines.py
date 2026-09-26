"""
Baseline numbers for the "Baseline / Initial" column of the results table.
Run:  python scripts/run_baselines.py --out results
      python scripts/run_baselines.py --detector learned --cache /content/drive/MyDrive/team08/detector_cache.npz
"""
import argparse, csv, os
import _path  # noqa
from env import InspectionEnv
from env.baselines import (RasterNav, ApproachRefiner, FixedOrbit, RuleSupervisor,
                           RandomPolicy, evaluate, summarize, KEYS)

p = argparse.ArgumentParser()
p.add_argument("--detector", default="analytic", choices=["analytic", "learned"])
p.add_argument("--cache", default=None)
p.add_argument("--episodes", type=int, default=20)
p.add_argument("--out", default="results")
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)

SUITES = {
    "navigate": lambda env: {"random": RandomPolicy(env.action_space),
                             "raster_mid": RasterNav(1), "raster_low": RasterNav(2)},
    "refine": lambda env: {"random": RandomPolicy(env.action_space),
                           "fixed_orbit": FixedOrbit(), "scripted_approach": ApproachRefiner()},
    "supervise": lambda env: {"random": RandomPolicy(env.action_space),
                              "rule_supervisor": RuleSupervisor()},
}

summary_path = os.path.join(args.out, f"baselines_summary_{args.detector}.csv")
with open(summary_path, "w", newline="") as fsum:
    ws = csv.writer(fsum)
    ws.writerow(["mode", "policy"] + [f"{k}_mean" for k in KEYS] + [f"{k}_std" for k in KEYS])
    for mode, make in SUITES.items():
        env = InspectionEnv(mode=mode, detector=args.detector, cache_path=args.cache)
        print(f"\n=== {mode} ({args.detector}) ===")
        for name, pol in make(env).items():
            rows = evaluate(env, pol, args.episodes)
            with open(os.path.join(args.out, f"baseline_{mode}_{name}_{args.detector}.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=KEYS)
                w.writeheader()
                w.writerows(rows)
            s = summarize(rows)
            ws.writerow([mode, name] + [s[k][0] for k in KEYS] + [s[k][1] for k in KEYS])
            print(f"{name:18s} return={s['return'][0]:7.2f}±{s['return'][1]:.2f}  "
                  f"success={s['success'][0]:.2f}  steps={s['steps'][0]:6.1f}  "
                  f"cov={s['coverage'][0]:.2f}  unc={s['mean_uncertainty'][0]:.3f}  "
                  f"found={s['defects_found'][0]:.1f}/{s['defects_total'][0]:.0f}  "
                  f"mIoU={s['miou'][0]:.3f}  energy={s['energy_used'][0]:.3f}")
print(f"\nSaved summary -> {summary_path}")
