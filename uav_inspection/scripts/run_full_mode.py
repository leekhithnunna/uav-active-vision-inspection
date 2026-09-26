"""
Integration (team demo) - owner: Leekhith
Plugs the three trained agents together in mode="full":
  Agent C (A2C) decides Explore / Refine / Return;  Explore runs Agent A (DQN);  Refine runs Agent B (PPO).
Also runs an ablation (which learned parts help?) on the same 20 seeded bridges.

Colab:
  !python scripts/run_full_mode.py --a $DRIVE/agentA/best/best_model.zip --b $DRIVE/agentB/best/best_model.zip \
        --c $DRIVE/agentC/best/best_model.zip --out $DRIVE/full
"""
import argparse, csv, os
import numpy as np
import _path  # noqa
from stable_baselines3 import DQN, PPO, A2C
from env import InspectionEnv, sb3_policy
from env.baselines import evaluate, summarize, KEYS, RasterNav, ApproachRefiner, RuleSupervisor

p = argparse.ArgumentParser()
p.add_argument("--a", required=True, help="Agent A DQN .zip")
p.add_argument("--b", required=True, help="Agent B PPO .zip")
p.add_argument("--c", required=True, help="Agent C A2C .zip")
p.add_argument("--detector", default="analytic", choices=["analytic", "learned"])
p.add_argument("--cache", default=None)
p.add_argument("--episodes", type=int, default=20)
p.add_argument("--out", default="runs/full")
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)

A = sb3_policy(DQN.load(args.a, device="cpu"))
B = sb3_policy(PPO.load(args.b, device="cpu"))
C_model = A2C.load(args.c, device="cpu")
C = lambda o: C_model.predict(o, deterministic=True)[0]


def env_with(nav, ref):
    return InspectionEnv("full", detector=args.detector, cache_path=args.cache, nav_policy=nav, refine_policy=ref)


# (name, navigator, refiner, supervisor) - fresh stub objects each time (they keep internal state)
configs = [
    ("ALL LEARNED  (C + A + B)", lambda: A, lambda: B, lambda: C),
    ("ALL SCRIPTED (rule + raster + approach)", lambda: RasterNav(1), lambda: ApproachRefiner(), lambda: RuleSupervisor()),
    ("ablation: C learned, A/B scripted", lambda: RasterNav(1), lambda: ApproachRefiner(), lambda: C),
    ("ablation: rule C, A + B learned", lambda: A, lambda: B, lambda: RuleSupervisor()),
]
with open(os.path.join(args.out, "full_mode_results.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["config"] + [f"{k}_mean" for k in KEYS] + [f"{k}_std" for k in KEYS])
    print(f"{'config':42s} {'return':>7} {'safe':>5} {'cov':>5} {'found':>6} {'mIoU':>6} {'energy':>7}")
    for name, nav, ref, sup in configs:
        s = summarize(evaluate(env_with(nav(), ref()), sup(), args.episodes))
        w.writerow([name] + [s[k][0] for k in KEYS] + [s[k][1] for k in KEYS])
        print(f"{name:42s} {s['return'][0]:7.2f} {s['success'][0]:5.2f} {s['coverage'][0]:5.2f} "
              f"{s['defects_found'][0]:6.1f} {s['miou'][0]:6.3f} {s['energy_used'][0]:7.3f}")

# ---------- one narrated end-to-end mission (team demo) ----------
OPTS = ["Explore (-> Agent A)", "Refine (-> Agent B)", "Return home"]
env = env_with(A, B)
obs, info = env.reset(seed=1000)
lines, total, done, t = [], 0.0, False, 0
while not done:
    a = int(C(obs))
    obs, r, term, trunc, info = env.step(a)
    total += r; done = term or trunc
    lines.append(f"decision {t:2d}: {OPTS[a]:22s} reward {r:+6.2f} | coverage {info['coverage']:.2f} "
                 f"defects {info['defects_found']:2d}/{info['defects_total']} battery {1 - info['energy_used']:.2f}")
    t += 1
lines.append(f"MISSION END: return {total:.2f}, safe return = {info['success']}, mIoU = {info['miou']:.3f}")
lines.append("final map (U=UAV, .=unseen, digit=uncertainty x10):\n" + env.render())
open(os.path.join(args.out, "demo_mission.txt"), "w").write("\n".join(lines))
print("\n" + "\n".join(lines))
print(f"\nSaved -> {args.out}")
