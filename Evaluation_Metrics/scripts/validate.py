"""Phase 1 - pipeline validation.

Re-evaluates every saved policy on the ORIGINAL seeds with env.baselines.evaluate() (exactly as the
training scripts did) and compares the means with the saved *_vs_baselines.csv / full_mode_results.csv.
Tolerance: |reproduced - saved| <= 1% of |saved| (+1e-6 absolute for values that are 0).
Output: results/validation.csv
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (ROOT, RESULTS, RUN_DIRS, InspectionEnv, CACHE, RasterNav, ApproachRefiner, FixedOrbit,  # noqa: E402
                    RuleSupervisor, RandomPolicy, evaluate, summarize, load_model, SB3Policy, sb3_policy)

METRICS = ["return", "success", "steps", "energy_used", "coverage", "defects_found", "miou"]
TOL = 0.01


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def env_for(mode, detector, nav=None, ref=None):
    return InspectionEnv(mode, detector=detector, cache_path=str(CACHE) if detector == "learned" else None,
                         nav_policy=nav, refine_policy=ref)


def specs():
    """(run, csv path, mode, detector, n_episodes, {row name: (policy factory, env factory)})"""
    out = []
    for key in ["A", "A_learned", "A_eps0.1", "A_eps0.3", "A_eps0.6"]:
        d = RUN_DIRS[key]
        det = "learned" if "learned" in key else "analytic"
        m = d / "best/best_model.zip"
        e = lambda det=det: env_for("navigate", det)
        out.append((key, d / "agentA_vs_baselines.csv", 20, {
            "DQN (Agent A)": (lambda m=m: SB3Policy(load_model(m)), e),
            "raster_mid": (lambda: RasterNav(1), e),
            "raster_low": (lambda: RasterNav(2), e)}))
    for key in ["B", "B_learned", "B_ent0.0", "B_ent0.01", "B_ent0.05", "B_clip0.1", "B_clip0.3"]:
        d = RUN_DIRS[key]
        det = "learned" if "learned" in key else "analytic"
        m = d / "best/best_model.zip"
        e = lambda det=det: env_for("refine", det)
        out.append((key, d / "agentB_vs_baselines.csv", 50, {
            "PPO (Agent B)": (lambda m=m: SB3Policy(load_model(m)), e),
            "fixed_orbit": (lambda: FixedOrbit(), e),
            "random": ("random", e),
            "scripted_approach (upper bound)": (lambda: ApproachRefiner(), e)}))
    for key in ["C", "C_learned", "C_withA", "C_ent0.0", "C_ent0.01", "C_ent0.05"]:
        d = RUN_DIRS[key]
        det = "learned" if "learned" in key else "analytic"
        m = d / "best/best_model.zip"
        nav = (lambda: sb3_policy(load_model("A"))) if key == "C_withA" else (lambda: None)
        e = lambda det=det, nav=nav: env_for("supervise", det, nav())
        out.append((key, d / "agentC_vs_baselines.csv", 20, {
            "A2C (Agent C)": (lambda m=m: SB3Policy(load_model(m)), e),
            "rule_supervisor": (lambda: RuleSupervisor(), e),
            "random": ("random", e)}))
    A = lambda: sb3_policy(load_model("A"))
    B = lambda: sb3_policy(load_model("B"))
    C = lambda: SB3Policy(load_model("C"))
    full = lambda nav, ref: (lambda: env_for("full", "analytic", nav(), ref()))
    out.append(("full", ROOT / "For_Leekhith/full/full_mode_results.csv", 20, {
        "ALL LEARNED  (C + A + B)": (C, full(A, B)),
        "ALL SCRIPTED (rule + raster + approach)": (lambda: RuleSupervisor(), full(lambda: RasterNav(1), ApproachRefiner)),
        "ablation: C learned, A/B scripted": (C, full(lambda: RasterNav(1), ApproachRefiner)),
        "ablation: rule C, A + B learned": (lambda: RuleSupervisor(), full(A, B))}))
    return out


def main():
    rows, n_fail = [], 0
    for run, path, n, pols in specs():
        saved = {r.get("policy", r.get("config")): r for r in read_csv(path)}
        for name, (pf, ef) in pols.items():
            env = ef()
            pol = RandomPolicy(env.action_space) if pf == "random" else pf()
            s = summarize(evaluate(env, pol, n, 1000))
            for k in METRICS:
                old, new = float(saved[name][f"{k}_mean"]), s[k][0]
                diff = abs(new - old)
                ok = diff <= TOL * abs(old) + 1e-6
                n_fail += not ok
                rows.append(dict(run=run, policy=name, metric=k, n_episodes=n, seeds=f"1000-{999 + n}",
                                 saved=old, reproduced=new, abs_diff=diff,
                                 rel_diff=diff / abs(old) if old else 0.0, within_1pct=ok,
                                 source=str(path.relative_to(ROOT)).replace("\\", "/")))
            bad = [r for r in rows[-len(METRICS):] if not r["within_1pct"]]
            print(f"{run:10s} {name:42s} {'OK' if not bad else 'MISMATCH ' + str([b['metric'] for b in bad])}",
                  flush=True)
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "validation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} checks, {n_fail} outside +-1%  -> {RESULTS / 'validation.csv'}")
    return n_fail


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
