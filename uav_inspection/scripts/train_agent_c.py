"""
Agent C - Mission Supervisor (A2C) - owner: Mukhesh
Trains on mode="supervise": every decision picks an OPTION (0 explore, 1 refine, 2 return home).
Sub-agents inside the options are scripted stubs by default (raster navigator + approach refiner),
or the trained Agent A model if you pass --nav_model (hierarchical RL / semi-MDP).

Colab:
  !python scripts/train_agent_c.py --steps 100000 --out $DRIVE/agentC
  !python scripts/train_agent_c.py --detector learned --cache $DRIVE/detector_cache.npz --out $DRIVE/agentC_learned
  Part II (CO5) entropy sensitivity:
  for ent in [0.0, 0.01, 0.05]: !python scripts/train_agent_c.py --steps 60000 --ent_coef {ent} --out $DRIVE/agentC_ent{ent}
"""
import argparse, csv, json, os, time
import _path  # noqa
from stable_baselines3 import A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from env import InspectionEnv, sb3_policy
from env.baselines import evaluate, summarize, KEYS, RuleSupervisor, RandomPolicy

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=100_000, help="supervisor decisions (each = up to 10 low-level steps)")
p.add_argument("--detector", default="analytic", choices=["analytic", "learned"])
p.add_argument("--cache", default=None)
p.add_argument("--nav_model", default=None, help="optional: trained Agent A (DQN) .zip used inside 'explore'")
p.add_argument("--out", default="runs/agentC")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--lr", type=float, default=7e-4)
p.add_argument("--gamma", type=float, default=0.99)
p.add_argument("--ent_coef", type=float, default=0.01, help="entropy bonus = exploration strength (CO5)")
p.add_argument("--n_envs", type=int, default=4)
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)

INFO_KEYS = ("coverage", "mean_uncertainty", "defects_found", "miou", "energy_used", "success")


def make_env():
    nav = sb3_policy(DQN.load(args.nav_model, device="cpu")) if args.nav_model else None
    return InspectionEnv(mode="supervise", detector=args.detector, cache_path=args.cache, nav_policy=nav)


env = make_vec_env(make_env, n_envs=args.n_envs, seed=args.seed, monitor_dir=os.path.join(args.out, "monitor"),
                   monitor_kwargs={"info_keywords": INFO_KEYS})
eval_env = Monitor(make_env(), os.path.join(args.out, "monitor_eval"), info_keywords=INFO_KEYS)

hp = dict(learning_rate=args.lr, gamma=args.gamma, ent_coef=args.ent_coef, n_steps=8, vf_coef=0.5,
          max_grad_norm=0.5, policy_kwargs=dict(net_arch=[64, 64]))
json.dump({**hp, "policy_kwargs": str(hp["policy_kwargs"]), **vars(args)},
          open(os.path.join(args.out, "hyperparams.json"), "w"), indent=2)

model = A2C("MlpPolicy", env, tensorboard_log=os.path.join(args.out, "tb"), device="cpu",
            seed=args.seed, verbose=1, **hp)
cb = EvalCallback(eval_env, eval_freq=max(1, 5_000 // args.n_envs), n_eval_episodes=10, deterministic=True,
                  best_model_save_path=os.path.join(args.out, "best"), log_path=os.path.join(args.out, "eval"))
t0 = time.time()
model.learn(total_timesteps=args.steps, callback=cb, log_interval=100)
train_min = (time.time() - t0) / 60
model.save(os.path.join(args.out, "agentC_final"))
print(f"\nTraining time: {train_min:.1f} min")

best = os.path.join(args.out, "best", "best_model.zip")
agent = A2C.load(best if os.path.exists(best) else os.path.join(args.out, "agentC_final"), device="cpu")
test_env = make_env()
results = {"A2C (Agent C)": evaluate(test_env, lambda o: agent.predict(o, deterministic=True)[0]),
           "rule_supervisor": evaluate(test_env, RuleSupervisor()),
           "random": evaluate(test_env, RandomPolicy(test_env.action_space))}
with open(os.path.join(args.out, "agentC_vs_baselines.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["policy"] + [f"{k}_mean" for k in KEYS] + [f"{k}_std" for k in KEYS] + ["train_minutes"])
    for name, rows in results.items():
        s = summarize(rows)
        w.writerow([name] + [s[k][0] for k in KEYS] + [s[k][1] for k in KEYS] + [round(train_min, 1)])
        print(f"{name:16s} return={s['return'][0]:7.2f}  safe_return={s['success'][0]:.2f}  "
              f"decisions={s['steps'][0]:5.1f}  cov={s['coverage'][0]:.2f}  found={s['defects_found'][0]:.1f}  "
              f"mIoU={s['miou'][0]:.3f}  energy={s['energy_used'][0]:.3f}")
print(f"\nAll outputs in: {args.out}")
