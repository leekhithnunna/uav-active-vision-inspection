"""
Agent B - Active View Refiner (PPO) - owner: Leekhith
Trains on mode="refine": the UAV hovers near ONE uncertain cell and adjusts distance / viewing angle /
lateral offset (continuous actions) until it gets the best possible view (q=4) of that cell.

Colab:
  !python scripts/train_agent_b.py --steps 100000 --out $DRIVE/agentB
  !python scripts/train_agent_b.py --detector learned --cache $DRIVE/detector_cache.npz --out $DRIVE/agentB_learned
  Part II (CO5) exploration sensitivity (entropy bonus):
  for ent in [0.0, 0.01, 0.05]: !python scripts/train_agent_b.py --steps 60000 --ent_coef {ent} --out $DRIVE/agentB_ent{ent}
"""
import argparse, csv, json, os, time
import _path  # noqa
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from env import InspectionEnv
from env.baselines import evaluate, summarize, KEYS, FixedOrbit, ApproachRefiner, RandomPolicy

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=100_000)
p.add_argument("--detector", default="analytic", choices=["analytic", "learned"])
p.add_argument("--cache", default=None)
p.add_argument("--out", default="runs/agentB")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--gamma", type=float, default=0.99)
p.add_argument("--clip", type=float, default=0.2, help="PPO clip range epsilon")
p.add_argument("--ent_coef", type=float, default=0.0, help="entropy bonus = exploration strength (CO5)")
p.add_argument("--n_envs", type=int, default=4)
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)

INFO_KEYS = ("mean_uncertainty", "energy_used", "success")


def make_env():
    return InspectionEnv(mode="refine", detector=args.detector, cache_path=args.cache)


env = make_vec_env(make_env, n_envs=args.n_envs, seed=args.seed, monitor_dir=os.path.join(args.out, "monitor"),
                   monitor_kwargs={"info_keywords": INFO_KEYS})
eval_env = Monitor(make_env(), os.path.join(args.out, "monitor_eval"), info_keywords=INFO_KEYS)

hp = dict(learning_rate=args.lr, gamma=args.gamma, clip_range=args.clip, ent_coef=args.ent_coef,
          n_steps=256, batch_size=64, n_epochs=10, gae_lambda=0.95, policy_kwargs=dict(net_arch=[64, 64]))
json.dump({**hp, "policy_kwargs": str(hp["policy_kwargs"]), **vars(args)},
          open(os.path.join(args.out, "hyperparams.json"), "w"), indent=2)

model = PPO("MlpPolicy", env, tensorboard_log=os.path.join(args.out, "tb"), device="cpu",
            seed=args.seed, verbose=1, **hp)
cb = EvalCallback(eval_env, eval_freq=max(1, 5_000 // args.n_envs), n_eval_episodes=20, deterministic=True,
                  best_model_save_path=os.path.join(args.out, "best"), log_path=os.path.join(args.out, "eval"))
t0 = time.time()
model.learn(total_timesteps=args.steps, callback=cb)
train_min = (time.time() - t0) / 60
model.save(os.path.join(args.out, "agentB_final"))
print(f"\nTraining time: {train_min:.1f} min")

best = os.path.join(args.out, "best", "best_model.zip")
agent = PPO.load(best if os.path.exists(best) else os.path.join(args.out, "agentB_final"), device="cpu")
test_env = make_env()
results = {"PPO (Agent B)": evaluate(test_env, lambda o: agent.predict(o, deterministic=True)[0], 50),
           "fixed_orbit": evaluate(test_env, FixedOrbit(), 50),
           "random": evaluate(test_env, RandomPolicy(test_env.action_space), 50),
           "scripted_approach (upper bound)": evaluate(test_env, ApproachRefiner(), 50)}
with open(os.path.join(args.out, "agentB_vs_baselines.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["policy"] + [f"{k}_mean" for k in KEYS] + [f"{k}_std" for k in KEYS] + ["train_minutes"])
    for name, rows in results.items():
        s = summarize(rows)
        w.writerow([name] + [s[k][0] for k in KEYS] + [s[k][1] for k in KEYS] + [round(train_min, 1)])
        print(f"{name:32s} return={s['return'][0]:6.2f}  success={s['success'][0]:.2f}  "
              f"steps={s['steps'][0]:5.1f}  energy={s['energy_used'][0]:.3f}")
print(f"\nAll outputs in: {args.out}")
