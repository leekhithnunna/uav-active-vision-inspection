"""
Agent A - Viewpoint Planner (DQN) - owner: Yeseswini
Trains on mode="navigate", saves checkpoints + best model + logs, then evaluates
the trained agent on the SAME 20 seeded worlds as the baselines.

Colab:
  !python scripts/train_agent_a.py --steps 200000 --out /content/drive/MyDrive/team08/agentA
  (after Mukhesh's cache arrives)
  !python scripts/train_agent_a.py --detector learned --cache /content/drive/MyDrive/team08/detector_cache.npz \
        --out /content/drive/MyDrive/team08/agentA_learned
"""
import argparse, csv, json, os, time
import _path  # noqa
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from env import InspectionEnv
from env.baselines import evaluate, summarize, KEYS, RasterNav

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=200_000)
p.add_argument("--detector", default="analytic", choices=["analytic", "learned"])
p.add_argument("--cache", default=None)
p.add_argument("--out", default="runs/agentA")
p.add_argument("--seed", type=int, default=0)
# hyper-parameters exposed for the Part II sensitivity study
p.add_argument("--lr", type=float, default=5e-4)
p.add_argument("--gamma", type=float, default=0.995)
p.add_argument("--eps_fraction", type=float, default=0.3, help="fraction of training for epsilon decay")
p.add_argument("--eps_final", type=float, default=0.05)
p.add_argument("--eps_start", type=float, default=1.0, help="initial epsilon (use ~0.3 when fine-tuning)")
p.add_argument("--init", default=None, help="warm-start from a trained model .zip (transfer: analytic -> real images)")
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)

INFO_KEYS = ("coverage", "mean_uncertainty", "defects_found", "miou", "energy_used", "success")


def make_env(tag):
    env = InspectionEnv(mode="navigate", detector=args.detector, cache_path=args.cache)
    return Monitor(env, os.path.join(args.out, f"monitor_{tag}"), info_keywords=INFO_KEYS)


env, eval_env = make_env("train"), make_env("eval")

hp = dict(learning_rate=args.lr, buffer_size=100_000, learning_starts=5_000, batch_size=128,
          gamma=args.gamma, train_freq=4, gradient_steps=1, target_update_interval=1_000,
          exploration_fraction=args.eps_fraction, exploration_initial_eps=args.eps_start,
          exploration_final_eps=args.eps_final, policy_kwargs=dict(net_arch=[256, 256]))
with open(os.path.join(args.out, "hyperparams.json"), "w") as f:
    json.dump({**hp, "policy_kwargs": str(hp["policy_kwargs"]), **vars(args)}, f, indent=2)

if args.init:   # transfer learning: keep the learned Q-network, fresh replay buffer + new exploration schedule
    from stable_baselines3.common.utils import get_linear_fn
    model = DQN.load(args.init, env=env, device="cpu", tensorboard_log=os.path.join(args.out, "tb"),
                     custom_objects={"learning_rate": args.lr, "gamma": args.gamma, "learning_starts": 5_000,
                                     "exploration_initial_eps": args.eps_start, "exploration_final_eps": args.eps_final,
                                     "exploration_fraction": args.eps_fraction})
    model.exploration_schedule = get_linear_fn(args.eps_start, args.eps_final, args.eps_fraction)
    model.verbose = 1
    print(f"warm-started from {args.init}")
else:
    model = DQN("MlpPolicy", env, tensorboard_log=os.path.join(args.out, "tb"),
                device="cpu", seed=args.seed, verbose=1, **hp)

callbacks = [
    CheckpointCallback(save_freq=25_000, save_path=os.path.join(args.out, "checkpoints"), name_prefix="agentA"),
    EvalCallback(eval_env, eval_freq=10_000, n_eval_episodes=10, deterministic=True,
                 best_model_save_path=os.path.join(args.out, "best"), log_path=os.path.join(args.out, "eval")),
]

t0 = time.time()
model.learn(total_timesteps=args.steps, callback=callbacks, log_interval=20)
train_min = (time.time() - t0) / 60
model.save(os.path.join(args.out, "agentA_final"))
print(f"\nTraining time: {train_min:.1f} min")

# ---------- final evaluation vs baselines on identical seeded worlds ----------
best_path = os.path.join(args.out, "best", "best_model.zip")
agent = DQN.load(best_path if os.path.exists(best_path) else os.path.join(args.out, "agentA_final"), device="cpu")
test_env = InspectionEnv(mode="navigate", detector=args.detector, cache_path=args.cache)
results = {
    "DQN (Agent A)": evaluate(test_env, lambda o: agent.predict(o, deterministic=True)[0]),
    "raster_mid": evaluate(test_env, RasterNav(1)),
    "raster_low": evaluate(test_env, RasterNav(2)),
}
with open(os.path.join(args.out, "agentA_vs_baselines.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["policy"] + [f"{k}_mean" for k in KEYS] + [f"{k}_std" for k in KEYS] + ["train_minutes"])
    for name, rows in results.items():
        s = summarize(rows)
        w.writerow([name] + [s[k][0] for k in KEYS] + [s[k][1] for k in KEYS] + [round(train_min, 1)])
        print(f"{name:14s} return={s['return'][0]:7.2f}  success={s['success'][0]:.2f}  "
              f"steps={s['steps'][0]:6.1f}  cov={s['coverage'][0]:.2f}  unc={s['mean_uncertainty'][0]:.3f}  "
              f"found={s['defects_found'][0]:.1f}  mIoU={s['miou'][0]:.3f}  energy={s['energy_used'][0]:.3f}")
print(f"\nAll outputs in: {args.out}")
