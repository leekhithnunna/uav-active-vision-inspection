"""d8 - Live learning burst: a short fresh training run so the UPDATE step (loss, reward, epsilon / entropy) is visible.

Trains a NEW model with demo-sized hyper-parameters into the temp folder from config.yaml. The trained
project models are never loaded for writing, and their modification times are checked before/after.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo_core import cli, loaders, narrate, paths  # noqa: E402

STEPS = {"A": (5000, 2500), "B": (6144, 2048), "C": (2400, 800)}     # (normal, --fast)
KEYS = {  # logger key -> label shown to the audience
    "A": [("rollout/ep_rew_mean", "ep reward (mean)"), ("train/loss", "TD loss (Huber)"),
          ("rollout/exploration_rate", "epsilon")],
    "B": [("rollout/ep_rew_mean", "ep reward (mean)"), ("train/value_loss", "value loss"),
          ("train/policy_gradient_loss", "policy-gradient loss"), ("train/entropy_loss", "entropy loss"),
          ("train/std", "policy std sigma"), ("train/clip_fraction", "clip fraction")],
    "C": [("rollout/ep_rew_mean", "ep reward (mean)"), ("train/value_loss", "value loss"),
          ("train/policy_loss", "policy loss"), ("train/entropy_loss", "entropy loss")],
}


def _model(key: str, env, seed: int, tb: Path):
    """Fresh model, same architecture as the real training script, demo-sized schedule."""
    import stable_baselines3 as sb3
    info = loaders.AGENT_INFO[key]
    if key == "A":
        return sb3.DQN("MlpPolicy", env, learning_rate=5e-4, buffer_size=20_000, learning_starts=1_000, batch_size=128,
                       gamma=0.995, train_freq=4, target_update_interval=500, exploration_fraction=0.5,
                       exploration_final_eps=0.05, policy_kwargs=dict(net_arch=[256, 256]), device="cpu", seed=seed)
    if key == "B":
        return sb3.PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=512, batch_size=64, n_epochs=10, gamma=0.99,
                       clip_range=0.2, policy_kwargs=dict(net_arch=[64, 64]), device="cpu", seed=seed)
    assert info["algo"] == "A2C"
    return sb3.A2C("MlpPolicy", env, learning_rate=7e-4, n_steps=8, gamma=0.99, ent_coef=0.01,
                   policy_kwargs=dict(net_arch=[64, 64]), device="cpu", seed=seed)


class _LivePlot:
    """Episode rewards + the logged training quantities, redrawn as training progresses."""

    def __init__(self, key: str, show: bool):
        from demo_core import render, style
        self.show, self.key = show, key
        labels = [lbl for _, lbl in KEYS[key]][1:4]
        self.fig = render.figure(17, 6, f"Demo 8 - {loaders.AGENT_INFO[key]['algo']} learning live "
                                        f"(fresh model, demo hyper-parameters)")
        axs = self.fig.subplots(1, 1 + len(labels))
        self.ax_r, self.axs = axs[0], dict(zip(labels, axs[1:]))
        self.ax_r.set_title("episode return"); self.ax_r.set_xlabel("env steps")
        (self.l_ep,) = self.ax_r.plot([], [], ".", color=style.LIGHT, ms=6)
        (self.l_avg,) = self.ax_r.plot([], [], "-", color=style.CRIMSON, lw=3)
        self.lines = {}
        for lbl, ax in self.axs.items():
            (self.lines[lbl],) = ax.plot([], [], "o-", color=style.GREY, ms=3, lw=2)
            ax.set_title(lbl); ax.set_xlabel("env steps")
        for ax in axs:
            ax.grid(alpha=0.3); ax.spines[["top", "right"]].set_visible(False)
        self.fig.tight_layout()
        if show:
            import matplotlib.pyplot as plt
            plt.ion(); plt.show(block=False)

    def update(self, eps: list[tuple[int, float]], logs: list[tuple[int, dict]]) -> None:
        import numpy as np
        if eps:
            x, y = zip(*eps)
            self.l_ep.set_data(x, y)
            w = max(1, len(y) // 10)
            self.l_avg.set_data(x, np.convolve(y, np.ones(w) / w, mode="full")[: len(y)])
        for key, lbl in KEYS[self.key]:
            if lbl in self.lines:
                pts = [(s, d[key]) for s, d in logs if key in d]
                if pts:
                    self.lines[lbl].set_data(*zip(*pts))
        for ax in [self.ax_r, *self.axs.values()]:
            ax.relim(); ax.autoscale_view()
        if self.show:
            import matplotlib.pyplot as plt
            plt.pause(0.001)


def _bellman_example(model) -> None:
    """One replay-buffer transition: Q(s,a) vs the TD target r + gamma * max_a' Q_target(s', a')."""
    import torch
    batch = model.replay_buffer.sample(1)
    with torch.no_grad():
        q_sa = model.q_net(batch.observations)[0, int(batch.actions[0])].item()
        q_next = model.q_net_target(batch.next_observations).max(1)[0][0].item()
    r, done = batch.rewards[0].item(), batch.dones[0].item()
    target = r + (1 - done) * model.gamma * q_next
    narrate.results(["r", "gamma", "done", "max Q_target(s',.)", "TD target y", "Q(s,a)", "TD error y - Q"],
                    [[f"{r:+.3f}", model.gamma, int(done), f"{q_next:+.3f}", f"{target:+.3f}", f"{q_sa:+.3f}",
                      f"{target - q_sa:+.3f}"]],
                    "DQN update on one sampled transition: loss = Huber(y - Q(s,a)), gradient step on Q-network")


def main(argv=None) -> dict:
    p = cli.parser(__doc__.splitlines()[0])
    p.add_argument("--agent", choices=list("ABC"), default="A")
    p.add_argument("--steps", type=int, default=None, help="training steps (default: A 5000, B 6144, C 2400)")
    args = cli.parse(p, argv)
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import CSVOutputFormat, KVWriter, Logger
    from stable_baselines3.common.monitor import Monitor
    from demo_core import animate

    key = args.agent
    steps = args.steps or STEPS[key][1 if args.fast else 0]
    out = paths.out_dir("tmp_train") / f"agent{key}_burst_{int(time.time())}"
    out.mkdir(parents=True, exist_ok=True)
    real = [paths.model_path(k) for k in ("A", "B", "C")]
    before = {m: m.stat().st_mtime for m in real if m and m.exists()}
    assert not any(str(out).startswith(str(m.parent)) for m in before), "burst output must not be a model folder"

    info = loaders.AGENT_INFO[key]
    narrate.header(f"DEMO 8 - Learning burst: {info['algo']} ({info['name']})",
                   f"{steps} steps, fresh network, output -> {out}")
    narrate.guideline(1, f"Fresh environment (mode='{info['mode']}') wrapped in a Monitor")
    env = Monitor(loaders.load_env(info["mode"], args.detector))
    model = _model(key, env, args.seed, out)
    eps: list[tuple[int, float]] = []
    logs: list[tuple[int, dict]] = []
    plot = _LivePlot(key, show=not args.headless)

    class Capture(KVWriter):
        def write(self, key_values, key_excluded, step=0):
            d = {k: float(v) for k, v in key_values.items() if isinstance(v, (int, float))}
            logs.append((model.num_timesteps, d))
            parts = [f"{lbl} {d[k]:+.4f}" for k, lbl in KEYS[key] if k in d]
            narrate.say(f"[bold]UPDATE[/] step {model.num_timesteps:5d} | " + " | ".join(parts))

        def close(self):
            pass

    model.set_logger(Logger(str(out), [CSVOutputFormat(str(out / "progress.csv")), Capture()]))

    class Watch(BaseCallback):
        def _on_step(self) -> bool:
            for inf in self.locals.get("infos", []):
                if "episode" in inf:
                    eps.append((self.num_timesteps, float(inf["episode"]["r"])))
            if self.num_timesteps % 250 == 0:
                plot.update(eps, logs)
            return True

    narrate.guideline(6, {"A": "DQN: act epsilon-greedy, store (s,a,r,s') in replay, every 4 steps a TD update",
                          "B": "PPO: collect 512 steps, then 10 epochs of clipped-surrogate updates",
                          "C": "A2C: every 8 decisions one actor-critic update (advantage = G - V)"}[key])
    t0 = time.time()
    model.learn(total_timesteps=steps, callback=Watch(), log_interval=2 if key == "A" else (1 if key == "B" else 20))
    dt = time.time() - t0
    plot.update(eps, logs)
    if key == "A":
        _bellman_example(model)
    path = out / f"agent{key}_burst.zip"
    model.save(path)
    after = {m: m.stat().st_mtime for m in before}
    untouched = before == after
    narrate.say(f"trained {steps} steps in {dt:.1f}s; {len(eps)} episodes; "
                f"{sum(1 for _, d in logs if any(k.startswith('train/') for k in d))} logged updates")
    narrate.say(f"saved burst model -> {path}")
    narrate.say(("[green]" if untouched else "[red]") + f"real trained models untouched: {untouched}")
    narrate.say("[grey50]This short run is NOT the trained agent: the real model trained for "
                f"{ {'A': '200k', 'B': '100k', 'C': '100k'}[key]} steps (see demo 9 for its curves).")
    if args.save:
        animate.save_png(plot.fig, f"learning_burst_{key}", "snapshots")
    if not args.headless:
        import matplotlib.pyplot as plt
        plt.ioff(); plt.show()
    else:
        import matplotlib.pyplot as plt
        plt.close(plot.fig)
    return {"episodes": len(eps), "updates": len(logs), "seconds": dt, "untouched": untouched, "out": str(out)}


if __name__ == "__main__":
    main()
