"""Look inside the trained networks: DQN Q-values, PPO Gaussian, A2C option probabilities."""
from __future__ import annotations

import numpy as np
import torch

A_ACTIONS = ["North", "South", "East", "West", "Climb (higher)", "Descend (lower)"]
B_DIMS = ["d distance", "d angle", "d lateral"]
C_OPTIONS = ["Explore -> Agent A", "Refine -> Agent B", "Return home"]
C_SHORT = ["EXPLORE", "REFINE", "RETURN"]


def _tensor(model, obs):
    return model.policy.obs_to_tensor(np.asarray(obs, np.float32))[0]


def q_values(model, obs) -> dict:
    """DQN: Q(s, a) for all 6 actions; the greedy action is the argmax."""
    with torch.no_grad():
        q = model.q_net(_tensor(model, obs))[0].numpy()
    return {"names": A_ACTIONS, "values": q, "chosen": int(np.argmax(q)), "kind": "Q(s,a)"}


def gaussian(model, obs) -> dict:
    """PPO: mean and std of the Gaussian policy; deterministic action = clip(mean, -1, 1)."""
    with torch.no_grad():
        dist = model.policy.get_distribution(_tensor(model, obs)).distribution
    mu, sigma = dist.mean[0].numpy(), dist.stddev[0].numpy()
    return {"names": B_DIMS, "mu": mu, "sigma": sigma, "action": np.clip(mu, -1, 1), "kind": "N(mu, sigma)"}


def option_probs(model, obs) -> dict:
    """A2C: pi(option | state) for Explore / Refine / Return."""
    with torch.no_grad():
        probs = model.policy.get_distribution(_tensor(model, obs)).distribution.probs[0].numpy()
    return {"names": C_OPTIONS, "values": probs, "chosen": int(np.argmax(probs)), "kind": "pi(o|s)"}


def for_agent(key: str):
    """The introspection function matching an agent key (None for baselines)."""
    return {"A": q_values, "B": gaussian, "C": option_probs}[key[0]]
