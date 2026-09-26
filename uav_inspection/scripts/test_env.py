"""Smoke test: every mode x both detectors. Run:  python scripts/test_env.py"""
import os, tempfile
import numpy as np
import _path  # noqa
from gymnasium.utils.env_checker import check_env
from env import InspectionEnv


def fake_cache(path, P=600, seed=0):
    """Random cache with the SAME format Mukhesh's real cache will have."""
    rng = np.random.default_rng(seed)
    is_def = rng.random(P) < 0.3
    u = np.sort(rng.random((P, 5)), axis=1)[:, ::-1].copy()
    np.savez(path, u=u.astype(np.float32), pdef=rng.random((P, 5)) < 0.5,
             iou=rng.random((P, 5)).astype(np.float32), is_defect=is_def)


def random_rollout(env, episodes=3):
    for ep in range(episodes):
        obs, info = env.reset(seed=ep)
        done, ret = False, 0.0
        while not done:
            obs, r, term, trunc, info = env.step(env.action_space.sample())
            assert env.observation_space.contains(obs), "obs out of bounds"
            ret += r
            done = term or trunc
    return ret, info


if __name__ == "__main__":
    tmp = os.path.join(tempfile.gettempdir(), "fake_cache.npz")
    fake_cache(tmp)
    for det in ("analytic", "learned"):
        for mode in ("navigate", "refine", "supervise"):
            env = InspectionEnv(mode=mode, detector=det, cache_path=tmp)
            check_env(env, skip_render_check=True)
            ret, info = random_rollout(env)
            print(f"[OK] {det:8s} {mode:9s} return={ret:7.2f} coverage={info['coverage']:.2f} "
                  f"steps={info['steps']}")
    env = InspectionEnv("navigate")
    env.reset(seed=0)
    for a in (5, 2, 2, 2):
        env.step(a)
    print("\nASCII render (U=UAV, .=unseen, digit=uncertainty*10):\n" + env.render())
    print("\nALL TESTS PASSED")
