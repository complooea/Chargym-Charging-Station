import os
import time
import argparse

import gym
import numpy as np
import torch

import Chargym_Charging_Station  # noqa: F401 - required to register the env
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--timesteps", type=int, default=20000, help="Timesteps per training chunk")
    parser.add_argument("--iters", type=int, default=50, help="Number of chunks to train for")
    parser.add_argument("--tensorboard-tag", type=str, default=None, help="Custom TB run name")
    parser.add_argument(
        "--no-cudnn-deterministic",
        action="store_true",
        help="Skip cuDNN deterministic settings (faster but less reproducible)",
    )
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Run gym check_env (consumes RNG; use only for debugging)",
    )
    args = parser.parse_args()

    seed = args.seed
    run_ts = int(time.time())
    run_name = f"seed-{seed}/{run_ts}"

    models_dir = os.path.join("models", "PPO", run_name)
    logdir = os.path.join("logs", "PPO", run_name)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)

    # Seed global RNGs used by SB3 and PyTorch
    set_random_seed(seed)
    if not args.no_cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Create and seed the environment BEFORE any reset/step/check
    env = gym.make("ChargingEnv-v0")
    env.seed(seed)
    try:
        env.action_space.seed(seed)
    except Exception:
        pass
    try:
        env.observation_space.seed(seed)
    except Exception:
        pass

    if args.check_env:
        from stable_baselines3.common.env_checker import check_env
        check_env(env)

    model = PPO("MlpPolicy", env, verbose=1, seed=seed, tensorboard_log=logdir)

    tb_tag = args.tensorboard_tag or f"PPO_s{seed}"
    timesteps = args.timesteps
    for i in range(1, args.iters + 1):
        model.learn(total_timesteps=timesteps, reset_num_timesteps=False, tb_log_name=tb_tag)
        model.save(os.path.join(models_dir, f"{timesteps * i}"))

    env.close()


if __name__ == "__main__":
    main()