import argparse
import copy
import glob
import os
import re
from typing import Dict, List, Optional, Tuple

import gym
import numpy as np

import Chargym_Charging_Station  # noqa: F401 - required to register the env
from stable_baselines3 import DDPG


def step_from_checkpoint_name(path: str) -> Optional[int]:
    base = os.path.basename(path)
    match = re.match(r"(\d+)\.zip$", base)
    if not match:
        return None
    return int(match.group(1))


def choose_run_dir(seed_dir: str, timestamp: str) -> Optional[str]:
    run_dirs = sorted([d for d in glob.glob(os.path.join(seed_dir, "*")) if os.path.isdir(d)])
    if not run_dirs:
        return None

    if timestamp == "latest":
        numeric_dirs = []
        for d in run_dirs:
            name = os.path.basename(d)
            if name.isdigit():
                numeric_dirs.append((int(name), d))
        if numeric_dirs:
            return sorted(numeric_dirs, key=lambda x: x[0])[-1][1]
        return run_dirs[-1]

    explicit_dir = os.path.join(seed_dir, timestamp)
    if os.path.isdir(explicit_dir):
        return explicit_dir
    return None


def discover_seed_checkpoints(models_dir: str, seed_glob: str, timestamp: str) -> List[Tuple[str, str, Dict[int, str]]]:
    seeds: List[Tuple[str, str, Dict[int, str]]] = []
    for seed_dir in sorted(glob.glob(os.path.join(models_dir, seed_glob))):
        if not os.path.isdir(seed_dir):
            continue
        seed_name = os.path.basename(seed_dir)
        run_dir = choose_run_dir(seed_dir, timestamp)
        if run_dir is None:
            print(f"[WARN] Skipping {seed_name}: no matching run dir for timestamp='{timestamp}'.")
            continue

        ckpt_map: Dict[int, str] = {}
        for path in glob.glob(os.path.join(run_dir, "*.zip")):
            step = step_from_checkpoint_name(path)
            if step is None:
                print(f"[WARN] Ignoring non-numeric checkpoint: {path}")
                continue
            ckpt_map[step] = path

        if not ckpt_map:
            print(f"[WARN] Skipping {seed_name}: no numeric checkpoints in {run_dir}.")
            continue
        seeds.append((seed_name, run_dir, ckpt_map))

    return seeds


def rollout_episode_from_obs(env: gym.Env, model: DDPG, obs: np.ndarray, deterministic: bool) -> float:
    """Roll out one full episode starting from a provided initial observation."""
    done = False
    total_reward = 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done, _ = env.step(action)
        total_reward += float(reward)
    return total_reward


def evaluate_seed(
    env_id: str,
    seed_name: str,
    ckpt_map: Dict[int, str],
    steps_union: np.ndarray,
    episodes: int,
    deterministic: bool,
) -> np.ndarray:
    del seed_name  # currently unused, kept for future per-seed diagnostics

    rewards = np.full((len(steps_union), episodes), np.nan, dtype=float)

    local_steps = sorted(ckpt_map.keys())
    step_to_idx = {int(s): i for i, s in enumerate(steps_union.tolist())}
    valid_steps = [s for s in local_steps if s in step_to_idx]
    if not valid_steps:
        return rewards

    env = gym.make(env_id)
    loaded_models: List[Tuple[int, DDPG]] = []
    for step in valid_steps:
        path = ckpt_map[step]
        try:
            model = DDPG.load(path, env=env)
            loaded_models.append((step, model))
        except Exception as exc:
            print(f"[WARN] Failed to load checkpoint {path}: {exc}")

    if not loaded_models:
        env.close()
        return rewards

    for ep in range(episodes):
        # Start a new day once, snapshot env internals, and replay that exact day for every checkpoint.
        first_obs = env.reset(reset_flag=0)
        base_invalues = copy.deepcopy(env.Invalues)
        base_energy = copy.deepcopy(env.Energy)

        first_step, first_model = loaded_models[0]
        first_idx = step_to_idx[first_step]
        rewards[first_idx, ep] = rollout_episode_from_obs(
            env=env,
            model=first_model,
            obs=first_obs,
            deterministic=deterministic,
        )

        for step, model in loaded_models[1:]:
            # Restore same day without relying on reset_flag=1 MAT reload path.
            env.timestep = 0
            env.day = 1
            env.done = False
            env.Invalues = copy.deepcopy(base_invalues)
            env.Energy = copy.deepcopy(base_energy)
            obs = env.get_obs()

            idx = step_to_idx[step]
            rewards[idx, ep] = rollout_episode_from_obs(
                env=env,
                model=model,
                obs=obs,
                deterministic=deterministic,
            )

    env.close()
    return rewards


def main() -> None:
    parser = argparse.ArgumentParser(description="Build post-hoc DDPG validation data from saved checkpoints.")
    parser.add_argument("--env", type=str, default="ChargingEnv-v0")
    parser.add_argument("--models-dir", type=str, default=os.path.join("models", "DDPG"))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-glob", type=str, default="seed-*")
    parser.add_argument(
        "--timestamp",
        type=str,
        default="latest",
        help="Use 'latest' or an explicit timestamp directory name.",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default=os.path.join("data", "ddpg_validation_from_checkpoints.npz"),
    )
    parser.add_argument(
        "--deterministic",
        dest="deterministic",
        action="store_true",
        default=True,
        help="Use deterministic policy actions (default: enabled).",
    )
    parser.add_argument(
        "--stochastic",
        dest="deterministic",
        action="store_false",
        help="Use stochastic policy actions for evaluation.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print discovered checkpoints and exit.")
    args = parser.parse_args()

    if args.episodes <= 0:
        raise SystemExit("--episodes must be positive.")

    seeds = discover_seed_checkpoints(args.models_dir, args.seed_glob, args.timestamp)
    if not seeds:
        raise SystemExit(
            f"No valid checkpoints found under {args.models_dir} with seed_glob='{args.seed_glob}'."
        )

    steps_union = sorted({step for _seed, _run, ckpt_map in seeds for step in ckpt_map.keys()})
    if not steps_union:
        raise SystemExit("Discovered seeds but no numeric checkpoints were found.")

    print(f"Discovered {len(seeds)} seed runs and {len(steps_union)} unique checkpoint steps.")
    for seed_name, run_dir, ckpt_map in seeds:
        print(f"  {seed_name}: run={run_dir}, checkpoints={len(ckpt_map)}")

    if args.dry_run:
        return

    steps = np.array(steps_union, dtype=float)
    seed_names = np.array([seed_name for seed_name, _run, _ckpt_map in seeds])
    rewards = np.full((len(seeds), len(steps_union), args.episodes), np.nan, dtype=float)

    for seed_idx, (seed_name, _run_dir, ckpt_map) in enumerate(seeds):
        print(f"Evaluating {seed_name} ({seed_idx + 1}/{len(seeds)})...")
        rewards[seed_idx] = evaluate_seed(
            env_id=args.env,
            seed_name=seed_name,
            ckpt_map=ckpt_map,
            steps_union=steps,
            episodes=args.episodes,
            deterministic=args.deterministic,
        )

    episode_mean_rewards = np.nanmean(rewards, axis=2)

    out_dir = os.path.dirname(args.outfile) or "."
    os.makedirs(out_dir, exist_ok=True)
    np.savez(
        args.outfile,
        steps=steps,
        seed_names=seed_names,
        rewards=rewards,
        episode_mean_rewards=episode_mean_rewards,
        env_id=args.env,
        episodes=args.episodes,
        models_dir=args.models_dir,
        timestamp_mode=args.timestamp,
        deterministic=args.deterministic,
    )
    print(f"Saved validation data to {args.outfile}")


if __name__ == "__main__":
    main()
