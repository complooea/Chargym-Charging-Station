import os
import re
import argparse
from glob import glob

import gym
import numpy as np
import matplotlib

# Headless backend
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure env is registered
import Chargym_Charging_Station  # noqa: F401

from stable_baselines3 import DDPG
from Solvers.RBC.RBC import RBC

plt.rcParams.update({
	"font.family": "serif",
	"font.serif": ["Times New Roman"],
	"font.size": 8,
})

def find_ddpg_model_paths(models_root: str) -> list:
	"""
	Discover best DDPG checkpoints under models_root.
	Expects structure like models/DDPG/seed-*/<timestamp>/*.zip
	Picks the largest step number .zip in each seed folder.
	Returns list of (seed_name, model_path).
	"""
	results = []

	seed_dirs = sorted(glob(os.path.join(models_root, "seed-*")))
	for seed_dir in seed_dirs:
		# Find timestamp subdirs
		ts_dirs = sorted(
			[d for d in glob(os.path.join(seed_dir, "*")) if os.path.isdir(d)]
		)
		if not ts_dirs:
			continue
		# Pick the latest timestamp directory (sorted lexical works since numbers only)
		ts_dir = ts_dirs[-1]

		zips = glob(os.path.join(ts_dir, "*.zip"))
		if not zips:
			continue

		# Choose checkpoint with largest numeric step prefix
		def step_num(p: str) -> int:
			# file names like 940000.zip
			base = os.path.basename(p)
			m = re.match(r"(\d+)\.zip$", base)
			return int(m.group(1)) if m else -1

		best_zip = max(zips, key=step_num)
		results.append((os.path.basename(seed_dir), best_zip))

	return results


def evaluate_episode_with_model(env, model) -> float:
	"""Run one full episode with given model, return sum of rewards."""
	done = False
	rewards = 0.0
	obs = env.reset(reset_flag=1)  # 1 -> keep the same simulated day
	while not done:
		action, _ = model.predict(obs)
		obs, r, done, _ = env.step(action)
		rewards += float(r)
	return rewards


def evaluate_episode_rbc(env) -> float:
	"""Run one full episode with RBC policy on current simulated day (reset_flag=1)."""
	done = False
	rewards = 0.0
	obs = env.reset(reset_flag=1)  # keep same day
	while not done:
		action_rbc = RBC.select_action(env.env, obs)
		obs, r, done, _ = env.step(action_rbc)
		rewards += float(r)
	return rewards


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--env", default="ChargingEnv-v0")
	parser.add_argument(
		"--ddpg_dir",
		default=os.path.join("models", "DDPG"),
		help="Root directory containing DDPG seed folders",
	)
	parser.add_argument("--episodes", type=int, default=100)
	parser.add_argument(
		"--outfile",
		default=os.path.join("plots", "ddpg_variation.pdf"),
		help="Where to save the figure",
	)
	args = parser.parse_args()

	# Discover DDPG checkpoints
	ddpg_models = find_ddpg_model_paths(args.ddpg_dir)
	if not ddpg_models:
		raise FileNotFoundError(
			f"No DDPG checkpoints found under {args.ddpg_dir}. Expected *.zip in seed-*/<timestamp>/"
		)

	# Create env
	env = gym.make(args.env)

	# Pre-load models (attach env for predict)
	loaded_models = []
	for seed_name, model_path in ddpg_models:
		model = DDPG.load(model_path, env=env)
		loaded_models.append((seed_name, model))

	num_models = len(loaded_models)
	episodes = args.episodes

	# Storage: rewards_per_episode[model_index, episode]
	ddpg_rewards = np.zeros((num_models, episodes), dtype=float)
	rbc_rewards = np.zeros((episodes,), dtype=float)

	# Evaluate
	for ep in range(episodes):
		# For each episode, advance to a new day for the first evaluation
		# We do that by running the FIRST DDPG model starting with reset_flag=0,
		# and all subsequent evals (other models + RBC) with reset_flag=1 so that
		# they all experience the same day within this episode.

		# Kick off a new day with the first model
		if num_models > 0:
			seed_name, first_model = loaded_models[0]
			done = False
			total = 0.0
			obs = env.reset(reset_flag=0)  # 0 -> new day
			while not done:
				action, _ = first_model.predict(obs)
				obs, r, done, _ = env.step(action)
				total += float(r)
			ddpg_rewards[0, ep] = total

		# Evaluate remaining models for the same day
		for i in range(1, num_models):
			_, model = loaded_models[i]
			ddpg_rewards[i, ep] = evaluate_episode_with_model(env, model)

		# RBC for the same day
		rbc_rewards[ep] = evaluate_episode_rbc(env)

	env.close()

	# Aggregate across models for each episode
	median = np.median(ddpg_rewards, axis=0)
	q25 = np.percentile(ddpg_rewards, 25, axis=0)
	q75 = np.percentile(ddpg_rewards, 75, axis=0)
	ymin = np.min(ddpg_rewards, axis=0)
	ymax = np.max(ddpg_rewards, axis=0)

	x = np.arange(1, episodes + 1)

	# Figure size: width = 3.487 inches, height = width * 0.5
	_width_in = 3.487
	_height_in = _width_in * 0.75
	plt.figure(figsize=(_width_in, _height_in))
	plt.fill_between(x, ymin, ymax, color="#f74d4d", edgecolor='none', alpha=0.16, label="DDPG min–max")
	plt.fill_between(x, q25, q75, color="#f74d4d", edgecolor='none', alpha=0.40, label="DDPG 25–75%")
	plt.plot(x, median, color="#f74d4d", linewidth=0.5, label="DDPG median")

	# Plot RBC
	plt.plot(x, rbc_rewards, color="#41b7ac", linewidth=0.5, label="RBC")

	plt.xlabel("Evaluation episodes")
	plt.ylabel("Reward")
	# Legend above plot, two columns (so 2x2 layout for 4 items)
	plt.legend(
		loc="upper center",
		bbox_to_anchor=(0.5, 1.3),
		ncol=2,
		frameon=True,
		fontsize=8,
		columnspacing=1.0,
		handlelength=1.5,
		borderaxespad=0.2,
	)
	plt.tight_layout()

	# Save figure
	os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
	plt.savefig(args.outfile, format="pdf", bbox_inches="tight")
	plt.close()

	# Also save the raw data for reproducibility
	np.savez(
		os.path.splitext(args.outfile)[0] + ".npz",
		ddpg_rewards=ddpg_rewards,
		rbc_rewards=rbc_rewards,
		episodes=episodes,
	)


if __name__ == "__main__":
	main()

