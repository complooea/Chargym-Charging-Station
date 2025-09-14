#!/usr/bin/env bash
set -euo pipefail

# Run from repo root directory

TIMESTEPS=${TIMESTEPS:-20000}
ITERS=${ITERS:-50}

echo "Training DDPG for seeds 0..9"
for s in $(seq 0 9); do
  echo "DDPG seed $s"
  python Solvers/RL/DDPG_train.py --seed "$s" --timesteps "$TIMESTEPS" --iters "$ITERS"
done

echo "Training PPO for seeds 0..9"
for s in $(seq 0 9); do
  echo "PPO seed $s"
  python Solvers/RL/PPO_train.py --seed "$s" --timesteps "$TIMESTEPS" --iters "$ITERS"
done

echo "All trainings completed."
