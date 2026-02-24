# 1) Activate env
conda activate chargym

# 2) Build validation data from all 10 seeds (defaults to all seed-* and latest run per seed)
python scripts/build_ddpg_validation_data.py \
  --models-dir models/DDPG \
  --outfile data/ddpg_validation_from_checkpoints_10seeds.npz

# 3) Plot training + validation together
python visualizers/ep_rew_mean.py \
  --logs-dir logs/DDPG \
  --tag ep_rew_mean \
  --val-npz data/ddpg_validation_from_checkpoints_10seeds.npz \
  --output plots/ep_rew_mean_with_validation_10seeds.pdf