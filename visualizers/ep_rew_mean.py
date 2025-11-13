import argparse
import os
from typing import List, Optional, Tuple, Any

import numpy as np
import matplotlib

# Use non-interactive backend by default (safe for headless servers)
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Global Matplotlib font configuration
# Note: Times New Roman must be installed on the system to take effect.
# You can add fallbacks after it if needed (e.g., "DejaVu Serif").
plt.rcParams.update({
	"font.family": "serif",
	"font.serif": ["Times New Roman"],
	"font.size": 8,
})

try:
	from tensorboard.backend.event_processing import event_accumulator
except Exception as e:  # pragma: no cover - import-time safeguard
	event_accumulator = None


def find_run_dirs(logs_dir: str) -> List[str]:
	"""Return directories under logs_dir that contain TensorBoard event files."""
	run_dirs = []
	for root, _dirs, files in os.walk(logs_dir):
		if any(f.startswith("events.out.tfevents.") for f in files):
			run_dirs.append(root)
	# Deduplicate and sort for deterministic order
	return sorted(set(run_dirs))


def resolve_tag(ea: Any, preferred: str) -> Optional[str]:
	"""Find the scalar tag to use.

	Tries exact match first. If not found, tries to find a tag that contains the
	preferred substring (case-insensitive), e.g., "ep_rew_mean" could match
	"rollout/ep_rew_mean".
	"""
	tags = ea.Tags().get("scalars", [])
	if preferred in tags:
		return preferred
	low = preferred.lower()
	for t in tags:
		if low in t.lower():
			return t
	return None


def load_scalar_series(run_dir: str, tag_hint: str) -> Optional[Tuple[np.ndarray, np.ndarray, str]]:
	"""Load (steps, values, resolved_tag) for a run directory.

	Returns None if the tag can't be found or reading fails.
	"""
	if event_accumulator is None:
		raise RuntimeError(
			"tensorboard is not available. Please install the dependencies from requirements.txt"
		)

	try:
		ea = event_accumulator.EventAccumulator(
			run_dir,
			size_guidance={event_accumulator.SCALARS: 0},
		)
		ea.Reload()
		tag = resolve_tag(ea, tag_hint)
		if not tag:
			return None
		scalar_events = ea.Scalars(tag)
		if not scalar_events:
			return None
		steps = np.array([e.step for e in scalar_events], dtype=float)
		values = np.array([e.value for e in scalar_events], dtype=float)
		# Ensure strictly increasing steps by deduping (keep last occurrence)
		_, unique_indices = np.unique(steps, return_index=True)
		# np.unique returns indices of first occurrence; we want last: recompute mapping
		last_indices = {}
		for idx, s in enumerate(steps):
			last_indices[s] = idx
		order = sorted(last_indices.keys())
		idxs = [last_indices[s] for s in order]
		steps_sorted = steps[idxs]
		values_sorted = values[idxs]
		return steps_sorted, values_sorted, tag
	except Exception:
		return None


def build_common_grid(series: List[Tuple[np.ndarray, np.ndarray]], grid_points: int) -> np.ndarray:
	"""Create a common step grid for interpolation across all runs."""
	if not series:
		return np.array([])
	min_step = min(float(s[0][0]) for s in series if len(s[0]) > 0)
	max_step = max(float(s[0][-1]) for s in series if len(s[0]) > 0)
	if max_step <= min_step:
		return series[0][0].copy()
	return np.linspace(min_step, max_step, num=max(10, grid_points))


def interpolate_to_grid(steps: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
	"""Interpolate a run's values onto the common grid; returns NaN outside run's range."""
	if len(steps) == 0:
		return np.full_like(grid, np.nan, dtype=float)
	y = np.interp(grid, steps, values)
	mask = (grid >= steps[0]) & (grid <= steps[-1])
	y[~mask] = np.nan
	return y


def compute_band_stats(curves: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	"""Compute median, q1, q3, min, max along axis=0, ignoring NaNs."""
	median = np.nanpercentile(curves, 50, axis=0)
	q1 = np.nanpercentile(curves, 25, axis=0)
	q3 = np.nanpercentile(curves, 75, axis=0)
	data_min = np.nanmin(curves, axis=0)
	data_max = np.nanmax(curves, axis=0)
	return median, q1, q3, data_min, data_max


def plot_bands(
	grid: np.ndarray,
	median: np.ndarray,
	q1: np.ndarray,
	q3: np.ndarray,
	data_min: np.ndarray,
	data_max: np.ndarray,
	title: str,
	output: Optional[str],
	show: bool,
	annotate_step: Optional[float] = None,
	annotate_label: Optional[str] = None,
):
	# Figure size: width = 3.487 inches, height = width * 0.5
	_width_in = 3.487
	_height_in = _width_in * 0.42
	plt.figure(figsize=(_width_in, _height_in))
	# Min-Max band
	plt.fill_between(grid, data_min, data_max, color="#0c84c6", edgecolor='none', alpha=0.16, label="min–max")
	# IQR band
	plt.fill_between(grid, q1, q3, color="#0c84c6", edgecolor='none', alpha=0.4, label="25–75%")
	# Median line
	plt.plot(grid, median, color="#0c84c6", linewidth=0.5, label="median")

	plt.xlabel("Steps")
	plt.ylabel("Episode reward")
	plt.grid(True, linestyle=":", alpha=0.5)


	# Optional red marker to indicate a specific step on the median curve
	if annotate_step is not None and grid.size > 0:
		# Find the nearest grid index to the requested step
		idx = int(np.argmin(np.abs(grid - float(annotate_step))))
		x_val = grid[idx]
		y_val = median[idx]
		if not (np.isnan(x_val) or np.isnan(y_val)):
			label = annotate_label if annotate_label is not None else f"selected agent ({int(annotate_step):,})"
			plt.scatter(
				x_val,
				y_val,
				color="#f74d4d",
				s=8,
				zorder=5,
				linewidths=0.3,
				edgecolors="white",
				label=label,
			)

	# Place legend after plotting all artists so the marker label shows up
	plt.legend()
	plt.tight_layout()

	if output:
		os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
		plt.savefig(output)
	if show:
		# Switch to a GUI backend if available; if not, this will no-op in Agg
		try:
			plt.show()
		except Exception:
			pass
	plt.close()


def main():
	parser = argparse.ArgumentParser(description="Plot ep_rew_mean across DDPG seeds with bands")
	parser.add_argument(
		"--logs-dir",
		type=str,
		default=os.path.join("logs", "DDPG"),
		help="Directory containing DDPG TensorBoard logs (default: logs/DDPG)",
	)
	parser.add_argument(
		"--tag",
		type=str,
		default="ep_rew_mean",
		help="Scalar tag to plot (substring matching allowed)",
	)
	parser.add_argument(
		"--grid-points",
		type=int,
		default=500,
		help="Number of step points for interpolation grid",
	)
	parser.add_argument(
		"--output",
		type=str,
		default=os.path.join("plots", "ep_rew_mean.pdf"),
		help="Path to save the plot image",
	)
	parser.add_argument("--show", action="store_true", help="Display the plot window")
	parser.add_argument("--dry-run", action="store_true", help="Only print discovery info and exit")
	parser.add_argument(
		"--mark-step",
		type=float,
		default=980000,
		help="If set, mark this step on the median curve with a red dot (default: 980000)",
	)
	args = parser.parse_args()

	run_dirs = find_run_dirs(args.logs_dir)
	if not run_dirs:
		raise SystemExit(f"No TensorBoard runs found under: {args.logs_dir}")

	series: List[Tuple[np.ndarray, np.ndarray]] = []
	resolved_tags = []
	loaded_runs = 0
	for rd in run_dirs:
		loaded = load_scalar_series(rd, args.tag)
		if loaded is None:
			continue
		steps, values, tag = loaded
		if len(steps) == 0:
			continue
		series.append((steps, values))
		resolved_tags.append(tag)
		loaded_runs += 1

	if args.dry_run:
		print(f"Discovered {len(run_dirs)} run dirs; loaded {loaded_runs} with tag '{args.tag}'.")
		if resolved_tags:
			unique_tags = sorted(set(resolved_tags))
			print("Resolved tag(s):", ", ".join(unique_tags))
		# Print a brief summary of first few runs
		for i, (steps, values) in enumerate(series[:3]):
			print(
				f"Run {i}: points={len(steps)}, step range=[{steps[0]}, {steps[-1]}], "
				f"value range=[{np.nanmin(values)}, {np.nanmax(values)}]"
			)
		return

	if not series:
		raise SystemExit(
			f"Found runs under {args.logs_dir} but none contained a scalar tag matching '{args.tag}'."
		)

	grid = build_common_grid(series, args.grid_points)
	curves = np.vstack([interpolate_to_grid(s, v, grid) for s, v in series])

	median, q1, q3, data_min, data_max = compute_band_stats(curves)

	title = "DDPG: ep_rew_mean vs steps"
	plot_bands(
		grid,
		median,
		q1,
		q3,
		data_min,
		data_max,
		title,
		args.output,
		args.show,
		annotate_step=args.mark_step,
		annotate_label="chosen agent",
	)

	print(
		f"Saved plot with {curves.shape[0]} runs and {curves.shape[1]} points to: {args.output}"
	)


if __name__ == "__main__":
	main()

