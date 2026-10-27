"""Dataset, normalizers and episode sampling.

Replicates le-wm/eval.py, so a run here selects the same 50 (episode, start)
pairs as the upstream LeWM evaluator for the same seed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import stable_worldmodel as swm
from sklearn import preprocessing


def episode_column(dataset) -> str:
    return "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"


def load_dataset(cfg):
    cache_dir = Path(swm.data.utils.get_cache_dir())
    return swm.data.HDF5Dataset(
        cfg.data.dataset_name,
        keys_to_cache=list(cfg.data.keys_to_cache),
        cache_dir=cache_dir,
    )


def build_process(dataset, keys: list[str]) -> dict:
    """StandardScaler per non-pixel column, aliased to goal_<col> for non-actions (eval.py:72-82)."""
    process = {}
    for col in keys:
        if col == "pixels":
            continue
        scaler = preprocessing.StandardScaler()
        data = dataset.get_col_data(col)
        data = data[~np.isnan(data).any(axis=1)]
        scaler.fit(data)
        process[col] = scaler
        if col != "action":
            process[f"goal_{col}"] = scaler
    return process


def sample_episodes(dataset, num_eval: int, goal_offset: int, seed: int):
    """Same draw as eval.py:109-134 (vectorised episode lengths, identical values).

    Returns (episode_ids, start_steps, n_valid_starts).
    """
    col = episode_column(dataset)
    episode_idx = np.asarray(dataset.get_col_data(col)).astype(np.int64).ravel()
    step_idx = np.asarray(dataset.get_col_data("step_idx")).astype(np.int64).ravel()

    lengths = np.zeros(int(episode_idx.max()) + 1, dtype=np.int64)
    np.maximum.at(lengths, episode_idx, step_idx + 1)          # max step_idx + 1 per episode
    max_start_per_row = lengths[episode_idx] - goal_offset - 1
    valid_indices = np.nonzero(step_idx <= max_start_per_row)[0]

    g = np.random.default_rng(seed)
    pick = g.choice(len(valid_indices) - 1, size=num_eval, replace=False)
    rows = np.sort(valid_indices[pick])
    if len(rows) < num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    data = dataset.get_row_data(rows)
    episodes = [int(x) for x in np.asarray(data[col]).ravel()]
    starts = [int(x) for x in np.asarray(data["step_idx"]).ravel()]
    return episodes, starts, int(len(valid_indices))
