"""
Dataset class for diverse maze data.

Extends PointMazeDataset with map metadata (map indices and maze specs).
The visual/state/action data format is identical to PointMazeDataset,
so training works out of the box.
"""
import torch
from pathlib import Path
from typing import Callable, Optional
from .point_maze_dset import PointMazeDataset
from .traj_dset import get_train_val_sliced


class DiverseMazeDataset(PointMazeDataset):
    """Dataset for diverse maze environments with multiple layouts.

    Loads the same data format as PointMazeDataset, plus:
    - map_indices.pth: (N,) tensor mapping each episode to its maze layout
    - maze_specs.pth: dict {map_idx: maze_spec_string}
    """

    def __init__(
        self,
        data_path: str = "data/diverse_maze/train",
        n_rollout: Optional[int] = None,
        transform: Optional[Callable] = None,
        normalize_action: bool = False,
        action_scale: float = 1.0,
        use_preprocessed: bool = False,
        use_frame_files: bool = False,
    ):
        super().__init__(
            data_path=data_path,
            n_rollout=n_rollout,
            transform=transform,
            normalize_action=normalize_action,
            action_scale=action_scale,
            use_preprocessed=use_preprocessed,
            use_frame_files=use_frame_files,
        )

        # Load map metadata if available
        map_indices_path = Path(data_path) / "map_indices.pth"
        maze_specs_path = Path(data_path) / "maze_specs.pth"

        if map_indices_path.exists():
            self.map_indices = torch.load(map_indices_path)
            if self.n_rollout:
                self.map_indices = self.map_indices[: self.n_rollout]
        else:
            self.map_indices = None

        if maze_specs_path.exists():
            self.maze_specs = torch.load(maze_specs_path)
        else:
            self.maze_specs = None

    def get_map_idx(self, episode_idx):
        """Return the map index for a given episode."""
        if self.map_indices is not None:
            return self.map_indices[episode_idx].item()
        return None

    def get_maze_spec(self, episode_idx):
        """Return the maze_spec string for a given episode."""
        if self.map_indices is not None and self.maze_specs is not None:
            map_idx = self.get_map_idx(episode_idx)
            return self.maze_specs[map_idx]
        return None

    def _load_episode_visual_tensor(self, idx):
        """Load one full episode tensor, supporting both 3-digit and 6-digit naming."""
        obs_dir = Path(self.data_path) / "obses"
        for fmt in [f"episode_{idx:06d}.pth", f"episode_{idx:03d}.pth"]:
            path = obs_dir / fmt
            if path.exists():
                return torch.load(path, map_location="cpu")
        raise ValueError(f"Failed to load image for episode {idx}")


def load_diverse_maze_slice_train_val(
    transform,
    n_rollout=None,
    data_path="data/diverse_maze",
    normalize_action=False,
    split_ratio=0.9,
    num_hist=0,
    num_pred=0,
    frameskip=0,
    use_preprocessed=False,
    use_frame_files=False,
):
    """Load diverse maze data with train/val split.

    If data_path contains train/ and val/ subdirectories, uses them directly.
    Otherwise falls back to random splitting like PointMazeDataset.
    """
    data_path = Path(data_path)
    train_dir = data_path / "train"
    val_dir = data_path / "val"

    # The released diverse_maze package ships val/ populated and train/ as an
    # empty stub. Treat an empty train/ as absent so we fall through to the
    # single-directory random split on val/ (below) instead of trying to load a
    # dataset from an empty directory. Goal generation (plan.py) reads val/
    # directly, so eval on val-only data is well-defined.
    def _has_pth(d):
        return d.exists() and any(d.glob("*.pth"))

    if _has_pth(train_dir) and _has_pth(val_dir):
        # Use pre-split directories (different maps for train vs val)
        dset_train = DiverseMazeDataset(
            data_path=str(train_dir),
            n_rollout=n_rollout,
            transform=transform,
            normalize_action=normalize_action,
            use_preprocessed=use_preprocessed,
            use_frame_files=use_frame_files,
        )
        dset_val = DiverseMazeDataset(
            data_path=str(val_dir),
            n_rollout=n_rollout,
            transform=transform,
            normalize_action=normalize_action,
            use_preprocessed=use_preprocessed,
            use_frame_files=use_frame_files,
        )
        # Propagate normalization stats from train to val
        dset_val.action_mean = dset_train.action_mean
        dset_val.action_std = dset_train.action_std
        dset_val.state_mean = dset_train.state_mean
        dset_val.state_std = dset_train.state_std
        dset_val.proprio_mean = dset_train.proprio_mean
        dset_val.proprio_std = dset_train.proprio_std

        # Re-normalize val actions/proprios with train stats
        if normalize_action:
            # Reload raw and re-normalize with train stats
            raw_val = DiverseMazeDataset(
                data_path=str(val_dir),
                n_rollout=n_rollout,
                transform=transform,
                normalize_action=False,
                use_preprocessed=use_preprocessed,
                use_frame_files=use_frame_files,
            )
            dset_val.actions = (raw_val.actions - dset_train.action_mean) / dset_train.action_std
            dset_val.proprios = (raw_val.proprios - dset_train.proprio_mean) / dset_train.proprio_std

        num_frames = num_hist + num_pred

        from .traj_dset import TrajSlicerDataset
        train_slices = TrajSlicerDataset(dset_train, num_frames, frameskip)
        val_slices = TrajSlicerDataset(dset_val, num_frames, frameskip)

        datasets = {'train': train_slices, 'valid': val_slices}
        traj_dset = {'train': dset_train, 'valid': dset_val}
        return datasets, traj_dset

    else:
        # Single directory, random split. If the populated trajectories live in
        # val/ (released val-only package with an empty train/ stub), load those.
        flat_dir = val_dir if (_has_pth(val_dir) and not _has_pth(data_path)) else data_path
        dset = DiverseMazeDataset(
            data_path=str(flat_dir),
            n_rollout=n_rollout,
            transform=transform,
            normalize_action=normalize_action,
            use_preprocessed=use_preprocessed,
            use_frame_files=use_frame_files,
        )
        dset_train, dset_val, train_slices, val_slices = get_train_val_sliced(
            traj_dataset=dset,
            train_fraction=split_ratio,
            num_frames=num_hist + num_pred,
            frameskip=frameskip,
        )
        datasets = {'train': train_slices, 'valid': val_slices}
        traj_dset = {'train': dset_train, 'valid': dset_val}
        return datasets, traj_dset
