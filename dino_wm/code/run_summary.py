"""Run summary: code fingerprint, environment, settings and per-episode outcomes.

plan.py writes <run dir>/summary.json after planning finishes. It only reads results, so it cannot change them.
Success follows the planner's per-replan records (success is cumulative within an episode):
  solved_by_k[k-1]  % of episodes solved within k replans, k = 1..max_iter
  success_rate      Final: solved_by_k at max_iter
  auc               mean of solved_by_k over replans 1..max_iter
"""

import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path

import numpy as np
import torch

CODE_ROOT = Path(__file__).resolve().parent
EXCLUDED_DIRS = {"__pycache__", "eval_outputs", "outputs", ".hydra", "wandb", "logs"}
PACKAGES = ("torch", "torchvision", "numpy", "gym", "mujoco-py", "opencv-python", "hydra-core", "omegaconf", "einops")


def code_fingerprint(root: Path = CODE_ROOT) -> str:
    """sha256 over every .py and .yaml file under the code root, in relative-path order; each file contributes its
    relative path, then its bytes. Run outputs and caches (EXCLUDED_DIRS) are skipped."""
    files = [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix in (".py", ".yaml") and not EXCLUDED_DIRS.intersection(p.relative_to(root).parts)
    ]
    h = hashlib.sha256()
    for p in sorted(files, key=lambda q: q.relative_to(root).as_posix()):
        h.update(p.relative_to(root).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def attention_mode() -> str:
    """The attention implementation models/vit.py is running with (set by OPTTA_ATTENTION, default "auto")."""
    try:
        from models.vit import ATTENTION
        return ATTENTION
    except Exception:  # noqa: BLE001 - the summary must never fail a finished run
        return "<unknown>"


def environment_record() -> dict:
    versions = {}
    for dist in PACKAGES:
        try:
            versions[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            versions[dist] = None
    return {
        "packages": versions,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "host": platform.node(),
        "python": platform.python_version(),
    }


def write_summary(cfg_dict: dict, workspace, final_logs: dict, wall_s: float, path: str = "summary.json") -> dict:
    planner = workspace.planner
    n = int(workspace.n_evals)
    max_iter = int(planner.max_iter)

    first_success = {}
    for rec in getattr(planner, "_records", None) or []:
        if rec["success"] and rec["sample_idx"] not in first_success:
            first_success[rec["sample_idx"]] = int(rec["step"])
    solved_by_k = [
        100.0 * sum(1 for i in range(n) if first_success.get(i, np.inf) <= k) / n for k in range(1, max_iter + 1)
    ]

    adapt = (cfg_dict.get("planner") or {}).get("adapt") or {}
    trainer = getattr(planner, "optta_trainer", None)
    sub = getattr(planner, "sub_planner", None)
    final_eval = final_logs.get("final_eval/success_rate")

    summary = {
        "run_name": cfg_dict.get("model_name"),
        "checkpoint": os.path.basename(os.path.normpath(str(cfg_dict.get("ckpt_base_path")))),
        "config_name": (cfg_dict.get("planner") or {}).get("name"),
        "seed": cfg_dict.get("seed"),
        "n_episodes": n,
        "max_iter": max_iter,
        "ood_corruption": cfg_dict.get("ood_corruption"),
        "ood_level": cfg_dict.get("ood_level"),
        "env_kwargs_override": cfg_dict.get("env_kwargs_override"),
        "goal_source": cfg_dict.get("goal_source"),
        "adapt": {
            "lr": adapt.get("lr"),
            "steps": adapt.get("steps"),
            "encoder_lr": adapt.get("encoder_lr"),
            "finetune_encoder": getattr(trainer, "finetune_encoder", adapt.get("finetune_encoder")),
            "replay_buffer": adapt.get("replay_buffer"),
            "rho": getattr(trainer, "ratio_target", None),
            "rho_schedule": getattr(trainer, "rho_schedule", None),
            "rho_K": getattr(trainer, "rho_schedule_steps", None),
            "optimism_elites": getattr(trainer, "optimism_elites", None),
            "optimism_all_samples": getattr(sub, "optimism_all_samples", None),
        },
        # What the run actually used, read from the live objects (the attention mode comes from an env var, so the
        # resolved Hydra config does not record it).
        "precision": {
            "attention": attention_mode(),
            "planning_dtype": getattr(sub, "planning_dtype", None),
            "optimism_dtype": getattr(trainer, "optimism_dtype", None),
            "optimism_chunk": getattr(trainer, "optimism_chunk", None),
        },
        "success_rate": solved_by_k[-1],
        "auc": float(np.mean(solved_by_k)),
        "final_eval_success_rate": None if final_eval is None else 100.0 * float(final_eval),
        "success_string": "".join("1" if i in first_success else "0" for i in range(n)),
        "replans_to_success": [first_success.get(i) for i in range(n)],
        "solved_by_k": solved_by_k,
        "wall_s": round(float(wall_s), 1),
        "code_sha256": code_fingerprint(),
        "environment": environment_record(),
    }
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"RUN SUMMARY {summary['run_name']}: success_rate {summary['success_rate']:.1f} auc {summary['auc']:.1f} "
          f"code_sha256 {summary['code_sha256'][:12]}", flush=True)
    return summary
