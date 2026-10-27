"""Evaluation driver.

    python -m lewm_optta.eval task=pusht arm=optta seed=1 [key=value ...]

Writes results/<run_name>/: config.yaml, run.log, episodes.jsonl (per-episode outcome, hashes and
adaptation records), adapt_losses.csv, summary.json (success rate, solved_by_k, code fingerprint,
environment).
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MUJOCO_GL", "egl")

import csv  # noqa: E402
import hashlib  # noqa: E402
import importlib.metadata  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import stable_pretraining as spt  # noqa: E402
import stable_worldmodel as swm  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from stable_worldmodel.solver import CEMSolver  # noqa: E402
from torchvision.transforms import v2 as transforms  # noqa: E402

from .adapter import OpTTAAdapter  # noqa: E402
from .config import load_config  # noqa: E402
from .corruption import wrap_transform  # noqa: E402
from .data import build_process, load_dataset, sample_episodes  # noqa: E402
from .determinism import pin  # noqa: E402
from .objective import CostModel, Objective  # noqa: E402
from .policy import OpTTAPolicy  # noqa: E402

log = logging.getLogger("lewm_optta")

ADAPT_FIELDS = [
    "episode", "replan", "k", "adapt_step", "n_frames", "pre_new_loss", "pre_loss_pred", "loss_pred",
    "loss_opt", "alpha", "n_pred", "n_opt", "rho_eff", "loss_total", "post_loss_pred", "enc_delta",
]
PACKAGES = ("torch", "stable-worldmodel", "stable-pretraining", "transformers", "numpy",
            "gymnasium", "pymunk", "mujoco")


class SeededWorld(swm.World):
    """swm.World whose dataset-driven reset gets a per-episode seed.

    The datasets have no seed column, so swm resets from OS entropy; for PushT that changes the order in
    which pymunk resolves contacts and makes episodes differ across processes.
    """

    episode_seed = None

    def reset(self, seed=None, options=None) -> None:
        if seed is None and self.episode_seed is not None:
            seed = int(self.episode_seed)
        return super().reset(seed=seed, options=options)


def code_fingerprint() -> str:
    """sha256 over the package's .py and .yaml files, in name order."""
    h = hashlib.sha256()
    pkg = Path(__file__).resolve().parent
    for p in sorted(pkg.glob("*.py")) + sorted(pkg.glob("*.yaml")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def environment_record(device: str) -> dict:
    versions = {}
    for dist in PACKAGES:
        try:
            versions[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            versions[dist] = None
    gpu = torch.cuda.get_device_name(torch.device(device)) if device.startswith("cuda") else None
    return {"packages": versions, "cuda": torch.version.cuda, "gpu": gpu, "host": platform.node(),
            "python": platform.python_version()}


def setup_logging(out_dir: Path) -> None:
    root = logging.getLogger("lewm_optta")
    root.setLevel(logging.INFO)
    root.propagate = False
    fmt = logging.Formatter("%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    for h in list(root.handlers):
        root.removeHandler(h)
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(out_dir / "run.log", mode="w")):
        h.setFormatter(fmt)
        root.addHandler(h)


def build_transform(cfg) -> dict:
    base = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=int(cfg.data.img_size)),
    ])
    tf = {"pixels": base, "goal": base}
    c = cfg.corruption
    if c.name:
        reported: set = set()
        # The goal image uses corruption.goal_mode when set (episode = one noise draw per episode).
        modes = {k: (str(c.goal_mode) if (k == "goal" and c.goal_mode) else str(c.mode)) for k in tf}
        tf = {k: wrap_transform(v, k, str(c.name), c.level, int(c.seed), modes[k], reported)
              for k, v in tf.items()}
    return tf


def main(argv=None) -> None:
    cfg = load_config(argv)
    out_dir = Path(cfg.results_root) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir)
    OmegaConf.save(cfg, out_dir / "config.yaml")
    code_sha = code_fingerprint()
    log.info("run %s code_sha256=%s\n%s", cfg.run_name, code_sha, OmegaConf.to_yaml(cfg))

    pin(int(cfg.seed))

    device = str(cfg.device)
    budget = int(cfg.eval.eval_budget)
    goal_offset = int(cfg.eval.goal_offset_steps)
    if int(cfg.plan.horizon) * int(cfg.plan.action_block) > budget:
        raise SystemExit("plan.horizon * plan.action_block must be <= eval.eval_budget")

    env_kwargs = OmegaConf.to_container(cfg.world.env_kwargs, resolve=True) or {}
    world = SeededWorld(
        str(cfg.world.env_name), num_envs=1,
        image_shape=tuple(int(x) for x in cfg.world.image_shape),
        max_episode_steps=2 * budget, **env_kwargs,
    )
    if env_kwargs:
        log.info("env kwargs: %s", env_kwargs)
    dataset = load_dataset(cfg)
    process = build_process(dataset, list(cfg.data.keys_to_cache))
    transform = build_transform(cfg)

    model = swm.wm.utils.load_pretrained(str(cfg.model.name)).to(device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    objective = Objective(float(cfg.plan.objective_base))
    cost_model = CostModel(model, objective)
    solver = CEMSolver(
        cost_model,
        batch_size=int(cfg.cem.batch_size), num_samples=int(cfg.cem.num_samples),
        var_scale=float(cfg.cem.var_scale), n_steps=int(cfg.cem.n_steps), topk=int(cfg.cem.topk),
        device=device, seed=int(cfg.seed),
    )
    plan_cfg = swm.PlanConfig(
        horizon=int(cfg.plan.horizon), receding_horizon=int(cfg.plan.receding_horizon),
        action_block=int(cfg.plan.action_block), warm_start=bool(cfg.plan.warm_start),
    )
    adapter = OpTTAAdapter(model, cost_model, cfg.adapt) if cfg.adapt.enabled else None
    policy = OpTTAPolicy(solver, plan_cfg, process, transform, cost_model, adapter,
                           base_seed=int(cfg.seed))
    world.set_policy(policy)
    log.info("task=%s arm=%s cell=%s horizon=%d rh=%d budget=%d (%d replans max) cem=%d/%d/%d adapter=%s",
             cfg.task, cfg.arm, cfg.cell, plan_cfg.horizon, plan_cfg.receding_horizon, budget,
             budget // (plan_cfg.receding_horizon * plan_cfg.action_block),
             int(cfg.cem.num_samples), int(cfg.cem.topk), int(cfg.cem.n_steps), "on" if adapter else "off")

    episodes, starts, n_valid = sample_episodes(dataset, int(cfg.eval.num_eval), goal_offset, int(cfg.seed))
    log.info("%d valid starting points; sampled %d episodes", n_valid, len(episodes))
    callables = OmegaConf.to_container(cfg.eval.callables, resolve=True)

    successes, n_replans_list, adapt_rows = [], [], []
    t_start = time.time()
    with open(out_dir / "episodes.jsonl", "w") as f:
        for i, (ep_idx, start) in enumerate(zip(episodes, starts)):
            t0 = time.time()
            ep = policy.begin_episode(i)
            world.episode_seed = policy.episode_seed(i)
            metrics = world.evaluate(
                dataset=dataset, start_steps=[start], goal_offset=goal_offset,
                eval_budget=budget, episodes_idx=[ep_idx], callables=callables, video=None,
            )
            success = bool(np.asarray(metrics["episode_successes"]).ravel()[0])
            successes.append(success)
            n_replans_list.append(len(ep.plan_hashes))
            adapt_rows.extend(ep.adapt_records)
            rec = {
                "index": i, "episode_idx": ep_idx, "start_step": start, "success": success,
                "n_replans": len(ep.plan_hashes),
                "replans_to_success": len(ep.plan_hashes) if success else None,
                "adapt_calls": len({r["replan"] for r in ep.adapt_records}),
                "plan_hashes": ep.plan_hashes, "frame_hashes": ep.frame_hashes,
                "track": ep.track, "adapt_records": ep.adapt_records,
                "wall_s": round(time.time() - t0, 1),
            }
            f.write(json.dumps(rec) + "\n")
            f.flush()
            log.info("episode %d/%d idx=%d start=%d success=%s replans=%d adapt_calls=%d %.0fs running=%.1f%%",
                     i + 1, len(episodes), ep_idx, start, success, rec["n_replans"], rec["adapt_calls"],
                     rec["wall_s"], 100.0 * np.mean(successes))
    world.close()

    with open(out_dir / "adapt_losses.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ADAPT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in adapt_rows:
            writer.writerow(row)

    k_max = max(1, budget // (plan_cfg.receding_horizon * plan_cfg.action_block))
    solved_by_k = [
        100.0 * sum(1 for s, n in zip(successes, n_replans_list) if s and n <= k) / len(successes)
        for k in range(1, k_max + 1)
    ]
    summary = {
        "run_name": cfg.run_name, "task": cfg.task, "arm": cfg.arm, "cell": cfg.cell, "seed": int(cfg.seed),
        "n_episodes": len(successes), "success_rate": 100.0 * float(np.mean(successes)),
        "success_string": "".join("1" if s else "0" for s in successes),
        "solved_by_k": solved_by_k, "n_adapt_steps": len(adapt_rows),
        "wall_s": round(time.time() - t_start, 1),
        "code_sha256": code_sha, "environment": environment_record(device),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("DONE %s success_rate=%.1f (%d/%d) adapt_steps=%d wall=%.0fs",
             cfg.run_name, summary["success_rate"], sum(successes), len(successes),
             len(adapt_rows), summary["wall_s"])
    print(f"RESULT {cfg.run_name}: success_rate {summary['success_rate']:.1f}", flush=True)


if __name__ == "__main__":
    main()
