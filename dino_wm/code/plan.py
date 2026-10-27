import os
import gym
import json
import time
import hydra
import random
import torch
import pickle
import wandb
import logging
import warnings
import numpy as np
import submitit
from itertools import product
from pathlib import Path
from einops import rearrange
from omegaconf import OmegaConf, open_dict

from env.venv import SubprocVectorEnv
from custom_resolvers import replace_slash
from preprocessor import Preprocessor
from planning.evaluator import PlanEvaluator
from planning.image_corruption import corrupt_obs_dict
from utils import cfg_to_dict, seed
from run_summary import write_summary

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)


ALL_MODEL_KEYS = [
    "encoder",
    "predictor",
    "decoder",
    "proprio_encoder",
    "action_encoder",
]

# Hard-goal sampling for PointMaze-Medium (goal_source=hard): start/goal drawn
# from free cells with a minimum separation, velocities sampled uniformly.
POINT_MAZE_MEDIUM_HARD_ALL_CELLS = np.array(
    [
        (1, 1), (1, 2), (1, 5), (1, 6),
        (2, 1), (2, 2), (2, 4), (2, 5), (2, 6),
        (3, 2), (3, 3), (3, 4),
        (4, 1), (4, 2), (4, 4), (4, 5), (4, 6),
        (5, 1), (5, 3), (5, 4), (5, 6),
        (6, 1), (6, 2), (6, 3), (6, 5), (6, 6),
    ],
    dtype=np.float32,
)
POINT_MAZE_MEDIUM_HARD_MIN_DIST = 3.0
POINT_MAZE_QVEL_RANGE = np.array(
    [[-5.2262554, 5.2262554], [-5.2262554, 5.2262554]], dtype=np.float32
)

def planning_main_in_dir(working_dir, cfg_dict):
    os.chdir(working_dir)
    return planning_main(cfg_dict=cfg_dict)

def launch_plan_jobs(
    epoch,
    cfg_dicts,
    plan_output_dir,
):
    with submitit.helpers.clean_env():
        jobs = []
        for cfg_dict in cfg_dicts:
            subdir_name = f"{cfg_dict['planner']['name']}_goal_source={cfg_dict['goal_source']}_goal_H={cfg_dict['goal_H']}_alpha={cfg_dict['objective']['alpha']}"
            subdir_path = os.path.join(plan_output_dir, subdir_name)
            executor = submitit.AutoExecutor(
                folder=subdir_path, slurm_max_num_timeout=20
            )
            executor.update_parameters(
                **{
                    k: v
                    for k, v in cfg_dict["hydra"]["launcher"].items()
                    if k != "submitit_folder"
                }
            )
            cfg_dict["saved_folder"] = subdir_path
            cfg_dict["wandb_logging"] = False  # don't init wandb
            job = executor.submit(planning_main_in_dir, subdir_path, cfg_dict)
            jobs.append((epoch, subdir_name, job))
            print(
                f"Submitted evaluation job for checkpoint: {subdir_path}, job id: {job.job_id}"
            )
        return jobs


def build_plan_cfg_dicts(
    plan_cfg_path="",
    ckpt_base_path="",
    model_name="",
    model_epoch="final",
    planner=["gd", "cem"],
    goal_source=["dset"],
    goal_H=[1, 5, 10],
    alpha=[0, 0.1, 1],
):
    """
    Return a list of plan overrides, for model_path, add a key in the dict {"model_path": model_path}.
    """
    config_path = os.path.dirname(plan_cfg_path)
    overrides = [
        {
            "planner": p,
            "goal_source": g_source,
            "goal_H": g_H,
            "ckpt_base_path": ckpt_base_path,
            "model_name": model_name,
            "model_epoch": model_epoch,
            "objective": {"alpha": a},
        }
        for p, g_source, g_H, a in product(planner, goal_source, goal_H, alpha)
    ]
    cfg = OmegaConf.load(plan_cfg_path)
    cfg_dicts = []
    for override_args in overrides:
        planner = override_args["planner"]
        planner_cfg = OmegaConf.load(
            os.path.join(config_path, f"planner/{planner}.yaml")
        )
        cfg["planner"] = OmegaConf.merge(cfg.get("planner", {}), planner_cfg)
        override_args.pop("planner")
        cfg = OmegaConf.merge(cfg, OmegaConf.create(override_args))
        cfg_dict = OmegaConf.to_container(cfg)
        cfg_dict["planner"]["horizon"] = cfg_dict["goal_H"]  # assume planning horizon equals to goal horizon
        cfg_dicts.append(cfg_dict)
    return cfg_dicts


class PlanWorkspace:
    def __init__(
        self,
        cfg_dict: dict,
        wm: torch.nn.Module,
        dset,
        env: SubprocVectorEnv,
        env_name: str,
        frameskip: int,
        wandb_run: wandb.run,
    ):
        self.cfg_dict = cfg_dict
        self.wm = wm
        self.dset = dset
        self.env = env
        self.env_name = env_name
        self.frameskip = frameskip
        self.wandb_run = wandb_run
        self.device = next(wm.parameters()).device

        # have different seeds for each planning instances
        self.eval_seed = [cfg_dict["seed"] * n + 1 for n in range(cfg_dict["n_evals"])]
        print("eval_seed: ", self.eval_seed)
        self.n_evals = cfg_dict["n_evals"]
        self.goal_source = cfg_dict["goal_source"]
        self.goal_H = cfg_dict["goal_H"]
        self.action_dim = self.dset.action_dim * self.frameskip
        self.debug_dset_init = cfg_dict["debug_dset_init"]

        objective_fn = hydra.utils.call(
            cfg_dict["objective"],
        )

        self.data_preprocessor = Preprocessor(
            action_mean=self.dset.action_mean,
            action_std=self.dset.action_std,
            state_mean=self.dset.state_mean,
            state_std=self.dset.state_std,
            proprio_mean=self.dset.proprio_mean,
            proprio_std=self.dset.proprio_std,
            transform=self.dset.transform,
        )

        if self.cfg_dict["goal_source"] == "file":
            self.prepare_targets_from_file(cfg_dict["goal_file_path"])
        elif self.cfg_dict["goal_source"] == "segments":
            self.prepare_targets_from_segments(cfg_dict["eval_data_path"])
        else:
            self.prepare_targets()
        self._corrupt_start_goal()

        self.evaluator = PlanEvaluator(
            obs_0=self.obs_0,
            obs_g=self.obs_g,
            state_0=self.state_0,
            state_g=self.state_g,
            env=self.env,
            wm=self.wm,
            frameskip=self.frameskip,
            seed=self.eval_seed,
            preprocessor=self.data_preprocessor,
            n_plot_samples=self.cfg_dict["n_plot_samples"],
            decode_for_viz=self.cfg_dict.get("decode_for_viz", True),
            ood_corruption=self.cfg_dict.get("ood_corruption", None),
            ood_level=self.cfg_dict.get("ood_level", None),
        )

        if self.wandb_run is None or isinstance(
            self.wandb_run, wandb.sdk.lib.disabled.RunDisabled
        ):
            self.wandb_run = DummyWandbRun()

        self.goal_H_model_steps = self.goal_H // self.frameskip
        self.log_filename = "logs.json"  # planner and final eval logs are dumped here
        planner_cfg = self.cfg_dict["planner"].copy()
        if "n_taken_actions" in planner_cfg:
            planner_cfg["n_taken_actions"] = planner_cfg["n_taken_actions"] // self.frameskip
        if "sub_planner" in planner_cfg and "horizon" in planner_cfg["sub_planner"]:
            planner_cfg["sub_planner"]["horizon"] = (
                planner_cfg["sub_planner"]["horizon"] // self.frameskip
            )
        elif "horizon" in planner_cfg:
            planner_cfg["horizon"] = planner_cfg["horizon"] // self.frameskip
        self.planner = hydra.utils.instantiate(
            planner_cfg,
            wm=self.wm,
            env=self.env,  # only for mpc
            action_dim=self.action_dim,
            objective_fn=objective_fn,
            preprocessor=self.data_preprocessor,
            evaluator=self.evaluator,
            wandb_run=self.wandb_run,
            log_filename=self.log_filename,
        )


        self.planner.horizon = self.goal_H_model_steps
        self.planner.run_seed = cfg_dict["seed"]  # per-episode reseeding (OpTTAMPCPlanner._reseed_episode)

        from planning.optta_mpc import OpTTAMPCPlanner

        # viz tag color: red for the adaptation planner, blue otherwise
        self.evaluator.viz_tag_color = (
            (214, 58, 46)
            if isinstance(self.planner, OpTTAMPCPlanner)
            else (44, 96, 206)
        )

        self.dump_targets()

    def prepare_targets(self):
        states = []
        actions = []
        observations = []

        if self.goal_source == "maze_dist":
            # Diverse maze (multi-layout): BFS distance-controlled start/goal
            # states were pre-sampled in planning_main. Each sub-env already has
            # its own maze_spec, so just render the agent at start -> obs_0 and
            # at goal -> obs_g.
            dm = self.cfg_dict["_dm_goals"]
            init_state = dm["start_states"]  # (n_evals, 4)
            goal_state = dm["goal_states"]   # (n_evals, 4)

            obs_0, _ = self.env.prepare(self.eval_seed, init_state)
            obs_g, _ = self.env.prepare(self.eval_seed, goal_state)

            for k in obs_0.keys():
                obs_0[k] = np.expand_dims(obs_0[k], axis=1)
                obs_g[k] = np.expand_dims(obs_g[k], axis=1)

            self.obs_0 = obs_0
            self.obs_g = obs_g
            self.state_0 = init_state
            self.state_g = goal_state
            self.gt_actions = None
            return

        if self.goal_source in {"random_state", "hard"}:
            # update env config from val trajs
            observations, states, actions, env_info = (
                self.sample_traj_segment_from_dset(traj_len=2)
            )
            self.env.update_env(env_info)

            # sample random states
            rand_init_state, rand_goal_state = self._sample_init_goal_states()
            if self.env_name == "deformable_env": # take rand init state from dset for deformable envs
                rand_init_state = np.array([x[0] for x in states])

            obs_0, state_0 = self.env.prepare(self.eval_seed, rand_init_state)
            obs_g, state_g = self.env.prepare(self.eval_seed, rand_goal_state)

            # add dim for t
            for k in obs_0.keys():
                obs_0[k] = np.expand_dims(obs_0[k], axis=1)
                obs_g[k] = np.expand_dims(obs_g[k], axis=1)

            self.obs_0 = obs_0
            self.obs_g = obs_g
            self.state_0 = rand_init_state  # (b, d)
            self.state_g = rand_goal_state
            self.gt_actions = None
        else:
            # update env config from val trajs
            observations, states, actions, env_info = (
                self.sample_traj_segment_from_dset(traj_len=self.goal_H + 1)
            )
            self.env.update_env(env_info)

            # get states from val trajs
            init_state = [x[0] for x in states]
            init_state = np.array(init_state)
            actions = torch.stack(actions)
            if self.goal_source == "random_action":
                actions = torch.randn_like(actions)
            wm_actions = rearrange(actions, "b (t f) d -> b t (f d)", f=self.frameskip)
            exec_actions = self.data_preprocessor.denormalize_actions(actions)
            # replay actions in env to get gt obses
            rollout_obses, rollout_states = self.env.rollout(
                self.eval_seed, init_state, exec_actions.numpy()
            )
            self.obs_0 = {
                key: np.expand_dims(arr[:, 0], axis=1)
                for key, arr in rollout_obses.items()
            }
            self.obs_g = {
                key: np.expand_dims(arr[:, -1], axis=1)
                for key, arr in rollout_obses.items()
            }
            self.state_0 = init_state  # (b, d)
            self.state_g = rollout_states[:, -1]  # (b, d)
            self.gt_actions = wm_actions

    def sample_traj_segment_from_dset(self, traj_len):
        states = []
        actions = []
        observations = []
        env_info = []

        # Check if any trajectory is long enough
        valid_traj = [
            self.dset[i][0]["visual"].shape[0]
            for i in range(len(self.dset))
            if self.dset[i][0]["visual"].shape[0] >= traj_len
        ]
        if len(valid_traj) == 0:
            raise ValueError("No trajectory in the dataset is long enough.")

        # sample init_states from dset
        for i in range(self.n_evals):
            max_offset = -1
            while max_offset < 0:  # filter out traj that are not long enough
                traj_id = random.randint(0, len(self.dset) - 1)
                obs, act, state, e_info = self.dset[traj_id]
                max_offset = obs["visual"].shape[0] - traj_len
            state = state.numpy()
            offset = random.randint(0, max_offset)
            obs = {
                key: arr[offset : offset + traj_len]
                for key, arr in obs.items()
            }
            state = state[offset : offset + traj_len]
            act = act[offset : offset + self.goal_H]
            actions.append(act)
            states.append(state)
            observations.append(obs)
            env_info.append(e_info)
        return observations, states, actions, env_info

    def _sample_init_goal_states(self):
        if self.goal_source == "random_state":
            return self.env.sample_random_init_goal_states(self.eval_seed)
        if self.goal_source == "hard":
            if self.env_name == "point_maze_medium":
                return self._sample_point_maze_hard_states()
            raise ValueError(
                f"goal_source='hard' is unsupported for env.name='{self.env_name}'."
            )
        raise ValueError(f"Unknown goal_source: {self.goal_source}")

    def _corrupt_start_goal(self):
        """Apply the image corruption to the start and goal observations once, for every goal source, so the goal
        image is fixed for the whole episode. Stochastic noise (snp) comes from an RNG keyed by the run seed;
        deterministic corruptions ignore it."""
        name = self.cfg_dict.get("ood_corruption", None)
        if not name:
            return
        level = self.cfg_dict.get("ood_level", None)
        rng = np.random.RandomState([int(self.cfg_dict["seed"]), 1])
        self.obs_0 = corrupt_obs_dict(self.obs_0, name, level, rng=rng)
        self.obs_g = corrupt_obs_dict(self.obs_g, name, level, rng=rng)
        print(f"start/goal corruption: {name} level={level} applied once to {self.obs_g['visual'].shape[0]} episodes "
              f"(goal_source={self.goal_source})", flush=True)

    def _sample_point_maze_hard_states(self):
        init_states = []
        goal_states = []
        # One RNG per (run seed, episode index), so episode sets are independent across run seeds
        # (seeding with eval_seed = seed*n + 1 made seeds 1-5 share episodes).
        for i in range(self.n_evals):
            rs = np.random.RandomState([int(self.cfg_dict["seed"]), i])
            rs.random()  # parity with umaze's start/goal flip draw
            start, goal = self._sample_point_maze_distant_cell_states(rs)
            init_states.append(start)
            goal_states.append(goal)
        return np.stack(init_states), np.stack(goal_states)

    def _sample_point_maze_distant_cell_states(self, rs):
        """Sample a start/goal pair from all hard cells with minimum distance."""
        cells = POINT_MAZE_MEDIUM_HARD_ALL_CELLS
        while True:
            i, j = rs.choice(len(cells), size=2, replace=False)
            if np.linalg.norm(cells[i] - cells[j]) >= POINT_MAZE_MEDIUM_HARD_MIN_DIST:
                break
        start = self._sample_point_maze_cell_state(rs, cells[i : i + 1])
        goal = self._sample_point_maze_cell_state(rs, cells[j : j + 1])
        return start, goal

    def _sample_point_maze_cell_state(self, rs, cells):
        loc = cells[rs.randint(len(cells))]
        qpos = loc + np.asarray(rs.uniform(low=-0.25, high=0.25, size=2), dtype=np.float32)
        qvel = np.array(
            [
                rs.uniform(low=POINT_MAZE_QVEL_RANGE[0][0], high=POINT_MAZE_QVEL_RANGE[0][1]),
                rs.uniform(low=POINT_MAZE_QVEL_RANGE[1][0], high=POINT_MAZE_QVEL_RANGE[1][1]),
            ],
            dtype=np.float32,
        )
        return np.concatenate([qpos, qvel], axis=0)

    def _apply_color_overrides(self, env_info):
        """Force eval-time color overrides for block, agent, and goal into
        per-sample env_info (cfg key env_kwargs_override.{color,agent_color,goal_color})."""
        ekw = self.cfg_dict.get("env_kwargs_override", None) or {}
        block_color = ekw.get("color")
        agent_color = ekw.get("agent_color")
        goal_color = ekw.get("goal_color")
        if not block_color and not agent_color and not goal_color:
            return
        for ei in env_info:
            if block_color:
                ei["color"] = block_color
            if agent_color:
                ei["agent_color"] = agent_color
            if goal_color:
                ei["goal_color"] = goal_color

    def prepare_targets_from_segments(self, file_path):
        """Load pre-sampled segments (init state + actions), replay in env to get observations."""
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        segments = data["segments"]
        self.goal_H = data["traj_len"]

        rng = random.Random(self.cfg_dict["seed"])
        chosen = rng.sample(segments, min(self.n_evals, len(segments)))

        # Pad 5-dim states to 7-dim (add zero velocities) if needed
        raw_init = np.array([s["states"][0] for s in chosen])
        if raw_init.shape[-1] == 5:
            init_states = np.zeros((len(chosen), 7), dtype=np.float32)
            init_states[:, :5] = raw_init
        else:
            init_states = raw_init
        seg_actions = torch.stack([torch.tensor(s["actions"], dtype=torch.float32) for s in chosen])

        env_info = [{"shape": s["shape"]} if "shape" in s else {} for s in chosen]
        self._apply_color_overrides(env_info)
        self.env.update_env(env_info)

        # PushT/PushObj store actions in pixel-space.
        action_scale = 100.0 if self.env_name in ("pusht", "pushobj") else 1.0
        env_actions = seg_actions.numpy() / action_scale
        rollout_obses, rollout_states = self.env.rollout(
            self.eval_seed, init_states, env_actions
        )
        # start/goal image corruption is applied once in _corrupt_start_goal, for every goal source
        self.obs_0 = {
            key: np.expand_dims(arr[:, 0], axis=1)
            for key, arr in rollout_obses.items()
        }
        self.obs_g = {
            key: np.expand_dims(arr[:, -1], axis=1)
            for key, arr in rollout_obses.items()
        }
        self.state_0 = init_states
        self.state_g = rollout_states[:, -1]
        # stored = denorm * action_scale, denorm = norm * std + mean
        denorm = seg_actions / action_scale
        normed = (denorm - self.data_preprocessor.action_mean) / self.data_preprocessor.action_std
        if normed.shape[1] % self.frameskip == 0:
            self.gt_actions = rearrange(normed, "b (t f) d -> b t (f d)", f=self.frameskip)
        else:
            self.gt_actions = None

    def prepare_targets_from_file(self, file_path):
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        self.obs_0 = data["obs_0"]
        self.obs_g = data["obs_g"]
        self.state_0 = data["state_0"]
        self.state_g = data["state_g"]
        self.gt_actions = data["gt_actions"]
        self.goal_H = data["goal_H"]

    def dump_targets(self):
        with open("plan_targets.pkl", "wb") as f:
            pickle.dump(
                {
                    "obs_0": self.obs_0,
                    "obs_g": self.obs_g,
                    "state_0": self.state_0,
                    "state_g": self.state_g,
                    "gt_actions": self.gt_actions,
                    "goal_H": self.goal_H,
                },
                f,
            )
        file_path = os.path.abspath("plan_targets.pkl")
        print(f"Dumped plan targets to {file_path}")

    def perform_planning(self):
        if self.debug_dset_init:
            actions_init = self.gt_actions
        else:
            actions_init = None
        actions, action_len = self.planner.plan(
            obs_0=self.obs_0,
            obs_g=self.obs_g,
            actions=actions_init,
        )
        logs, successes, _, _ = self.evaluator.eval_actions(
            actions.detach(), action_len, save_video=True, filename="output_final"
        )
        logs = {f"final_eval/{k}": v for k, v in logs.items()}
        self.wandb_run.log(logs)
        logs_entry = {
            key: (
                value.item()
                if isinstance(value, (np.float32, np.int32, np.int64))
                else value
            )
            for key, value in logs.items()
        }
        with open(self.log_filename, "a") as file:
            file.write(json.dumps(logs_entry) + "\n")
        return logs


def load_ckpt(snapshot_path, device):
    from models.dino import DinoV2Encoder
    _ = DinoV2Encoder('dinov2_vits14', 'x_norm_patchtokens')
    with snapshot_path.open("rb") as f:
        payload = torch.load(f, map_location=device, weights_only=False)
    loaded_keys = []
    result = {}
    for k, v in payload.items():
        if k in ALL_MODEL_KEYS:
            loaded_keys.append(k)
            result[k] = v.to(device)
    result["epoch"] = payload["epoch"]
    return result


def load_model(model_ckpt, train_cfg, num_action_repeat, device):
    result = {}
    if model_ckpt.exists():
        result = load_ckpt(model_ckpt, device)
        print(f"Resuming from epoch {result['epoch']}: {model_ckpt}")

    if "encoder" not in result:
        result["encoder"] = hydra.utils.instantiate(
            train_cfg.encoder,
        )
    if "predictor" not in result:
        raise ValueError("Predictor not found in model checkpoint")

    if train_cfg.has_decoder and "decoder" not in result:
        base_path = os.path.dirname(os.path.abspath(__file__))
        if train_cfg.env.decoder_path is not None:
            decoder_path = os.path.join(base_path, train_cfg.env.decoder_path)
            ckpt = torch.load(decoder_path, weights_only=False)
            if isinstance(ckpt, dict):
                result["decoder"] = ckpt["decoder"]
            else:
                result["decoder"] = torch.load(decoder_path, weights_only=False)
        elif not train_cfg.model.get("train_decoder", True):
            # Run trained with model.train_decoder=false: checkpoint legitimately
            # has no decoder weights. Load without one (use decode_for_viz=false).
            print(
                "WARNING: model.train_decoder=false and no decoder in checkpoint; "
                "loading model WITHOUT decoder (set decode_for_viz=false)."
            )
            result["decoder"] = None
        else:
            raise ValueError(
                "Decoder path not found in model checkpoint \
                                and is not provided in config"
            )
    elif not train_cfg.has_decoder:
        result["decoder"] = None

    model = hydra.utils.instantiate(
        train_cfg.model,
        encoder=result["encoder"],
        proprio_encoder=result["proprio_encoder"],
        action_encoder=result["action_encoder"],
        predictor=result["predictor"],
        decoder=result["decoder"],
        proprio_dim=train_cfg.proprio_emb_dim,
        action_dim=train_cfg.action_emb_dim,
        concat_dim=train_cfg.concat_dim,
        num_action_repeat=num_action_repeat,
        num_proprio_repeat=train_cfg.num_proprio_repeat,
    )
    model.to(device)
    return model


class DummyWandbRun:
    def __init__(self):
        self.mode = "disabled"

    def log(self, *args, **kwargs):
        pass

    def watch(self, *args, **kwargs):
        pass

    def config(self, *args, **kwargs):
        pass

    def finish(self):
        pass


def planning_main(cfg_dict):
    t_start = time.perf_counter()
    t_after_model = None
    t_after_workspace = None
    t_after_planning = None

    output_dir = cfg_dict["saved_folder"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if cfg_dict["wandb_logging"]:
        wandb_run = wandb.init(
            project=f"plan_{cfg_dict['planner']['name']}", config=cfg_dict
        )
        wandb.run.name = "{}".format(output_dir.split("plan_outputs/")[-1])
    else:
        wandb_run = None

    ckpt_base_path = cfg_dict["ckpt_base_path"]
    model_name = cfg_dict.get("model_name")
    if ckpt_base_path.startswith("/"):
        model_path = ckpt_base_path
    else:
        model_path = f"{ckpt_base_path}/{cfg_dict['model_name']}/"
    model_path = os.path.abspath(model_path)
    with open(os.path.join(model_path, "hydra.yaml"), "r") as f:
        model_cfg = OmegaConf.load(f)
    
    seed(cfg_dict["seed"])
    if cfg_dict.get("goal_source") == "segments":
        # Segment targets carry states+actions; build a stats-only dset stub so
        # no training data is needed at eval time.
        from types import SimpleNamespace
        from datasets.img_transforms import default_transform
        dataset_transform_cfg = model_cfg.env.dataset.get("transform", None)
        transform = (
            hydra.utils.instantiate(dataset_transform_cfg)
            if dataset_transform_cfg is not None
            else default_transform()
        )
        if model_cfg.env.name not in ("pusht", "pushobj"):
            raise ValueError(f"segments mode not supported for env '{model_cfg.env.name}'")
        from datasets.pusht_dset import (
            ACTION_MEAN, ACTION_STD, STATE_MEAN, STATE_STD, PROPRIO_MEAN, PROPRIO_STD,
        )
        dset = SimpleNamespace(
            action_dim=2,
            action_mean=ACTION_MEAN, action_std=ACTION_STD,
            state_mean=STATE_MEAN[:5], state_std=STATE_STD[:5],
            proprio_mean=PROPRIO_MEAN[:4], proprio_std=PROPRIO_STD[:4],
            transform=transform,
        )
    else:
        # Allow the plan config to override the eval dataset path / obs layout
        # (e.g. checkpoints trained against a different data root).
        if cfg_dict.get("eval_data_path"):
            model_cfg.env.dataset.data_path = cfg_dict["eval_data_path"]
        if cfg_dict.get("eval_use_frame_files") is not None:
            model_cfg.env.dataset.use_frame_files = bool(cfg_dict["eval_use_frame_files"])
        _, dset = hydra.utils.call(
            model_cfg.env.dataset,
            num_hist=model_cfg.num_hist,
            num_pred=model_cfg.num_pred,
            frameskip=model_cfg.frameskip,
        )
        dset = dset["valid"]

    num_action_repeat = model_cfg.num_action_repeat
    model_ckpt = (
        Path(model_path) / "checkpoints" / f"model_{cfg_dict['model_epoch']}.pth"
    )
    model = load_model(model_ckpt, model_cfg, num_action_repeat, device=device)
    t_after_model = time.perf_counter()
    print(f"[timing] setup_model_s={t_after_model - t_start:.3f}", flush=True)

    # Allow plan config to inject/override env kwargs (e.g. eval-time color shifts).
    env_kwargs = dict(model_cfg.env.kwargs) if model_cfg.env.kwargs else {}
    if cfg_dict.get("env_kwargs_override"):
        env_kwargs.update(dict(cfg_dict["env_kwargs_override"]))

    if model_cfg.env.name == "diverse_maze" and cfg_dict.get("goal_source") == "maze_dist":
        # Multi-layout planning: BFS distance-controlled (start, goal) pairs on
        # held-out val mazes. Generate goals up front so each sub-env is built
        # with the correct per-episode maze_spec.
        from datasets.diverse_maze_goals import generate_diverse_maze_goals

        eval_root = cfg_dict.get("eval_data_path") or str(model_cfg.env.dataset.data_path)
        dm_goals = generate_diverse_maze_goals(
            val_data_path=os.path.join(eval_root, "val"),
            n_evals=cfg_dict["n_evals"],
            seed=cfg_dict["seed"],
            min_block_radius=cfg_dict.get("min_block_radius", 4),
            max_block_radius=cfg_dict.get("max_block_radius", 10000),
        )
        cfg_dict["_dm_goals"] = dm_goals
        bd = dm_goals["block_dists"]
        print(
            f"[diverse_maze] {len(dm_goals['map_specs'])} goals over "
            f"{len(set(dm_goals['map_indices'].tolist()))} val maps | "
            f"block_dist min/mean/max={bd.min()}/{bd.mean():.1f}/{bd.max()}",
            flush=True,
        )
        env = SubprocVectorEnv(
            [
                (
                    lambda spec=dm_goals["map_specs"][i], kw=env_kwargs: gym.make(
                        "diverse_maze", maze_spec=spec, **kw
                    )
                )
                for i in range(cfg_dict["n_evals"])
            ]
        )
    # use dummy vector env for wall and deformable envs
    elif model_cfg.env.name == "wall" or model_cfg.env.name == "deformable_env":
        from env.serial_vector_env import SerialVectorEnv
        env = SerialVectorEnv(
            [
                gym.make(
                    model_cfg.env.name, *model_cfg.env.args, **env_kwargs
                )
                for _ in range(cfg_dict["n_evals"])
            ]
        )
    else:
        env = SubprocVectorEnv(
            [
                lambda: gym.make(
                    model_cfg.env.name, *model_cfg.env.args, **env_kwargs
                )
                for _ in range(cfg_dict["n_evals"])
            ]
        )

    plan_workspace = PlanWorkspace(
        cfg_dict=cfg_dict,
        wm=model,
        dset=dset,
        env=env,
        env_name=model_cfg.env.name,
        frameskip=model_cfg.frameskip,
        wandb_run=wandb_run,
    )
    t_after_workspace = time.perf_counter()
    print(f"[timing] setup_workspace_s={t_after_workspace - t_after_model:.3f}", flush=True)

    logs = plan_workspace.perform_planning()
    t_after_planning = time.perf_counter()
    print(f"[timing] perform_planning_s={t_after_planning - t_after_workspace:.3f}", flush=True)
    print(f"[timing] total_planning_main_s={t_after_planning - t_start:.3f}", flush=True)
    write_summary(cfg_dict, plan_workspace, logs, t_after_planning - t_start)  # summary.json; reads results only
    return logs


@hydra.main(config_path="conf", config_name="plan_gd")
def main(cfg: OmegaConf):
    with open_dict(cfg):
        cfg["saved_folder"] = os.getcwd()
        log.info(f"Planning result saved dir: {cfg['saved_folder']}")
    cfg_dict = cfg_to_dict(cfg)
    cfg_dict["wandb_logging"] = bool(cfg_dict.get("wandb_logging", True))
    planning_main(cfg_dict)


if __name__ == "__main__":
    main()
