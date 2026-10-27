"""MPC planner with OpTTA: test-time adaptation of the world model between replans.

Samples are planned one at a time by the inherited MPC loop, each starting from
the pre-adaptation weights (independent adaptation). The adaptation plugs into
the loop via the _post_env_feedback hook: after each replan, the executed
environment segment is added to a replay buffer and the world model is
finetuned on it (see OpTTATrainer).
"""

import csv
import json
import os
import re
import time

import numpy as np
import torch

from utils import move_to_device

from .optta import OpTTATrainer
from .mpc import MPCPlanner, _cuda_sync


class OpTTAMPCPlanner(MPCPlanner):
    # adapt.* is read with dict.get(), so unknown keys are rejected rather than silently ignored.
    _ADAPT_KEYS = frozenset(
        {
            "lr", "steps", "optimizer", "finetune_every", "replay_buffer",
            "finetune_encoder", "encoder_lr", "last_layer_only", "encoder_last_layer_only",
            "optimism_ratio_target", "optimism_elites", "optimism_chunk", "optimism_dtype",
            "optimism_rho_schedule", "optimism_rho_schedule_steps",
        }
    )

    def __init__(self, adapt=None, **kwargs):
        super().__init__(**kwargs)
        adapt = dict(adapt) if adapt else {}
        unknown = set(adapt) - self._ADAPT_KEYS
        if unknown:
            raise ValueError(
                f"Unknown planner.adapt key(s): {sorted(unknown)}. "
                f"Valid keys: {sorted(self._ADAPT_KEYS)}"
            )
        self.optta_finetune_every = adapt.get("finetune_every", 1)
        self.replay_mode, self.replay_size = self._parse_replay_buffer(adapt.get("replay_buffer", False))
        # Optimism ramp length K. null (or absent) = the replan budget, so the ramp spans the episode
        # whatever max_iter is; an explicit value overrides.
        rho_K = adapt.get("optimism_rho_schedule_steps")
        if rho_K is None:
            if not np.isfinite(self.max_iter):
                raise ValueError("optimism_rho_schedule_steps=null needs a finite planner.max_iter")
            rho_K = int(self.max_iter)
        print(f"optimism ramp K={rho_K} (planner.max_iter={self.max_iter})", flush=True)
        self.optta_trainer = OpTTATrainer(
            wm=self.wm,
            lr=adapt.get("lr", 5e-4),
            steps=adapt.get("steps", 1),
            optimizer_name=adapt.get("optimizer", "adam"),
            finetune_encoder=adapt.get("finetune_encoder", True),
            last_layer_only=adapt.get("last_layer_only", True),
            encoder_lr=adapt.get("encoder_lr", None),
            encoder_last_layer_only=adapt.get("encoder_last_layer_only", True),
            optimism_ratio_target=adapt.get("optimism_ratio_target", 0.0),
            optimism_elites=adapt.get("optimism_elites", 0),
            optimism_chunk=adapt.get("optimism_chunk"),
            optimism_dtype=adapt.get("optimism_dtype", "float32"),
            optimism_rho_schedule=adapt.get("optimism_rho_schedule", "const"),
            optimism_rho_schedule_steps=rho_K,
            objective_fn=self.objective_fn,
        )
        self._optta_loss_records = []
        self._records = []
        self._sample_idx = 0
        self._obs_buffer, self._act_buffer, self._segment_scores = [], [], []

    @staticmethod
    def _parse_replay_buffer(cfg):
        """false -> keep all segments; 'recent<N>' keeps the last N;
        'hard<N>' keeps the N with the highest prediction loss."""
        if not cfg:
            return None, 0
        m = re.match(r"(recent|hard)(\d+)$", str(cfg))
        if not m:
            raise ValueError(
                f"Invalid adapt.replay_buffer: {cfg!r}; expected false, 'recent<N>', or 'hard<N>'"
            )
        return m.group(1), int(m.group(2))

    def plan(self, obs_0, obs_g, actions=None):
        """Plan one sample at a time, resetting the weights before each sample."""
        n_evals = obs_0["visual"].shape[0]
        all_actions, all_action_lens = [], []
        self._records = []

        for i in range(n_evals):
            self.optta_trainer.reset()
            self._reseed_episode(i)
            self._sample_idx = i
            self._obs_buffer, self._act_buffer, self._segment_scores = [], [], []
            self.plan_suffix = f"_s{i}"
            actions_i, action_len_i = self._plan_single(
                obs_0_i={k: v[i : i + 1] for k, v in obs_0.items()},
                obs_g_i={k: v[i : i + 1] for k, v in obs_g.items()},
                env_i=_EnvWorkerProxy(self.env, i),
                seed_i=[self.evaluator.seed[i]],
                state_0_i=self.evaluator.state_0[i : i + 1],
                state_g_i=self.evaluator.state_g[i : i + 1],
            )
            all_actions.append(actions_i)
            all_action_lens.append(action_len_i)

        self.plan_suffix = ""
        self.optta_trainer.reset()
        self._dump_per_iter_logs(self._records, n_evals)
        self._dump_optta_loss_csv()

        # Pad per-sample action sequences to a common length before concatenating;
        # action_len keeps the true number of actions per sample.
        max_T = max(a.shape[1] for a in all_actions)
        padded = []
        for a in all_actions:
            if a.shape[1] < max_T:
                pad = torch.zeros(
                    a.shape[0], max_T - a.shape[1], a.shape[2],
                    device=a.device, dtype=a.dtype,
                )
                a = torch.cat([a, pad], dim=1)
            padded.append(a)
        return torch.cat(padded, dim=0), np.concatenate(all_action_lens)

    def _reseed_episode(self, i):
        """Seed CEM noise and per-call image-corruption noise from (run seed, episode index), so
        episode i does not depend on how many random draws earlier episodes made, and every arm gets
        the same first plan. Dropout still draws from the global CUDA RNG."""
        s = (int(self.run_seed) * 1_000_003 + i) % (2**31)
        if hasattr(self.sub_planner, "reseed"):
            self.sub_planner.reseed(s)
        self.evaluator.reseed_noise(s + 7)
        print(f"episode {i}: episode_seed={s}", flush=True)

    def _plan_single(self, obs_0_i, obs_g_i, env_i, seed_i, state_0_i, state_g_i):
        """Point the shared evaluator at one sample, run the inherited MPC loop, restore."""
        saved = (
            self.evaluator.env,
            self.evaluator.seed,
            self.evaluator.obs_0,
            self.evaluator.obs_g,
            self.evaluator.state_0,
            self.evaluator.state_g,
        )
        self.evaluator.env = env_i
        self.evaluator.seed = seed_i
        self.evaluator.obs_0 = obs_0_i
        self.evaluator.obs_g = obs_g_i
        self.evaluator.state_0 = state_0_i
        self.evaluator.state_g = state_g_i
        try:
            return super().plan(obs_0_i, obs_g_i)
        finally:
            (
                self.evaluator.env,
                self.evaluator.seed,
                self.evaluator.obs_0,
                self.evaluator.obs_g,
                self.evaluator.state_0,
                self.evaluator.state_g,
            ) = saved

    def _build_plan_ctx(self, plan_obs_0, plan_actions, obs_g, plan_elites=None):
        """Package the just-finished plan for the optimism term.

        Mirrors what GDPlanner.plan does to its inputs (gd.py:105-110) so the
        imagined rollout is scored under exactly the conditions the controller
        optimized it under. Returns None when optimism is off, so AdaJEPA costs
        nothing extra.
        """
        if not self.optta_trainer.optimism_on or plan_obs_0 is None:
            return None
        return {
            "plan_obs_0": move_to_device(
                self.preprocessor.transform_obs(plan_obs_0), self.device
            ),
            "obs_g": move_to_device(
                self.preprocessor.transform_obs(obs_g), self.device
            ),
            "plan_actions": plan_actions.to(self.device),
            # CEM only. (n_evals, K, T, act_dim) final-iteration plans; None
            # under GD. Used when optimism_elites > 0 to average the optimism loss
            # over real plans instead of scoring the elite mean.
            "plan_elites": None if plan_elites is None else plan_elites.to(self.device),
            "step": self.iter,
        }

    def _post_env_feedback(self, taken_actions, e_obses, plan_obs_0=None,
                           plan_actions=None, obs_g=None, plan_elites=None):
        """Adapt the world model on the newly executed segment (MPC-loop hook)."""
        obs_seq, act_seq = self._extract_optta_data(e_obses, taken_actions, self.iter)
        self._obs_buffer.append(obs_seq)
        self._act_buffer.append(act_seq)
        self._segment_scores = self._trim_buffer(
            self._obs_buffer, self._act_buffer, self._segment_scores
        )
        rec = {
            "sample_idx": self._sample_idx,
            "step": self.iter + 1,
            "success": bool(self.is_success[0]),
            "t_plan_s": self.t_plan_s,
        }
        extra_logs = None
        if (self.iter + 1) % self.optta_finetune_every == 0:
            _cuda_sync()
            t0 = time.perf_counter()
            plan_ctx = self._build_plan_ctx(plan_obs_0, plan_actions, obs_g, plan_elites)
            self._segment_scores, last_loss, step_losses = self._finetune_with_buffer(
                self._obs_buffer, self._act_buffer, self._segment_scores, plan_ctx
            )
            _cuda_sync()
            rec["t_optta_s"] = time.perf_counter() - t0
            rec["optta/pred_loss"] = last_loss["pred"]
            self._record_optta_losses(self._sample_idx, self.iter + 1, step_losses)
            extra_logs = {"optta/pred_loss": last_loss["pred"]}
            if last_loss["optimism"] is not None:
                rec["optta/optimism_loss"] = last_loss["optimism"]
                extra_logs["optta/optimism_loss"] = last_loss["optimism"]
                # alpha_t*optimism relative to the grounding term: >>1 means
                # optimism dominates the update.
                extra_logs["optta/optimism_ratio"] = (
                    last_loss["alpha_t"] * last_loss["optimism"] / max(last_loss["pred"], 1e-12)
                )
                rec["optta/alpha_t"] = last_loss["alpha_t"]
                extra_logs["optta/alpha_t"] = last_loss["alpha_t"]
                extra_logs["optta/grad_pred_norm"] = last_loss["n_pred"]
                extra_logs["optta/grad_opt_norm"] = last_loss["n_opt"]
        self._records.append(rec)
        return extra_logs

    def _extract_optta_data(self, e_obses, taken_actions, iter_idx: int):
        """Slice this iteration's T*frameskip+1 env frames from the cumulative
        rollout at the model stride: T+1 frames aligned with the T taken actions."""
        fs = self.evaluator.frameskip
        T = taken_actions.shape[1]
        start = iter_idx * T * fs
        obs_chunk = {k: v[:, start : start + T * fs + 1 : fs] for k, v in e_obses.items()}
        return self.preprocessor.transform_obs(obs_chunk), taken_actions.cpu()

    def _trim_buffer(self, obs_buffer, act_buffer, scores):
        """Prune the replay buffer in place. Returns the updated scores list."""
        if not self.replay_mode or len(obs_buffer) <= self.replay_size:
            return scores
        if self.replay_mode == "recent" or not scores:
            del obs_buffer[: -self.replay_size]
            del act_buffer[: -self.replay_size]
            return scores[-self.replay_size :]
        # hard: keep top-N by previous scores; newest (unscored) segments always kept.
        scores = scores + [float("inf")] * (len(obs_buffer) - len(scores))
        top_idx = sorted(
            sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: self.replay_size]
        )
        obs_buffer[:] = [obs_buffer[i] for i in top_idx]
        act_buffer[:] = [act_buffer[i] for i in top_idx]
        return [scores[i] for i in top_idx]

    def _finetune_with_buffer(self, obs_buffer, act_buffer, scores, plan_ctx=None):
        """Finetune on the buffer. Returns (updated_scores, last_loss, step_losses).

        last_loss is a dict {"pred", "optimism", "total"}; "optimism" is None when
        the optimistic term is disabled.
        """
        # hard-buffer segments are non-contiguous, so they cannot be merged.
        merge = self.replay_mode != "hard"
        step_losses = self.optta_trainer.finetune(
            obs_buffer, act_buffer, merge=merge, plan_ctx=plan_ctx
        )
        last_loss = (
            step_losses[-1] if step_losses
            else {"pred": 0.0, "optimism": None, "total": 0.0}
        )
        if self.replay_mode == "hard":
            scores = self.optta_trainer.score_segments(obs_buffer, act_buffer)
        return scores, last_loss, step_losses

    def _record_optta_losses(self, sample_idx, mpc_iter, step_losses):
        for s, loss in enumerate(step_losses):
            self._optta_loss_records.append(
                {
                    "sample_idx": int(sample_idx),
                    "mpc_iter": int(mpc_iter),
                    "optta_step": int(s + 1),
                    "loss": float(loss["pred"]),
                    "optimism_loss": (
                        None if loss["optimism"] is None else float(loss["optimism"])
                    ),
                    "total_loss": float(loss["total"]),
                    "alpha_t": (None if loss.get("alpha_t") is None else float(loss["alpha_t"])),
                    "grad_pred_norm": (None if loss.get("n_pred") is None else float(loss["n_pred"])),
                    "grad_opt_norm": (None if loss.get("n_opt") is None else float(loss["n_opt"])),
                }
            )

    def _dump_optta_loss_csv(self):
        if not self._optta_loss_records or self.log_filename is None:
            return
        path = os.path.join(os.path.dirname(self.log_filename) or ".", "optta_losses.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "sample_idx", "mpc_iter", "optta_step",
                    "loss", "optimism_loss", "total_loss",
                    "alpha_t", "grad_pred_norm", "grad_opt_norm",
                ],
            )
            writer.writeheader()
            for rec in self._optta_loss_records:
                writer.writerow(rec)

    def _dump_per_iter_logs(self, records, n_evals):
        if self.log_filename is None or not records:
            return
        out_dir = os.path.dirname(self.log_filename) or "."
        with open(os.path.join(out_dir, "per_iter_logs.json"), "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        # Per-step success rate across samples (success carries forward once reached).
        success_step = {}
        for rec in records:
            if rec["success"] and rec["sample_idx"] not in success_step:
                success_step[rec["sample_idx"]] = rec["step"]
        for step in sorted({rec["step"] for rec in records}):
            n_success = sum(
                1 for i in range(n_evals) if success_step.get(i, np.inf) <= step
            )
            self.dump_logs({"mpc/success_rate": n_success / n_evals, "step": step})


class _EnvWorkerProxy:
    """Single-worker view of a SubprocVectorEnv or SerialVectorEnv, exposing the
    vectorized-env API with batch-1 arrays."""

    def __init__(self, parent_env, env_idx: int):
        from env.serial_vector_env import SerialVectorEnv

        if isinstance(parent_env, SerialVectorEnv):
            self._env = parent_env.envs[env_idx]
            self._mode = "serial"
        else:
            self._worker = parent_env.workers[env_idx]
            self._mode = "subproc"

    def rollout(self, seed, init_state, actions):
        if self._mode == "serial":
            obs, state = self._env.rollout(seed[0], init_state[0], actions[0])
        else:
            self._worker.rollout(seed[0], init_state[0], actions[0])
            obs, state = self._worker.recv()
        return {k: v[np.newaxis] for k, v in obs.items()}, state[np.newaxis]

    def eval_state(self, goal_state, cur_state):
        if self._mode == "serial":
            result = self._env.eval_state(goal_state[0], cur_state[0])
        else:
            result = self._worker.eval_state(goal_state[0], cur_state[0])
        return {k: np.expand_dims(np.asarray(v), 0) for k, v in result.items()}

    def prepare(self, seed, init_state):
        if self._mode == "serial":
            obs, state = self._env.prepare(seed[0], init_state[0])
        else:
            obs, state = self._worker.prepare(seed[0], init_state[0])
        return {k: v[np.newaxis] for k, v in obs.items()}, state[np.newaxis]

    def update_env(self, env_info):
        if self._mode == "serial":
            self._env.update_env(env_info[0])
        else:
            self._worker.update_env(env_info[0])
