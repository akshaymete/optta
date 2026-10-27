"""Receding-horizon policy (swm WorldModelPolicy, one env) that adapts the model at every replan.

At a replan boundary: close the executed segment, adapt on the buffer with the executed plan's final
CEM population as the optimism context, then plan with the updated weights, warm-started from the
unexecuted tail.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from stable_worldmodel.policy import WorldModelPolicy

from .corruption import nphash
from .determinism import plan_hash
from .objective import CostModel, PlanContext


@dataclass
class EpisodeState:
    index: int
    replan_idx: int = 0                 # plans made so far
    env_step: int = 0
    total_steps: int = 0
    seg_frames: list = field(default_factory=list)
    seg_actions: torch.Tensor | None = None
    seg_ctx: PlanContext | None = None
    plan_hashes: list = field(default_factory=list)
    frame_hashes: list = field(default_factory=list)   # per model step: env state, raw pixels, transformed frame
    track: list = field(default_factory=list)          # per replan: latent vs true goal distance
    adapt_records: list = field(default_factory=list)


class OpTTAPolicy(WorldModelPolicy):
    def __init__(self, solver, config, process, transform, cost_model: CostModel,
                 adapter=None, base_seed: int = 0):
        super().__init__(solver=solver, config=config, process=process, transform=transform)
        self.type = "lewm_optta"
        self.cost_model = cost_model
        self.adapter = adapter
        self.base_seed = int(base_seed)
        self.ep: EpisodeState | None = None

    def episode_seed(self, index: int) -> int:
        return (self.base_seed * 1_000_003 + index) % (2**31)

    def begin_episode(self, index: int) -> EpisodeState:
        assert self._action_buffer is not None, "set_env() must run before begin_episode()"
        for buf in self._action_buffer:
            buf.clear()
        self._next_init = None
        self.cost_model.step = 0
        self.cost_model.last = None
        if self.adapter is not None:
            self.adapter.reset()
        # Reseed the CEM generator and corruption stream per episode, so an episode does not depend on
        # how long earlier episodes ran and every arm gets the same first plan.
        s = self.episode_seed(index)
        gen = getattr(self.solver, "torch_gen", None)
        if gen is not None:
            gen.manual_seed(s)
        for i, key in enumerate(sorted(self.transform)):
            reseed = getattr(self.transform[key], "reseed", None)
            if reseed is not None:
                reseed(s + 7 * (i + 1))
        self.ep = EpisodeState(index=index)
        return self.ep

    def get_action(self, info_dict: dict, **kwargs):
        assert self.ep is not None, "call begin_episode() first"
        assert self.env.num_envs == 1, "OpTTAPolicy evaluates one env at a time"
        info = self._prepare_info(info_dict)
        info.pop("_needs_flush", None)
        obs = info["pixels"].detach().clone()           # (1, 1, C, H, W), transformed
        ep, buf = self.ep, self._action_buffer[0]

        if len(buf) == 0:                               # replan boundary
            self._record_frame(info_dict, obs)
            self._close_segment(obs)
            self._replan(info, obs, info_dict)
        else:
            ep.env_step += 1
            if ep.env_step % self.cfg.action_block == 0:  # one frame per model step
                self._record_frame(info_dict, obs)
                ep.seg_frames.append(obs)
        ep.total_steps += 1

        action = buf.popleft()
        action = action.reshape(*self.env.action_space.shape).float().numpy()
        if "action" in self.process:
            action = self.process["action"].inverse_transform(action)
        return action

    def _record_frame(self, info_dict: dict, obs: torch.Tensor) -> None:
        """Hash env state, raw pixels and transformed frame, so lewm_optta.gate can name the first differing layer."""
        rec = {"t": self.ep.total_steps, "obs": plan_hash(obs)}
        for key in ("state", "pixels"):
            v = info_dict.get(key)
            if v is not None:
                rec[key] = nphash(v.detach().cpu().numpy() if torch.is_tensor(v) else v)
        self.ep.frame_hashes.append(rec)

    def _close_segment(self, obs: torch.Tensor) -> None:
        ep = self.ep
        if ep.seg_actions is None:                      # nothing executed yet
            return
        ep.seg_frames.append(obs)
        if self.adapter is None:
            return
        frames = torch.cat(ep.seg_frames, dim=1)        # (1, T+1, C, H, W)
        self.adapter.add_segment(frames, ep.seg_actions)
        recs = self.adapter.step(ep.seg_ctx, episode=ep.index, replan=ep.replan_idx)
        ep.adapt_records.extend(recs)

    def _track(self, info_dict: dict) -> None:
        """Log latent goal distance next to true state distance (does the latent cost track progress?)."""
        ctx = self.cost_model.last
        if ctx is None:
            return
        rec = {"t": self.ep.total_steps, "replan": self.ep.replan_idx}
        z0 = ctx.emb.reshape(-1, ctx.emb.shape[-1])[0]
        zg = ctx.goal_emb.reshape(-1, ctx.goal_emb.shape[-1])[0]
        rec["latent_d"] = float(((z0 - zg) ** 2).sum())
        # PushT: first 4 dims of `state` (agent and block position); other tasks: `proprio`.
        for key, ndim in (("state", 4), ("proprio", None)):
            st, gs = info_dict.get(key), info_dict.get(f"goal_{key}")
            if st is None or gs is None:
                continue
            st = np.asarray(st.detach().cpu() if torch.is_tensor(st) else st, dtype=np.float64).ravel()
            gs = np.asarray(gs.detach().cpu() if torch.is_tensor(gs) else gs, dtype=np.float64).ravel()
            n = min(len(st), len(gs)) if ndim is None else min(ndim, len(st), len(gs))
            rec["state_d"] = float(np.linalg.norm(gs[:n] - st[:n]))
            break
        self.ep.track.append(rec)

    def _replan(self, info: dict, obs: torch.Tensor, info_dict: dict) -> None:
        ep = self.ep
        self.cost_model.step = ep.replan_idx
        init = self._next_init if self.cfg.warm_start else None
        out = self.solver(info, init_action=init)
        self._track(info_dict)
        actions = out["actions"]                        # (1, horizon, A), normalised
        rh = self.cfg.receding_horizon
        plan, rest = actions[:, :rh], actions[:, rh:]
        self._next_init = rest.clone() if (self.cfg.warm_start and rest.shape[1] > 0) else None

        flat = plan.reshape(1, self.flatten_receding_horizon, -1)
        self._action_buffer[0].extend(flat[0])

        ep.plan_hashes.append(plan_hash(actions))
        ep.seg_frames = [obs]
        ep.seg_actions = plan.detach().clone()
        ep.seg_ctx = self.cost_model.last               # this plan's population, scored after execution
        ep.env_step = 0
        ep.replan_idx += 1
