"""AdaJEPA's staged planning objective on LeWM latents, and the cost model the CEM solver calls."""

from __future__ import annotations

from dataclasses import dataclass

import torch


class Objective:
    """Staged objective: terminal frame only while step < T, then all T+1 frames weighted base^t.

    Costs sum the squared error over D (LeWM's scale); CEM ranking and the optimism controller are
    invariant to scale.
    """

    def __init__(self, base: float = 2.0):
        self.base = float(base)

    @staticmethod
    def _goal(pred: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        B, S, T1, D = pred.shape
        if goal.ndim == 3:
            goal = goal.reshape(B, -1, 1, D)
        if goal.shape[1] not in (1, S) or goal.shape[-1] != D:
            raise ValueError(f"goal_emb {tuple(goal.shape)} incompatible with pred {tuple(pred.shape)}")
        return goal

    def __call__(self, pred: torch.Tensor, goal: torch.Tensor, step: int | None = None) -> torch.Tensor:
        """pred (B, S, T+1, D), goal (B, 1, D) -> (B, S) costs, lower is better."""
        g = self._goal(pred, goal)
        per_frame = (pred - g).pow(2).sum(-1)    # (B, S, T+1)
        T1 = per_frame.shape[-1]
        if step is not None and step < T1 - 1:
            return per_frame[..., -1]
        w = self.base ** torch.arange(T1, device=per_frame.device, dtype=per_frame.dtype)
        w = w / w.sum()
        return (per_frame * w).sum(-1)


@dataclass
class PlanContext:
    """Final CEM population of one solve, detached.

    emb        (B, S, 1, D)  encoded start observation
    goal_emb   (B, 1, D)
    candidates (B, S, H, A)  all samples of the last CEM iteration
    costs      (B, S)
    step       replan index of the solve (selects the objective stage)
    """

    emb: torch.Tensor
    goal_emb: torch.Tensor
    candidates: torch.Tensor
    costs: torch.Tensor
    step: int


class CostModel:
    """Objective plus population capture around a LeWM model.

    Not an nn.Module and without `get_action`, so the solver pads warm starts with zeros.
    """

    def __init__(self, model, objective: Objective):
        self.model = model
        self.objective = objective
        self.step = 0                 # replan index of the solve in progress
        self.last: PlanContext | None = None

    def parameters(self):
        return self.model.parameters()

    @property
    def device(self):
        return next(self.model.parameters()).device

    def _ensure_goal_emb(self, info: dict) -> None:
        # As LeWM.get_cost: the goal image is encoded once per solve.
        if "goal_emb" in info:
            return
        goal = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        goal["pixels"] = goal["goal"]
        for k in list(goal.keys()):
            if k.startswith("goal_"):
                goal[k[len("goal_"):]] = goal.pop(k)
        goal.pop("action", None)
        goal = self.model.encode(goal)
        info["goal_emb"] = goal["emb"].detach()

    def get_cost(self, info: dict, candidates: torch.Tensor) -> torch.Tensor:
        dev = self.device
        for k, v in list(info.items()):
            if torch.is_tensor(v):
                info[k] = v.to(dev)
        self._ensure_goal_emb(info)
        info = self.model.rollout(info, candidates)          # caches info["emb"] on first call
        cost = self.objective(info["predicted_emb"], info["goal_emb"], step=self.step)
        # The solver runs in inference mode; clone outside it so the population can enter autograd later.
        with torch.inference_mode(False):
            self.last = PlanContext(
                emb=info["emb"].clone(),
                goal_emb=info["goal_emb"].clone(),
                candidates=candidates.clone(),
                costs=cost.clone(),
                step=int(self.step),
            )
        return cost

    def score_population(self, ctx: PlanContext, pool: str = "all", elites: int = 30) -> torch.Tensor:
        """J = mean planning cost of the recorded population under the current weights.

        Candidates are constants (envelope theorem) and emb / goal_emb are detached, so the gradient
        reaches the predictor only.
        """
        cand, costs, emb = ctx.candidates, ctx.costs, ctx.emb
        if pool == "elites":
            K = min(int(elites), cand.shape[1])
            idx = torch.topk(costs, k=K, dim=1, largest=False).indices
            b = torch.arange(cand.shape[0], device=cand.device).unsqueeze(1).expand(-1, K)
            cand, emb = cand[b, idx], emb[b, idx]
        elif pool != "all":
            raise ValueError(f"pool must be all|elites, got {pool!r}")
        B, S = cand.shape[:2]
        info = {
            # rollout() reads the history length from pixels.size(2) and skips encoding when "emb" is given.
            "pixels": torch.zeros(B, S, 1, 1, device=cand.device),
            "emb": emb,
            "goal_emb": ctx.goal_emb,
        }
        info = self.model.rollout(info, cand)
        J = self.objective(info["predicted_emb"], info["goal_emb"], step=ctx.step)
        return J.mean()
