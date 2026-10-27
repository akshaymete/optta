import logging

import torch
import torch.nn as nn

from utils import move_to_device

log = logging.getLogger(__name__)


class OpTTATrainer:
    """OpTTA trainer: test-time adaptation of the predictor (and optionally
    the encoder) on trajectory segments observed from the environment during
    planning, using the same sliding-window 1-step latent prediction loss as
    training. Defaults follow AdaJEPA: one adaptation step on the predictor's
    last transformer layer and the encoder's head.

    OpTTA adds an optimism term to the same step:
        L = L_pred + alpha_t * J,
        alpha_t = rho_eff * ||grad L_pred|| / (||grad J|| + eps),
    where J is the planning objective of the just-optimized plan(s) rolled out by
    the current model and both norms are over the adapted predictor parameters.
    rho_eff = rho ("const") or rho * sqrt(min(1, k/K)) ("increasing"), with k the
    number of adaptations completed this episode. rho = optimism_ratio_target;
    rho = 0 is AdaJEPA.
    """

    def __init__(
        self,
        wm,
        lr: float,
        steps: int = 1,
        optimizer_name: str = "adam",
        finetune_encoder: bool = True,
        last_layer_only: bool = True,
        encoder_lr: float = None,
        encoder_last_layer_only: bool = True,
        optimism_ratio_target: float = 0.0,
        optimism_elites: int = 0,
        optimism_chunk: int = None,
        optimism_dtype: str = "float32",
        optimism_rho_schedule: str = "const",
        optimism_rho_schedule_steps: int = 100,
        objective_fn=None,
    ):
        self.wm = wm
        self.lr = lr
        self.encoder_lr = encoder_lr if encoder_lr is not None else lr
        self.steps = steps
        self.optimizer_name = optimizer_name.lower()
        self.finetune_encoder = finetune_encoder
        self.last_layer_only = last_layer_only
        self.encoder_last_layer_only = encoder_last_layer_only

        self.ratio_target = float(optimism_ratio_target)
        # CEM only: >0 scores J on up to this many final-iteration plans (mean of
        # their costs); 0 scores the elite mean.
        self.optimism_elites = int(optimism_elites)
        # >0: J and its gradient over the pooled plans in chunks of this many plans (memory scales with the chunk,
        # not the pool); None = one batch.
        self.optimism_chunk = None if optimism_chunk is None else int(optimism_chunk)
        if self.optimism_chunk is not None and self.optimism_chunk <= 0:
            raise ValueError(f"optimism_chunk must be null or > 0, got {optimism_chunk!r}")
        # Precision of J's forward (rollout and objective): "float32" (default) or "bfloat16" (torch.autocast).
        # Weights, gradients and L_pred stay float32.
        self.optimism_dtype = str(optimism_dtype)
        if self.optimism_dtype not in ("float32", "bfloat16"):
            raise ValueError(f"optimism_dtype must be float32|bfloat16, got {optimism_dtype!r}")
        if self.optimism_dtype != "float32" and self.ratio_target > 0:
            print(f"OpTTA J forward under torch.autocast({self.optimism_dtype})", flush=True)
        self.rho_schedule = str(optimism_rho_schedule)
        self.rho_schedule_steps = int(optimism_rho_schedule_steps)
        if self.rho_schedule not in ("const", "increasing"):
            raise ValueError(
                f"optimism_rho_schedule must be const|increasing, got {self.rho_schedule!r}"
            )
        self.ratio_eps = 1e-12
        self.objective_fn = objective_fn
        if self.optimism_on and objective_fn is None:
            raise ValueError("optimism_ratio_target > 0 requires an objective_fn.")
        self._replan_idx = 0             # k: finetune() calls this episode
        self._opt_scope_checked = False  # one-time encoder-scope assertion

        self.device = next(wm.parameters()).device
        self.criterion = nn.MSELoss()
        self._adapted_predictor_params = self._select_predictor_params()
        self._adapted_encoder_params = self._select_encoder_params() if self.finetune_encoder else []
        self._snapshot = self._take_snapshot()

    @property
    def optimism_on(self) -> bool:
        return self.ratio_target > 0

    def _take_snapshot(self):
        """Snapshot the tensors adaptation can touch: selected params + train-mode buffers (e.g. BN stats)."""
        tensors = list(self._adapted_predictor_params)
        tensors += [b for _, b in self.wm.predictor.named_buffers()]
        if self.finetune_encoder:
            tensors += self._adapted_encoder_params
            tensors += [b for _, b in self.wm.encoder.named_buffers()]
        return [(t, t.detach().clone()) for t in tensors]

    @torch.no_grad()
    def reset(self):
        """Restore the pre-adaptation values of the adapted tensors."""
        for tensor, saved in self._snapshot:
            tensor.copy_(saved)
        self._replan_idx = 0

    def finetune(self, obs_seqs: list, act_seqs: list, merge: bool = True,
                 plan_ctx: dict = None) -> list:
        """Run `steps` optimization steps on the given trajectory segments.

        merge=True concatenates temporally contiguous segments into one long
        sequence; use merge=False for non-contiguous segments.
        plan_ctx (plan_obs_0, obs_g, plan_actions detached, plan_elites, step)
        adds the optimism term when optimism is on.
        Returns one dict per step: pred, optimism, total, alpha_t, n_pred, n_opt.
        """
        if not obs_seqs or self.steps <= 0:
            return []
        if merge and len(obs_seqs) > 1:
            obs_seqs, act_seqs = self._merge_segments(obs_seqs, act_seqs)
        segments = [self._prepare_segment(o, a) for o, a in zip(obs_seqs, act_seqs)]

        use_optimism = self.optimism_on and plan_ctx is not None
        if use_optimism:
            # Encode the rollout start and goal BEFORE train() is switched on (BatchNorm).
            z_0, z_g, num_obs_init = self._encode_plan_endpoints(
                plan_ctx["plan_obs_0"], plan_ctx["obs_g"], plan_ctx["plan_actions"]
            )
        if not self.finetune_encoder:
            # Encoder frozen: embeddings can be precomputed once.
            with torch.no_grad():
                all_z = [self.wm.encode(o, a).detach() for o, a in segments]

        self._set_requires_grad(self.wm.predictor, self._adapted_predictor_params, True)
        self.wm.predictor.train()
        if self.finetune_encoder:
            self._set_requires_grad(self.wm.encoder, self._adapted_encoder_params, True)
            self.wm.encoder.train()
            base_model = getattr(self.wm.encoder, "base_model", None)
            if base_model is not None:
                base_model.eval()  # keep the frozen backbone in eval mode

        optimizer = self._make_optimizer()
        detach_src = not self.finetune_encoder
        detach_tgt = True if not self.finetune_encoder else bool(getattr(self.wm, "stop_grad", True))
        step_losses = []
        for step in range(self.steps):
            optimizer.zero_grad()
            if self.finetune_encoder:
                # Re-encode each step so gradients reach the encoder.
                all_z = [self.wm.encode(o, a) for o, a in segments]
            loss_pred = torch.stack(
                [
                    self._prediction_loss(z, detach_src=detach_src, detach_tgt=detach_tgt)
                    for z in all_z
                ]
            ).mean()
            rec = {"optimism": None, "alpha_t": None, "n_pred": None, "n_opt": None}
            if use_optimism and self.optimism_chunk:
                grads, alpha, n_pred, n_opt, loss_opt = self._optimism_gradient_chunked(
                    loss_pred, z_0, z_g, plan_ctx["plan_actions"], num_obs_init,
                    step=plan_ctx.get("step"), plan_elites=plan_ctx.get("plan_elites"),
                )
            elif use_optimism:
                loss_opt = self._optimism_loss(
                    z_0, z_g, plan_ctx["plan_actions"], num_obs_init,
                    step=plan_ctx.get("step"), plan_elites=plan_ctx.get("plan_elites"),
                )
                if not self._opt_scope_checked:
                    self._assert_optimism_scope(loss_opt)
                grads, alpha, n_pred, n_opt = self._optimism_gradient(loss_pred, loss_opt)
            if use_optimism:
                for p, g in zip(self._adapted_predictor_params + self._adapted_encoder_params, grads):
                    p.grad = g
                loss_total = loss_pred.detach() + alpha * loss_opt.detach()
                rec.update(optimism=float(loss_opt.item()), alpha_t=float(alpha),
                           n_pred=float(n_pred), n_opt=float(n_opt))
            else:
                loss_pred.backward()
                loss_total = loss_pred.detach()
            optimizer.step()
            rec.update(pred=float(loss_pred.item()), total=float(loss_total.item()))
            step_losses.append(rec)
            if rec["optimism"] is None:
                log.info("Adaptation step %d/%d  pred_loss=%.6f", step + 1, self.steps, rec["pred"])
            else:
                log.info(
                    "Adaptation step %d/%d  pred=%.6f  optimism=%.6f  "
                    "alpha_t=%.4g  |g_pred|=%.4g  |g_opt|=%.4g  rho_eff=%.4g",
                    step + 1, self.steps, rec["pred"], rec["optimism"],
                    rec["alpha_t"], rec["n_pred"], rec["n_opt"], self.rho_eff(),
                )

        self._replan_idx += 1
        self.wm.predictor.eval()
        self._set_requires_grad(self.wm.predictor, self._adapted_predictor_params, False)
        if self.finetune_encoder:
            self.wm.encoder.eval()
            self._set_requires_grad(self.wm.encoder, self._adapted_encoder_params, False)
        return step_losses

    def rho_eff(self) -> float:
        if self.rho_schedule == "increasing":
            K = max(1, self.rho_schedule_steps)
            frac = min(1.0, self._replan_idx / K)
            return self.ratio_target * frac**0.5
        return self.ratio_target

    def _encode_plan_endpoints(self, plan_obs_0, obs_g, plan_actions):
        """Encode the rollout start and the goal, detached, so the optimism
        gradient reaches the predictor only."""
        num_obs_init = plan_obs_0["visual"].shape[1]
        with torch.no_grad():
            z_0 = self.wm.encode(plan_obs_0, plan_actions[:, :num_obs_init]).detach()
            z_g = {k: v.detach() for k, v in self.wm.encode_obs(obs_g).items()}
        return z_0, z_g, num_obs_init

    def _optimism_loss(self, z_0, z_g, plan_actions, num_obs_init, step=None, plan_elites=None):
        """J: the planning objective of the fixed plan(s), rolled out by the model.

        Runs in eval() mode, the mode the planner optimized the plan under
        (the predictor has dropout).
        """
        was_training = self.wm.predictor.training
        self.wm.predictor.eval()
        try:
            if plan_elites is not None and self.optimism_elites > 0:
                K = min(self.optimism_elites, plan_elites.shape[1])
                el = plan_elites[0, :K]                       # (K, T, act_dim)
                # Put each plan's own first action(s) into its start state, as CEM's
                # wm.rollout encoded it; z_0 carries the executed mean plan's.
                with torch.no_grad():
                    z0K = self.wm.replace_actions_from_z(
                        z_0.expand(K, *z_0.shape[1:]).clone(), el[:, :num_obs_init]
                    )
                zgK = {k: v.expand(K, *v.shape[1:]) for k, v in z_g.items()}
                with self._optimism_autocast():
                    z_obses, _ = self.wm.rollout_from_z(z0K, el[:, num_obs_init:])
                    loss = self.objective_fn(z_obses, zgK, step=step).mean()
                return loss.float()
            with self._optimism_autocast():
                z_obses, _ = self.wm.rollout_from_z(z_0, plan_actions[:, num_obs_init:])
                loss = self.objective_fn(z_obses, z_g, step=step).mean()
            return loss.float()
        finally:
            if was_training:
                self.wm.predictor.train()

    def _optimism_autocast(self):
        """torch.autocast for J's forward when optimism_dtype is bfloat16, a no-op otherwise. The backward follows the
        recorded dtypes; weights and their gradients stay float32."""
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.optimism_dtype == "bfloat16")

    @staticmethod
    def _global_norm(grads):
        """L2 norm over a list of grad tensors; None entries contribute zero."""
        sq = None
        for g in grads:
            if g is not None:
                s = g.pow(2).sum()
                sq = s if sq is None else sq + s
        if sq is None:
            return torch.zeros((), device="cpu")
        return sq.sqrt()

    def _optimism_gradient(self, loss_pred, loss_opt):
        """grad L_pred + alpha_t grad J on the predictor, grad L_pred on the encoder head (if adapted),
        from one backward pass per loss, as lewm_optta. alpha_t = rho_eff * ||grad L_pred|| /
        (||grad J|| + eps), both norms over the adapted predictor params; J never reaches the encoder.

        Returns (grads for predictor + encoder params, alpha_t, ||grad L_pred||, ||grad J||).
        """
        pred = self._adapted_predictor_params
        n_p = len(pred)
        g_all = torch.autograd.grad(loss_pred, pred + self._adapted_encoder_params, allow_unused=True)
        g_pred, g_enc = g_all[:n_p], g_all[n_p:]
        g_opt = torch.autograd.grad(loss_opt, pred, allow_unused=True)
        return self._combine_optimism(g_pred, g_enc, g_opt)

    def _optimism_gradient_chunked(self, loss_pred, z_0, z_g, plan_actions, num_obs_init, step=None, plan_elites=None):
        """_optimism_loss + _optimism_gradient with the pooled J evaluated in chunks of `optimism_chunk` plans.

        J is the mean of per-plan costs and the predictor treats plans independently, so the chunk terms
        (sum of the chunk's costs) / K and their gradients add up to J and grad J, up to floating point. Each chunk's graph
        is freed before the next, so memory scales with the chunk, not the pool. Without a pool (elite mean) this is the
        one-batch path. Returns (grads, alpha_t, ||grad L_pred||, ||grad J||, J).
        """
        if plan_elites is None or self.optimism_elites <= 0:
            loss_opt = self._optimism_loss(z_0, z_g, plan_actions, num_obs_init, step=step, plan_elites=plan_elites)
            if not self._opt_scope_checked:
                self._assert_optimism_scope(loss_opt)
            return (*self._optimism_gradient(loss_pred, loss_opt), loss_opt)
        pred = self._adapted_predictor_params
        n_p = len(pred)
        g_all = torch.autograd.grad(loss_pred, pred + self._adapted_encoder_params, allow_unused=True)
        g_pred, g_enc = g_all[:n_p], g_all[n_p:]
        K = min(self.optimism_elites, plan_elites.shape[1])
        el_all = plan_elites[0, :K]                           # (K, T, act_dim)
        g_opt = [None] * n_p
        loss_opt = None
        was_training = self.wm.predictor.training
        self.wm.predictor.eval()
        try:
            for start in range(0, K, self.optimism_chunk):
                el = el_all[start:start + self.optimism_chunk]
                k = el.shape[0]
                # As _optimism_loss: each plan's own first action(s) go into its start state.
                with torch.no_grad():
                    z0k = self.wm.replace_actions_from_z(
                        z_0.expand(k, *z_0.shape[1:]).clone(), el[:, :num_obs_init]
                    )
                zgk = {name: v.expand(k, *v.shape[1:]) for name, v in z_g.items()}
                with self._optimism_autocast():
                    z_obses, _ = self.wm.rollout_from_z(z0k, el[:, num_obs_init:])
                    part = self.objective_fn(z_obses, zgk, step=step).sum() / K
                part = part.float()
                if not self._opt_scope_checked:
                    self._assert_optimism_scope(part)
                g_part = torch.autograd.grad(part, pred, allow_unused=True)
                g_opt = [b if a is None else (a if b is None else a + b) for a, b in zip(g_opt, g_part)]
                loss_opt = part.detach() if loss_opt is None else loss_opt + part.detach()
        finally:
            if was_training:
                self.wm.predictor.train()
        return (*self._combine_optimism(g_pred, g_enc, g_opt), loss_opt)

    def _combine_optimism(self, g_pred, g_enc, g_opt):
        """alpha_t from the two gradient norms; returns (g_pred + alpha_t g_opt on the predictor, then g_enc,
        alpha_t, ||grad L_pred||, ||grad J||)."""
        n_pred = self._global_norm(g_pred)
        n_opt = self._global_norm(g_opt)
        alpha = (self.rho_eff() * n_pred / (n_opt + self.ratio_eps)).detach()
        grads = []
        for gp, go in zip(g_pred, g_opt):
            if go is None:
                grads.append(gp)
            elif gp is None:
                grads.append(alpha * go)
            else:
                grads.append(gp + alpha * go)
        grads.extend(g_enc)
        return grads, alpha, n_pred, n_opt

    def _assert_optimism_scope(self, loss_opt):
        """Checked once: the optimism term must not reach any adapted encoder param."""
        self._opt_scope_checked = True
        if not self.finetune_encoder or not self._adapted_encoder_params:
            return
        g_enc = torch.autograd.grad(
            loss_opt, self._adapted_encoder_params,
            retain_graph=True, allow_unused=True,
        )
        leaked = [i for i, g in enumerate(g_enc) if g is not None and g.abs().sum() > 0]
        if leaked:
            raise AssertionError(
                f"Optimism gradient reached {len(leaked)} encoder param(s) (indices {leaked})."
            )

    @torch.no_grad()
    def score_segments(self, obs_seqs: list, act_seqs: list) -> list:
        """Prediction loss per segment under the current weights (higher = harder)."""
        return [
            float(self._prediction_loss(self.wm.encode(*self._prepare_segment(o, a))))
            for o, a in zip(obs_seqs, act_seqs)
        ]

    def _select_predictor_params(self):
        predictor = self.wm.predictor
        if self.last_layer_only:
            params = list(predictor.transformer.layers[-1].parameters())
            params += list(predictor.transformer.norm.parameters())
            log.info(
                "Predictor adaptation restricted to last transformer layer (%d tensors).",
                len(params),
            )
            return params
        return list(predictor.parameters())

    def _select_encoder_params(self):
        """Mirror train-time freezing: never adapt a frozen backbone."""
        encoder = self.wm.encoder
        if hasattr(encoder, "base_model"):
            if hasattr(encoder, "projector") and getattr(encoder, "projector_name", None) in (
                "channel",
                "global",
            ):
                params = list(encoder.projector.parameters())
            else:
                params = [
                    p for name, p in encoder.named_parameters()
                    if not name.startswith("base_model.")
                ]
            log.info("Encoder adaptation uses non-backbone params (%d tensors).", len(params))
            return params
        if self.encoder_last_layer_only:
            children = list(encoder.named_children())
            if children:
                name, module = children[-1]
                params = list(module.parameters())
                log.info(
                    "Encoder adaptation restricted to last submodule '%s' (%d tensors).",
                    name, len(params),
                )
                return params
        params = list(encoder.parameters())
        log.info("Encoder adaptation uses full encoder params (%d tensors).", len(params))
        return params

    @staticmethod
    def _set_requires_grad(module, adapted_params, enabled: bool):
        """Freeze all params of `module`; if enabled, re-enable the adapted subset."""
        for p in module.parameters():
            p.requires_grad_(False)
        if enabled:
            for p in adapted_params:
                p.requires_grad_(True)

    def _make_optimizer(self):
        param_groups = [{"params": self._adapted_predictor_params, "lr": self.lr}]
        if self.finetune_encoder:
            param_groups.append({"params": self._adapted_encoder_params, "lr": self.encoder_lr})
            log.info("Adaptation optimizer: predictor_lr=%.2e  encoder_lr=%.2e", self.lr, self.encoder_lr)
        optimizers = {
            "adam": torch.optim.Adam,
            "adamw": torch.optim.AdamW,
            "sgd": torch.optim.SGD,
        }
        if self.optimizer_name not in optimizers:
            raise ValueError(f"Unknown adaptation optimizer: {self.optimizer_name!r}")
        return optimizers[self.optimizer_name](param_groups)

    def _prepare_segment(self, obs, act):
        """Move one (obs, act) segment to device and pad a dummy action for the
        last frame so encode() sees T+1 (frame, action) pairs. The dummy action
        never enters the prediction loss: action dims are excluded from the loss
        and the last frame never appears in a source window."""
        obs = move_to_device({k: v.clone() for k, v in obs.items()}, self.device)
        act = act.to(self.device)
        act = torch.cat([act, torch.zeros_like(act[:, :1])], dim=1)
        return obs, act

    @staticmethod
    def _merge_segments(obs_seqs, act_seqs):
        """Concatenate contiguous segments; the last frame of segment i equals
        the first frame of segment i+1, so the duplicate frame is dropped."""
        obs = {k: v.clone() for k, v in obs_seqs[0].items()}
        act = act_seqs[0]
        for obs_i, act_i in zip(obs_seqs[1:], act_seqs[1:]):
            for k in obs:
                obs[k] = torch.cat([obs[k], obs_i[k][:, 1:]], dim=1)
            act = torch.cat([act, act_i], dim=1)
        return [obs], [act]

    def _prediction_loss(
        self,
        z: torch.Tensor,
        detach_src: bool = True,
        detach_tgt: bool = True,
    ) -> torch.Tensor:
        """Sliding-window 1-step MSE on obs tokens (visual+proprio, no action).

        z: (b, T+1, p, d) embeddings of T+1 frames.
        """
        T = z.shape[1] - 1
        if T < 1:
            return torch.tensor(0.0, device=self.device)
        window = min(self.wm.num_hist, T)
        losses = []
        for t in range(T - window + 1):
            z_src = z[:, t : t + window]
            z_tgt = z[:, t + 1 : t + 1 + window]
            if detach_src:
                z_src = z_src.detach()
            if detach_tgt:
                z_tgt = z_tgt.detach()
            z_pred = self.wm.predict(z_src)
            if self.wm.concat_dim == 0:
                loss = self.criterion(z_pred[:, :, :-1, :], z_tgt[:, :, :-1, :])
            else:
                drop = self.wm.action_dim
                loss = self.criterion(z_pred[:, :, :, :-drop], z_tgt[:, :, :, :-drop])
            losses.append(loss)
        return torch.stack(losses).mean()
