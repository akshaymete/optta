"""Test-time adaptation of LeWM: AdaJEPA grounding plus the OpTTA optimism term.

Adapted: the predictor's last transformer block, its final norm, and pred_proj (LeWM's predict() is
predictor + pred_proj). With adapt.finetune_encoder (default off) the encoder head is also
adapted: the projector (the MLP from the ViT CLS token to the embedding), at adapt.encoder_lr (null = adapt.lr).
The grounding loss is then re-encoded each step so its gradient reaches the projector; targets stay detached.
The projector's BatchNorm stays in eval mode.

Optimism weight alpha_t = rho_eff |grad L_pred| / (|grad J| + eps), with rho_eff = rho sqrt(k/K) and k the number
of adaptations completed this episode. J and both norms use the predictor parameters only; J never reaches the
encoder.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn

from .objective import CostModel, PlanContext

log = logging.getLogger("lewm_optta.adapter")


class OpTTAAdapter:
    def __init__(self, model, cost_model: CostModel, cfg):
        self.model = model
        self.cost_model = cost_model

        self.lr = float(cfg.lr)
        self.steps = int(cfg.steps)
        self.buffer_size = int(cfg.buffer_size)

        opt = cfg.optimism
        self.rho = float(opt.rho)
        self.schedule = str(opt.schedule)
        self.K = int(opt.schedule_steps)
        self.pool = str(opt.pool)
        self.elites = int(opt.elites)
        self.eps = float(opt.ratio_eps)
        if self.schedule not in ("const", "increasing"):
            raise ValueError(f"optimism.schedule must be const|increasing, got {self.schedule!r}")
        if self.pool not in ("all", "elites"):
            raise ValueError(f"optimism.pool must be all|elites, got {self.pool!r}")

        self.history_size = int(getattr(model.predictor, "num_frames", 3))
        self.criterion = nn.MSELoss()

        transformer = model.predictor.transformer
        self.pred_params = (
            list(transformer.layers[-1].parameters())
            + list(transformer.norm.parameters())
            + list(model.pred_proj.parameters())
        )
        tensors = list(self.pred_params)
        tensors += [b for _, b in model.predictor.named_buffers()]
        tensors += [b for _, b in model.pred_proj.named_buffers()]

        self.finetune_encoder = bool(cfg.finetune_encoder)
        self.encoder_lr = self.lr if cfg.encoder_lr is None else float(cfg.encoder_lr)
        self.enc_params = []
        if self.finetune_encoder:
            self.enc_params = list(model.projector.parameters())
            tensors += self.enc_params
            tensors += [b for _, b in model.encoder.named_buffers()]
            tensors += [b for _, b in model.projector.named_buffers()]
        self._enc_saved = [p.detach().clone() for p in self.enc_params]
        self._scope_checked = False
        self._snapshot = [(t, t.detach().clone()) for t in tensors]

        self.buffer: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.n_updates = 0            # k: adaptations completed this episode

        log.info(
            "adapted: %d tensors (%.2fM params); history=%d lr=%.1e steps=%d buffer=%d "
            "rho=%.3g schedule=%s K=%d pool=%s",
            len(self.pred_params), sum(p.numel() for p in self.pred_params) / 1e6,
            self.history_size, self.lr, self.steps, self.buffer_size,
            self.rho, self.schedule, self.K, self.pool,
        )
        log.info("finetune_encoder=%s: %d projector tensors (%.2fM params) encoder_lr=%.1e",
                 self.finetune_encoder, len(self.enc_params),
                 sum(p.numel() for p in self.enc_params) / 1e6, self.encoder_lr)

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def optimism_on(self) -> bool:
        # alpha_t = rho_eff |grad L_pred| / (|grad J| + eps) has both norms >= 0 and no clamp, so sign(alpha) =
        # sign(rho): rho > 0 is OpTTA and rho = 0 is AdaJEPA.
        return self.rho != 0

    @torch.no_grad()
    def reset(self) -> None:
        """Restore the pretrained adapted tensors and clear per-episode state."""
        for tensor, saved in self._snapshot:
            tensor.copy_(saved)
        self.buffer.clear()
        self.n_updates = 0

    def add_segment(self, frames: torch.Tensor, actions: torch.Tensor) -> None:
        """frames (1, T+1, C, H, W) at model stride; actions (1, T, A), normalised."""
        dev = self.device
        frames, actions = frames.to(dev), actions.to(dev)
        n = min(frames.shape[1] - 1, actions.shape[1])
        if n < 1:
            return
        self.buffer.append((frames[:, : n + 1], actions[:, :n]))
        if len(self.buffer) > self.buffer_size:
            del self.buffer[: len(self.buffer) - self.buffer_size]

    def merged(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Contiguous segments as one trajectory (shared boundary frames dropped)."""
        frames, acts = [self.buffer[0][0]], [self.buffer[0][1]]
        for f, a in self.buffer[1:]:
            frames.append(f[:, 1:])
            acts.append(a)
        return torch.cat(frames, dim=1), torch.cat(acts, dim=1)

    def rho_eff(self) -> float:
        if not self.optimism_on:
            return 0.0
        if self.schedule == "const":
            return self.rho
        return self.rho * math.sqrt(min(1.0, self.n_updates / max(1, self.K)))

    def _encode(self, frames: torch.Tensor, actions: torch.Tensor):
        info = self.model.encode({"pixels": frames, "action": actions})
        return info["emb"], info["act_emb"]

    def grounding_loss(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        """Sliding-window 1-step latent prediction loss with detached targets (AdaJEPA, LeWM training)."""
        T = min(emb.shape[1] - 1, act_emb.shape[1])
        if T < 1:
            return torch.zeros((), device=emb.device)
        w = min(self.history_size, T)
        losses = []
        for t in range(T - w + 1):
            pred = self.model.predict(emb[:, t : t + w], act_emb[:, t : t + w])
            tgt = emb[:, t + 1 : t + 1 + w].detach()
            losses.append(self.criterion(pred, tgt))
        return torch.stack(losses).mean()

    def _set_mode(self, grounding: bool) -> None:
        # Dropout on for the grounding forward. pred_proj and projector BatchNorm stay in eval mode, as in
        # planning. The ViT has dropout 0.
        self.model.predictor.train(bool(grounding))
        self.model.pred_proj.eval()
        self.model.projector.eval()

    def _make_optimizer(self):
        groups = [{"params": self.pred_params, "lr": self.lr}]
        if self.finetune_encoder:
            groups.append({"params": self.enc_params, "lr": self.encoder_lr})
        return torch.optim.Adam(groups)

    @staticmethod
    def _global_norm(grads) -> torch.Tensor:
        sq = None
        for g in grads:
            if g is not None:
                s = g.pow(2).sum()
                sq = s if sq is None else sq + s
        return torch.zeros(()) if sq is None else sq.sqrt()

    def _check_optimism_scope(self, loss_opt: torch.Tensor) -> None:
        """Once per run: J must not reach any adapted encoder tensor."""
        self._scope_checked = True
        g_enc = torch.autograd.grad(loss_opt, self.enc_params, retain_graph=True, allow_unused=True)
        leaked = [i for i, g in enumerate(g_enc) if g is not None and bool(g.abs().sum() > 0)]
        if leaked:
            raise AssertionError(f"optimism gradient reached {len(leaked)} encoder tensors: {leaked}")
        log.info("optimism scope OK: J reaches 0 of %d encoder tensors", len(self.enc_params))

    def optimism_gradient(self, loss_pred: torch.Tensor, loss_opt: torch.Tensor):
        """grad L_pred + alpha_t grad J on the predictor, grad L_pred on the projector (if adapted).

        Returns (gradients for pred_params + enc_params, alpha_t, |grad L_pred|, |grad J|, rho_eff); both norms
        are over the predictor parameters.
        """
        rho = self.rho_eff()
        n_p = len(self.pred_params)
        g_all = torch.autograd.grad(loss_pred, self.pred_params + self.enc_params, allow_unused=True)
        g_pred, g_enc = g_all[:n_p], g_all[n_p:]
        g_opt = torch.autograd.grad(loss_opt, self.pred_params, allow_unused=True)
        n_pred, n_opt = self._global_norm(g_pred), self._global_norm(g_opt)
        alpha = (rho * n_pred / (n_opt + self.eps)).detach()
        grads = []
        for gp, go in zip(g_pred, g_opt):
            if go is None:
                grads.append(gp)
            elif gp is None:
                grads.append(alpha * go)
            else:
                grads.append(gp + alpha * go)
        grads.extend(g_enc)
        return grads, alpha, float(n_pred), float(n_opt), rho

    @torch.no_grad()
    def _enc_delta(self) -> float | None:
        """L2 distance of the adapted projector tensors from their pretrained values (None if not adapted)."""
        if not self.finetune_encoder:
            return None
        return math.sqrt(sum(float((p - s).pow(2).sum()) for p, s in zip(self.enc_params, self._enc_saved)))

    @torch.no_grad()
    def prequential_loss(self) -> float:
        """1-step loss of the current weights on the newest segment, before it is trained on."""
        frames, actions = self.buffer[-1]
        self._set_mode(grounding=False)
        emb, act_emb = self._encode(frames, actions)
        return float(self.grounding_loss(emb, act_emb))

    def step(self, plan_ctx: PlanContext | None, *, episode: int, replan: int) -> list[dict]:
        """Run `steps` optimiser steps on the buffer; one record per step. steps=0 only logs losses."""
        if not self.buffer:
            return []
        frames, actions = self.merged()
        with torch.inference_mode(False):
            pre_new = self.prequential_loss()
        if self.steps <= 0:
            with torch.no_grad(), torch.inference_mode(False):
                self._set_mode(grounding=False)
                emb_eval, act_eval = self._encode(frames, actions)
                pre = float(self.grounding_loss(emb_eval, act_eval))
            self.n_updates += 1
            rec = {
                "episode": episode, "replan": replan, "k": self.n_updates - 1, "adapt_step": 0,
                "n_frames": int(frames.shape[1]), "pre_new_loss": pre_new, "pre_loss_pred": pre,
                "loss_pred": pre, "loss_opt": None, "alpha": None, "n_pred": None, "n_opt": None,
                "rho_eff": 0.0, "loss_total": pre, "post_loss_pred": pre, "enc_delta": None,
            }
            log.info("adapt ep=%d replan=%d k=%d frames=%d pre_new=%.6f pre=%.6f (log only)",
                     episode, replan, rec["k"], rec["n_frames"], pre_new, pre)
            return [rec]
        use_opt = self.optimism_on and plan_ctx is not None
        records: list[dict] = []
        params = self.pred_params + self.enc_params

        with torch.inference_mode(False), torch.enable_grad():
            for p in params:
                p.requires_grad_(True)
            optimizer = self._make_optimizer()

            with torch.no_grad():
                self._set_mode(grounding=False)
                emb_eval, act_eval = self._encode(frames, actions)
                pre = float(self.grounding_loss(emb_eval, act_eval))
            emb_fixed, act_fixed = emb_eval.detach(), act_eval.detach()     # frozen encoder: encode once

            for s in range(self.steps):
                optimizer.zero_grad(set_to_none=True)
                self._set_mode(grounding=True)
                if self.finetune_encoder:
                    emb_g, act_g = self._encode(frames, actions)                # re-encode so gradient reaches the projector
                    loss_pred = self.grounding_loss(emb_g, act_g)
                else:
                    loss_pred = self.grounding_loss(emb_fixed, act_fixed)
                self._set_mode(grounding=False)                              # J is scored in eval mode

                rec = {
                    "episode": episode, "replan": replan, "k": self.n_updates, "adapt_step": s + 1,
                    "n_frames": int(frames.shape[1]), "pre_new_loss": pre_new, "pre_loss_pred": pre,
                    "loss_pred": float(loss_pred.detach()), "loss_opt": None, "alpha": None,
                    "n_pred": None, "n_opt": None, "rho_eff": self.rho_eff(),
                }
                if use_opt:
                    loss_opt = self.cost_model.score_population(plan_ctx, self.pool, self.elites)
                    if self.finetune_encoder and not self._scope_checked:
                        self._check_optimism_scope(loss_opt)
                    grads, alpha, n_pred, n_opt, rho = self.optimism_gradient(loss_pred, loss_opt)
                    for p, g in zip(params, grads):
                        p.grad = g
                    loss_total = loss_pred.detach() + alpha * loss_opt.detach()
                    rec.update(loss_opt=float(loss_opt.detach()), alpha=float(alpha), n_pred=n_pred, n_opt=n_opt, rho_eff=rho)
                else:
                    loss_pred.backward()
                    loss_total = loss_pred.detach()
                optimizer.step()
                rec["loss_total"] = float(loss_total)
                rec["enc_delta"] = self._enc_delta()
                records.append(rec)

            with torch.no_grad():
                self._set_mode(grounding=False)
                emb_eval, act_eval = self._encode(frames, actions)
                post = float(self.grounding_loss(emb_eval, act_eval))
            for p in params:
                p.requires_grad_(False)

        self.model.eval()
        for rec in records:
            rec["post_loss_pred"] = post
        self.n_updates += 1

        last = records[-1]
        msg = "adapt ep=%d replan=%d k=%d frames=%d pre_new=%.6f pre=%.6f post=%.6f" % (
            episode, replan, last["k"], last["n_frames"], pre_new, pre, post)
        if last["loss_opt"] is not None:
            msg += " J=%.4g alpha=%.4g |g_pred|=%.4g |g_J|=%.4g rho_eff=%.3f" % (
                last["loss_opt"], last["alpha"], last["n_pred"], last["n_opt"], last["rho_eff"])
        if last["enc_delta"] is not None:
            msg += " |d_enc|=%.4g" % last["enc_delta"]
        log.info(msg)
        return records
