"""
Trainer: orchestrates the full training loop for TwitWave RSSM.

Logs per-step: total loss, BCE, MSE, KL, β, LR.
Logs per-epoch: validation losses.
Saves checkpoints every N epochs.
Optionally logs to wandb.
Also saves KL_t time series and A_t attention snapshots to output_dir/logs/.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.twit_wave import TwitWave
from training.loss import elbo_loss
from training.scheduler import BetaScheduler, CosineWarmupScheduler


class Trainer:
    def __init__(
        self,
        model: TwitWave,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: torch.device,
    ) -> None:
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.cfg          = config
        self.device       = device

        self.output_dir = Path(config.get("output_dir", "outputs"))
        self.ckpt_dir   = self.output_dir / "checkpoints"
        self.log_dir    = self.output_dir / "logs"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config.get("weight_decay", 1e-6),
        )

        total_steps = config["max_epochs"] * len(train_loader)
        warmup_steps = config.get("warmup_epochs", 5) * len(train_loader)
        self.lr_sched = CosineWarmupScheduler(
            self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )
        self.beta_sched = BetaScheduler(
            beta_start=config["beta_start"],
            beta_end=config["beta_end"],
            total_steps=int(config["beta_anneal_epochs"] * len(train_loader)),
            hold_steps=int(config.get("beta_hold_epochs", 0) * len(train_loader)),
        )

        self.global_step = 0
        self.best_val_loss = float("inf")
        self.best_val_recon = float("inf")   # tracks bce+mse only for early stopping
        self.start_epoch = 1
        self.patience = config.get("patience", 20)
        self.epochs_no_improve = 0

        # wandb
        self.use_wandb = config.get("use_wandb", False)
        if self.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=config.get("project_name", "twit_wave"),
                    config=config,
                )
                self.wandb = wandb
            except ImportError:
                print("[trainer] wandb not installed, disabling")
                self.use_wandb = False

        # KL log
        self.kl_log: list[dict] = []
        self.attn_snapshots: list[dict] = []

    # ------------------------------------------------------------------
    def train(self) -> None:
        for epoch in range(self.start_epoch, self.cfg["max_epochs"] + 1):
            t0 = time.time()
            train_metrics = self._train_epoch(epoch)
            val_metrics   = self._val_epoch()

            elapsed = time.time() - t0
            print(
                f"[epoch {epoch:03d}] "
                f"train: total={train_metrics['total']:.4f}  bce={train_metrics['bce']:.4f}  mse={train_metrics['mse']:.4f}  kl={train_metrics['kl']:.4f}  "
                f"| val: total={val_metrics['total']:.4f}  bce={val_metrics['bce']:.4f}  mse={val_metrics['mse']:.4f}  kl={val_metrics['kl']:.4f}  "
                f"| β={self.beta_sched.value:.3f}  ({elapsed:.0f}s)"
            )

            if self.use_wandb:
                self.wandb.log(
                    {"epoch": epoch, **{f"val/{k}": v for k, v in val_metrics.items()}}
                )

            # Always write last.pt so we can resume from the most recent epoch
            self.save_checkpoint("last.pt", epoch=epoch)

            if epoch % self.cfg.get("checkpoint_every", 5) == 0:
                self.save_checkpoint(f"epoch_{epoch:03d}.pt", epoch=epoch)

            if val_metrics["total"] < self.best_val_loss:
                self.best_val_loss = val_metrics["total"]
                self.save_checkpoint("best.pt", epoch=epoch)

            # Early stopping on reconstruction only — avoids false triggers during
            # β-annealing where the ELBO increases even when reconstruction improves.
            val_recon = val_metrics["bce"] + val_metrics["mse"]
            if val_recon < self.best_val_recon:
                self.best_val_recon = val_recon
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.patience:
                    print(f"[trainer] early stopping at epoch {epoch} "
                          f"(no improvement for {self.patience} epochs)")
                    break

        # Save KL log
        with open(self.log_dir / "kl_log.json", "w") as f:
            json.dump(self.kl_log, f)
        print(f"[trainer] KL log saved ({len(self.kl_log)} entries)")

    # ------------------------------------------------------------------
    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        running = {k: 0.0 for k in ["total", "bce", "mse", "kl"]}
        n_batches = 0

        cfg = self.cfg
        window_k = self.model.cfg.window_k

        for batch in tqdm(self.train_loader, desc=f"train {epoch}", leave=False):
            features   = batch["features"].to(self.device)    # (B, T+k, N, 5)
            ticker_ids = batch["ticker_ids"].to(self.device)  # (B, T+k, N)
            presence   = batch["presence"].to(self.device)    # (B, T+k, vocab_size)
            week_of_year = batch.get("week_of_year")
            if week_of_year is not None:
                week_of_year = week_of_year.to(self.device)

            beta = self.beta_sched.step()

            out = self.model.forward_train(features, ticker_ids, presence, window_k, week_of_year)

            loss_dict = elbo_loss(
                presence_logits  = out["presence_logits"],
                presence_targets = out["presence_true"],
                feat_pred        = out["feat_pred"],
                feat_true        = out["feat_true"],
                ticker_ids       = ticker_ids[:, window_k:],
                post_mean        = out["post_mean"],
                post_logvar      = out["post_logvar"],
                prior_mean       = out["prior_mean"],
                prior_logvar     = out["prior_logvar"],
                lambda_          = cfg["lambda_"],
                beta             = beta,
                free_nats        = cfg["free_nats"],
                pos_weight       = cfg["pos_weight"],
                label_smoothing  = cfg.get("label_smoothing", 0.0),
            )

            self.optimizer.zero_grad()
            loss_dict["total"].backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), cfg["grad_clip"])
            self.optimizer.step()
            self.lr_sched.step()

            # log KL_t
            kl_vals = [k.item() for k in out["kl_t"]]
            self.kl_log.append({
                "step":     self.global_step,
                "epoch":    epoch,
                "kl_mean":  float(np.mean(kl_vals)),
                "kl_max":   float(np.max(kl_vals)),
                "kl_t":     kl_vals,
            })

            if self.global_step % cfg.get("log_every", 100) == 0:
                lr_now = self.lr_sched.get_last_lr()[0]
                log = {
                    "step": self.global_step,
                    "loss/total": loss_dict["total"].item(),
                    "loss/bce":   loss_dict["bce"].item(),
                    "loss/mse":   loss_dict["mse"].item(),
                    "loss/kl":    loss_dict["kl"].item(),
                    "train/beta": beta,
                    "train/lr":   lr_now,
                }
                if self.use_wandb:
                    self.wandb.log(log)

            for k in running:
                running[k] += loss_dict[k].item()
            n_batches += 1
            self.global_step += 1

        return {k: v / max(n_batches, 1) for k, v in running.items()}

    # ------------------------------------------------------------------
    def _val_epoch(self) -> dict:
        self.model.eval()
        running = {k: 0.0 for k in ["total", "bce", "mse", "kl"]}
        n_batches = 0
        window_k = self.model.cfg.window_k

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="val", leave=False):
                features   = batch["features"].to(self.device)
                ticker_ids = batch["ticker_ids"].to(self.device)
                presence   = batch["presence"].to(self.device)
                week_of_year = batch.get("week_of_year")
                if week_of_year is not None:
                    week_of_year = week_of_year.to(self.device)

                out = self.model.forward_train(features, ticker_ids, presence, window_k, week_of_year)

                loss_dict = elbo_loss(
                    presence_logits  = out["presence_logits"],
                    presence_targets = out["presence_true"],
                    feat_pred        = out["feat_pred"],
                    feat_true        = out["feat_true"],
                    ticker_ids       = ticker_ids[:, window_k:],
                    post_mean        = out["post_mean"],
                    post_logvar      = out["post_logvar"],
                    prior_mean       = out["prior_mean"],
                    prior_logvar     = out["prior_logvar"],
                    lambda_          = self.cfg["lambda_"],
                    beta             = self.beta_sched.value,
                    free_nats        = self.cfg["free_nats"],
                    pos_weight       = self.cfg["pos_weight"],
                    label_smoothing  = self.cfg.get("label_smoothing", 0.0),
                )

                for k in running:
                    running[k] += loss_dict[k].item()
                n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in running.items()}

    # ------------------------------------------------------------------
    def save_checkpoint(self, name: str, epoch: int = 0) -> None:
        path = self.ckpt_dir / name
        torch.save({
            "model_state":       self.model.state_dict(),
            "optimizer_state":   self.optimizer.state_dict(),
            "lr_sched_state":    self.lr_sched.state_dict(),
            "beta_sched_state":  self.beta_sched.state_dict(),
            "global_step":       self.global_step,
            "epoch":             epoch,
            "best_val_loss":     self.best_val_loss,
            "best_val_recon":    self.best_val_recon,
            "epochs_no_improve": self.epochs_no_improve,
            "model_cfg":         self.model.cfg.__dict__,
            "train_cfg":         self.cfg,
            "kl_log":            self.kl_log,
        }, path)
        print(f"[trainer] checkpoint → {path}")

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        if "lr_sched_state" in ckpt:
            self.lr_sched.load_state_dict(ckpt["lr_sched_state"])
        if "beta_sched_state" in ckpt:
            self.beta_sched.load_state_dict(ckpt["beta_sched_state"])
        self.global_step      = ckpt.get("global_step", 0)
        self.best_val_loss    = ckpt.get("best_val_loss", float("inf"))
        self.best_val_recon   = ckpt.get("best_val_recon", float("inf"))
        self.epochs_no_improve = ckpt.get("epochs_no_improve", 0)
        self.kl_log           = ckpt.get("kl_log", [])
        self.start_epoch      = ckpt.get("epoch", 0) + 1
        print(
            f"[trainer] resumed from {path} "
            f"(epoch {ckpt.get('epoch', '?')} → starting epoch {self.start_epoch}, "
            f"step {self.global_step})"
        )
