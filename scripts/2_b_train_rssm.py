"""
Step 2b: Train the Twit Wave RSSM

Usage:
    python scripts/2_b_train_rssm.py --cfg configs/rssm_base.yaml --out_dir outputs/rssm_base
    python scripts/2_b_train_rssm.py --cfg configs/debug.yaml     --out_dir outputs/debug
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from configs import load_config
from data.dataset import TwitWaveDataset, collate_dynamic
from data.vocab import Vocabulary
from model.twit_wave import TwitWave, ModelConfig
from training.trainer import Trainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cfg",      type=str, required=True, help="Path to unified YAML config")
    p.add_argument("--data_dir", type=str, default="data/processed")
    p.add_argument("--out_dir",  type=str, default="outputs/rssm_base")
    p.add_argument("--resume",   type=str, default=None, help="Checkpoint path to resume from")
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--wandb",    action="store_true")
    p.add_argument("--wandb_project", type=str, default="twit_wave")
    p.add_argument("--wandb_run",     type=str, default=None)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    cfg  = load_config(args.cfg)
    mcfg = cfg["model"]
    tcfg = cfg["train"]

    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save exact config used for reproducibility
    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)

    # ── device ───────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    if device.type == "cuda":
        log.info("GPU: %s  |  VRAM: %.1f GB",
                 torch.cuda.get_device_name(0),
                 torch.cuda.get_device_properties(0).total_memory / 1e9)

    # ── data ─────────────────────────────────────────────────────────────────
    data_dir = Path(args.data_dir)
    vocab = Vocabulary.load(data_dir / "vocab.json")
    log.info("Vocab size: %d", len(vocab))

    panel_train = pd.read_parquet(data_dir / "panel_train.parquet")
    panel_val   = pd.read_parquet(data_dir / "panel_val.parquet")

    ds_train = TwitWaveDataset(
        panel=panel_train, vocab=vocab,
        seq_len=tcfg["seq_len"], window_k=mcfg["window_k"],
        mode="dynamic", split="train", normalise=True,
    )
    ds_val = TwitWaveDataset(
        panel=panel_val, vocab=vocab,
        seq_len=tcfg["seq_len"], window_k=mcfg["window_k"],
        mode="dynamic", split="val", normalise=True,
        norm_stats=ds_train.norm_stats,
    )

    with open(out_dir / "norm_stats.json", "w") as f:
        json.dump({k: v.tolist() for k, v in ds_train.norm_stats.items()}, f, indent=2)

    train_loader = DataLoader(
        ds_train, batch_size=tcfg["batch_size"], shuffle=True,
        num_workers=tcfg["num_workers"], collate_fn=collate_dynamic,
        pin_memory=(device.type == "cuda"), drop_last=True,
    )
    val_loader = DataLoader(
        ds_val, batch_size=tcfg["batch_size"], shuffle=False,
        num_workers=tcfg["num_workers"], collate_fn=collate_dynamic,
        pin_memory=(device.type == "cuda"),
    )
    log.info("Train batches: %d  |  Val batches: %d", len(train_loader), len(val_loader))

    # ── model ─────────────────────────────────────────────────────────────────
    model_cfg = ModelConfig(
        vocab_size  = vocab.size,       # includes padding row
        embed_dim   = mcfg["embed_dim"],
        d_enc       = mcfg["d_enc"],
        h_dim       = mcfg["h_dim"],
        s_dim       = mcfg["s_dim"],
        n_heads     = mcfg["n_heads"],
        n_layers    = mcfg["n_layers"],
        window_k    = mcfg["window_k"],
        mlp_hidden  = mcfg["mlp_hidden"],
        feature_dim = mcfg["feature_dim"],
        top_k       = mcfg["top_k"],
        dropout     = mcfg["dropout"],
    )
    model  = TwitWave(model_cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("TwitWave  embed_dim=%d  z_dim=%d  params=%s",
             mcfg["embed_dim"], model_cfg.z_dim, f"{n_params:,}")

    # ── trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        device=device, output_dir=out_dir,
        lr                 = tcfg["lr"],
        weight_decay       = tcfg["weight_decay"],
        max_epochs         = tcfg["max_epochs"],
        grad_clip          = tcfg["grad_clip"],
        beta_start         = tcfg["beta_start"],
        beta_end           = tcfg["beta_end"],
        beta_anneal_epochs = tcfg["beta_anneal_epochs"],
        free_nats          = tcfg["free_nats"],
        lambda_mse         = tcfg["lambda_mse"],
        bce_pos_weight     = tcfg["bce_pos_weight"],
        warmup_epochs      = tcfg["warmup_epochs"],
        patience           = tcfg["patience"],
        use_wandb          = args.wandb,
        wandb_project      = args.wandb_project,
        wandb_run_name     = args.wandb_run,
        resume_ckpt        = args.resume,
    )

    log.info("Starting training → %s", out_dir)
    trainer.train()


if __name__ == "__main__":
    main()
