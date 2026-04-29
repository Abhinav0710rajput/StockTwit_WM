"""Shared utilities for evaluation scripts."""

from __future__ import annotations

from pathlib import Path

import torch

from configs import load_config
from model.twit_wave import TwitWave, ModelConfig


def load_rssm(model_dir: str | Path, vocab_size: int, device: torch.device) -> TwitWave:
    """
    Load a trained TwitWave model from an output directory.
    Expects model_dir/config.yaml and model_dir/best_model.pt.
    """
    model_dir = Path(model_dir)
    cfg  = load_config(model_dir / "config.yaml")
    mcfg = cfg["model"]

    model_cfg = ModelConfig(
        vocab_size  = vocab_size,
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
        dropout     = 0.0,   # disable dropout at eval time
    )
    model = TwitWave(model_cfg)
    ckpt  = torch.load(model_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()
    return model.to(device)
