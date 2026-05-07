"""Shared utilities for evaluation scripts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from configs import load_config
from model.twit_wave import TwitWave, ModelConfig


def load_rssm(model_dir: str | Path, vocab_size: int, device: torch.device) -> TwitWave:
    """
    Load a trained TwitWave model from an output directory.
    Expects model_dir/config.yaml and model_dir/checkpoints/best.pt.
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
    ckpt  = torch.load(model_dir / "checkpoints" / "best.pt", map_location=device)
    model.load_state_dict(ckpt.get("model_state", ckpt))
    model.eval()
    return model.to(device)


def export_embeddings(model: TwitWave, vocab, out_dir: str | Path) -> Path:
    """
    Extract both learned ticker embedding tables from a trained model and write
    them to <out_dir>/embeddings/ in three formats:

      e_ret.npy           (vocab_size, embed_dim) float32  — retrieval table
      e_dec.npy           (vocab_size, embed_dim) float32  — encoder/feature table
      ticker_to_idx.json  {"AAPL": 1, ...}
      idx_to_ticker.json  {"1": "AAPL", "0": "<pad>", ...}
      meta.json           vocab_size, embed_dim, n_tickers

    Row 0 is the padding index. e_dec row 0 stays near zero (padding tickers
    are masked in the set encoder). e_ret row 0 may drift slightly because
    all_ret() is used directly in the presence head matrix multiply, bypassing
    the padding_idx gradient-zeroing mechanism. Ignore row 0 in both tables.
    Real tickers occupy rows 1 … vocab_size-1 in the order defined by vocab.

    Returns the embeddings directory path.
    """
    emb_dir = Path(out_dir) / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)

    e_ret = model.embeddings.all_ret().detach().cpu().numpy().astype(np.float32)
    e_dec = model.embeddings.all_dec().detach().cpu().numpy().astype(np.float32)

    np.save(emb_dir / "e_ret.npy", e_ret)
    np.save(emb_dir / "e_dec.npy", e_dec)

    tok2idx = vocab._tok2idx.copy()
    idx2tok = {str(v): k for k, v in tok2idx.items()}
    idx2tok["0"] = "<pad>"

    with open(emb_dir / "ticker_to_idx.json", "w") as f:
        json.dump(tok2idx, f, indent=2, sort_keys=True)
    with open(emb_dir / "idx_to_ticker.json", "w") as f:
        json.dump(idx2tok, f, indent=2)
    with open(emb_dir / "meta.json", "w") as f:
        json.dump({
            "vocab_size":  int(e_ret.shape[0]),
            "embed_dim":   int(e_ret.shape[1]),
            "n_tickers":   len(tok2idx),
            "padding_idx": 0,
            "tables": {
                "e_ret": "retrieval geometry — used by the presence head (inner-product lookup)",
                "e_dec": "encoder/feature geometry — used by the set encoder and feature MLPs",
            },
        }, f, indent=2)

    return emb_dir
