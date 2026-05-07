"""
Export learned ticker embeddings from a trained TwitWave checkpoint.

Useful for:
  - Downstream similarity / clustering analysis outside the main pipeline
  - Re-exporting embeddings from any saved epoch (not just the final best.pt)
  - Sharing embeddings with other researchers without the full model weights

Usage:
    # From the default best checkpoint:
    python scripts/export_embeddings.py \
        --model_dir outputs/rssm_base \
        --data_dir  data/processed_week

    # From a specific epoch checkpoint:
    python scripts/export_embeddings.py \
        --model_dir outputs/rssm_base \
        --ckpt      outputs/rssm_base/checkpoints/epoch_050.pt \
        --data_dir  data/processed_week \
        --out_dir   outputs/rssm_base/embeddings_epoch050

Output files written to <out_dir>/embeddings/ (or <model_dir>/embeddings/ by default):

    e_ret.npy           (vocab_size, embed_dim) float32
                        Retrieval geometry — used by the presence head.
                        Row i is the embedding for ticker vocab[i].
                        Cosine similarity between rows reflects how similarly
                        a ticker is positioned in the presence-detection space.

    e_dec.npy           (vocab_size, embed_dim) float32
                        Encoder / feature geometry — used by the set encoder
                        and per-feature MLPs.
                        Row i is the embedding for ticker vocab[i].

    ticker_to_idx.json  {"AAPL": 1, "GME": 2, ...}
                        Map ticker symbol → embedding row index.
                        Row 0 is always <pad> (all zeros).

    idx_to_ticker.json  {"0": "<pad>", "1": "AAPL", "2": "GME", ...}
                        Reverse map — embedding row index → ticker symbol.

    meta.json           vocab_size, embed_dim, n_tickers, table descriptions
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from data.vocab import Vocabulary
from eval.utils import load_rssm, export_embeddings
from configs import load_config
from model.twit_wave import TwitWave, ModelConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", type=str, required=True,
                   help="Training output directory (contains config.yaml, norm_stats.json)")
    p.add_argument("--data_dir",  type=str, default="data/processed_week",
                   help="Directory containing vocab.json")
    p.add_argument("--ckpt",      type=str, default=None,
                   help="Specific checkpoint to load. Defaults to <model_dir>/checkpoints/best.pt")
    p.add_argument("--out_dir",   type=str, default=None,
                   help="Where to write embeddings/. Defaults to <model_dir>/embeddings/")
    return p.parse_args()


def main() -> None:
    args  = parse_args()
    model_dir = Path(args.model_dir)
    data_dir  = Path(args.data_dir)
    out_dir   = Path(args.out_dir) if args.out_dir else model_dir
    ckpt_path = Path(args.ckpt)   if args.ckpt    else model_dir / "checkpoints" / "best.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    vocab = Vocabulary.load(data_dir / "vocab.json")
    log.info("Vocab: %d tickers (table size %d incl. padding)", len(vocab), vocab.size)

    # Load config and build model from scratch, then load weights
    cfg  = load_config(model_dir / "config.yaml")
    mcfg = cfg["model"]
    model_cfg = ModelConfig(
        vocab_size  = vocab.size,
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
        dropout     = 0.0,
    )
    model = TwitWave(model_cfg)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt.get("model_state", ckpt))
    model.eval()
    log.info("Loaded checkpoint: %s  (step %d)", ckpt_path, ckpt.get("global_step", -1))

    emb_dir = export_embeddings(model, vocab, out_dir)

    # Print quick stats
    e_ret = np.load(emb_dir / "e_ret.npy")
    e_dec = np.load(emb_dir / "e_dec.npy")
    log.info("e_ret shape: %s  norm mean=%.4f", e_ret.shape, float(np.linalg.norm(e_ret[1:], axis=1).mean()))
    log.info("e_dec shape: %s  norm mean=%.4f", e_dec.shape, float(np.linalg.norm(e_dec[1:], axis=1).mean()))
    log.info("Embeddings exported → %s/", emb_dir)
    log.info("  e_ret.npy, e_dec.npy, ticker_to_idx.json, idx_to_ticker.json, meta.json")


if __name__ == "__main__":
    main()
