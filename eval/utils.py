"""Shared utilities for evaluation scripts."""

from __future__ import annotations

from pathlib import Path

import torch

from configs import load_config
from model.twit_wave import TwitWave, ModelConfig


def _checkpoint_path(model_dir: Path) -> Path:
    candidates = [
        model_dir / "checkpoints" / "best.pt",
        model_dir / "best.pt",
        model_dir / "best_model.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find checkpoint. Tried: "
        + ", ".join(str(path) for path in candidates)
    )


def _state_dict_from_checkpoint(ckpt: object) -> dict:
    if isinstance(ckpt, dict):
        for key in ("model_state", "model_state_dict", "state_dict", "model"):
            value = ckpt.get(key)
            if isinstance(value, dict) and any(torch.is_tensor(v) for v in value.values()):
                return value
        if ckpt and all(torch.is_tensor(v) for v in ckpt.values()):
            return ckpt
    raise ValueError("Checkpoint does not contain an obvious model state_dict")


def _infer_gru_input_extra_dim(state_dict: dict, s_dim: int) -> int:
    weight = state_dict.get("rssm.gru.weight_ih")
    if weight is None:
        return 0
    extra_dim = int(weight.shape[1]) - int(s_dim)
    if extra_dim < 0:
        raise ValueError(
            f"Checkpoint GRU input width {weight.shape[1]} is smaller than s_dim={s_dim}"
        )
    return extra_dim


def _normalize_state_dict_keys(state_dict: dict) -> dict:
    replacements = {
        "rssm.posterior_net.3.": "rssm.posterior_net.2.",
        "rssm.posterior_net.6.": "rssm.posterior_net.4.",
        "rssm.prior_net.3.": "rssm.prior_net.2.",
        "rssm.prior_net.6.": "rssm.prior_net.4.",
    }
    normalized = {}
    for key, value in state_dict.items():
        new_key = key.removeprefix("module.")
        for old, new in replacements.items():
            if new_key.startswith(old):
                new_key = new + new_key[len(old):]
                break
        normalized[new_key] = value
    return normalized


def load_rssm(model_dir: str | Path, vocab_size: int, device: torch.device) -> TwitWave:
    """
    Load a trained TwitWave model from an output directory.
    Expects model_dir/config.yaml and a checkpoint at checkpoints/best.pt,
    best.pt, or best_model.pt.
    """
    model_dir = Path(model_dir)
    cfg  = load_config(model_dir / "config.yaml")
    mcfg = cfg["model"]
    ckpt_path = _checkpoint_path(model_dir)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = _normalize_state_dict_keys(_state_dict_from_checkpoint(ckpt))
    gru_input_extra_dim = _infer_gru_input_extra_dim(state_dict, mcfg["s_dim"])

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
        gru_input_extra_dim = gru_input_extra_dim,
    )
    model = TwitWave(model_cfg)
    model.load_state_dict(state_dict)
    model.eval()
    return model.to(device)
