"""
Ticker vocabulary: assigns a stable integer index to every ticker
that ever appears in the panel, and provides save/load utilities.
"""

from __future__ import annotations

import json
from pathlib import Path


PADDING_IDX = 0   # reserved; real tickers start at 1


class Vocabulary:
    def __init__(self) -> None:
        self._tok2idx: dict[str, int] = {}
        self._idx2tok: dict[int, str] = {}

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    @classmethod
    def build(cls, symbols: list[str]) -> "Vocabulary":
        """Build vocab from a list of ticker symbols."""
        vocab = cls()
        for i, sym in enumerate(sorted(symbols), start=1):   # 0 is padding
            vocab._tok2idx[sym] = i
            vocab._idx2tok[i] = sym
        print(f"[vocab] built: {len(vocab)} tickers (indices 1–{len(vocab)})")
        return vocab

    # ------------------------------------------------------------------
    # lookup
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._tok2idx)

    @property
    def size(self) -> int:
        """Total embedding table size including padding token."""
        return len(self._tok2idx) + 1

    def encode(self, symbol: str) -> int:
        return self._tok2idx.get(symbol, PADDING_IDX)

    def decode(self, idx: int) -> str:
        return self._idx2tok.get(idx, "<unk>")

    def has(self, symbol: str) -> bool:
        return symbol in self._tok2idx

    def encode_list(self, symbols: list[str]) -> list[int]:
        return [self.encode(s) for s in symbols]

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"tok2idx": self._tok2idx}, f, indent=2)
        print(f"[vocab] saved {len(self)} tickers → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        with open(path) as f:
            data = json.load(f)
        vocab = cls()
        vocab._tok2idx = data["tok2idx"]
        vocab._idx2tok = {v: k for k, v in vocab._tok2idx.items()}
        print(f"[vocab] loaded {len(vocab)} tickers from {path}")
        return vocab
