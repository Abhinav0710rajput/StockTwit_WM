#!/usr/bin/env python3
"""
Convert StockTwits CSV exports to Parquet.

There are two CSV families and they should be parsed differently:
1. feature_wo_messages/*.csv
   - structured tabular data
   - parsed with Dask for scalable batch conversion
   - adds year/month partitions from created_at

2. messages/*.csv
   - contains free-text message bodies
   - may include commas, quotes, and multiline content
   - parsed one file at a time with pandas + Python CSV engine for safety

Example:
    python csv_to_parquet_clean.py --base /scratch/xl2860/stocktwits_dataset
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from pathlib import Path

import pandas as pd
import dask.dataframe as dd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert StockTwits CSV data to Parquet.")
    parser.add_argument(
        "--base",
        type=str,
        default="/scratch/xl2860/stocktwits_dataset",
        help="Base dataset directory containing csv/ and parquet/ folders.",
    )
    parser.add_argument(
        "--feature-blocksize",
        type=str,
        default="512MB",
        help="Dask blocksize for feature CSVs.",
    )
    parser.add_argument(
        "--messages-compression",
        type=str,
        default="snappy",
        help="Compression codec for messages parquet files.",
    )
    parser.add_argument(
        "--features-compression",
        type=str,
        default="snappy",
        help="Compression codec for feature parquet dataset.",
    )
    parser.add_argument(
        "--overwrite-messages",
        action="store_true",
        help="Rewrite message parquet files even if they already exist.",
    )
    parser.add_argument(
        "--overwrite-features",
        action="store_true",
        help="Rewrite the feature parquet dataset if it already exists.",
    )
    return parser.parse_args()


def convert_features(base: str, blocksize: str, compression: str, overwrite: bool) -> None:
    feature_glob = os.path.join(base, "csv", "feature_wo_messages", "*.csv")
    feature_out = os.path.join(base, "parquet", "feature_wo_messages")
    Path(feature_out).mkdir(parents=True, exist_ok=True)

    in_files = sorted(glob.glob(feature_glob))
    if not in_files:
        print(f"[features] no input files found: {feature_glob}")
        return

    if os.path.exists(feature_out) and os.listdir(feature_out) and not overwrite:
        print(f"[features] output already exists, skipping: {feature_out}")
        print("[features] use --overwrite-features to rewrite it.")
        return

    if overwrite and os.path.exists(feature_out):
        for root, dirs, files in os.walk(feature_out, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))

    print(f"[features] reading {len(in_files)} files from {feature_glob}")

    feature_dtypes = {
        "message_id": "object",
        "user_id": "float64",
        "sentiment": "object",
        "parent_message_id": "object",
        "in_reply_to_message_id": "object",
        "symbol_list": "object",
    }

    df = dd.read_csv(
        feature_glob,
        dtype=feature_dtypes,
        assume_missing=True,
        blocksize=blocksize,
    )

    df["created_at"] = dd.to_datetime(df["created_at"], errors="coerce", utc=True)
    df["year"] = df["created_at"].dt.year
    df["month"] = df["created_at"].dt.month

    print(f"[features] writing parquet dataset to {feature_out}")
    df.to_parquet(
        feature_out,
        engine="pyarrow",
        compression=compression,
        write_index=False,
        partition_on=["year", "month"],
        overwrite=overwrite,
    )
    print("[features] done")



def convert_messages(base: str, compression: str, overwrite: bool) -> None:
    messages_glob = os.path.join(base, "csv", "messages", "*.csv")
    messages_out = os.path.join(base, "parquet", "messages")
    Path(messages_out).mkdir(parents=True, exist_ok=True)

    in_files = sorted(glob.glob(messages_glob))
    if not in_files:
        print(f"[messages] no input files found: {messages_glob}")
        return

    print(f"[messages] converting {len(in_files)} files")

    for i, in_file in enumerate(in_files):
        out_file = os.path.join(messages_out, f"part_{i:03d}.parquet")

        if os.path.exists(out_file) and not overwrite:
            print(f"[messages] skip existing {out_file}")
            continue

        print(f"[messages] [{i + 1}/{len(in_files)}] reading {in_file}")

        # Important:
        # message_body is free text and may contain commas, quotes, or line breaks.
        # The Python CSV engine is slower but safer for this kind of data.
        df = pd.read_csv(
            in_file,
            dtype={"message_id": "object", "message_body": "object"},
            engine="python",
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL,
            on_bad_lines="warn",
        )

        df.to_parquet(
            out_file,
            engine="pyarrow",
            compression=compression,
            index=False,
        )
        print(f"[messages] wrote {out_file}, shape={df.shape}")

    print("[messages] done")



def main() -> None:
    args = parse_args()
    convert_features(
        base=args.base,
        blocksize=args.feature_blocksize,
        compression=args.features_compression,
        overwrite=args.overwrite_features,
    )
    convert_messages(
        base=args.base,
        compression=args.messages_compression,
        overwrite=args.overwrite_messages,
    )


if __name__ == "__main__":
    main()
