"""
Step 2c: Build weekly ticker-ticker fact networks from post-level StockTwits data.

This creates observed networks that can be compared with the model's learned
cross-ticker attention matrix A_t:

1. message_co_mentions:
   Two tickers are linked when they appear together in the same message's
   symbol_list during the same week.

2. user_week_co_mentions:
   Two tickers are linked when the same user mentions both tickers at least
   once during the same week. This is a broader shared-attention signal.

The script uses DuckDB over partitioned parquet and processes one year at a
time, so it does not load the full raw dataset into memory.

Example:
    python scripts/2_c_week_ticker_network_fact.py ^
        --raw_dir C:\\stocktwits_2026\\parquet\\feature_wo_messages ^
        --panel_path data\\processed_week\\panel_all.parquet ^
        --out_dir data\\processed_week\\ticker_network_fact
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
import uuid
from pathlib import Path

import duckdb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--raw_dir",
        type=str,
        default=r"C:\stocktwits_2026\parquet\feature_wo_messages",
        help="Partitioned parquet root with year=*/month=*/part.*.parquet files.",
    )
    p.add_argument(
        "--panel_path",
        type=str,
        default="data/processed_week/panel_all.parquet",
        help="Weekly top-K panel. Used to restrict network nodes to model-observed tickers.",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="data/processed_week/ticker_network_fact",
        help="Output folder for weekly node/edge parquet files.",
    )
    p.add_argument("--start_year", type=int, default=2008)
    p.add_argument("--end_year", type=int, default=2022)
    p.add_argument(
        "--all_tickers",
        action="store_true",
        help="Do not restrict to panel top-K symbols. This can be much larger.",
    )
    p.add_argument(
        "--max_tickers_per_user_week",
        type=int,
        default=50,
        help="Cap user-week symbol sets before pair expansion to avoid pathological users.",
    )
    p.add_argument(
        "--min_edge_weight",
        type=int,
        default=1,
        help="Drop final edges with both message and user weights below this value.",
    )
    p.add_argument(
        "--keep_tmp",
        action="store_true",
        help="Keep per-year partial parquet files for debugging.",
    )
    return p.parse_args()


def sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def partition_glob(path: Path) -> str:
    return (path / "*.parquet").resolve().as_posix().replace("'", "''")


def remove_tree_with_retries(path: Path, attempts: int = 5, delay: float = 0.5) -> None:
    if not path.exists():
        return
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            shutil.rmtree(path)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(delay)
    raise PermissionError(f"Could not remove temporary directory after retries: {path}") from last_error


def build_year_partials(
    con: duckdb.DuckDBPyConnection,
    raw_dir: Path,
    panel_path: Path,
    tmp_dir: Path,
    year: int,
    all_tickers: bool,
    max_tickers_per_user_week: int,
    month_dir: Path | None = None,
) -> None:
    if month_dir is None:
        month_dirs = sorted((raw_dir / f"year={year}").glob("month=*"))
        if not month_dirs:
            raise FileNotFoundError(f"No month partitions found for year={year}")
        # Backward-compatible fallback: process a whole year only when called
        # explicitly without a month_dir. The main path below uses months.
        glob = (raw_dir / f"year={year}" / "**" / "*.parquet").resolve().as_posix().replace("'", "''")
        label = f"year={year}"
    else:
        glob = partition_glob(month_dir)
        label = f"year={year}_{month_dir.name}"

    safe_label = label.replace("=", "").replace("/", "_").replace("\\", "_")
    edge_path = tmp_dir / f"edges_{safe_label}.parquet"
    node_path = tmp_dir / f"nodes_{safe_label}.parquet"

    panel_join = ""
    panel_where = ""
    if not all_tickers:
        panel_join = """
        INNER JOIN panel_symbols p
          ON p.week = s.week AND p.symbol = s.symbol
        """
        panel_where = "WHERE p.symbol IS NOT NULL"

    log.info("Processing %s", label)

    con.execute("DROP TABLE IF EXISTS panel_symbols")
    if not all_tickers:
        con.execute(
            f"""
            CREATE TEMP TABLE panel_symbols AS
            SELECT DISTINCT CAST(week AS DATE) AS week, UPPER(symbol) AS symbol
            FROM read_parquet('{sql_path(panel_path)}')
            WHERE EXTRACT(year FROM CAST(week AS DATE)) BETWEEN {year - 1} AND {year + 1}
            """
        )

    con.execute("DROP TABLE IF EXISTS symbols_year")
    con.execute(
        f"""
        CREATE TEMP TABLE symbols_year AS
        WITH raw AS (
            SELECT
                message_id,
                CAST(user_id AS VARCHAR) AS user_id,
                CAST(date_trunc('week', created_at) AS DATE) AS week,
                sentiment,
                symbol_list
            FROM read_parquet('{glob}', hive_partitioning=true)
            WHERE symbol_list IS NOT NULL
              AND symbol_list <> '[]'
              AND created_at IS NOT NULL
        ),
        exploded AS (
            SELECT
                week,
                message_id,
                user_id,
                sentiment,
                UPPER(sym) AS symbol
            FROM raw,
            UNNEST(
                string_split(
                    regexp_replace(symbol_list, '[\\[\\]''" ]', '', 'g'),
                    ','
                )
            ) AS u(sym)
            WHERE sym IS NOT NULL AND sym <> ''
        ),
        s AS (
            SELECT DISTINCT week, message_id, user_id, sentiment, symbol
            FROM exploded
            WHERE symbol <> ''
        )
        SELECT s.*
        FROM s
        {panel_join}
        {panel_where}
        """
    )

    n_symbols = con.execute("SELECT COUNT(*) FROM symbols_year").fetchone()[0]
    log.info("  exploded ticker-message rows after filter: %s", f"{n_symbols:,}")

    con.execute(
        f"""
        COPY (
            SELECT
                week,
                symbol,
                COUNT(DISTINCT message_id) AS message_count,
                COUNT(DISTINCT user_id) AS user_count,
                SUM(CASE WHEN sentiment = 'Bullish' THEN 1 ELSE 0 END) AS bullish_mentions,
                SUM(CASE WHEN sentiment = 'Bearish' THEN 1 ELSE 0 END) AS bearish_mentions,
                SUM(CASE WHEN sentiment IS NULL OR sentiment NOT IN ('Bullish', 'Bearish') THEN 1 ELSE 0 END) AS unlabeled_mentions
            FROM symbols_year
            GROUP BY week, symbol
        ) TO '{sql_path(node_path)}' (FORMAT PARQUET)
        """
    )

    con.execute(
        f"""
        COPY (
            WITH message_pairs AS (
                SELECT
                    a.week,
                    LEAST(a.symbol, b.symbol) AS ticker_i,
                    GREATEST(a.symbol, b.symbol) AS ticker_j,
                    COUNT(DISTINCT a.message_id) AS message_co_mentions
                FROM symbols_year a
                JOIN symbols_year b
                  ON a.week = b.week
                 AND a.message_id = b.message_id
                 AND a.symbol < b.symbol
                GROUP BY 1, 2, 3
            ),
            user_week_sets AS (
                SELECT week, user_id, symbol
                FROM symbols_year
                QUALIFY COUNT(DISTINCT symbol) OVER (PARTITION BY week, user_id)
                    BETWEEN 2 AND {max_tickers_per_user_week}
            ),
            user_pairs AS (
                SELECT
                    a.week,
                    LEAST(a.symbol, b.symbol) AS ticker_i,
                    GREATEST(a.symbol, b.symbol) AS ticker_j,
                    COUNT(DISTINCT a.user_id) AS user_week_co_mentions
                FROM user_week_sets a
                JOIN user_week_sets b
                  ON a.week = b.week
                 AND a.user_id = b.user_id
                 AND a.symbol < b.symbol
                GROUP BY 1, 2, 3
            )
            SELECT
                COALESCE(m.week, u.week) AS week,
                COALESCE(m.ticker_i, u.ticker_i) AS ticker_i,
                COALESCE(m.ticker_j, u.ticker_j) AS ticker_j,
                COALESCE(m.message_co_mentions, 0) AS message_co_mentions,
                COALESCE(u.user_week_co_mentions, 0) AS user_week_co_mentions
            FROM message_pairs m
            FULL OUTER JOIN user_pairs u
              ON m.week = u.week
             AND m.ticker_i = u.ticker_i
             AND m.ticker_j = u.ticker_j
        ) TO '{sql_path(edge_path)}' (FORMAT PARQUET)
        """
    )

    n_edges = con.execute(f"SELECT COUNT(*) FROM read_parquet('{sql_path(edge_path)}')").fetchone()[0]
    log.info("  partial edges: %s", f"{n_edges:,}")


def finalize_outputs(
    con: duckdb.DuckDBPyConnection,
    tmp_dir: Path,
    out_dir: Path,
    min_edge_weight: int,
    metadata: dict,
) -> None:
    edges_glob = (tmp_dir / "edges_*.parquet").resolve().as_posix()
    nodes_glob = (tmp_dir / "nodes_*.parquet").resolve().as_posix()

    edges_out = out_dir / "weekly_ticker_edges.parquet"
    nodes_out = out_dir / "weekly_ticker_nodes.parquet"
    summary_out = out_dir / "weekly_network_summary.parquet"

    log.info("Finalizing edge aggregates")
    con.execute(
        f"""
        COPY (
            SELECT
                week,
                ticker_i,
                ticker_j,
                SUM(message_co_mentions) AS message_co_mentions,
                SUM(user_week_co_mentions) AS user_week_co_mentions,
                SUM(message_co_mentions) + SUM(user_week_co_mentions) AS total_weight
            FROM read_parquet('{edges_glob}')
            GROUP BY 1, 2, 3
            HAVING
                SUM(message_co_mentions) >= {min_edge_weight}
                OR SUM(user_week_co_mentions) >= {min_edge_weight}
            ORDER BY week, total_weight DESC, ticker_i, ticker_j
        ) TO '{sql_path(edges_out)}' (FORMAT PARQUET)
        """
    )

    log.info("Finalizing node aggregates")
    con.execute(
        f"""
        COPY (
            SELECT
                week,
                symbol,
                SUM(message_count) AS message_count,
                SUM(user_count) AS user_count,
                SUM(bullish_mentions) AS bullish_mentions,
                SUM(bearish_mentions) AS bearish_mentions,
                SUM(unlabeled_mentions) AS unlabeled_mentions
            FROM read_parquet('{nodes_glob}')
            GROUP BY 1, 2
            ORDER BY week, message_count DESC, symbol
        ) TO '{sql_path(nodes_out)}' (FORMAT PARQUET)
        """
    )

    log.info("Writing weekly summary")
    con.execute(
        f"""
        COPY (
            WITH e AS (
                SELECT
                    week,
                    COUNT(*) AS n_edges,
                    SUM(message_co_mentions) AS message_co_mentions,
                    SUM(user_week_co_mentions) AS user_week_co_mentions,
                    SUM(total_weight) AS total_weight
                FROM read_parquet('{sql_path(edges_out)}')
                GROUP BY week
            ),
            n AS (
                SELECT
                    week,
                    COUNT(*) AS n_nodes,
                    SUM(message_count) AS message_count,
                    SUM(user_count) AS user_count
                FROM read_parquet('{sql_path(nodes_out)}')
                GROUP BY week
            )
            SELECT
                COALESCE(n.week, e.week) AS week,
                COALESCE(n.n_nodes, 0) AS n_nodes,
                COALESCE(e.n_edges, 0) AS n_edges,
                COALESCE(n.message_count, 0) AS message_count,
                COALESCE(n.user_count, 0) AS user_count,
                COALESCE(e.message_co_mentions, 0) AS message_co_mentions,
                COALESCE(e.user_week_co_mentions, 0) AS user_week_co_mentions,
                COALESCE(e.total_weight, 0) AS total_weight
            FROM n
            FULL OUTER JOIN e ON n.week = e.week
            ORDER BY week
        ) TO '{sql_path(summary_out)}' (FORMAT PARQUET)
        """
    )

    metadata.update(
        {
            "edges_path": str(edges_out),
            "nodes_path": str(nodes_out),
            "summary_path": str(summary_out),
        }
    )
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    n_edges = con.execute(f"SELECT COUNT(*) FROM read_parquet('{sql_path(edges_out)}')").fetchone()[0]
    n_nodes = con.execute(f"SELECT COUNT(*) FROM read_parquet('{sql_path(nodes_out)}')").fetchone()[0]
    n_weeks = con.execute(f"SELECT COUNT(DISTINCT week) FROM read_parquet('{sql_path(summary_out)}')").fetchone()[0]
    log.info("Done: %s nodes, %s edges, %s weeks", f"{n_nodes:,}", f"{n_edges:,}", f"{n_weeks:,}")


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    panel_path = Path(args.panel_path)
    out_dir = Path(args.out_dir)
    tmp_dir = out_dir / f"_tmp_year_partials_{uuid.uuid4().hex[:8]}"

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw parquet directory not found: {raw_dir}")
    if not args.all_tickers and not panel_path.exists():
        raise FileNotFoundError(f"Panel path not found: {panel_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True)

    con = duckdb.connect()
    con.execute("PRAGMA threads=4")

    metadata = {
        "raw_dir": str(raw_dir),
        "panel_path": None if args.all_tickers else str(panel_path),
        "out_dir": str(out_dir),
        "start_year": args.start_year,
        "end_year": args.end_year,
        "all_tickers": args.all_tickers,
        "max_tickers_per_user_week": args.max_tickers_per_user_week,
        "min_edge_weight": args.min_edge_weight,
        "edge_definitions": {
            "message_co_mentions": "Number of same-week messages containing ticker_i and ticker_j in symbol_list.",
            "user_week_co_mentions": "Number of same-week users who mentioned both ticker_i and ticker_j at least once.",
        },
    }

    try:
        for year in range(args.start_year, args.end_year + 1):
            year_dir = raw_dir / f"year={year}"
            if not year_dir.exists():
                log.warning("Skipping missing %s", year_dir)
                continue
            month_dirs = sorted(year_dir.glob("month=*"), key=lambda p: int(p.name.split("=")[-1]))
            for month_dir in month_dirs:
                build_year_partials(
                    con=con,
                    raw_dir=raw_dir,
                    panel_path=panel_path,
                    tmp_dir=tmp_dir,
                    year=year,
                    all_tickers=args.all_tickers,
                    max_tickers_per_user_week=args.max_tickers_per_user_week,
                    month_dir=month_dir,
                )

        finalize_outputs(con, tmp_dir, out_dir, args.min_edge_weight, metadata)
    finally:
        con.close()
        if tmp_dir.exists() and not args.keep_tmp:
            try:
                remove_tree_with_retries(tmp_dir)
            except PermissionError as exc:
                log.warning("%s", exc)
                log.warning("Temporary files remain at %s", tmp_dir)


if __name__ == "__main__":
    main()
