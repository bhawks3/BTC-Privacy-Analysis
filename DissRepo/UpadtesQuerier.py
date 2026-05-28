import argparse
import csv
import duckdb
import os
from typing import Dict, Tuple


class CoinSwapEstimator:
    def __init__(self, db_path: str, threads: int = 8, memory_limit: str = "2000GB"):
        self.conn = duckdb.connect(database=db_path, read_only=False)
        self.conn.execute(f"SET memory_limit = '{memory_limit}';")
        self.conn.execute(f"SET threads TO {threads};")

    def get_block_bounds(self) -> Tuple[int, int]:
        min_height, max_height = self.conn.execute(
            "SELECT MIN(height), MAX(height) FROM blocks"
        ).fetchone()
        if min_height is None or max_height is None:
            raise RuntimeError("Could not read block height bounds from the database.")
        return min_height, max_height

    def count_coinswap_period(
        self,
        start_block: int,
        end_block: int,
        lookahead: int = 30,
    ) -> Dict[str, int]:
        """
        Count candidate CoinSwap outputs in a block range.

        Heuristic:
          1. Select 2-of-2 multisig outputs created in the period.
          2. Keep only those spent within 30 blocks after creation.
          3. Prune transactions containing 3 or more such outputs.
          4. Count outputs in the pruned set that have another pruned output
             within the next 30 blocks. AND within 20000 satoshis
  
        """
        sql = f"""
            WITH candidate_outputs AS (
                SELECT
                    o.txid,
                    o.indexOut,
                    o.tValue AS value,
                    cb.height AS creation_height
                FROM tx_out o
                JOIN transactions t ON o.txid = t.txid
                JOIN blocks cb ON t.hashBlock = cb.block_hash
                JOIN tx_in ti
                    ON ti.hashPrevOut = o.txid
                   AND ti.indexPrevOut = o.indexOut
                JOIN transactions st ON ti.txid = st.txid
                JOIN blocks sb ON st.hashBlock = sb.block_hash
                WHERE cb.height BETWEEN {start_block} AND {end_block}
                  AND sb.height BETWEEN cb.height AND cb.height + {lookahead}
                  AND (
                      lower(o.scriptPubKey) LIKE '%op_checkmultisig%'
                      OR lower(o.scriptPubKey) LIKE '%op_checkmultisigverify%'
                      OR lower(ti.scriptSig) LIKE '%op_checkmultisig%'
                      OR lower(ti.scriptSig) LIKE '%op_checkmultisigverify%'
                      OR lower(o.scriptPubKey) LIKE '%ae'
                      OR lower(o.scriptPubKey) LIKE '%af'
                      OR lower(ti.scriptSig) LIKE '%ae'
                      OR lower(ti.scriptSig) LIKE '%af'
                  )
            ),
            tx_counts AS (
                SELECT txid, COUNT(*) AS output_count
                FROM candidate_outputs
                GROUP BY txid
            ),
            pruned AS (
                SELECT c.*
                FROM candidate_outputs c
                JOIN tx_counts tc ON c.txid = tc.txid
                WHERE tc.output_count < 3
            ),
            matches AS (
                SELECT p1.txid || ':' || p1.indexOut AS output_id
                FROM pruned p1
                JOIN pruned p2
                  ON p2.creation_height > p1.creation_height
                 AND p2.creation_height <= p1.creation_height + {lookahead}
                 AND ABS(p1.value - p2.value) <= 20000
            )
            SELECT
                (SELECT COUNT(*) FROM candidate_outputs) AS total_candidate_outputs,
                (SELECT COUNT(*) FROM pruned) AS pruned_outputs,
                COUNT(*) AS matching_pairs,
                COUNT(DISTINCT output_id) AS outputs_with_match
            FROM matches;
        """

        result = self.conn.execute(sql).fetchone()
        if result is None:
            raise RuntimeError("CoinSwap query returned no result.", flush=True)

        return {
            "start_block": start_block,
            "end_block": end_block,
            "total_candidate_outputs": result[0],
            "pruned_outputs": result[1],
            "matching_pairs": result[2],
            "outputs_with_match": result[3],
        }

    def close(self) -> None:
        self.conn.close()


def run_periods(
    db_path: str,
    output_csv: str,
    period_length: int = 1008,
    lookahead: int = 30,
    start_block: int = None,
    end_block: int = None,
) -> None:
    estimator = CoinSwapEstimator(db_path=db_path)
    try:
        min_height, max_height = estimator.get_block_bounds()
        if start_block is None:
            start_block = min_height
        if end_block is None:
            end_block = max_height

        if start_block < min_height or end_block > max_height:
            raise ValueError(
                f"Requested block range [{start_block}, {end_block}] is outside database bounds [{min_height}, {max_height}]"
            )

        print(
            f"Starting CoinSwap estimation: db={db_path}, output={output_csv}, "
            f"blocks={start_block}-{end_block}, period_length={period_length}, lookahead={lookahead}",
            flush=True,
        )

        fieldnames = [
            "start_block",
            "end_block",
            "total_candidate_outputs",
            "pruned_outputs",
            "matching_pairs",
            "outputs_with_match",
        ]

        write_header = not os.path.exists(output_csv)

        with open(output_csv, mode="a", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()

            current_block = start_block
            while current_block <= end_block:
                period_end = min(current_block + period_length - 1, end_block)
                print(f"Processing blocks {current_block}-{period_end}...", flush=True)
                row = estimator.count_coinswap_period(
                    current_block, period_end, lookahead=lookahead
                )
                print(
                    f"  candidates={row['total_candidate_outputs']}, "
                    f"pruned={row['pruned_outputs']}, "
                    f"pairs={row['matching_pairs']}, "
                    f"outputs_with_match={row['outputs_with_match']}"
                ,flush=True)
                writer.writerow(row)
                csvfile.flush()
                current_block = period_end + 1

    finally:
        estimator.close()


if __name__ == "__main__":
    import os

    parser = argparse.ArgumentParser(description="Estimate CoinSwap candidates in 1008-block windows.")
    parser.add_argument("--db", default='database.db', help="Path to the DuckDB database file.")
    parser.add_argument("--output", default="coinswap_estimate.csv", help="CSV file to append results.")
    parser.add_argument("--period-length", type=int, default=1008, help="Number of blocks per period.")
    parser.add_argument("--lookahead", type=int, default=30, help="Lookahead window in blocks.")
    parser.add_argument("--start-block", type=int, default=None, help="Optional starting block height.")
    parser.add_argument("--end-block", type=int, default=None, help="Optional ending block height.")
    args = parser.parse_args()

    run_periods(
        db_path=args.db,
        output_csv=args.output,
        period_length=args.period_length,
        lookahead=args.lookahead,
        start_block=758608,
        end_block=args.end_block,
    )
