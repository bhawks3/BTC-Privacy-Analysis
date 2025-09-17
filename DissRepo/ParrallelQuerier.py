import duckdb
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Dict, Any
import concurrent.futures


class BlockchainAnalyzer:
    def __init__(self, db_path: str):
        """
        Initialize the analyzer with path to DuckDB database
        """
        self.conn = duckdb.connect(db_path)
        self.block_stats = []  # Store block period statistics
        self.heuristic_results = {}  # Store results for each heuristic

    def get_block_range(self) -> Tuple[int, int]:
        """
        Get the minimum and maximum block numbers in the database
        """
        result = self.conn.execute("SELECT MIN(height), MAX(height) FROM blocks").fetchone()
        return result[0], result[1]

    def collect_block_statistics(self, period_length: int = 1008) -> List[dict]:
        """
        PHASE 1: Collect block statistics (run once)
        Returns: block periods with total transactions and basic metrics
        """
        min_block, max_block = self.get_block_range()
        print(f"Collecting block statistics from {min_block} to {max_block}")

        self.block_stats = []
        current_block = min_block

        while current_block <= max_block:
            end_block = min(current_block + period_length - 1, max_block)

            # Get total transactions in this block range
            stats_query = f"""
            SELECT 
                COUNT(DISTINCT t.txid) AS total_transactions
            FROM transactions t
            JOIN blocks b ON t.hashBlock = b.block_hash
            WHERE b.height BETWEEN {current_block} AND {end_block}
            """

            result = self.conn.execute(stats_query).fetchone()
            total_transactions = result[0] if result else 0

            if total_transactions > 0:
                blocks_in_period = end_block - current_block + 1
                avg_tx_per_block = total_transactions / blocks_in_period

                period_data = {
                    'start_block': current_block,
                    'end_block': end_block,
                    'total_transactions': total_transactions,
                    'blocks_in_period': blocks_in_period,
                    'avg_tx_per_block': avg_tx_per_block
                }

                self.block_stats.append(period_data)
                print(
                    f"Blocks {current_block}-{end_block}: {total_transactions} transactions, {avg_tx_per_block:.2f} avg/block")

            current_block = end_block + 1

        print(f"Collected statistics for {len(self.block_stats)} block periods")
        return self.block_stats

    def run_heuristic(self, heuristic_name: str, heuristic_query_template: str) -> Dict[str, Any]:
        """
        PHASE 2: Run a specific heuristic on pre-collected block statistics
        """
        if not self.block_stats:
            self.collect_block_statistics()

        results = []
        total_heuristic_count = 0

        print(f"\nRunning heuristic: {heuristic_name}")

        for period in self.block_stats:
            start_block, end_block = period['start_block'], period['end_block']

            # Execute the heuristic-specific query
            heuristic_query = heuristic_query_template.format(
                start_block=start_block,
                end_block=end_block
            )

            result = self.conn.execute(heuristic_query).fetchone()
            heuristic_count = result[0] if result else 0
            total_heuristic_count += heuristic_count

            # Calculate metrics using pre-computed block statistics
            avg_per_block = heuristic_count / period['blocks_in_period']
            normalized_ratio = avg_per_block / period['avg_tx_per_block'] if period['avg_tx_per_block'] > 0 else 0

            period_result = {
                'start_block': start_block,
                'end_block': end_block,
                'heuristic_count': heuristic_count,
                'total_transactions': period['total_transactions'],  # From pre-computed stats
                'avg_per_block': avg_per_block,
                'avg_tx_per_block': period['avg_tx_per_block'],  # From pre-computed stats
                'normalized_ratio': normalized_ratio
            }

            results.append(period_result)

            if heuristic_count > 0:
                print(f"Blocks {start_block}-{end_block}: {heuristic_count} matches, {avg_per_block:.6f} avg/block")

        # Store results
        self.heuristic_results[heuristic_name] = {
            'period_results': results,
            'total_heuristic_count': total_heuristic_count,
            'total_transactions': sum(period['total_transactions'] for period in self.block_stats)
        }

        return self.heuristic_results[heuristic_name]

    def create_optimized_queries(self):
        """Create optimized versions of all heuristic queries using only existing columns"""
    
        # Enable parallel processing
        self.conn.execute("SET threads TO 64;")
        self.conn.execute("SET enable_progress_bar=true;")

        # Create optimized materialized views for common joins
        print("Creating optimized materialized views...")

        # View for transaction details with block information (only using existing columns)
        self.conn.execute("""
            CREATE OR REPLACE VIEW tx_with_blocks AS
            SELECT 
                t.txid, t.hashBlock,
                b.height, b.block_hash
            FROM transactions t
            JOIN blocks b ON t.hashBlock = b.block_hash
        """)

        # View for transaction inputs with previous output details (only using existing columns)
        self.conn.execute("""
            CREATE OR REPLACE VIEW tx_inputs_detailed AS
            SELECT 
                ti.txid, ti.indexPrev, ti.hashPrev,
                prev_out.address as prev_address, prev_out.tValue as prev_value,
                prev_out.scriptPubKey as prev_scriptPubKey
            FROM tx_in ti
            JOIN tx_out prev_out ON ti.hashPrev = prev_out.txid AND ti.indexPrev = prev_out.indexOut
        """)

        # View for transaction outputs (only using existing columns)
        self.conn.execute("""
            CREATE OR REPLACE VIEW tx_outputs_detailed AS
            SELECT 
                txid, indexOut, address, tValue, scriptPubKey
            FROM tx_out
        """)

        print("Materialized views created successfully!")


    # ---- OPTIMIZED HEURISTIC QUERIES ----
    NEW_COINSWAP_QUERY_OPTIMIZED = """
    WITH filtered_transactions AS (
        SELECT DISTINCT t.txid
        FROM tx_with_blocks t
        WHERE t.height BETWEEN {start_block} AND {end_block}
        AND NOT EXISTS (
            SELECT 1 FROM tx_outputs_detailed to2 
            WHERE to2.txid = t.txid AND to2.scriptPubKey LIKE '%OP_RETURN%'
        )
    ),
    transaction_stats AS (
        SELECT 
            t.txid,
            COUNT(DISTINCT ti.prev_address) as unique_prev_addresses,
            COUNT(DISTINCT to_out.address) as unique_output_addresses,
            STDDEV(to_out.tValue) / AVG(to_out.tValue) as value_variation,
            MAX(to_out.tValue) - MIN(to_out.tValue) as value_range
        FROM filtered_transactions t
        JOIN tx_inputs_detailed ti ON t.txid = ti.txid
        JOIN tx_outputs_detailed to_out ON t.txid = to_out.txid
        GROUP BY t.txid
        HAVING unique_prev_addresses >= 2
        AND unique_output_addresses >= 2
        AND value_variation < 0.15
        AND value_range <= 200000
    )
    SELECT COUNT(DISTINCT txid) FROM transaction_stats
    """

    COINJOIN_QUERY_OPTIMIZED = """
    WITH transaction_stats AS (
        SELECT 
            t.txid,
            COUNT(DISTINCT to_out.address) as output_address_count,
            COUNT(DISTINCT ti.hashPrev) as unique_inputs
        FROM tx_with_blocks t
        JOIN tx_inputs_detailed ti ON t.txid = ti.txid
        JOIN tx_outputs_detailed to_out ON t.txid = to_out.txid
        WHERE t.height BETWEEN {start_block} AND {end_block}
        GROUP BY t.txid
        HAVING output_address_count >= 4 
        AND unique_inputs >= output_address_count / 2
    )
    SELECT COUNT(DISTINCT txid) FROM transaction_stats
    """

    COINSHUFFLE_QUERY_OPTIMIZED = """
    WITH transaction_stats AS (
        SELECT 
            t.txid,
            COUNT(DISTINCT ti.hashPrev) as unique_inputs,
            COUNT(DISTINCT to_out.address) as unique_outputs,
            STDDEV(to_out.tValue) / AVG(to_out.tValue) as value_variation
        FROM tx_with_blocks t
        JOIN tx_inputs_detailed ti ON t.txid = ti.txid
        JOIN tx_outputs_detailed to_out ON t.txid = to_out.txid
        WHERE t.height BETWEEN {start_block} AND {end_block}
        GROUP BY t.txid
        HAVING unique_inputs >= 3
        AND unique_inputs = unique_outputs
        AND value_variation < 0.0001
    )
    SELECT COUNT(DISTINCT txid) FROM transaction_stats
    """

    STEALTHADDRESS_QUERY_OPTIMIZED = """
    SELECT COUNT(DISTINCT to_out.txid)
    FROM tx_outputs_detailed to_out
    JOIN tx_with_blocks t ON to_out.txid = t.txid
    WHERE t.height BETWEEN {start_block} AND {end_block}
    AND (
        (to_out.scriptPubKey LIKE '6a26%' AND LENGTH(to_out.scriptPubKey) = 70)
    OR
    (to_out.scriptPubKey LIKE '21%' AND (to_out.scriptPubKey LIKE '2102%' OR to_out.scriptPubKey LIKE '2103%')
    AND LENGTH(to_out.scriptPubKey) = 68)
    )
    """

    def run_heuristic_parallel(self, heuristic_query_template: str, 
                              period_length: int = 1008, 
                              max_workers: int = 32) -> List[dict]:
        """
        Run heuristic in parallel across block ranges
        """
        if not self.block_stats:
            self.collect_block_statistics(period_length)
        
        # Create a list of all block ranges to process
        block_ranges = [(period['start_block'], period['end_block']) 
                       for period in self.block_stats]
        
        results = []
        
        # Function to process a single block range
        def process_block_range(start_block, end_block):
            heuristic_query = heuristic_query_template.format(
                start_block=start_block, end_block=end_block
            )
            result = self.conn.execute(heuristic_query).fetchone()
            heuristic_count = result[0] if result else 0
            
            # Find the corresponding period data
            period_data = next((p for p in self.block_stats 
                              if p['start_block'] == start_block and p['end_block'] == end_block), None)
            
            if period_data:
                avg_per_block = heuristic_count / period_data['blocks_in_period']
                normalized_ratio = avg_per_block / period_data['avg_tx_per_block'] if period_data['avg_tx_per_block'] > 0 else 0
                
                return {
                    'start_block': start_block,
                    'end_block': end_block,
                    'heuristic_count': heuristic_count,
                    'total_transactions': period_data['total_transactions'],
                    'avg_per_block': avg_per_block,
                    'avg_tx_per_block': period_data['avg_tx_per_block'],
                    'normalized_ratio': normalized_ratio
                }
            return None
        
        # Process block ranges in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_range = {
                executor.submit(process_block_range, start, end): (start, end) 
                for start, end in block_ranges
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_range):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        print(f"Blocks {result['start_block']}-{result['end_block']}: "
                              f"{result['heuristic_count']} matches, {result['avg_per_block']:.6f} avg/block")
                except Exception as e:
                    print(f"Error processing block range: {e}")
        
        return results

    def run_coinswap_heuristic_optimized(self, parallel: bool = True, max_workers: int = 32):
        """Run optimized coinswap heuristic"""
        if parallel:
            results = self.run_heuristic_parallel(self.NEW_COINSWAP_QUERY_OPTIMIZED, max_workers=max_workers)
            self.heuristic_results["coinswap_optimized"] = {
                'period_results': results,
                'total_heuristic_count': sum(r['heuristic_count'] for r in results),
                'total_transactions': sum(r['total_transactions'] for r in results)
            }
            return self.heuristic_results["coinswap_optimized"]
        else:
            return self.run_heuristic("coinswap_optimized", self.NEW_COINSWAP_QUERY_OPTIMIZED)

    def run_coinjoin_optimized(self, parallel: bool = True, max_workers: int = 32):
        """Run optimized coinjoin heuristic"""
        if parallel:
            results = self.run_heuristic_parallel(self.COINJOIN_QUERY_OPTIMIZED, max_workers=max_workers)
            self.heuristic_results["coinjoin_optimized"] = {
                'period_results': results,
                'total_heuristic_count': sum(r['heuristic_count'] for r in results),
                'total_transactions': sum(r['total_transactions'] for r in results)
            }
            return self.heuristic_results["coinjoin_optimized"]
        else:
            return self.run_heuristic("coinjoin_optimized", self.COINJOIN_QUERY_OPTIMIZED)

    def run_coinshuffle_optimized(self, parallel: bool = True, max_workers: int = 32):
        """Run optimized coinshuffle heuristic"""
        if parallel:
            results = self.run_heuristic_parallel(self.COINSHUFFLE_QUERY_OPTIMIZED, max_workers=max_workers)
            self.heuristic_results["coinshuffle_optimized"] = {
                'period_results': results,
                'total_heuristic_count': sum(r['heuristic_count'] for r in results),
                'total_transactions': sum(r['total_transactions'] for r in results)
            }
            return self.heuristic_results["coinshuffle_optimized"]
        else:
            return self.run_heuristic("coinshuffle_optimized", self.COINSHUFFLE_QUERY_OPTIMIZED)

    def run_stealth_optimized(self, parallel: bool = True, max_workers: int = 32):
        """Run optimized stealth address heuristic"""
        if parallel:
            results = self.run_heuristic_parallel(self.STEALTHADDRESS_QUERY_OPTIMIZED, max_workers=max_workers)
            self.heuristic_results["stealthaddress_optimized"] = {
                'period_results': results,
                'total_heuristic_count': sum(r['heuristic_count'] for r in results),
                'total_transactions': sum(r['total_transactions'] for r in results)
            }
            return self.heuristic_results["stealthaddress_optimized"]
        else:
            return self.run_heuristic("stealthaddress_optimized", self.STEALTHADDRESS_QUERY_OPTIMIZED)

    def export_heuristic_results(self, heuristic_name: str, filename: str = None):
        """Export results for a specific heuristic"""
        if heuristic_name not in self.heuristic_results:
            print(f"No results found for heuristic: {heuristic_name}")
            return

        if filename is None:
            filename = f'{heuristic_name}_analysis.csv'

        import pandas as pd
        df = pd.DataFrame(self.heuristic_results[heuristic_name]['period_results'])
        df.to_csv(filename, index=False)
        print(f"Results for {heuristic_name} exported to {filename}")

    def get_heuristic_summary(self, heuristic_name: str):
        """Get summary for a specific heuristic"""
        if heuristic_name not in self.heuristic_results:
            print(f"No results found for heuristic: {heuristic_name}")
            return

        results = self.heuristic_results[heuristic_name]
        total_blocks = sum(period['blocks_in_period'] for period in self.block_stats)

        print(f"\n=== {heuristic_name.upper()} SUMMARY ===")
        print(f"Total matches: {results['total_heuristic_count']}")
        print(f"Total transactions: {results['total_transactions']}")
        print(f"Total blocks: {total_blocks}")
        if total_blocks > 0:
            print(f"Avg per block: {results['total_heuristic_count'] / total_blocks:.6f}")
        if results['total_transactions'] > 0:
            print(f"Overall ratio: {results['total_heuristic_count'] / results['total_transactions']:.6f}")

    def run_optimisations(self):
        self.conn.execute("""set memory_limit ='1.8TB';""")
        self.conn.execute("SET threads TO 64;")

    def export_blockstats(self):
        import pandas as pd
        df = pd.DataFrame(self.collect_block_statistics(period_length=1008))
        df.to_csv("blockStats.csv", index=False)

    def close(self):
        self.conn.close()


# Usage example
if __name__ == "__main__":
    analyzer = BlockchainAnalyzer('DuckTest.db')

    try:
        # Set up optimizations
        analyzer.run_optimisations()
        analyzer.create_optimized_queries()
        
        #Collect block statistics
        analyzer.export_blockstats()
        
        # Run heuristics in parallel
        print("\n" + "="*50)
        print("Running optimized heuristics in parallel...")
        
        # Run with parallel processing (adjust max_workers based on your cluster)
        coinswap_results = analyzer.run_coinswap_heuristic_optimized(parallel=True, max_workers=64)
        analyzer.export_heuristic_results("coinswap_optimized")
        
        coinjoin_results = analyzer.run_coinjoin_optimized(parallel=True, max_workers=64)
        analyzer.export_heuristic_results("coinjoin_optimized")
        
        coinshuffle_results = analyzer.run_coinshuffle_optimized(parallel=True, max_workers=64)
        analyzer.export_heuristic_results("coinshuffle_optimized")
        
        stealth_results = analyzer.run_stealth_optimized(parallel=True, max_workers=64)
        analyzer.export_heuristic_results("stealthaddress_optimized")
        
        print("All heuristics completed successfully!")
        
    finally:
        analyzer.close()