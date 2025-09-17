# Bitcoin Privacy-Preserving Protocol Analysis

This repository contains the code and data pipeline for the Master's thesis **"An Empirical Analysis of the Adoption and Decline of Privacy-Preserving Protocols on the Bitcoin Blockchain"**.

The study measures the prevalence of key privacy-enhancing technologies (PETs)—namely **CoinJoin, CoinSwap, CoinShuffle, and Stealth Addresses (BIP47)**—over a longitudinal period from block 400,000 (June 2016) to block 900,000 (est. June 2025). It investigates correlations between adoption trends and significant real-world events.

## Methodology Overview

The research follows a structured data pipeline:

1.  **Data Collection:** A full Bitcoin Core node was used to obtain a raw, validated copy of the blockchain.
2.  **Data Parsing:** The `rusty-blockparser` tool was used to parse `.dat` files into structured CSV files (`blocks`, `transactions`, `inputs`, `outputs`).
3.  **Database Construction:** CSV files were imported into a **DuckDB** database for efficient analytical querying. The schema was pruned to essential columns to optimize storage and performance (see table below).
4.  **Heuristic Querying:** A series of SQL queries, based on established academic literature (Möser & Böhme), were executed to detect transactions matching the patterns of each privacy protocol.
5.  **Analysis & Visualization:** Detected transactions were aggregated over 1008-block periods, normalized against total transaction volume, and plotted as a time series. Significant external events were annotated to contextualize trends.

### Pruned Database Schema
| Table | Retained Columns |
| :--- | :--- |
| `blocks` | `height`, `nTime` |
| `transactions` | `txid`, `hashBlock`, `lockTime` |
| `tx_in` | `txid`, `hashPrevOut`, `indexPrevOut`, `scriptSig` |
| `tx_out` | `txid`, `indexOut`, `value`, `address`, `scriptPubKey` |

## Heuristics Summary

The SQL heuristics implement the following criteria to identify potential privacy-protocol transactions:

| Protocol | Key Detection Criteria |
| :--- | :--- |
| **CoinJoin** | `# inputs >= # outputs / 2` and `# outputs >= 4` |
| **CoinSwap** | 2-of-2 multisig, no graph connection, value variation < 15%, no `OP_RETURN` |
| **CoinShuffle** | `# inputs == # outputs >= 3`, near-identical output values (`variation < 0.0001`) |
| **Stealth Addresses (BIP47)** | Exact `OP_RETURN` payload matching `'6a26%'` (BIP47) |

*Note: The Stealth Address heuristic underwent three iterations to balance precision and recall, moving from broad pattern matching to a precise BIP47-specific filter.*

## Usage

### Prerequisites
- Python 3.8+
- DuckDB
- A parsed Bitcoin blockchain dataset in the pruned DuckDB format described above.

### Running the Analysis
1.  Clone the repository:
    ```bash
    git clone https://github.com/<your-username>/BTC-Privacy-Analysis.git
    cd BTC-Privacy-Analysis
    ```
2.  Ensure your DuckDB database (`bitcoin.db`) is in the correct location.
3.  Install Python dependencies:
    ```bash
    pip install duckdb pandas
    ```
4.  Run the main analysis script:
    ```bash
    python src/analysis.py
    ```
    This script will execute the SQL queries, aggregate the results, and generate the time-series plots.

## Key Findings

The analysis reveals distinct adoption trends for each protocol, heavily influenced by external events:
- **CoinJoin:** Shows a clear lifecycle, with adoption peaks correlating with the availability of user-friendly services (e.g., Wasabi Wallet) and sharp declines following regulatory actions (e.g., Tornado Cash sanctions).
- **Stealth Addresses (BIP47):** Despite a targeted heuristic, no transactions were detected, indicating negligible adoption of the standardized BIP47 protocol and suggesting a pivot towards more opaque privacy technologies.
- **General Trend:** A significant portion of initially detected "privacy" activity was later attributed to **false positives** from non-financial data embedding (e.g., Ordinals inscriptions), highlighting the methodological challenges of on-chain privacy research.

## 📮 Contact

For questions regarding this research, please open an Issue on this repository.

---
