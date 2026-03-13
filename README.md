# Prediction Market Analysis

A framework for analyzing prediction market data, including the largest publicly available dataset of Polymarket and Kalshi market and trade data. Provides tools for data collection, storage, and running analysis scripts that generate figures and statistics.

## Overview

This project enables research and analysis of prediction markets by providing:
- Pre-collected datasets from Polymarket and Kalshi
- Data collection indexers for gathering new data
- An extensible analysis framework for generating figures and statistics
- A Jupyter notebook (`analysis.ipynb`) consolidating key findings
- An interactive Streamlit dashboard for exploring analysis outputs

Currently supported features:
- Market metadata collection (Kalshi & Polymarket)
- Trade history collection via API and blockchain
- Parquet-based storage with automatic progress saving
- 28 built-in analysis scripts across Kalshi, Polymarket, and cross-platform comparisons
- Interactive dashboard with filterable charts and downloadable data

## Installation & Usage

Requires Python 3.9+. Install dependencies with [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

Download and extract the pre-collected dataset (36GiB compressed):

```bash
make setup
```

This downloads `data.tar.zst` from [Cloudflare R2 Storage](https://s3.jbecker.dev/data.tar.zst), installs required system tools, and extracts the data to `data/`.

### Environment Variables

Copy the example env and fill in your values:

```bash
cp .env.example .env
```

| Variable | Description |
|---|---|
| `POLYGON_RPC` | Polygon RPC endpoint URL (required for Polymarket blockchain indexing) |
| `POLYMARKET_START_BLOCK` | Starting block for Polymarket trade indexing (default: `33605403`) |

### Data Collection

Collect market and trade data from prediction market APIs:

```bash
make index
```

This opens an interactive menu to select which indexer to run. Data is saved to `data/kalshi/` and `data/polymarket/` directories. Progress is saved automatically, so you can interrupt and resume collection.

### Running Analyses

```bash
make analyze
```

This opens an interactive menu to select which analysis to run. You can run all analyses or select a specific one. Output files (PNG, PDF, CSV, JSON, MD) are saved to `output/`.

To run a specific analysis directly:

```bash
make run <analysis_name>
```

### Interactive Dashboard

Launch the Streamlit dashboard to explore analysis outputs interactively:

```bash
uv run streamlit run dashboard.py
```

The dashboard provides filterable interactive charts, raw data tables, and downloadable JSON chart configurations for each analysis.

### Packaging Data

To compress the data directory for storage/distribution:

```bash
make package
```

This creates a zstd-compressed tar archive (`data.tar.zst`) and removes the `data/` directory.

### Development

```bash
make lint      # Run ruff linter and format checker
make format    # Auto-fix lint issues and format code
make test      # Run the test suite with pytest
```

## Project Structure

```
├── main.py                 # CLI entrypoint (analyze, index, package)
├── dashboard.py            # Streamlit interactive dashboard
├── analysis.ipynb          # Jupyter notebook with consolidated analysis
├── Makefile                # Common commands
├── src/
│   ├── analysis/           # Analysis scripts
│   │   ├── kalshi/         # Kalshi-specific analyses (23 scripts)
│   │   ├── polymarket/     # Polymarket-specific analyses (4 scripts)
│   │   └── comparison/     # Cross-platform comparison analyses
│   ├── indexers/           # Data collection indexers
│   │   ├── kalshi/         # Kalshi API client and indexers
│   │   └── polymarket/     # Polymarket API/blockchain indexers
│   └── common/             # Shared utilities and interfaces
├── data/                   # Data directory (extracted from data.tar.zst)
│   ├── kalshi/
│   │   ├── markets/
│   │   └── trades/
│   └── polymarket/
│       ├── blocks/
│       ├── markets/
│       └── trades/
├── tests/                  # Test suite
├── docs/                   # Documentation
├── output/                 # Analysis outputs (figures, CSVs, reports)
└── scripts/                # Setup and download scripts
```

## Available Analyses

### Kalshi

| Analysis | Description |
|---|---|
| Calibration Deviation Over Time | Tracks calibration accuracy over time |
| EV Yes vs No | Expected value comparison of Yes vs No positions |
| Longshot Volume Share Over Time | Volume share of longshot contracts over time |
| Maker Returns By Direction | Maker returns segmented by trade direction |
| Maker Taker Gap Over Time | Spread between maker and taker returns |
| Maker Taker Returns By Category | Returns by market category for makers vs takers |
| Maker vs Taker Returns | Overall maker vs taker return comparison |
| Maker Win Rate By Direction | Win rate analysis by maker trade direction |
| Market Types | Distribution and breakdown of market types |
| Meta Stats | High-level dataset statistics |
| Mispricing By Price | Mispricing analysis across price levels |
| NBA Game Accuracy | Calibration curve for NBA game markets |
| Political Speech Mention Accuracy | Calibration for political speech mention markets |
| Political Mention Calibration | Calibration for political-mention markets |
| Returns By Hour | Trading returns by hour of day |
| Speech Mention Subgroup Accuracy | Subgroup-level calibration for speech markets |
| Statistical Tests | Statistical significance tests on market data |
| Trade Size By Role | Trade size distribution by maker/taker role |
| Volume Over Time | Trading volume trends |
| VWAP By Hour | Volume-weighted average price by hour |
| Win Rate By Price | Win rate as a function of trade price |
| Win Rate By Trade Size | Win rate by trade size |
| Yes vs No By Price | Yes vs No distribution by price level |

### Polymarket

| Analysis | Description |
|---|---|
| Speech Mention Subgroup Accuracy | Subgroup calibration for Polymarket speech markets |
| Trades Over Time | Polymarket trade activity trends |
| Volume Over Time | Polymarket volume trends |
| Win Rate By Price | Polymarket win rate by contract price |

### Cross-Platform

| Analysis | Description |
|---|---|
| Win Rate By Price Animated | Animated comparison of win rate by price across platforms |

## Documentation

- [Data Schemas](docs/SCHEMAS.md) - Parquet file schemas for markets and trades
- [Writing Analyses](docs/ANALYSIS.md) - Guide for writing custom analysis scripts

## Contributing

If you'd like to contribute to this project, please open a pull-request with your changes, as well as detailed information on what is changed, added, or improved.

For more information, see the [contributing guide](CONTRIBUTING.md).

## Issues

If you've found an issue or have a question, please open an issue [here](https://github.com/jon-becker/prediction-market-analysis/issues).

## Research & Citations

- Becker, J. (2026). _The Microstructure of Wealth Transfer in Prediction Markets_. Jbecker. https://jbecker.dev/research/prediction-market-microstructure
- Le, N. A. (2026). _Decomposing Crowd Wisdom: Domain-Specific Calibration Dynamics in Prediction Markets_. arXiv. https://arxiv.org/abs/2602.19520

If you have used or plan to use this dataset in your research, please reach out via [email](mailto:jonathan@jbecker.dev) or [Twitter](https://x.com/BeckerrJon) -- i'd love to hear about what you're using the data for! Additionally, feel free to open a PR and update this section with a link to your paper.

## License

This project is licensed under the [MIT License](LICENSE).
