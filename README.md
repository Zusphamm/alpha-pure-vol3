# Pure Vol 3

Cross-sectional alpha for crypto USDT-M perpetual futures. Short high-volatility, long low-volatility coins — a volatility mean-reversion strategy.

## Factors

| Factor | Transform | Direction | Rationale |
|--------|-----------|-----------|-----------|
| parkinson_vol30 | chg20 | short (-) | Coins with rising realized vol tend to revert |
| ivol | z20 | short (-) | Idiosyncratic vol z-score captures abnormal risk |
| residual_vol | z20 | short (-) | Market-residual vol spike = overreaction to noise |

**No QV dependency** — universe selection uses Volume rank (not quote_volume). All factors use only OHLC data.

## Backtest (clean 1d data, 7bp cost)

| Period | Net Sharpe | Return | Max DD | Win Rate | Days |
|--------|-----------|--------|--------|----------|------|
| IS 2021-2023 | 1.44 | +145% | -47% | 57.4% | 1095 |
| **OS 2024+** | **1.94** | **+208%** | **-46%** | **56.6%** | 1008 |
| 2026 YTD | 4.11 | +136% | -24% | 60.3% | 277 |

### Yearly Sharpe (net of 7bp cost)

| 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|------|------|------|------|------|------|------|------|
| -0.78 | -0.20 | +0.87 | +3.14 | +2.16 | +1.21 | +1.98 | +4.11 |

### IS/OS Protocol

| Check | Value | Threshold | Result |
|-------|-------|-----------|--------|
| IS Sharpe | 1.44 | >= 1.4 | PASS |
| OS Sharpe | 1.94 | >= 0.0 | PASS |
| OS Max DD | -46% | > -50% | PASS |
| IS/OS Decay | 1.35 | >= 0.3 | PASS |

## Setup

```bash
pip install numpy pandas ccxt python-dotenv
cp .env.example .env   # fill in API keys
```

## Run

```bash
# Dry run (paper trade)
LIVE_DRY_RUN=1 python run_trader.py

# Live
LIVE_DRY_RUN=0 python run_trader.py
```

## Risk Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| LEVERAGE | 5x | Position leverage |
| GROSS_TARGET | 5.5 | Total gross exposure multiplier |
| FREE_PCT | 33% | Minimum free margin |
| TOP_N | 30 | Positions per side |
| FUNDING_FILTER | 0.03% | Skip high-funding symbols |
| REBALANCE | Daily | After 00:00 UTC candle close |
| COST (backtest) | 7bp one-way | Taker 0.04% + slippage 0.03% |

## Data

Requires `~/Desktop/datacryp/derived/panel_wide/1d/` with `close.parquet`, `high.parquet`, `low.parquet`, `volume.parquet`, `open.parquet`. Refreshed by `run_data_refresh.py` every 6h via launchd.
