# ETH 1-minute trend strategy — AIProve `eth-pnl-fixed`

Highest net PnL after costs on the frozen ETH 1-minute dataset
(`data/bars.csv`, 1,440 bars, Binance ETH/USDT, 24h ending 2026-11-13
12:17 UTC). Prize: 2,000,000 PROVE (1st 1,000,000 / 2nd 600,000 /
3rd 400,000). Closes 20 Nov 2026 20:49 WIB.

## Result (local backtest, $10,000 starting capital)

| Metric | Value |
|---|---|
| **Net PnL after costs** | **+$481.54 (+4.82%)** |
| Gross PnL | +$549.55 |
| Total costs | $68.01 (fees $35.00 + slippage $33.00) |
| **Max drawdown** | **1.98%** |
| Trades | 4 |
| Final equity | $10,481.54 |
| Split-half PnL | first 720 bars −$3.77, last 720 bars +$485.31 |

Benchmarks on the same data and cost model: buy-and-hold −$343.59,
short-and-hold +$327.59, SMA 20/100 crossover +$191.43, EMA 20/100
−$192.75, Donchian 55 +$84.15, RSI 25/75 +$56.53. Full comparison
table in `../backtest.md`.

## Strategy

Long-window OLS trend with a **significance hysteresis band** (Schmitt
trigger on the regression slope):

1. Every bar, fit an OLS line to the last **280** closes (log closes,
   ~4.7 hours) and compute the slope's t-statistic,
   `t = slope / SE(slope)`.
2. Go long (+1) when `t > +2.0`, short (−1) when `t < −2.0`.
3. While `|t| <= 2.0`, **hold the previous position** (flat only before
   the first signal). The dead band is the whole edge: 1-minute ETH here
   is chop-dominated (lag-1 return autocorrelation ≈ −0.08), so plain
   slope-sign / EMA-crossover systems flip 20–130 times and bleed out on
   fees. This system traded 4 times in 1,440 bars.

Trades on this dataset (filled at next bar open): short @ bar 280
($2,007.45) → long @ 326 ($2,011.03) → short @ 724 ($2,015.45) →
long @ 1197 ($1,937.27), held long to the close ($1,960.59).
The short from bar 724 rides the
main down-move; the final long catches the recovery.

### Why these parameters (anti-overfitting note)

The challenge README warns that overfitting this sample is easy and a
held-out tail punishes it. The parameters were picked on the **plateau**,
not the peak: windows 220–360 with thresholds 1.5–3.0 all net +3.0% to
+5.1% with 3–6 trades and ~2% max drawdown, and every window ≥220 is
profitable in the second half while roughly flat in the first. No stops,
no second indicator, no position sizing by signal strength — each of
those was tested and made it worse (trailing ATR stops: −$555;
exit-on-zero-crossing: +$477 with 9 trades under the exploration
engine; two-window vote: +$408–493).
Two free parameters, one trade idea.

## Execution / cost model

The challenge brief publishes no numeric fee schedule — only "maker and
taker fees, plus slippage scaled by size" — so the model is stated
explicitly here and the strategy is deliberately insensitive to it:

- Signal computed on bar `i`'s close; fill at bar `i+1`'s **open**.
  No phantom fills, no lookahead (verified by test: truncating the
  series never changes earlier positions).
- All fills taker: **0.05%** of notional per fill.
- Slippage **0.01% + 0.02% × |position change|** of notional, charged
  against the fill (buy higher, sell lower). Slippage grows with size.
- Position ±1× the $10,000 capital. **No leverage.** Trade notional is
  fixed at $10,000 per unit of position.
- Fee sensitivity: net PnL stays positive even at a 20bp taker fee with
  10bp base slippage (+$209.46); at zero costs it is +$549.55.

## How to run

```bash
python strategy.py            # uses data/bars.csv, writes results.json etc.
python strategy.py bars.csv out_dir
python backtest.py            # full printed report
python -m unittest discover -s tests -v
```

Python 3.10+ standard library only — no numpy, pandas, or network.
Deterministic: same bars → same positions.

## Interface for the sandbox replay

- `strategy.generate_positions(closes)` → `list[int]` in `{-1, 0, +1}`,
  one per bar. Accepts a list of closes, a list of bar dicts, or a
  pandas Series of closes. Aliases: `generate_signals`, `get_positions`.
- `strategy.backtest(bars, positions=None, ...)` → dict with `net_pnl`,
  `max_drawdown`, `num_trades`, `equity_curve`, `trade_log`.
- `strategy.t_statistics(closes, window=280)` → per-bar slope t-stats
  (`None` during warmup).
- `python strategy.py [bars.csv] [out_dir]` writes `positions.csv`,
  `trades.csv`, `results.json`.
- `data/bars.csv` is a verbatim copy of the challenge dataset
  (`https://github.com/AIProveRH/aiprove/tree/main/challenges/eth-backtest`,
  `data/README-challenge.md` is that folder's README).

## Files

| File | What it is |
|---|---|
| `strategy.py` | Signal + backtest engine + CLI (stdlib only) |
| `backtest.py` | Report runner |
| `tests/test_strategy.py` | 13 unit tests (no-lookahead, hysteresis, cost, determinism) |
| `data/bars.csv` | Frozen challenge dataset (1,440 bars) |
| `requirements.txt` | Empty — standard library only |
