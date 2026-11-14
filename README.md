# ETH 1-minute trend + range strategy — AIProve `eth-pnl-fixed`

Highest net PnL after costs on the frozen ETH 1-minute dataset
(`data/bars.csv`, 1,440 bars, Binance ETH/USDT, 24h ending 2026-11-13
12:17 UTC). Prize: 2,000,000 PROVE (1st 1,000,000 / 2nd 600,000 /
3rd 400,000). Closes 20 Nov 2026 20:49 WIB.

## Result (local backtest, $10,000 starting capital)

| Metric | v1 | v2 (this entry) |
|---|---|---|
| **Net PnL after costs** | +$481.54 (+4.82%) | **+$701.17 (+7.01%)** |
| Gross PnL | +$549.55 | +$797.18 |
| Total costs | $68.01 (fees $35.00 + slippage $33.00) | $96.00 (fees $55.00 + slippage $41.00) |
| **Max drawdown** | 1.98% | **1.30%** |
| Trades | 4 | 9 |
| Final equity | $10,481.54 | $10,701.17 |
| Split-half PnL | first 720 bars −$3.77, last 720 bars +$485.31 | first 720 bars **+$144.98**, last 720 bars **+$556.19** |
| Quarterly PnL | −$6, +$3, +$436, +$49 | **+$142.30, +$2.68, +$435.22, +$120.97** |

Benchmarks on the same data and cost model: buy-and-hold −$343.59,
short-and-hold +$327.59, SMA 20/100 crossover +$191.43, EMA 20/100
−$192.75, Donchian 55 +$84.15, RSI 25/75 +$56.53. Full comparison
table in `../backtest.md` (v2 report lives in `../v2/backtest-v2.md`;
`../backtest.md` is the v1 report, kept for reference).

## Strategy

Three layers, one position machine (positions in {−1, 0, +1}):

1. **Trend layer — long-window OLS trend with an asymmetric
   significance hysteresis band** (Schmitt trigger on the regression
   slope):
   - Every bar, fit an OLS line to the last **280** closes (log closes,
     ~4.7 hours) and compute the slope's t-statistic,
     `t = slope / SE(slope)`.
   - Go long (+1) when `t > +2.25`, short (−1) when `t < −1.5`.
   - While inside the band, **hold the previous position** (flat only
     before the first signal). The dead band is the whole edge:
     1-minute ETH here is chop-dominated (lag-1 return autocorrelation
     ≈ −0.08), so plain slope-sign / EMA-crossover systems flip 20–130
     times and bleed out on fees.
   - The asymmetry sits on a broad plateau: long thresholds 2.0–3.0
     with short thresholds 1.25–2.0 all net +$482 to +$539 trend-only.
2. **Range layer — Bollinger mean-reversion, active only while the
   trend layer is flat:**
   - 120-bar SMA ± 2 standard deviations. Long below the lower band,
     short above the upper band, exit to flat at the middle band.
   - This harvests the range-bound first half (price swings 1985–2029
     while the trend layer is flat in its warmup): long 121 @ $1,998.73
     → flat 139 @ $2,015.24, short 228 @ $2,020.49 → flat 260 @
     $2,003.60, +$142 in Q1. It never fights a trend — the moment the
     trend layer takes a position it is overridden, and it never
     re-enters while a trend leg is open. Standalone it loses (−$224);
     an independent RSI(14) 25/75 version of the same layer gives the
     same +$145, so the two swings are not a one-indicator artifact.
3. **Bounce exit (short legs only) — take-profit protection:**
   - Once a short leg is ≥ 2% in profit vs its entry close it is
     armed; while armed, if the close rises ≥ 1.0% above the lowest
     close since the leg began, exit to flat. This banks most of the
     second-half downtrend (short 723 → exit 1082 @ $1,929.21 instead
     of riding the bounce back to $1,937) and cuts max drawdown from
     1.84% to 1.30%.
   - The exit latches: no re-short and no range trades until the
     t-stat recovers above −1.5 (hysteresis reset) or flips long.
     Without the latch the same exit churns (0.7% version: +$289).

Trades on this dataset (filled at next bar open): range long @ bar 121
($1,998.73) → flat @ 139 ($2,015.24) → range short @ 228 ($2,020.49) →
flat @ 260 ($2,003.60) → trend short @ 280 ($2,007.45) → long @ 327
($2,010.83) → short @ 723 ($2,016.20) → bounce exit flat @ 1082
($1,929.21) → long @ 1198 ($1,932.22), held long to the close
($1,960.59). The short from bar 723 rides the main down-move; the
final long catches the recovery.

### Why these parameters (anti-overfitting note)

The challenge README warns that overfitting this sample is easy and a
held-out tail punishes it. Every parameter sits mid-plateau, none at a
cliff: trend long 2.0–3.0 × short 1.25–2.0 → +$482–539; Bollinger
window 90/120/150 → +$611–681 and k 1.75/2.0/2.25 → +$630–701; bounce
0.7%/1.0%/1.5% → +$651–701. The Bollinger 2.75σ cell (+$717.90) was
deliberately NOT shipped — its extra $37 is dip-shape luck and 3σ
collapses to +$611. The range layer only trades while the trend layer
is flat, so on any extension past bar 1440 the strategy behaves as the
trend-only plateau. No stops, no second trend indicator, no fractional
sizing — each was tested and made it worse or added nothing.

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
  10bp base slippage (+$317.14); at zero costs it is +$797.18.

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
- `strategy.trend_positions(closes)` → trend layer alone.
- `strategy.mr_positions(closes)` → Bollinger range layer alone.
- `strategy.backtest(bars, positions=None, ...)` → dict with `net_pnl`,
  `max_drawdown`, `num_trades`, `equity_curve`, `trade_log`.
- `strategy.t_statistics(closes, window=280)` → per-bar slope t-stats
  (`None` during warmup).
- `strategy.bollinger_bands(closes, window=120, k=2.0)` → per-bar
  `(mid, upper, lower)` (`None` during warmup).
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
| `tests/test_strategy.py` | 16 unit tests (no-lookahead, hysteresis, trend priority, bounce latch, cost, determinism) |
| `data/bars.csv` | Frozen challenge dataset (1,440 bars) |
| `requirements.txt` | Empty — standard library only |
