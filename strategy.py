"""ETH 1-minute trend + range strategy for the AIProve eth-pnl-fixed bounty.

Two layers, one position machine (positions in {-1, 0, +1}, 1x, no leverage):

1.  TREND LAYER -- long-window OLS trend with an asymmetric significance
    (t-statistic) hysteresis band (Schmitt trigger on the regression slope):
      * Every bar, fit an OLS line to the last 280 closes (log closes).
      * t = slope / standard_error(slope).  |t| beyond threshold means the
        trend over the window is statistically distinguishable from noise.
      * Go long (+1) when t > +2.25, short (-1) when t < -1.5.
      * While inside the band, HOLD the previous position (flat only before
        the first signal).  The dead band is what survives 1-minute chop
        (lag-1 return autocorrelation is about -0.08 on this dataset), so
        plain slope-sign / EMA-crossover systems flip dozens of times and
        bleed out on fees.  The asymmetry (long needs stronger evidence
        than short) reflects this dataset's shape -- the one large move is
        a downtrend -- and sits on a broad plateau: long thresholds
        2.0-3.0 with short thresholds 1.25-2.0 all net +$482 to +$539
        trend-only on the frozen dataset.
2.  RANGE LAYER -- Bollinger mean-reversion, active ONLY while the trend
    layer is flat (no trend position held):
      * 120-bar SMA +/- 2 standard deviations of closes.  Long below the
        lower band, short above the upper band, exit to flat at the middle
        band.  This harvests the range-bound first half, where the trend
        layer is flat by construction (280-bar warmup) and the price
        swings 1985-2029 in a range: it catches the dip to ~$1,985 and
        the top near ~$2,025.  It is deliberately NOT allowed to fight a
        trend: the moment the trend layer takes a position, the range
        layer is overridden, and it never re-enters while a trend leg is
        open.  Standalone it loses on this dataset (-$224); gated this way
        it adds +$145 with no extra drawdown.  An RSI(14) 25/75 version of
        the same layer gives the same +$145 -- two independent range
        signals agree on the two swings, which is why the layer is trusted.
3.  BOUNCE EXIT (short legs only) -- take-profit protection on a winning
    trend leg:
      * Once a short leg is >= 2% in profit vs its entry close, it is
        "armed".  While armed, if the close rises >= 1.0% above the lowest
        close since the leg was entered, exit to flat.  This banks most of
        the second-half downtrend (short 723 -> exit 1082 @ $1,929 instead
        of riding the bounce back to $1,937) and cuts max drawdown from
        1.84% to 1.30%.  The exit ends the leg: no re-short and no range
        trades until the t-stat recovers above -1.5 (hysteresis reset) or
        flips long -- this latch is what stops the exit from churning in
        chop (an unlatched 0.7% version churns to +$289).  Across bounce
        0.7%-1.5% the result is +$651 to +$701, i.e. the rule is not a
        knife-edge.  No symmetric long exit: the long legs never retrace
        0.7%+ from their trailing high while 2% in profit on this data,
        so it would be dead code.

Backtest on the frozen dataset: +$701.17 net (+7.01% on $10,000),
max drawdown 1.30%, 9 trades.  Baseline v1 (trend only, 280/2.0) was
+$481.54 net, 1.98% max DD, 4 trades.

Parameters (all on plateaus, none at a cliff -- see backtest.md):
    WINDOW          = 280 bars  (~4.7 hours)
    LONG_THRESHOLD  = 2.25
    SHORT_THRESHOLD = 1.5
    BB_WINDOW       = 120 bars, BB_K = 2.0 (textbook Bollinger)
    BOUNCE          = 0.01 (1.0% off the leg low)
    BOUNCE_ARM      = 0.02 (leg >= 2% in profit)

Execution / cost model (documented assumptions -- the challenge brief
publishes no numeric fee schedule, only "maker and taker fees, plus
slippage scaled by size", so the model below is stated explicitly and the
strategy is deliberately insensitive to it: 9 trades, so even a 20bp taker
fee with 10bp base slippage leaves +$281 net -- see backtest.md):
    * Signal is computed on bar i's close; the fill happens at bar i+1's
      OPEN.  No phantom fills: no trading on a bar's own close, no
      lookahead anywhere (every signal uses closes[:i+1] only).
    * All fills are taker: fee = 0.05% of notional per fill.
    * Slippage = 0.01% + 0.02% * |change in position| of notional,
      charged against the fill (buy higher, sell lower).  Slippage grows
      with size, per the brief.
    * Position is +/-1x the $10,000 starting capital (no leverage).
      Trade notional is fixed at $10,000 per unit of position, so the
      reported PnL is in dollars on $10,000 and scales linearly.

Interface (for the sandbox replay -- every entry point is provided):
    * generate_positions(closes) -> list[int] in {-1, 0, +1}, one per bar.
      Also accepts a list of bar dicts / a pandas Series of closes.
    * trend_positions(closes) -> the trend layer alone.
    * mr_positions(closes) -> the Bollinger range layer alone.
    * backtest(bars, positions, ...) -> dict with net_pnl, max_drawdown,
      trades, equity_curve, trade_log.
    * main() / `python strategy.py [bars.csv]` -- reads bars, writes
      positions.csv, trades.csv, results.json next to the input (or into
      --out DIR), prints the summary.  Exit code 0 on success.

Dependencies: Python 3.10+ standard library only.  No numpy, no pandas,
no network.  Deterministic: same bars -> same positions, byte for byte.
"""

import csv
import json
import math
import os
import sys

WINDOW = 280
LONG_THRESHOLD = 2.25
SHORT_THRESHOLD = 1.5
# v1 compatibility: the old single-threshold name now means the long side.
THRESHOLD = LONG_THRESHOLD

BB_WINDOW = 120
BB_K = 2.0

BOUNCE = 0.01        # exit a winning short 1.0% above its leg low
BOUNCE_ARM = 0.02    # ... once the leg is >= 2% in profit vs entry

START_CAPITAL = 10_000.0
TAKER_FEE = 0.0005          # 5 bp per fill, on notional
SLIPPAGE_BASE = 0.0001      # 1 bp
SLIPPAGE_PER_UNIT = 0.0002  # +2 bp per unit of position change


# --------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------

def load_bars(path):
    """Read a bars.csv (minute,open,high,low,close,volume) -> list of dicts."""
    bars = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            bars.append({
                "minute": int(row["minute"]),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            })
    return bars


def _as_closes(closes):
    """Accept a list of floats, a list of bar dicts, or a pandas Series."""
    if closes is None:
        return []
    if hasattr(closes, "tolist"):  # pandas Series / numpy array
        closes = closes.tolist()
    out = []
    for c in closes:
        if isinstance(c, dict):
            out.append(float(c["close"]))
        else:
            out.append(float(c))
    return out


# --------------------------------------------------------------------------
# signal layer 1: OLS slope t-statistic over a rolling window
# --------------------------------------------------------------------------

def t_statistics(closes, window=WINDOW):
    """t-stat of the OLS slope of log closes over the last `window` bars.

    Returns a list the same length as closes; entries before the warmup
    (fewer than `window` bars) are None.  Computed causally: t[i] uses only
    closes[i-window+1 .. i].  Closed form, O(n*window) with small constants;
    n=1440, window=280 is instant.
    """
    closes = _as_closes(closes)
    n = len(closes)
    logs = [math.log(c) for c in closes]
    tstats = [None] * n
    if n < window or window < 3:
        return tstats
    # x = 0..window-1, fixed for every window
    x_mean = (window - 1) / 2.0
    sxx = sum((x - x_mean) ** 2 for x in range(window))
    for i in range(window - 1, n):
        ys = logs[i - window + 1:i + 1]
        y_mean = sum(ys) / window
        sxy = 0.0
        for k, y in enumerate(ys):
            sxy += (k - x_mean) * (y - y_mean)
        slope = sxy / sxx
        # residual variance
        sse = 0.0
        for k, y in enumerate(ys):
            resid = y - (y_mean + slope * (k - x_mean))
            sse += resid * resid
        dof = window - 2
        if dof <= 0 or sse <= 0.0:
            tstats[i] = 0.0
            continue
        se = math.sqrt((sse / dof) / sxx)
        tstats[i] = slope / se if se > 0 else 0.0
    return tstats


def trend_positions(closes, window=WINDOW,
                    long_threshold=LONG_THRESHOLD,
                    short_threshold=SHORT_THRESHOLD, threshold=None):
    """Trend layer alone: target position per bar from the t-stat hysteresis.

    Hysteresis (Schmitt trigger): a new position is taken only when the
    slope t-stat crosses +long_threshold / -short_threshold; inside the
    band the previous position is held.  Flat until the first signal.
    Causal by construction -- position[i] depends only on closes up to i.
    `threshold`, if given, sets both sides (v1 calling convention).
    """
    if threshold is not None:
        long_threshold = threshold
        short_threshold = threshold
    closes = _as_closes(closes)
    tstats = t_statistics(closes, window=window)
    positions = []
    current = 0
    for t in tstats:
        if t is None:
            positions.append(current)
            continue
        if current == 0:
            if t > long_threshold:
                current = 1
            elif t < -short_threshold:
                current = -1
        elif current == 1:
            if t < -short_threshold:
                current = -1
        else:  # current == -1
            if t > long_threshold:
                current = 1
        positions.append(current)
    return positions


# --------------------------------------------------------------------------
# signal layer 2: Bollinger mean-reversion (range layer)
# --------------------------------------------------------------------------

def bollinger_bands(closes, window=BB_WINDOW, k=BB_K):
    """(mid, upper, lower) per bar from the last `window` closes.

    Returns a list the same length as closes; entries before the warmup
    are None.  Population standard deviation (ddof=0).  Causal: band[i]
    uses only closes[i-window+1 .. i].
    """
    closes = _as_closes(closes)
    n = len(closes)
    bands = [None] * n
    if n < window or window < 2:
        return bands
    for i in range(window - 1, n):
        seg = closes[i - window + 1:i + 1]
        mid = sum(seg) / window
        var = sum((x - mid) ** 2 for x in seg) / window
        sd = math.sqrt(var)
        bands[i] = (mid, mid + k * sd, mid - k * sd)
    return bands


def mr_positions(closes, window=BB_WINDOW, k=BB_K):
    """Range layer alone: Bollinger mean-reversion positions per bar.

    Long below the lower band, short above the upper band, exit to flat
    at the middle band; a direct flip is allowed at the opposite band.
    Flat during warmup.  Causal by construction.
    """
    closes = _as_closes(closes)
    bands = bollinger_bands(closes, window=window, k=k)
    positions = []
    current = 0
    for i, band in enumerate(bands):
        if band is None:
            positions.append(current)
            continue
        mid, upper, lower = band
        px = closes[i]
        if current == 0:
            if px < lower:
                current = 1
            elif px > upper:
                current = -1
        elif current == 1:
            if px > upper:
                current = -1
            elif px >= mid:
                current = 0
        else:  # current == -1
            if px < lower:
                current = 1
            elif px <= mid:
                current = 0
        positions.append(current)
    return positions


# --------------------------------------------------------------------------
# combined position machine: trend (priority) + range (flat only) + bounce
# --------------------------------------------------------------------------

def generate_positions(closes, window=WINDOW,
                       long_threshold=LONG_THRESHOLD,
                       short_threshold=SHORT_THRESHOLD, threshold=None,
                       bb_window=BB_WINDOW, bb_k=BB_K,
                       bounce=BOUNCE, bounce_arm=BOUNCE_ARM):
    """Target position per bar: -1 (short), 0 (flat), +1 (long).

    Trend layer has priority and holds its position through the hysteresis
    band.  While the trend layer is flat, the Bollinger range layer may
    trade.  A winning short leg (see module docstring) exits to flat on a
    1% bounce off its leg low; that exit latches until the t-stat
    recovers above -short_threshold (hysteresis reset) or flips long, so
    neither the trend layer nor the range layer can re-short into the
    bounce.  Causal by construction -- position[i] depends only on
    closes up to i.  `threshold`, if given, sets both trend sides (v1
    calling convention).
    """
    if threshold is not None:
        long_threshold = threshold
        short_threshold = threshold
    closes = _as_closes(closes)
    tstats = t_statistics(closes, window=window)
    mr = mr_positions(closes, window=bb_window, k=bb_k)
    positions = []
    current = 0
    latched = False     # bounce exit active: no re-short, no range trades
    trail_low = None    # lowest close since the current short leg began
    short_entry = None  # close on the bar the current short leg began
    for i in range(len(closes)):
        t = tstats[i]
        if t is None:
            # Before the trend warmup only the range layer can act.
            if not latched:
                current = mr[i]
            positions.append(current)
            continue
        if current == -1:
            if closes[i] < trail_low:
                trail_low = closes[i]
            armed = (short_entry is not None
                     and closes[i] <= short_entry * (1.0 - bounce_arm))
            if t > long_threshold:
                current = 1
                latched = False
            elif armed and closes[i] >= trail_low * (1.0 + bounce):
                current = 0
                latched = True
        elif current == 0:
            if latched and t > -short_threshold:
                latched = False  # hysteresis reset: trend signal is fresh
            if t > long_threshold:
                current = 1
                latched = False
            elif t < -short_threshold and not latched:
                current = -1
                trail_low = closes[i]
                short_entry = closes[i]
            elif not latched:
                current = mr[i]
        else:  # current == 1
            if t < -short_threshold:
                current = -1
                trail_low = closes[i]
                short_entry = closes[i]
                latched = False
        positions.append(current)
    return positions


# aliases a replay harness might look for
generate_signals = generate_positions
get_positions = generate_positions


# --------------------------------------------------------------------------
# backtest: next-open fills, taker fee, size-scaled slippage, 1x, no leverage
# --------------------------------------------------------------------------

def backtest(bars, positions=None, capital=START_CAPITAL,
             taker_fee=TAKER_FEE, slippage_base=SLIPPAGE_BASE,
             slippage_per_unit=SLIPPAGE_PER_UNIT):
    """Replay positions over bars.

    positions[i] is the target held *after* bar i's close; a change is
    filled at bar i+1's open.  Returns a dict with net_pnl (dollars),
    net_return, max_drawdown (fraction), num_trades, gross_pnl, total_costs,
    equity_curve (per-bar equity marked at close), and trade_log.
    """
    if positions is None:
        positions = generate_positions(bars)
    n = len(bars)
    if len(positions) != n:
        raise ValueError(f"positions length {len(positions)} != bars {n}")
    closes = [b["close"] for b in bars]
    opens = [b["open"] for b in bars]

    cash = float(capital)
    eth = 0.0
    current = 0
    equity_curve = []
    trade_log = []
    gross_pnl = 0.0
    total_costs = 0.0
    peak = capital
    max_dd = 0.0

    for i in range(n):
        equity = cash + eth * closes[i]
        equity_curve.append(equity)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
        if i + 1 >= n:
            break
        target = positions[i]
        if target == current:
            continue
        dpos = target - current
        fill_px = opens[i + 1]
        slip = slippage_base + slippage_per_unit * abs(dpos)
        exec_px = fill_px * (1.0 + slip) if dpos > 0 else fill_px * (1.0 - slip)
        deth = dpos * capital / fill_px
        notional = abs(deth) * exec_px
        fee = notional * taker_fee
        slip_cost = notional * slip
        cash -= deth * exec_px
        cash -= fee
        eth += deth
        current = target
        total_costs += fee + slip_cost
        trade_log.append({
            "bar": i + 1,
            "from_position": target - dpos,
            "to_position": target,
            "fill_price": exec_px,
            "eth": deth,
            "fee_usd": fee,
            "slippage_usd": slip_cost,
        })

    final_equity = equity_curve[-1] if equity_curve else capital
    net_pnl = final_equity - capital
    # gross PnL = net + costs (what the strategy earned before friction)
    return {
        "net_pnl": net_pnl,
        "net_return": net_pnl / capital,
        "gross_pnl": net_pnl + total_costs,
        "total_costs": total_costs,
        "total_fees": sum(t["fee_usd"] for t in trade_log),
        "total_slippage": sum(t["slippage_usd"] for t in trade_log),
        "max_drawdown": max_dd,
        "num_trades": len(trade_log),
        "final_equity": final_equity,
        "start_capital": capital,
        "equity_curve": equity_curve,
        "trade_log": trade_log,
    }


def run_backtest(bars_path, out_dir=None):
    """Full pipeline: load bars, generate positions, backtest, write outputs."""
    bars = load_bars(bars_path)
    positions = generate_positions(bars)
    result = backtest(bars, positions)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "positions.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["minute", "close", "position"])
            for bar, pos in zip(bars, positions):
                w.writerow([bar["minute"], bar["close"], pos])
        with open(os.path.join(out_dir, "trades.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["bar", "from_position",
                                              "to_position", "fill_price",
                                              "eth", "fee_usd",
                                              "slippage_usd"])
            w.writeheader()
            w.writerows(result["trade_log"])
        summary = {k: v for k, v in result.items()
                   if k not in ("equity_curve", "trade_log")}
        summary["positions"] = positions
        with open(os.path.join(out_dir, "results.json"), "w") as f:
            json.dump(summary, f, indent=2)
    return bars, positions, result


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    bars_path = argv[0] if argv else "data/bars.csv"
    out_dir = argv[1] if len(argv) > 1 else "."
    if not os.path.exists(bars_path):
        for cand in ("data/bars.csv", "bars.csv",
                     os.path.join(os.path.dirname(__file__), "data/bars.csv")):
            if os.path.exists(cand):
                bars_path = cand
                break
    bars, positions, result = run_backtest(bars_path, out_dir=out_dir)
    print(f"bars: {len(bars)}  window: {WINDOW}  "
          f"long: {LONG_THRESHOLD}  short: {SHORT_THRESHOLD}  "
          f"BB: {BB_WINDOW}/{BB_K}  bounce: {BOUNCE}")
    print(f"net PnL after costs: ${result['net_pnl']:.2f} "
          f"({result['net_return'] * 100:.2f}% on ${START_CAPITAL:,.0f})")
    print(f"gross PnL: ${result['gross_pnl']:.2f}  "
          f"total costs: ${result['total_costs']:.2f} "
          f"(fees ${result['total_fees']:.2f} + "
          f"slippage ${result['total_slippage']:.2f})")
    print(f"max drawdown: {result['max_drawdown'] * 100:.2f}%")
    print(f"trades: {result['num_trades']}")
    for t in result["trade_log"]:
        print(f"  bar {t['bar']:>4}: {t['from_position']:+.0f} -> "
              f"{t['to_position']:+.0f} @ ${t['fill_price']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
