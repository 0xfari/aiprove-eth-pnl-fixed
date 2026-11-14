"""Run the backtest and print a full report.

Usage:
    python backtest.py [bars.csv] [out_dir]

Defaults to data/bars.csv and writes positions.csv / trades.csv /
results.json into out_dir (default: current directory).
"""

import sys

from strategy import (WINDOW, THRESHOLD, START_CAPITAL,
                      run_backtest)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    bars_path = argv[0] if argv else "data/bars.csv"
    out_dir = argv[1] if len(argv) > 1 else "."
    bars, positions, r = run_backtest(bars_path, out_dir=out_dir)
    eq = r["equity_curve"]
    first_half = eq[719] - START_CAPITAL
    second_half = eq[-1] - eq[719]
    print("=" * 64)
    print("ETH 1-minute trend strategy -- backtest report")
    print("=" * 64)
    print(f"bars: {len(bars)}  window: {WINDOW}  threshold: {THRESHOLD}")
    print(f"start capital: ${START_CAPITAL:,.2f}")
    print(f"net PnL after costs: ${r['net_pnl']:.2f} "
          f"({r['net_return'] * 100:.2f}%)")
    print(f"gross PnL: ${r['gross_pnl']:.2f}")
    print(f"total costs: ${r['total_costs']:.2f} "
          f"(fees ${r['total_fees']:.2f}, slippage ${r['total_slippage']:.2f})")
    print(f"max drawdown: {r['max_drawdown'] * 100:.2f}%")
    print(f"trades: {r['num_trades']}")
    print(f"final equity: ${r['final_equity']:.2f}")
    print(f"split-half net PnL: first 720 bars ${first_half:+.2f}, "
          f"last 720 bars ${second_half:+.2f}")
    print("-" * 64)
    print("trades (filled at next bar open, taker fee 5bp, size-scaled slip):")
    for t in r["trade_log"]:
        print(f"  bar {t['bar']:>4}: {t['from_position']:+.0f} -> "
              f"{t['to_position']:+.0f} @ ${t['fill_price']:>8.2f}  "
              f"fee ${t['fee_usd']:.2f}  slip ${t['slippage_usd']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
