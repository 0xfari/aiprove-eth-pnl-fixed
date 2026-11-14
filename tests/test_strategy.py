"""Tests for strategy.py v2 -- stdlib unittest, no network, no extra deps.

Run:  python -m unittest discover -s tests -v

v1 had 13 tests; v2 has 16.  Three v1 tests changed, with reasons:
  * test_warmup_is_flat -> test_trend_warmup_is_flat: v2's *combined*
    positions are NOT flat before bar 280 by design -- the Bollinger
    range layer trades the range-bound first half while the trend layer
    is still warming up.  The trend layer alone is still flat in warmup.
  * test_real_data_matches_reported_numbers: v1 asserted +$481.54 /
    1.98% / 4 trades; v2 asserts +$701.17 / 1.30% / 9 trades.
  * test_zero_cost_gross_exceeds_net: v1 asserted $549.55 zero-cost net;
    v2 asserts $797.18.
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy import (WINDOW, LONG_THRESHOLD, SHORT_THRESHOLD, backtest,
                      bollinger_bands, generate_positions, load_bars,
                      mr_positions, t_statistics, trend_positions)

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "bars.csv")


class TestSignal(unittest.TestCase):
    def test_trend_warmup_is_flat(self):
        closes = [2000.0 + i for i in range(300)]
        pos = trend_positions(closes)
        self.assertEqual(pos[:WINDOW - 1], [0] * (WINDOW - 1))

    def test_strong_uptrend_goes_long(self):
        closes = [2000.0 * (1.001 ** i) for i in range(400)]
        pos = generate_positions(closes)
        self.assertEqual(pos[-1], 1)

    def test_strong_downtrend_goes_short(self):
        closes = [2000.0 * (0.999 ** i) for i in range(400)]
        pos = generate_positions(closes)
        self.assertEqual(pos[-1], -1)

    def test_flat_market_stays_flat(self):
        closes = [2000.0] * 400
        pos = generate_positions(closes)
        self.assertEqual(set(pos), {0})

    def test_no_lookahead(self):
        """Position at bar i must be identical whether later bars exist."""
        bars = load_bars(DATA)
        closes = [b["close"] for b in bars]
        full = generate_positions(closes)
        for cut in (300, 700, 1000, 1439):
            partial = generate_positions(closes[:cut])
            self.assertEqual(full[:cut], partial,
                             f"lookahead detected at cut {cut}")

    def test_hysteresis_holds_inside_band(self):
        # t-stat inside (-1.5, 2.25) must not change the trend position.
        closes = [2000.0 * (0.999 ** i) for i in range(400)]
        pos = trend_positions(closes)
        self.assertEqual(pos[-1], -1)
        # append a flat stretch: slope decays, t falls inside the band,
        # position must stay short (not flip to flat or long).
        closes2 = closes + [closes[-1]] * 100
        pos2 = trend_positions(closes2)
        self.assertEqual(pos2[-1], -1)

    def test_positions_only_in_minus_one_zero_one(self):
        bars = load_bars(DATA)
        pos = generate_positions(bars)
        self.assertTrue(all(p in (-1, 0, 1) for p in pos))
        self.assertEqual(len(pos), len(bars))

    def test_mr_layer_buys_dip(self):
        # Flat 2000 for 200 bars, then a sharp dip to 1900: the Bollinger
        # range layer must go long below its lower band.
        closes = [2000.0] * 200 + [1900.0] * 50
        pos = mr_positions(closes)
        self.assertEqual(pos[200], 1)

    def test_bounce_exit_protects_short(self):
        # Steady -0.1%/bar downtrend (short from the trend layer), then a
        # flat base and a +1.5% bounce: the bounce exit must take the short
        # to flat, and the latch must block any re-short while the t-stat
        # is still deeply negative.
        closes = [2000.0 * (0.999 ** i) for i in range(400)]
        base = closes[-1]
        closes += [base] * 10
        closes += [base * (1.001 ** j) for j in range(1, 21)]
        pos = generate_positions(closes)
        self.assertIn(-1, pos)
        first_flat_after_short = next(
            i for i in range(280, len(pos)) if pos[i] == 0 and pos[i - 1] == -1)
        self.assertTrue(all(p != -1 for p in pos[first_flat_after_short:]),
                        "re-short after bounce exit -- latch failed")

    def test_trend_priority_on_real_data(self):
        # Wherever the trend layer holds a position, the combined machine
        # must agree with it -- except inside a bounce-exit latch window,
        # where the bounce rule deliberately overrides a short to flat.
        # The override must be flat-only, short-only, and one contiguous
        # window (the 1082 bounce exit through the 1198 long flip).
        bars = load_bars(DATA)
        closes = [b["close"] for b in bars]
        trend = trend_positions(closes)
        combined = generate_positions(closes)
        diffs = [i for i, tp in enumerate(trend)
                 if tp != 0 and combined[i] != tp]
        self.assertTrue(
            all(combined[i] == 0 and trend[i] == -1 for i in diffs),
            "trend overridden by something other than a short bounce exit")
        self.assertEqual(diffs, list(range(diffs[0], diffs[-1] + 1)),
                         "bounce override is not one contiguous window")
        self.assertLess(len(diffs), 200)


class TestBacktest(unittest.TestCase):
    def test_real_data_matches_reported_numbers(self):
        bars = load_bars(DATA)
        r = backtest(bars)
        self.assertAlmostEqual(r["net_pnl"], 701.17, places=1)
        self.assertAlmostEqual(r["max_drawdown"], 0.0130, places=3)
        self.assertEqual(r["num_trades"], 9)

    def test_zero_cost_gross_exceeds_net(self):
        bars = load_bars(DATA)
        r = backtest(bars, taker_fee=0.0, slippage_base=0.0,
                     slippage_per_unit=0.0)
        self.assertAlmostEqual(r["net_pnl"], 797.18, places=1)
        self.assertGreater(r["net_pnl"], 701.0)

    def test_high_costs_still_profitable(self):
        # 20bp taker fee + 10bp base slippage: the strategy must survive
        # a much harsher cost model than the base one.
        bars = load_bars(DATA)
        r = backtest(bars, taker_fee=0.002, slippage_base=0.001,
                     slippage_per_unit=0.001)
        self.assertGreater(r["net_pnl"], 0.0)

    def test_determinism(self):
        bars = load_bars(DATA)
        r1 = backtest(bars)
        r2 = backtest(bars)
        self.assertEqual(r1["net_pnl"], r2["net_pnl"])
        self.assertEqual(r1["equity_curve"], r2["equity_curve"])

    def test_flat_positions_no_trades_no_pnl(self):
        bars = load_bars(DATA)
        r = backtest(bars, positions=[0] * len(bars))
        self.assertEqual(r["num_trades"], 0)
        self.assertAlmostEqual(r["net_pnl"], 0.0, places=6)

    def test_short_hold_beats_long_hold_on_this_data(self):
        bars = load_bars(DATA)
        short = backtest(bars, positions=[-1] * len(bars))
        long_ = backtest(bars, positions=[1] * len(bars))
        self.assertGreater(short["net_pnl"], long_["net_pnl"])


if __name__ == "__main__":
    unittest.main()
