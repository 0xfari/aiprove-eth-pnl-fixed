"""Tests for strategy.py -- stdlib unittest, no network, no extra deps.

Run:  python -m unittest discover -s tests -v
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy import (WINDOW, THRESHOLD, backtest, generate_positions,
                      load_bars, t_statistics)

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "bars.csv")


class TestSignal(unittest.TestCase):
    def test_warmup_is_flat(self):
        closes = [2000.0 + i for i in range(300)]
        pos = generate_positions(closes)
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
        # t-stat inside (-2, 2) must not change the position.
        closes = [2000.0 * (0.999 ** i) for i in range(400)]
        pos = generate_positions(closes)
        self.assertEqual(pos[-1], -1)
        # append a flat stretch: slope decays, t falls inside the band,
        # position must stay short (not flip to flat or long).
        closes2 = closes + [closes[-1]] * 100
        pos2 = generate_positions(closes2)
        self.assertEqual(pos2[-1], -1)

    def test_positions_only_in_minus_one_zero_one(self):
        bars = load_bars(DATA)
        pos = generate_positions(bars)
        self.assertTrue(all(p in (-1, 0, 1) for p in pos))
        self.assertEqual(len(pos), len(bars))


class TestBacktest(unittest.TestCase):
    def test_real_data_matches_reported_numbers(self):
        bars = load_bars(DATA)
        r = backtest(bars)
        self.assertAlmostEqual(r["net_pnl"], 481.54, places=1)
        self.assertAlmostEqual(r["max_drawdown"], 0.0198, places=3)
        self.assertEqual(r["num_trades"], 4)

    def test_zero_cost_gross_exceeds_net(self):
        bars = load_bars(DATA)
        r = backtest(bars, taker_fee=0.0, slippage_base=0.0,
                     slippage_per_unit=0.0)
        self.assertAlmostEqual(r["net_pnl"], 549.55, places=1)
        self.assertGreater(r["net_pnl"], 481.0)

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
