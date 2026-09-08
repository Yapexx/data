'''Small deterministic regressions. Run with python -m unittest <module>.'''

import unittest
import warnings
from io import StringIO

import pandas as pd

from crypto_quant_engine.signals.pivot_detector import ZigZagATR


def detector(**kwargs):
    return ZigZagATR('1h', atr_mult=1.0, **kwargs)


def bar(zz, high, low, close, atr=1.0, **kwargs):
    return zz.update(close, high, low, close, 1.0, atr, **kwargs)


def bridge_example(zz):
    return tuple(bar(zz, *values) for values in (
        (100, 95, 98), (99, 90, 92), (97, 91, 95), (98, 92, 94)))


class PivotDetectorTests(unittest.TestCase):
    def test_exact_improvement_replaces_candidate_and_restarts_count(self):
        zz = detector()
        for values in ((10, 8, 9), (9.8, 8, 8.5), (10.1, 8, 9.1),
                       (9, 7, 8), (8, 6, 7)):
            bar(zz, *values)
        self.assertEqual(len(zz.events), 1)
        self.assertEqual(zz.events[0]['pivot_index'], 2)
        self.assertEqual(zz.events[0]['pivot_price'], 10.1)
        self.assertEqual(zz.events[0]['confirmation_index'], 4)

    def test_equal_high_keeps_first_occurrence(self):
        zz = detector()
        for values in ((10, 8, 9), (10, 8, 8.5), (9, 7, 8)):
            bar(zz, *values)
        self.assertEqual(zz.events[0]['pivot_index'], 0)
        self.assertEqual(zz.events[0]['pivot_type'], 'high')

    def test_extreme_bar_cannot_confirm_itself(self):
        zz = detector(confirm_closes=1)
        out = bar(zz, 20, 1, 10)
        self.assertFalse(out['pivot_high'] or out['pivot_low'])
        self.assertTrue(zz.bootstrap)

    def test_close_and_wick_confirmation_have_explicit_different_rules(self):
        close, wick = detector(confirm_closes=1), detector(confirm_closes=1, use_close=False)
        for zz in (close, wick):
            bar(zz, 10, 8, 9)
        self.assertFalse(bar(close, 9.9, 8, 9.5)['pivot_high'])
        self.assertTrue(bar(wick, 9.9, 8, 9.5)['pivot_high'])

    def test_bootstrap_uses_real_wicks_and_waits_for_valid_atr(self):
        zz = detector()
        bar(zz, 10, 8, 9, atr=None)
        bar(zz, 12, 8, 8.5, atr=None)
        bar(zz, 11, 7, 8, atr=float('nan'))
        self.assertFalse(zz.events)
        bar(zz, 10, 7, 8)
        out = bar(zz, 9, 6, 7)
        self.assertEqual(zz.events[0]['pivot_price'], 12)
        self.assertEqual(zz.events[0]['pivot_index'], 1)
        self.assertFalse(out['features_ready'])
        self.assertIsNone(out['microcycle_density'])

    def test_missing_atr_breaks_confirmation_streak(self):
        zz = detector()
        for high, low, close, atr in ((10, 8, 9, 1), (9, 7, 8, 1),
                                      (8, 6, 7, None), (7, 5, 6, 1), (6, 4, 5, 1)):
            bar(zz, high, low, close, atr)
        self.assertEqual(zz.events[0]['confirmation_index'], 4)

    def test_confirmation_bridge_retains_earlier_opposite_extreme(self):
        zz = detector()
        rows = bridge_example(zz)
        self.assertEqual(tuple(event['pivot_index'] for event in zz.events), (0, 1))
        self.assertEqual(tuple(event['confirmation_index'] for event in zz.events), (2, 3))
        self.assertEqual(zz.events[1]['pivot_price'], 90)
        self.assertEqual(rows[3]['potential_pivot_time'], 1)
        self.assertEqual(rows[3]['direction'], 'down')
        self.assertEqual(rows[3]['last_pivot_type'], 'H')
        self.assertEqual(rows[3]['local_low'], None)
        self.assertEqual(rows[3]['current_local_low'], 90)
        self.assertTrue(rows[3]['early_reversal'])
        self.assertEqual(rows[3]['reversal_count'], 2)
        self.assertEqual(zz.state['direction'], 'up')

    def test_full_leg_path_and_microcycles_have_matching_origins(self):
        zz = detector()
        rows = bridge_example(zz)
        # High 100 -> close 98 -> 92 -> 95 -> 94: path 12, displacement 6.
        self.assertAlmostEqual(rows[-1]['leg_inefficiency'], 0.5)
        self.assertEqual(rows[-1]['microcycle_count'], 2)
        self.assertAlmostEqual(rows[-1]['microcycle_density'], 2 / 3)
        self.assertAlmostEqual(rows[-1]['slope_decay'], 1.5)
        # New low 90 -> close 92 -> 95 -> 94 -> 96: path 8, displacement 6.
        next_row = bar(zz, 99, 93, 96)
        self.assertAlmostEqual(next_row['leg_inefficiency'], 0.25)
        self.assertAlmostEqual(next_row['microcycle_density'], 2 / 3)

    def test_equal_price_double_top_is_allowed_and_alternates(self):
        zz = detector(confirm_closes=1)
        for close in (8, 10, 8, 10, 8, 10, 8):
            bar(zz, close + 0.1, close - 0.1, close)
        types = tuple(event['pivot_type'] for event in zz.events)
        self.assertTrue(all(a != b for a, b in zip(types, types[1:])))
        highs = tuple(event['pivot_price'] for event in zz.events if event['pivot_type'] == 'high')
        self.assertGreaterEqual(len(highs), 2)
        self.assertEqual(len(set(highs)), 1)

    def test_legacy_gates_warn_and_cannot_freeze_a_short_swing(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            zz = detector(confirm_closes=1, min_runext_step_atr=100,
                          min_pivot_separation_atr=100, min_bars_between_pivots=100,
                          min_leg_span_atr=100, range_if_small_span=True)
        self.assertEqual(len(caught), 1)
        for close in (8, 10, 8, 10, 8):
            bar(zz, close + 0.1, close - 0.1, close)
        self.assertEqual(len(zz.events), 4)

    def test_macro_requires_hh_and_hl_and_uses_current_extreme(self):
        zz = detector(confirm_closes=1)
        rows = tuple(bar(zz, close + 0.1, close - 0.1, close)
                     for close in (100, 110, 103, 115, 107))
        self.assertTrue(all(row['macro_trend'] is None for row in rows))
        self.assertEqual(zz.state['macro_trend'], 'up')
        self.assertAlmostEqual(zz.state['macro_ext_price'], 115.1)
        self.assertAlmostEqual(zz.state['macro_anchor_price'], 99.9)
        self.assertEqual(bar(zz, 120.1, 119.9, 120)['macro_trend'], 'up')

    def test_retracement_and_fibonacci_geometry(self):
        zz = detector()
        self.assertEqual(zz._retracement(80, 100, 100, 'up'), 0)
        self.assertEqual(zz._retracement(80, 100, 90, 'up'), 0.5)
        self.assertEqual(zz._retracement(100, 80, 90, 'down'), 0.5)
        self.assertEqual(zz._retracement(80, 100, 70, 'up'), 1.5)
        self.assertEqual(zz._nearest_fib(0.38), 0.382)
        self.assertEqual(zz._nearest_fib(1.5), 1)
        self.assertIsNone(zz._nearest_fib(None))

    def test_range_exits_when_price_leaves_band(self):
        zz = detector(confirm_closes=1)
        bar(zz, 101, 99, 100)
        bar(zz, 111, 109, 110)  # Confirm low 99, seed high 111.
        # Large ATR prevents a pivot while we examine the range classification.
        mid = bar(zz, 110, 104, 105, atr=100)
        self.assertEqual(mid['structure_micro'], 'range')
        end = bar(zz, 111, 109, 110, atr=100)
        self.assertEqual(end['structure_micro'], 'up')

    def test_labels_are_nullable_and_separate_from_confirmation_events(self):
        zz = detector()
        first = bar(zz, 100, 95, 98)
        before = first.copy()
        self.assertTrue(zz.get_result_df()['is_pivot'].isna().all())
        bar(zz, 99, 90, 92)
        bar(zz, 97, 91, 95)
        df = zz.get_result_df()
        self.assertEqual(df.loc[0, 'is_pivot'], 1)
        self.assertEqual(df.loc[0, 'label_available_time'], 2)
        self.assertTrue(pd.isna(df.loc[2, 'is_pivot']))
        bar(zz, 98, 92, 94)
        self.assertEqual(first, before)
        self.assertEqual(zz.get_result_df().loc[1, 'pivot_confirmation_time'], 3)
        X, y, meta = zz.get_research_dataset()
        self.assertFalse(set(zz.label_col) & set(X.columns))
        self.assertNotIn('confirmation_time', X)
        self.assertEqual(tuple(X.index), tuple(y.index))
        self.assertEqual(tuple(X.index), tuple(meta.index))

    def test_reset_replay_matches_fresh_instance(self):
        zz = detector()
        bridge_example(zz)
        expected = zz.get_result_df()
        zz.reset()
        self.assertTrue(zz.bootstrap)
        self.assertFalse(zz.events or zz.value_by_dates or zz.pivots_ts)
        bridge_example(zz)
        pd.testing.assert_frame_equal(zz.get_result_df(), expected)
        self.assertEqual(len(zz.pivots_ts), 2)

    def test_bad_bar_does_not_advance_state(self):
        zz = detector()
        bar(zz, 10, 8, 9, timestamp=10)
        with self.assertRaises(ValueError):
            bar(zz, 10, 8, 9, timestamp=10)
        with self.assertRaises(ValueError):
            bar(zz, 8, 10, 9, timestamp=11)
        with self.assertRaises(ValueError):
            bar(zz, float('nan'), 8, 9, timestamp=11)
        self.assertEqual(zz.state['i'], 0)

    def test_invalid_parameters(self):
        for kwargs in (dict(confirm_closes=0), dict(confirm_closes=1.5),
                       dict(atr_period=1), dict(range_band=(0.7, 0.3)),
                       dict(confirm_hysteresis_mult=0.5)):
            with self.assertRaises(ValueError):
                detector(**kwargs)

    def test_enricher_preserves_negative_and_unresolved_labels_across_csv(self):
        from crypto_quant_engine.analysis.dataset_builder import pivot_enricher

        frame = pd.DataFrame(dict(open=9., high=10., low=8., close=9., volume=1.,
                                  pivot_type=('high', '', 'low', None, None),
                                  label_resolved=(True, True, True, True, False)),
                             index=pd.date_range('2024-01-01', periods=5, freq='h', name='datetime'))
        loaded = pd.read_csv(StringIO(frame.to_csv()), index_col=0, parse_dates=True)
        for data in (frame, loaded):
            result = pivot_enricher(data)
            self.assertEqual(result.is_pivot.iloc[:4].tolist(), [1, 0, 1, 0])
            self.assertTrue(pd.isna(result.is_pivot.iloc[4]))
            self.assertTrue(pd.isna(result.is_high_pivot.iloc[4]))
            self.assertTrue(pd.isna(result.is_low_pivot.iloc[4]))

    def test_short_empty_and_rebound_market_data(self):
        from crypto_quant_engine.market_data import MarketData

        frame = pd.DataFrame(dict(open=(9.,), high=(10.,), low=(8.,), close=(9.,), volume=(1.,)),
                             index=pd.date_range('2024-01-01', periods=1, name='datetime'))
        frame['datetime'] = frame.index
        first = MarketData('TEST', exchange='test', market_type='s')
        first.market_data_by_frequencies['1h'] = frame
        second = MarketData('SECOND', exchange='test', market_type='s')
        second.market_data_by_frequencies['1h'] = frame.iloc[:0]
        zz = detector()
        zz.set_market_data(first)
        zz.signal_value_on_date(frame.index[0])
        self.assertEqual(len(zz.value_by_dates), 1)
        self.assertTrue(zz.get_result_df().is_pivot.isna().all())
        self.assertTrue(zz.get_research_dataset()[0].empty)
        with self.assertRaises(ValueError):
            zz.set_market_data(second, clear_data=False)
        zz.set_market_data(second)
        self.assertFalse(zz.value_by_dates or zz.events)
        zz.compute_and_cache_full_period(fail_on_missing_dates=False)
        self.assertTrue(zz.get_research_dataset()[0].empty)
        with self.assertRaises(ValueError):
            zz.compute_and_cache_full_period()


if __name__ == '__main__':
    unittest.main()
