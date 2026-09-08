'''Deterministic daily/hourly research checks using the existing MarketData loader.

Run scripts/pivot_test.py --audit, or run this module from the repository root.
No files are written unless --output-dir is supplied.
'''

import argparse
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from crypto_quant_engine.market_data import MarketData
from crypto_quant_engine.tools import OPEN, HIGH, LOW, CLOSE, VOLUME, DATE
from crypto_quant_engine.signals.pivot_detector import ZigZagATR


SAMPLES = (
    ('1d', datetime(2024, 1, 1), datetime(2025, 12, 31)),
    ('1h', datetime(2024, 1, 1), datetime(2024, 3, 1, 23)),
)
BOUNDED_FIELDS = (
    'pullback_ratio', 'leg_age_norm', 'leg_amp_norm', 'wick_upper_norm',
    'wick_lower_norm', 'adx_factor', 'leg_compression', 'leg_inefficiency',
    'microcycle_density', 'nearest_fib', 'macro_nearest_fib', 'pivot_score', 'pivot_score_adj',
)


def make_detector(frequency):
    return ZigZagATR(frequency, range_persistence=2, min_leg_atr=0.5)


def validate_events(zz, market):
    events = zz.get_events_df()
    previous_index, previous_type = -1, None
    for event in events.itertuples(index=False):
        assert event.pivot_type != previous_type, 'Confirmed pivots must alternate'
        assert previous_index < event.pivot_index < event.confirmation_index
        field = HIGH if event.pivot_type == 'high' else LOW
        window = market.iloc[previous_index + 1:event.confirmation_index + 1][field]
        extreme_time = window.idxmax() if field == HIGH else window.idxmin()
        assert event.pivot_time == extreme_time, 'Pivot must locate the first exact extreme'
        assert event.pivot_price == market.loc[extreme_time, field]
        count = zz.config['confirm_closes']
        assert event.confirmation_delay >= count
        for j in range(event.confirmation_index - count + 1, event.confirmation_index + 1):
            output = zz.value_by_dates[market.index[j]]
            close = market.iloc[j][CLOSE]
            distance = event.pivot_price - close if field == HIGH else close - event.pivot_price
            threshold = zz.config['atr_mult'] * zz.config['confirm_hysteresis_mult'] * output['atr']
            assert distance + 1e-9 >= threshold, 'Confirmation needs consecutive qualifying closes'
        confirmation_row = zz.value_by_dates[event.confirmation_time]
        assert confirmation_row['last_pivot_idx'] == previous_index, 'Feature snapshot includes new pivot'
        if previous_index >= 0:
            assert confirmation_row['early_reversal'], 'Confirmation counter reset before snapshot'
        previous_index, previous_type = event.pivot_index, event.pivot_type


def validate_features(zz, market):
    rows = pd.DataFrame.from_dict(zz.value_by_dates, orient='index')
    numeric = rows.select_dtypes(include='number')
    assert not np.isinf(numeric.to_numpy(dtype=float)).any()
    for key in BOUNDED_FIELDS:
        values = rows[key].dropna()
        assert values.between(0, 1).all(), f'{key} outside [0, 1]'
    for i, row in enumerate(rows.to_dict('records')):
        p = row['last_pivot_idx']
        if p < 0:
            continue
        closes = market[CLOSE].iloc[p:i + 1].to_numpy()
        path = abs(closes[0] - row['last_pivot_price']) + np.abs(np.diff(closes)).sum()
        displacement = abs(closes[-1] - row['last_pivot_price'])
        assert displacement <= path + 1e-8, 'Path origin differs from displacement origin'
        expected = 0 if path == 0 else 1 - displacement / path
        assert np.isclose(row['leg_inefficiency'], expected, atol=1e-10), 'Leg path omitted observed bars'
        moves = np.sign(np.diff(closes))
        moves = moves[moves != 0]
        cycles = int(np.count_nonzero(moves[1:] != moves[:-1]))
        assert row['microcycle_count'] == cycles
        assert np.isclose(row['microcycle_density'], cycles / (i - p))
    return rows


def validate_replay(zz, market, rows):
    indicators = zz._indicator_frame(market)
    replay = make_detector(zz.frequency)
    for data, values in zip(market.to_dict('records'), indicators.to_dict('records')):
        replay.update(*(data[key] for key in (OPEN, HIGH, LOW, CLOSE, VOLUME)),
                      timestamp=data[DATE], **values)
    pd.testing.assert_frame_equal(pd.DataFrame.from_dict(replay.value_by_dates, orient='index'), rows)
    pd.testing.assert_frame_equal(replay.get_result_df(), zz.get_result_df())
    expected = replay.get_result_df()
    replay.reset()
    assert not replay.events and replay.bootstrap and not replay.value_by_dates
    for data, values in zip(market.to_dict('records'), indicators.to_dict('records')):
        replay.update(*(data[key] for key in (OPEN, HIGH, LOW, CLOSE, VOLUME)),
                      timestamp=data[DATE], **values)
    pd.testing.assert_frame_equal(replay.get_result_df(), expected)


def validate_prefix_and_append(zz, market, rows, trading_pair, exchange, market_type):
    # Check both sides of a real confirmation and a later independent cutoff.
    events = zz.get_events_df()
    confirmation = int(events.iloc[min(4, len(events) - 1)].confirmation_index) if len(events) else len(market) // 2
    stops = sorted(set((max(1, confirmation - 1), confirmation, len(market) // 2)))
    for stop in stops:
        md = MarketData(trading_pair, exchange=exchange, market_type=market_type,
                        start_time=market.index[0], end_time=market.index[stop])
        prefix = make_detector(zz.frequency)
        prefix.set_market_data(md)
        prefix.compute_and_cache_full_period()
        before = {ts: values.copy() for ts, values in prefix.value_by_dates.items()}
        assert before == {ts: zz.value_by_dates[ts] for ts in before}, 'Future bars changed past features'
        labels = prefix.get_result_df()
        resolved = labels.index[labels.label_resolved]
        pd.testing.assert_frame_equal(labels.loc[resolved, list(zz.label_col)],
                                      zz.get_result_df().loc[resolved, list(zz.label_col)])
        # Exercise the actual Signal append entry point with full Wilder history.
        md.append_bar(zz.frequency, market.iloc[stop + 1].to_dict())
        prefix.compute_and_cache_value_on_bar_append(market.index[stop + 1])
        after = prefix.value_by_dates
        assert after == {ts: zz.value_by_dates[ts] for ts in after}, 'Batch and append differ'
        assert {ts: after[ts] for ts in before} == before, 'Append rewrote prior features'
        prefix.compute_and_cache_value_on_bar_append(market.index[stop + 1])
        assert len(prefix.value_by_dates) == stop + 2, 'Repeated append changed detector history'


def audit_sample(frequency, start, end, trading_pair='BTC_USDT', exchange='binance', market_type='s'):
    started = perf_counter()
    md = MarketData(trading_pair, exchange=exchange, market_type=market_type, start_time=start, end_time=end)
    market = md.get_market_data(frequency)
    if len(market) < 30:
        raise ValueError('Audit sample requires at least 30 bars')
    assert market.index.is_unique and market.index.is_monotonic_increasing
    delta = pd.Timedelta(days=1) if frequency == '1d' else pd.Timedelta(hours=1)
    assert (market.index.to_series().diff().dropna() == delta).all(), 'Missing market bars'
    zz = make_detector(frequency)
    zz.set_market_data(md)
    zz.compute_and_cache_full_period()
    calculation_seconds = perf_counter() - started
    validate_events(zz, market)
    rows = validate_features(zz, market)
    validate_replay(zz, market, rows)
    validate_prefix_and_append(zz, market, rows, trading_pair, exchange, market_type)
    result, events = zz.get_result_df(), zz.get_events_df()
    X, y, metadata = zz.get_research_dataset()
    assert tuple(X.columns) == zz.training_fields
    assert not set(zz.label_col + zz.output_fields).intersection(X.columns)
    assert X.index.equals(y.index) and X.index.equals(metadata.index)
    assert (metadata.label_available_time >= metadata.bar_time).all()
    missing = rows.isna().sum()
    stats = dict(
        frequency=frequency, start=str(market.index[0]), end=str(market.index[-1]),
        bars=len(market), highs=int((events.pivot_type == 'high').sum()),
        lows=int((events.pivot_type == 'low').sum()),
        prevalence_all_bars=len(events) / len(market),
        unresolved_bars=int((~result.label_resolved).sum()),
        research_rows=len(X), research_positives=int(y.sum()),
        research_prevalence=float(y.mean()),
        confirmation_delay=events.confirmation_delay.describe().to_dict(),
        missing_counts={key: int(value) for key, value in missing.items() if value},
        constant_features={key: int(X[key].nunique()) for key in X if X[key].nunique() <= 1},
        calculation_seconds=round(calculation_seconds, 3),
        audit_seconds=round(perf_counter() - started, 3),
    )
    return stats, result, events


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--frequency', choices=('1d', '1h', 'both'), default='both')
    parser.add_argument('--pair', default='BTC_USDT')
    parser.add_argument('--exchange', default='binance')
    parser.add_argument('--market-type', default='s')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    for frequency, start, end in SAMPLES:
        if args.frequency not in ('both', frequency):
            continue
        stats, result, events = audit_sample(frequency, start, end, args.pair, args.exchange, args.market_type)
        print(f'PASS {frequency}: {stats}')
        print(events.head(5).to_string(index=False))
        if args.output_dir is not None:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            stem = f'pivot_{args.pair}_{frequency}_{args.exchange}_{args.market_type}'
            result.to_csv(args.output_dir / f'{stem}_features_labels.csv')
            events.to_csv(args.output_dir / f'{stem}_events.csv', index=False)


if __name__ == '__main__':
    main()
