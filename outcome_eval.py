"""Generic OHLC event outcomes. Proposed target: event_study/outcome_eval.py.

One output row per (event_id, evaluated_direction, outcome_config_id).
Original event context is preserved. This is a barrier/path study, not a fill
simulator: no fees, slippage, position sizing or assumed intrabar ordering.
"""

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from numbers import Integral, Real

import numpy as np
import pandas as pd

from crypto_quant_engine.event_study.event_schema import standardize_event_dataframe


@dataclass(frozen=True)
class OutcomeConfig:
    # All timing/entry and barrier choices are required, rather than inferred.
    entry: str                  # 'event_close' or 'next_open'
    threshold_unit: str         # 'return' (0.01 = 1%), 'atr', or 'price'
    tp: float
    sl: float
    horizon: int                # Number of complete forward candles
    bar_duration: str           # Fixed duration, e.g. '1h'
    bar_timestamp: str          # Market index denotes 'open' or 'close'
    directions: str = 'event_or_both'  # Or 'both' to test both for every event
    ambiguity: str = 'mark'      # Only supported policy in this first version
    atr_source: str = 'event'    # 'event' snapshot or 'market' detection candle
    atr_column: str = 'atr'

    def __post_init__(self):
        choices = {
            'entry': ('event_close', 'next_open'),
            'threshold_unit': ('return', 'atr', 'price'),
            'bar_timestamp': ('open', 'close'),
            'directions': ('event_or_both', 'both'),
            'ambiguity': ('mark',), 'atr_source': ('event', 'market'),
        }
        for name, allowed in choices.items():
            if getattr(self, name) not in allowed:
                raise ValueError(f'{name} must be one of {allowed}')
        for name in ('tp', 'sl'):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if isinstance(self.horizon, (bool, np.bool_)) or not isinstance(self.horizon, Integral) or self.horizon < 1:
            raise ValueError('horizon must be a positive integer')
        if not isinstance(self.bar_duration, (str, pd.Timedelta)):
            raise ValueError('bar_duration must be an explicit fixed duration')
        duration = pd.Timedelta(self.bar_duration)
        if pd.isna(duration) or duration <= pd.Timedelta(0):
            raise ValueError('bar_duration must be positive')
        if not isinstance(self.atr_column, str) or not self.atr_column:
            raise ValueError('atr_column must be a non-empty string')

    def definition(self):
        values = asdict(self)
        values.update(version=1, bar_duration=str(pd.Timedelta(self.bar_duration)),
                      tp=float(self.tp), sl=float(self.sl), horizon=int(self.horizon))
        return json.dumps(values, sort_keys=True, separators=(',', ':'))

    @property
    def config_id(self):
        return 'outcome_v1_' + sha256(self.definition().encode()).hexdigest()[:20]


INTEGER_COLUMNS = ('evaluated_direction', 'observed_bars', 'tp_hit_bar',
                   'sl_hit_bar', 'resolution_bar', 'mfe_bar', 'mae_bar')
TIME_COLUMNS = ('entry_time', 'tp_hit_bar_time', 'sl_hit_bar_time',
                'resolution_bar_time', 'mfe_bar_time', 'mae_bar_time',
                'observed_end_bar_time', 'horizon_bar_time')
FLOAT_COLUMNS = ('resolved_entry_price', 'atr_at_event', 'tp_price', 'sl_price',
                 'mfe_return', 'mae_return', 'mfe_atr', 'mae_atr',
                 'observed_return', 'horizon_return')
OUTPUT_COLUMNS = (
    'outcome_config_id', 'outcome_definition', 'outcome_label', 'horizon_complete',
) + INTEGER_COLUMNS + TIME_COLUMNS + FLOAT_COLUMNS


def _market_arrays(market, config):
    if not isinstance(market, pd.DataFrame) or not isinstance(market.index, pd.DatetimeIndex):
        raise ValueError('market_data_df must have a DatetimeIndex of candle keys')
    index = market.index
    if index.hasnans or not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError('Market timestamps must be sorted, unique and non-missing')
    if not market.columns.is_unique or not {'open', 'high', 'low', 'close'} <= set(market.columns):
        raise ValueError('Market requires unique columns including open/high/low/close')
    # Never silently turn missing candles into a shorter horizon.
    duration = pd.Timedelta(config.bar_duration)
    if len(index) > 1 and not (index[1:] - index[:-1] == duration).all():
        raise ValueError('Market must contain contiguous bars matching bar_duration')
    values = market[['open', 'high', 'low', 'close']].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('OHLC prices must be finite and positive')
    opening, high, low, close = values.T
    if ((low > np.minimum(opening, close)) | (high < np.maximum(opening, close)) | (high < low)).any():
        raise ValueError('Invalid OHLC candle geometry')
    return opening, high, low, close


def _first_hit(mask):
    matches = np.flatnonzero(mask)
    return int(matches[0]) + 1 if len(matches) else None


def _evaluate_path(index, high, low, close, i, entry, direction, atr, config):
    # Both conventions start monitoring at i+1; next_open includes its entry
    # candle, while event_close excludes the already completed detection candle.
    start = i + 1
    stop = min(start + config.horizon, len(index))
    n = stop - start
    full = n == config.horizon
    scale = atr if config.threshold_unit == 'atr' else entry if config.threshold_unit == 'return' else 1.0
    tp_distance, sl_distance = config.tp * scale, config.sl * scale
    tp_price, sl_price = entry + direction * tp_distance, entry - direction * sl_distance
    if not np.isfinite([tp_price, sl_price]).all() or min(tp_price, sl_price) <= 0:
        raise ValueError('Configured barriers must be finite and positive prices')
    out = dict(observed_bars=n, horizon_complete=full, tp_price=tp_price,
               sl_price=sl_price, outcome_label='insufficient_data')
    if n == 0:
        return out

    best = high[start:stop] if direction == 1 else low[start:stop]
    worst = low[start:stop] if direction == 1 else high[start:stop]
    favorable = direction * (best - entry)
    adverse = direction * (worst - entry)
    tp_bar = _first_hit(favorable >= tp_distance)
    sl_bar = _first_hit(adverse <= -sl_distance)
    if tp_bar is not None and sl_bar == tp_bar:
        label, resolved = 'ambiguous', tp_bar
    elif tp_bar is not None and (sl_bar is None or tp_bar < sl_bar):
        label, resolved = 'target', tp_bar
    elif sl_bar is not None:
        label, resolved = 'stop', sl_bar
    else:
        label, resolved = ('timeout', config.horizon) if full else ('insufficient_data', None)

    def bar_time(offset):
        return index[i + offset] if offset is not None else pd.NaT

    # Include the zero excursion at entry: MFE >= 0, MAE <= 0. Timing 0 means
    # entry itself (no forward candle); ties choose the earliest observation.
    best_path = np.r_[0.0, favorable]
    worst_path = np.r_[0.0, adverse]
    mfe_bar, mae_bar = int(best_path.argmax()), int(worst_path.argmin())
    mfe, mae = best_path[mfe_bar], worst_path[mae_bar]
    terminal = direction * (close[stop - 1] - entry) / entry
    out.update(
        outcome_label=label, tp_hit_bar=tp_bar, sl_hit_bar=sl_bar,
        resolution_bar=resolved, tp_hit_bar_time=bar_time(tp_bar),
        sl_hit_bar_time=bar_time(sl_bar), resolution_bar_time=bar_time(resolved),
        mfe_return=mfe / entry, mae_return=mae / entry,
        mfe_atr=mfe / atr if np.isfinite(atr) else np.nan,
        mae_atr=mae / atr if np.isfinite(atr) else np.nan,
        mfe_bar=mfe_bar, mae_bar=mae_bar,
        mfe_bar_time=bar_time(mfe_bar) if mfe_bar else pd.NaT,
        mae_bar_time=bar_time(mae_bar) if mae_bar else pd.NaT,
        observed_end_bar_time=index[stop - 1], observed_return=terminal,
        horizon_bar_time=index[stop - 1] if full else pd.NaT,
        horizon_return=terminal if full else np.nan,
    )
    return out


def label_event_outcomes(market_data_df, events_df, config: OutcomeConfig):
    """Preserve each event and attach one configured economic path evaluation.

    Market: one instrument/timeframe, sorted contiguous DatetimeIndex, OHLC.
    Events: canonical schema, with event_bar_time required for exact lookup.
    Supplied context (including event ATR) must be known at event_time.

    Entry is priced at the detection close or next candle open. entry_time is
    the physical time under bar_timestamp/bar_duration, never the candle key
    by assumption; entries before event_time raise rather than shift silently.
    event_close is a theoretical close-price benchmark, not a guaranteed fill.

    Bar offsets are 1..H for the following H candles. *_bar_time are candle
    keys, not intrabar execution timestamps. MFE/MAE cover the entire observed
    path even after resolution; horizon_complete distinguishes partial paths.
    First hits of both barriers in the same candle remain ambiguous, including
    gap candles; this conservative initial policy does not use open ordering.
    Both hit times are retained, even when the second is after resolution.

    No-entry tail events are retained as no_entry. An entered but unresolved
    partial path is insufficient_data; a partial path can still have a known
    target/stop/ambiguous label. horizon_return exists only for complete paths.
    ATR metrics are populated only for ATR configurations, using a frozen,
    positive event snapshot or detection-candle market ATR (explicit source).
    Optional event entry_price/reference_price are preserved as context only.
    """
    if not isinstance(config, OutcomeConfig):
        raise TypeError('config must be OutcomeConfig')
    opening, high, low, close = _market_arrays(market_data_df, config)
    collisions = set(events_df.columns).intersection(OUTPUT_COLUMNS)
    if collisions:
        raise ValueError(f'Event context collides with outcome columns: {sorted(collisions)}')
    # Validate with the existing event contract, preserving row order/context.
    events = standardize_event_dataframe(events_df, sort_by_time=False)
    if 'event_bar_time' not in events:
        raise ValueError('events_df requires event_bar_time; no legacy time inference')
    index = market_data_df.index
    positions = index.get_indexer(events['event_bar_time'])
    if (positions < 0).any():
        raise ValueError('Some event_bar_time values do not match market timestamps')
    if not events.empty:
        for col in ('event_time', 'event_bar_time'):
            if str(events[col].dt.tz) != str(index.tz):
                raise ValueError('Event and market timestamps must use the same timezone')

    duration = pd.Timedelta(config.bar_duration)
    definition, config_id = config.definition(), config.config_id
    source_rows, rows = [], []
    for row_number, (i, event) in enumerate(zip(positions, events.to_dict('records'))):
        close_time = index[i] + (duration if config.bar_timestamp == 'open' else pd.Timedelta(0))
        if config.entry == 'event_close':
            entry_time, entry = close_time, close[i]
        elif i + 1 < len(index):
            entry_time = index[i + 1] - (duration if config.bar_timestamp == 'close' else pd.Timedelta(0))
            entry = opening[i + 1]
        else:
            entry_time, entry = pd.NaT, np.nan
        if pd.notna(entry_time) and entry_time < event['event_time']:
            raise ValueError(f"Entry would precede event_time for event {event['event_id']!r}")

        atr = np.nan
        if config.threshold_unit == 'atr':
            source = event if config.atr_source == 'event' else market_data_df.iloc[i]
            atr = source.get(config.atr_column, np.nan)
            if isinstance(atr, (bool, np.bool_)) or not isinstance(atr, Real) or not np.isfinite(atr) or atr <= 0:
                raise ValueError(f"Positive finite ATR required for event {event['event_id']!r}")
        directions = (1, -1) if config.directions == 'both' or pd.isna(event['direction']) else (int(event['direction']),)
        for direction in directions:
            out = dict(outcome_config_id=config_id, outcome_definition=definition,
                       evaluated_direction=direction, entry_time=entry_time,
                       resolved_entry_price=entry, atr_at_event=atr,
                       observed_bars=0, horizon_complete=False, outcome_label='no_entry')
            if np.isfinite(entry):
                out.update(_evaluate_path(index, high, low, close, i, entry, direction, atr, config))
            source_rows.append(row_number)
            rows.append(out)

    metrics = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    for col in INTEGER_COLUMNS:
        metrics[col] = metrics[col].astype('Int64')
    for col in FLOAT_COLUMNS:
        metrics[col] = metrics[col].astype('float64')
    for col in TIME_COLUMNS:
        metrics[col] = pd.array(metrics[col], dtype=index.dtype)
    metrics['horizon_complete'] = metrics['horizon_complete'].astype(bool)
    return pd.concat([events.iloc[source_rows].reset_index(drop=True), metrics], axis=1)
