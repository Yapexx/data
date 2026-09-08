'''Causal ATR swing detection with retrospective extreme labels.

See pivot_detector.md for the swing definition, feature timing and migration notes.
'''

from collections import deque
from dataclasses import dataclass
from math import isfinite
import warnings

import pandas as pd

from crypto_quant_engine.tools import OPEN, HIGH, LOW, CLOSE, VOLUME, DATE, is_aligned
from crypto_quant_engine.trade_signal import Signal
from crypto_quant_engine.signals.talib_volatility import ATRSignal
from crypto_quant_engine.signals.talib_trend import ADXSignal
from crypto_quant_engine.signals.talib_momentum import RSISignal


@dataclass(frozen=True)
class _Bar:
    o: float
    h: float
    l: float
    c: float
    atr: float | None
    rsi: float | None
    timestamp: object


@dataclass(frozen=True)
class _Extreme:
    index: int
    price: float
    rsi: float | None


class ZigZagATR(Signal):
    '''Track exact bar extrema; confirm after an ATR reversal persists.

    Features describe the current closed bar before its pivot transition. Events
    describe that transition; labels are attached separately to historical bars.
    All prices/indicators supplied to update() must belong to the completed bar.
    '''

    FIB_LEVELS = (0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0)
    categorical_fields = ('direction', 'macro_trend', 'structure_macro',
                          'structure_micro', 'last_pivot_type', 'adx_regime')
    numerical_fields = (
        'bars_since_pivot', 'atr_pct', 'distance_to_pivot_atr',
        'pullback_ratio', 'early_reversal', 'leg_age_norm', 'leg_amp_norm',
        'wick_upper_norm', 'wick_lower_norm', 'divergence_flag',
        'adx', 'adx_factor', 'adx_rising', 'leg_exhaust_atr', 'slope_decay',
        'exhaustion_score', 'leg_compression', 'leg_inefficiency',
        'microcycle_count', 'microcycle_density', 'retracement', 'nearest_fib',
        'macro_retracement', 'macro_nearest_fib', 'bars_since_macro_anchor',
    )
    training_fields = categorical_fields + numerical_fields
    output_fields = (
        'pivot_score', 'pivot_score_adj', 'local_high', 'local_low',
        'last_H_time', 'last_L_time', 'potential_pivot', 'potential_pivot_time',
        'pivot_high', 'pivot_low', 'current_local_high', 'current_local_low',
        'new_pivot_type', 'new_pivot_time', 'new_pivot_price',
        'confirmation_time', 'confirmation_delay', 'hh', 'hl', 'lh', 'll',
    )
    label_col = ('is_pivot', 'pivot_type', 'label_resolved',
                 'label_available_time', 'pivot_confirmation_time')

    def __init__(self, frequency: str, *, atr_mult: float = 1.5,
                 use_close: bool = True, range_band: tuple = (0.30, 0.70),
                 range_persistence: int = 0, min_leg_atr: float = 0.0,
                 atr_period: int = 14, confirm_closes: int = 2,
                 confirm_hysteresis_mult: float = 1.0,
                 min_runext_step_atr=None, min_pivot_separation_atr=None,
                 min_bars_between_pivots=None, min_leg_span_atr=None,
                 range_if_small_span=None):
        self._validate_config(atr_mult, confirm_closes, confirm_hysteresis_mult,
                              atr_period, range_band, range_persistence, min_leg_atr)
        retired = dict(min_runext_step_atr=min_runext_step_atr,
                       min_pivot_separation_atr=min_pivot_separation_atr,
                       min_bars_between_pivots=min_bars_between_pivots,
                       min_leg_span_atr=min_leg_span_atr,
                       range_if_small_span=range_if_small_span)
        supplied = tuple(key for key, value in retired.items() if value is not None)
        if supplied:
            warnings.warn(f'Ignored retired pivot filters: {", ".join(supplied)}. '
                          'See pivot_detector.md.', FutureWarning, stacklevel=2)
        self.atr_signal_object = ATRSignal(frequency, atr_period)
        self.rsi_signal_object = RSISignal(frequency, atr_period)
        self.adx_signal_object = ADXSignal(frequency, max(2, atr_period // 2))
        super().__init__('TrendAnalyzer', frequency,
                         (self.atr_signal_object, self.rsi_signal_object, self.adx_signal_object))
        self.config = dict(
            atr_mult=float(atr_mult), use_close=use_close, range_band=tuple(range_band),
            range_persistence=range_persistence, min_leg_atr=float(min_leg_atr),
            confirm_closes=confirm_closes, confirm_hysteresis_mult=float(confirm_hysteresis_mult),
            age_scale_bars=20, amp_scale_atr=2.0, adx_low=20.0, adx_high=55.0,
            score_w_pullback=0.55, score_w_earlyflag=0.15, score_w_leg_age=0.10,
            score_w_leg_amp=0.10, score_w_wick=0.05, score_w_div=0.02,
            score_w_adx_mod=0.4,
        )
        self.reset()

    @staticmethod
    def _validate_config(atr_mult, count, hysteresis, period, band, persistence, min_leg):
        for name, value in (('confirm_closes', count), ('atr_period', period),
                            ('range_persistence', persistence)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f'{name} must be an integer')
        if count < 1 or period < 2 or persistence < 0:
            raise ValueError('Require confirm_closes >= 1, atr_period >= 2, range_persistence >= 0')
        if not all(isfinite(v) for v in (atr_mult, hysteresis, min_leg)):
            raise ValueError('Thresholds must be finite')
        if atr_mult <= 0 or hysteresis < 1 or min_leg < 0:
            raise ValueError('Require atr_mult > 0, hysteresis >= 1, min_leg_atr >= 0')
        if len(band) != 2 or not 0 <= band[0] < band[1] <= 1:
            raise ValueError('range_band must satisfy 0 <= low < high <= 1')

    def reset(self):
        '''Clear detector history, events and outputs; retain market/configuration.'''
        self.value_by_dates.clear()
        self._bars = []
        self._candidate = None
        self._previous_candidate = None
        self._boot = {'H': None, 'L': None}
        self._boot_counts = {'H': 0, 'L': 0}
        self._swings = {'H': deque(maxlen=2), 'L': deque(maxlen=2)}
        self._last_pivot = None
        self.events = []
        self.pivots_ts = []
        self.pivot_types = []
        self.bootstrap = True
        self.state = dict(i=-1, direction='unknown', rev_close_count=0,
                          last_p_idx=-1, last_p_typ=None, macro_trend=None,
                          macro_anchor_price=None, macro_anchor_idx=-1,
                          macro_ext_price=None, prev_adx=None)
        self._reset_leg()

    def _reset_leg(self):
        self._path = 0.0
        self._path_close = None
        self._previous_slope = None
        self._short_direction = None
        self._microcycles = 0
        self._compression = deque(maxlen=5)
        self._range_count = 0

    @staticmethod
    def _finite(value, positive=False):
        if value is None or not isfinite(value) or (positive and value <= 0):
            return None
        return float(value)

    @staticmethod
    def _clip01(value):
        return max(0.0, min(1.0, value))

    def _time(self, extreme):
        return None if extreme is None else self._bars[extreme.index].timestamp

    def _last(self, typ):
        return self._swings[typ][-1] if self._swings[typ] else None

    def _price(self, typ):
        pivot = self._last(typ)
        return None if pivot is None else pivot.price

    def _threshold(self, bar):
        if bar.atr is None:
            return None
        return self.config['atr_mult'] * self.config['confirm_hysteresis_mult'] * bar.atr

    def _beyond_threshold(self, bar, candidate, typ):
        threshold = self._threshold(bar)
        if threshold is None:
            return False
        price = bar.c if self.config['use_close'] else (bar.l if typ == 'H' else bar.h)
        distance = candidate.price - price if typ == 'H' else price - candidate.price
        return distance >= threshold

    def _advance_candidate(self, candidate, count, index, typ):
        '''Strict improvements replace the extreme; only later bars may confirm it.'''
        bar = self._bars[index]
        price = bar.h if typ == 'H' else bar.l
        improved = candidate is None or (price > candidate.price if typ == 'H' else price < candidate.price)
        if improved:
            return _Extreme(index, price, bar.rsi), 0, True
        count = count + 1 if self._beyond_threshold(bar, candidate, typ) else 0
        return candidate, count, False

    def _track(self, index):
        if self.bootstrap:
            eligible = []
            for typ in ('H', 'L'):
                candidate, count, _ = self._advance_candidate(self._boot[typ], self._boot_counts[typ], index, typ)
                self._boot[typ], self._boot_counts[typ] = candidate, count
                if count >= self.config['confirm_closes']:
                    eligible.append((candidate, typ))
            # If both qualify, the older extreme wins; H is the deterministic tie-break.
            return min(eligible, key=lambda item: item[0].index) if eligible else None
        typ = 'H' if self.state['direction'] == 'up' else 'L'
        previous = self._candidate
        candidate, count, improved = self._advance_candidate(previous, self.state['rev_close_count'], index, typ)
        if improved:
            self._previous_candidate = previous
        self._candidate = candidate
        self.state['rev_close_count'] = count
        return (candidate, typ) if count >= self.config['confirm_closes'] else None

    @staticmethod
    def _retracement(anchor, extreme, price, trend):
        if anchor is None or extreme is None or trend not in ('up', 'down'):
            return None
        span = extreme - anchor if trend == 'up' else anchor - extreme
        if span <= 0:
            return None
        distance = extreme - price if trend == 'up' else price - extreme
        return max(0.0, distance / span)

    def _nearest_fib(self, retracement):
        if retracement is None:
            return None
        return min(self.FIB_LEVELS, key=lambda level: abs(level - self._clip01(retracement)))

    def _leg_geometry(self, index):
        if self._last_pivot is None or self._candidate is None:
            return None, None, None
        anchor, extreme = self._last_pivot.price, self._candidate.price
        sign = 1 if self.state['direction'] == 'up' else -1
        amplitude = max(0.0, sign * (extreme - anchor))
        age = index - self._last_pivot.index
        retracement = self._retracement(anchor, extreme, self._bars[index].c, self.state['direction'])
        return amplitude, age, retracement

    def _advance_leg(self, index):
        '''Update close-path and leg-local accumulators, also used for bridge replay.'''
        bar = self._bars[index]
        if self._path_close is not None:
            move = bar.c - self._path_close
            self._path += abs(move)
            direction = 1 if move > 0 else -1 if move < 0 else None
            if direction is not None:
                if self._short_direction is not None and direction != self._short_direction:
                    self._microcycles += 1
                self._short_direction = direction
        self._path_close = bar.c
        self._compression.append(None if bar.atr is None else float(bar.h - bar.l < 0.5 * bar.atr))
        amplitude, age, retracement = self._leg_geometry(index)
        slope = amplitude / age if amplitude is not None and age > 0 else None
        decay = self._previous_slope / slope if self._previous_slope is not None and slope and slope > 0 else None
        self._previous_slope = slope
        lo, hi = self.config['range_band']
        in_band = retracement is not None and lo <= retracement <= hi
        self._range_count = self._range_count + 1 if in_band else 0
        return decay

    def _macro_transition(self, bar):
        highs, lows = self._swings['H'], self._swings['L']
        if len(highs) < 2 or len(lows) < 2:
            return
        up = highs[-1].price > highs[-2].price and lows[-1].price > lows[-2].price
        down = highs[-1].price < highs[-2].price and lows[-1].price < lows[-2].price
        trend = 'up' if up else 'down' if down else 'range'
        span = abs(highs[-1].price - lows[-1].price)
        if bar.atr is not None and span < self.config['min_leg_atr'] * bar.atr:
            trend = 'range'
        old = self.state['macro_trend']
        self.state['macro_trend'] = trend
        if trend == 'range':
            self.state.update(macro_anchor_price=None, macro_anchor_idx=-1, macro_ext_price=None)
            return
        if old != trend:
            # The older same-side anchor precedes the HH/LL that establishes trend.
            anchor = lows[-2] if trend == 'up' else highs[-2]
            self.state.update(macro_anchor_price=anchor.price, macro_anchor_idx=anchor.index,
                              macro_ext_price=highs[-1].price if trend == 'up' else lows[-1].price)
        elif trend == 'up':
            self.state['macro_ext_price'] = max(self.state['macro_ext_price'], highs[-1].price)
        else:
            self.state['macro_ext_price'] = min(self.state['macro_ext_price'], lows[-1].price)

    def _confirm(self, candidate, typ, index):
        previous = self._last(typ)
        price = candidate.price
        flags = dict(hh=False, hl=False, lh=False, ll=False)
        if previous is not None:
            flags['hh' if typ == 'H' else 'hl'] = price > previous.price
            flags['lh' if typ == 'H' else 'll'] = price < previous.price
        event = dict(pivot_index=candidate.index, pivot_time=self._time(candidate),
                     pivot_type='high' if typ == 'H' else 'low', pivot_price=price,
                     confirmation_index=index, confirmation_time=self._bars[index].timestamp,
                     confirmation_delay=index - candidate.index)
        self.events.append(event)
        self.pivots_ts.append(event['pivot_time'])
        self.pivot_types.append(event['pivot_type'])
        self._swings[typ].append(candidate)
        self._last_pivot = candidate
        self.bootstrap = False
        self.state.update(last_p_idx=candidate.index, last_p_typ=typ,
                          direction='down' if typ == 'H' else 'up', rev_close_count=0)
        self._macro_transition(self._bars[index])
        self._rebuild_leg(candidate, index)
        return dict(pivot_high=typ == 'H', pivot_low=typ == 'L',
                    new_pivot_type='HIGH' if typ == 'H' else 'LOW',
                    new_pivot_time=event['pivot_time'], new_pivot_price=price,
                    confirmation_time=event['confirmation_time'],
                    confirmation_delay=event['confirmation_delay'], **flags)

    def _rebuild_leg(self, pivot, index):
        '''Initialize new state from observed t*..t_c; never rewrite feature rows.'''
        self._reset_leg()
        pivot_close = self._bars[pivot.index].c
        self._path = abs(pivot_close - pivot.price)
        self._path_close = pivot_close
        self._candidate = self._previous_candidate = None
        typ = 'H' if self.state['direction'] == 'up' else 'L'
        for j in range(pivot.index + 1, index + 1):
            previous = self._candidate
            candidate, count, improved = self._advance_candidate(previous, self.state['rev_close_count'], j, typ)
            if improved:
                self._previous_candidate = previous
            self._candidate = candidate
            self.state['rev_close_count'] = count
            self._advance_leg(j)

    def _divergence(self):
        previous, candidate = self._previous_candidate, self._candidate
        if previous is None or candidate is None or previous.rsi is None or candidate.rsi is None:
            return None
        if self.state['direction'] == 'up':
            return candidate.price > previous.price and candidate.rsi < previous.rsi
        return candidate.price < previous.price and candidate.rsi > previous.rsi

    def _adx_features(self, adx):
        previous = self.state['prev_adx']
        self.state['prev_adx'] = adx
        if adx is None:
            return dict(adx=None, adx_factor=None, adx_rising=None, adx_regime=None)
        lo, hi = self.config['adx_low'], self.config['adx_high']
        factor = self._clip01((adx - lo) / (hi - lo))
        regime = ('flat' if adx < lo else 'forming' if adx < lo + 0.47 * (hi - lo)
                  else 'trending' if adx < hi else 'overheated')
        return dict(adx=adx, adx_factor=factor,
                    adx_rising=None if previous is None else adx > previous, adx_regime=regime)

    def _score(self, features):
        if not features['features_ready']:
            return None, None
        terms = (
            ('pullback', features['pullback_ratio']),
            ('earlyflag', float(features['early_reversal'])),
            ('leg_age', features['leg_age_norm']), ('leg_amp', features['leg_amp_norm']),
            ('wick', features['wick_upper_norm'] if features['direction'] == 'up' else features['wick_lower_norm']),
            ('div', float(features['divergence_flag'] or False)),
        )
        weights = tuple(self.config[f'score_w_{name}'] for name, _ in terms)
        score = sum(weight * term for weight, (_, term) in zip(weights, terms)) / sum(weights)
        adjusted = score
        if features['adx_rising']:
            adjusted *= 0.8 + self.config['score_w_adx_mod'] * (1 - features['adx_factor'])
        return self._clip01(score), self._clip01(adjusted)

    def _features(self, index, adx, slope_decay):
        bar, state = self._bars[index], self.state
        direction = state['direction']
        candidate, pivot = self._candidate, self._last_pivot
        amplitude, age, retracement = self._leg_geometry(index)
        ready = pivot is not None and candidate is not None and bar.atr is not None
        span = max(bar.h - bar.l, 1e-12)
        exhaustion = amplitude / bar.atr if ready else None
        path_displacement = abs(bar.c - pivot.price) if pivot is not None else None
        inefficiency = None
        if path_displacement is not None:
            inefficiency = 0.0 if self._path == 0 else self._clip01(1 - path_displacement / self._path)
        compression = tuple(v for v in self._compression if v is not None)
        in_range = self._range_count >= self.config['range_persistence'] + 1
        macro_retracement = self._retracement(state['macro_anchor_price'], state['macro_ext_price'], bar.c, state['macro_trend'])
        pullback = None
        if ready:
            distance = candidate.price - bar.c if direction == 'up' else bar.c - candidate.price
            pullback = self._clip01(distance / self._threshold(bar))
        leg_high = (candidate.price if direction == 'up' else pivot.price) if ready else None
        leg_low = (pivot.price if direction == 'up' else candidate.price) if ready else None
        out = dict(
            i=index, timestamp=bar.timestamp, features_ready=ready, direction=direction,
            macro_trend=state['macro_trend'],
            structure_macro='range' if in_range else state['macro_trend'],
            structure_micro=None if pivot is None else 'range' if in_range else direction,
            last_pivot_idx=state['last_p_idx'], last_pivot_type=state['last_p_typ'],
            last_pivot_price=None if pivot is None else pivot.price, last_pivot_time=self._time(pivot),
            bars_since_pivot=age, local_high=self._price('H'), local_low=self._price('L'),
            last_H_time=self._time(self._last('H')), last_L_time=self._time(self._last('L')),
            leg_high=leg_high, leg_low=leg_low,
            potential_pivot=None if candidate is None else candidate.price,
            potential_pivot_time=self._time(candidate), atr=bar.atr,
            atr_pct=bar.atr / bar.c if bar.atr is not None and bar.c != 0 else None,
            distance_to_pivot_atr=(bar.c - pivot.price) / bar.atr if ready else None,
            pullback_ratio=pullback, early_reversal=state['rev_close_count'] > 0 if ready else None,
            reversal_count=state['rev_close_count'],
            leg_age_norm=self._clip01(age / self.config['age_scale_bars']) if age is not None else None,
            leg_amp_norm=self._clip01(exhaustion / self.config['amp_scale_atr']) if ready else None,
            wick_upper_norm=max(0.0, bar.h - max(bar.o, bar.c)) / span,
            wick_lower_norm=max(0.0, min(bar.o, bar.c) - bar.l) / span,
            volume_z=None, divergence_flag=self._divergence(),
            leg_exhaust_atr=exhaustion, slope_decay=slope_decay,
            exhaustion_score=exhaustion * (1 + slope_decay) if ready and slope_decay is not None else None,
            leg_compression=sum(compression) / len(compression) if compression else None,
            leg_inefficiency=inefficiency, microcycle_count=self._microcycles if pivot is not None else None,
            microcycle_density=self._microcycles / age if age else None,
            retracement=retracement, nearest_fib=self._nearest_fib(retracement),
            macro_retracement=macro_retracement, macro_nearest_fib=self._nearest_fib(macro_retracement),
            bars_since_macro_anchor=index - state['macro_anchor_idx'] if state['macro_anchor_idx'] >= 0 else None,
        )
        out.update(self._adx_features(adx))
        out['pivot_score'], out['pivot_score_adj'] = self._score(out)
        return out

    def update(self, o, h, l, c, v=None, atr=None, rsi14=None, adx=None, timestamp=None):
        '''Consume one bar once, returning features plus any confirmation event.'''
        if not all(isfinite(value) for value in (o, h, l, c)):
            raise ValueError('OHLC prices must be finite')
        if h < max(o, l, c) or l > min(o, h, c):
            raise ValueError('Inconsistent OHLC prices')
        index = len(self._bars)
        timestamp = index if timestamp is None else timestamp
        if pd.isna(timestamp):
            raise ValueError('Bar timestamp must not be missing')
        if self._bars and timestamp <= self._bars[-1].timestamp:
            raise ValueError('Bars must have unique, strictly increasing timestamps')
        bar = _Bar(float(o), float(h), float(l), float(c), self._finite(atr, positive=True),
                   self._finite(rsi14), timestamp)
        self._bars.append(bar)
        self.state['i'] = index
        pending = self._track(index)
        slope_decay = self._advance_leg(index) if self._last_pivot is not None else None
        out = self._features(index, self._finite(adx), slope_decay)
        event = dict(pivot_high=False, pivot_low=False, new_pivot_type='', new_pivot_time=None,
                     new_pivot_price=None, confirmation_time=None, confirmation_delay=None,
                     hh=False, hl=False, lh=False, ll=False)
        if pending is not None:
            event = self._confirm(*pending, index)
        out.update(event, current_local_high=self._price('H'), current_local_low=self._price('L'))
        self.value_by_dates[timestamp] = out
        return out

    def set_market_data(self, market_data, clear_data=True):
        if not clear_data and self.market_data is not None and self.market_data is not market_data:
            raise ValueError('Cannot carry pivot state to a different MarketData object')
        super().set_market_data(market_data, clear_data)
        if clear_data:
            self.reset()

    def _indicator_frame(self, market_data):
        '''Use full causal TA-Lib series, avoiding finite-window Wilder restarts.'''
        indicators = pd.DataFrame(index=market_data.index)
        for name, signal in (('atr', self.atr_signal_object), ('rsi14', self.rsi_signal_object),
                             ('adx', self.adx_signal_object)):
            values = signal.talib_func(*(market_data[key] for key in signal.input_keys),
                                       timeperiod=signal.entry_observation_window)
            indicators[name] = values.reindex(market_data.index)
            signal.value_by_dates = values.dropna().to_dict()
        return indicators

    def compute_with_vectorized_calculation(self):
        if self.market_data is None:
            raise ValueError('Set market data before computing the detector')
        frame = self.market_data.get_market_data(self.frequency)
        self.reset()
        if frame.empty:
            return self.value_by_dates
        indicators = self._indicator_frame(frame)
        for row, values in zip(frame.to_dict('records'), indicators.to_dict('records')):
            self.update(row[OPEN], row[HIGH], row[LOW], row[CLOSE], row[VOLUME],
                        timestamp=row[DATE], **values)
        return self.value_by_dates

    def compute_and_cache_full_period(self, fail_on_missing_dates=True):
        if self.market_data is None:
            raise ValueError('Set market data before computing the detector')
        if self.market_data.get_market_data(self.frequency).empty and fail_on_missing_dates:
            raise ValueError('No market bars available')
        self.compute_with_vectorized_calculation()

    def compute_and_cache_signal_value_on_date(self, on_date):
        # The batch path also handles prefixes shorter than indicator warmup.
        if on_date not in self.value_by_dates:
            self.compute_and_cache_full_period()

    def compute_signal_value_on_bar_append(self, bar_date, required_data):
        return self.update(*(required_data[key] for key in (OPEN, HIGH, LOW, CLOSE, VOLUME)),
                           atr=required_data[self.atr_signal_object],
                           rsi14=required_data[self.rsi_signal_object],
                           adx=required_data[self.adx_signal_object], timestamp=bar_date)

    def compute_and_cache_value_on_bar_append(self, bar_ts):
        if not is_aligned(bar_ts, self.frequency) or bar_ts in self.value_by_dates:
            return
        frame = self.market_data.get_market_data(self.frequency).loc[:bar_ts]
        if frame.empty or frame.index[-1] != bar_ts:
            raise ValueError('Appended bar is absent from market data')
        if len(frame) != len(self._bars) + 1 or (self._bars and frame.index[-2] != self._bars[-1].timestamp):
            raise ValueError('Initialize the detector, then append bars in order')
        values = self._indicator_frame(frame).iloc[-1]
        row = frame.iloc[-1]
        return self.update(*(row[key] for key in (OPEN, HIGH, LOW, CLOSE, VOLUME)),
                           atr=values['atr'], rsi14=values['rsi14'], adx=values['adx'], timestamp=bar_ts)

    def get_events_df(self):
        return pd.DataFrame(self.events, columns=(
            'pivot_index', 'pivot_time', 'pivot_type', 'pivot_price',
            'confirmation_index', 'confirmation_time', 'confirmation_delay'))

    def ordered_col_names(self):
        return self.training_fields + self.output_fields + self.label_col

    def get_result_df(self):
        '''Snapshot features plus conservative, nullable retrospective labels.

        A confirmation resolves the interval through t*. The later tail remains
        unresolved until a subsequent confirmation closes it. Availability times
        belong to label metadata, never to the model's feature matrix.
        '''
        df = pd.DataFrame.from_dict(self.value_by_dates, orient='index').rename(columns={'timestamp': 'bar_time'})
        df['is_pivot'] = pd.Series(pd.NA, index=df.index, dtype='Int64')
        df['pivot_type'] = pd.Series(pd.NA, index=df.index, dtype='string')
        df['label_resolved'] = False
        df['label_available_time'] = pd.Series(None, index=df.index, dtype=object)
        df['pivot_confirmation_time'] = pd.Series(None, index=df.index, dtype=object)
        start = 0
        for event in self.events:
            end = event['pivot_index'] + 1
            resolved = df.index[start:end]
            df.loc[resolved, 'is_pivot'] = 0
            df.loc[resolved, 'label_resolved'] = True
            df.loc[resolved, 'label_available_time'] = event['confirmation_time']
            ts = event['pivot_time']
            df.loc[ts, 'is_pivot'] = 1
            df.loc[ts, 'pivot_type'] = event['pivot_type']
            df.loc[ts, 'pivot_confirmation_time'] = event['confirmation_time']
            start = end
        preferred = tuple(key for key in self.ordered_col_names() if key in df)
        return df.loc[:, preferred + tuple(key for key in df if key not in preferred)]

    def get_research_dataset(self):
        '''Return (X, y, metadata); exclude warmup and unresolved rows explicitly.'''
        df = self.get_result_df()
        if 'features_ready' not in df:
            metadata = pd.DataFrame(columns=('i', 'bar_time', 'pivot_type', 'label_available_time',
                                             'pivot_confirmation_time'), index=df.index)
            return pd.DataFrame(columns=self.training_fields, index=df.index), df['is_pivot'], metadata
        rows = df.loc[df['features_ready'] & df['label_resolved']]
        X = rows.loc[:, self.training_fields].copy()
        for key in self.numerical_fields:
            X[key] = pd.to_numeric(X[key], errors='raise').astype(float)
        for key in self.categorical_fields:
            X[key] = X[key].astype('string')
        y = rows['is_pivot'].astype(int).copy()
        metadata = rows.loc[:, ('i', 'bar_time', 'pivot_type', 'label_available_time',
                                'pivot_confirmation_time')].copy()
        return X, y, metadata


pivot_detector_1h = ZigZagATR('1h', range_persistence=2, min_leg_atr=0.5)
