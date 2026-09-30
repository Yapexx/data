from __future__ import annotations

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Canonical Event DataFrame contract
# -----------------------------------------------------------------------------
# Layer C consumes an EVENT DATAFRAME, not a raw signal dataframe.
#
# One row = one event.
#
# Required columns
# ----------------
# event_id
#     Unique identifier for the event row.
#
# event_time
#     Actual time the event becomes knowable and a decision can be made.
#     An outcome evaluator must ensure entry_time >= event_time.
#
# direction
#     Signed event direction.
#     Expected values are +1 for bullish / long events and -1 for bearish /
#     short events. Missing means no directional hypothesis; zero is invalid.
#
# Recommended standard columns
# ----------------------------
# event_bar_time
#     Exact market-data key identifying the candle on which detection occurs.
#     Produced by the scenario adapter; optional only for legacy event frames.
#     For open-stamped candles, this precedes the completed candle's event_time.
#
# event_type
#     Human-readable setup label such as 'wick', 'pivot_reversal',
#     'breakout_retest', 'scenario_match'.
#
# entry_price
#     Optional explicit price used as the event-study entry anchor.
#     The outcome evaluator must resolve price and actual entry time together
#     under an explicit entry rule. A market-bar key is not an execution time.
#
# reference_price
#     Optional structural / execution level linked to the event.
#     Used for retest, touch, overshoot and reference-aware path analysis.
#
# signal_value
#     Optional raw signal magnitude at event time.
#
# strength
#     Optional standardized event strength score.
#
# Optional metadata columns
# -------------------------
# Any additional feature snapshot is allowed, for example:
#     regime, atr, volume_ratio, zscore, macro_trend, setup_name, source_module
#
# Design rule
# -----------
# The event dataframe should only describe the event at trigger time.
# It should NOT contain future returns, MFE, MAE or lifecycle outputs.
# Those belong to the event outcome dataframe produced later by Layer C.
# -----------------------------------------------------------------------------

EVENT_ID = 'event_id'
EVENT_TIME = 'event_time'
EVENT_BAR_TIME = 'event_bar_time'
DIRECTION = 'direction'
EVENT_TYPE = 'event_type'
ENTRY_PRICE = 'entry_price'
REFERENCE_PRICE = 'reference_price'
SIGNAL_VALUE = 'signal_value'
STRENGTH = 'strength'
EVENT_BAR_INDEX = 'event_bar_index'
ENTRY_RULE = 'entry_rule'
CLUSTER_ID = 'cluster_id'
CLUSTER_SIZE = 'cluster_size'
EVENT_IN_CLUSTER_RANK = 'event_in_cluster_rank'

EVENT_DF_REQUIRED_COLUMNS = (EVENT_ID, EVENT_TIME, DIRECTION)
EVENT_DF_STANDARD_COLUMNS = EVENT_DF_REQUIRED_COLUMNS + (EVENT_BAR_TIME, EVENT_TYPE, ENTRY_PRICE, REFERENCE_PRICE, SIGNAL_VALUE, STRENGTH)


def empty_event_dataframe() -> pd.DataFrame:
    return pd.DataFrame(columns=EVENT_DF_STANDARD_COLUMNS)


def _ensure_datetime(series: pd.Series, column_name: str) -> pd.Series:
    out = pd.to_datetime(series, errors='coerce')
    if out.isna().any():
        raise ValueError(f'Column {column_name!r} contains non-parsable datetimes')
    return out


def _ensure_direction(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors='coerce')
    if (out.isna() & series.notna()).any():
        raise ValueError('Direction column contains non-numeric values')
    if series.map(lambda value: isinstance(value, (bool, np.bool_))).any():
        raise ValueError('Direction column must contain -1, 1 or missing values, not booleans')

    bad = out.notna() & ~out.isin((-1, 1))
    if bad.any():
        bad_values = tuple(sorted(set(out[bad].tolist())))
        raise ValueError(f'Direction column must only contain -1 or 1, got {bad_values}')
    return out.astype('Int64')


def add_event_id_column(event_df: pd.DataFrame, *, event_id_col: str = EVENT_ID, start: int = 0, inplace: bool = False) -> pd.DataFrame:
    df = event_df if inplace else event_df.copy()
    df[event_id_col] = np.arange(start, start + len(df), dtype=int)
    return df


def validate_event_dataframe(event_df: pd.DataFrame, allow_empty: bool = True) -> None:
    if not isinstance(event_df, pd.DataFrame):
        raise TypeError('event_df must be a pandas DataFrame')

    if not event_df.columns.is_unique:
        raise ValueError('event_df contains duplicated column names')

    missing = tuple(col for col in EVENT_DF_REQUIRED_COLUMNS if col not in event_df.columns)
    if missing:
        raise ValueError(f'event_df is missing required columns: {missing}')

    if event_df.empty:
        if allow_empty:
            return
        raise ValueError('event_df is empty')
    if event_df[EVENT_ID].isna().any():
        raise ValueError('event_id must not be missing')

    duplicated_ids = event_df[EVENT_ID].duplicated(keep=False)
    if duplicated_ids.any():
        dupes = tuple(event_df.loc[duplicated_ids, EVENT_ID].tolist())
        raise ValueError(f'event_df contains duplicated event_id values: {dupes}')

    event_times = _ensure_datetime(event_df[EVENT_TIME], EVENT_TIME)
    if EVENT_BAR_TIME in event_df.columns:
        bar_times = _ensure_datetime(event_df[EVENT_BAR_TIME], EVENT_BAR_TIME)
        try:
            too_early = event_times < bar_times
        except TypeError as error:
            raise ValueError('event_time and event_bar_time must have compatible timezones') from error
        if too_early.any():
            raise ValueError('event_time cannot precede event_bar_time')
    _ensure_direction(event_df[DIRECTION])


def standardize_event_dataframe(event_df: pd.DataFrame, *,
    sort_by_time: bool = True,
    reset_index: bool = True,
    keep_standard_columns_first: bool = True,
) -> pd.DataFrame:
    validate_event_dataframe(event_df)

    df = event_df.copy()
    df[EVENT_TIME] = _ensure_datetime(df[EVENT_TIME], EVENT_TIME)
    if EVENT_BAR_TIME in df.columns:
        df[EVENT_BAR_TIME] = _ensure_datetime(df[EVENT_BAR_TIME], EVENT_BAR_TIME)
    df[DIRECTION] = _ensure_direction(df[DIRECTION])

    if keep_standard_columns_first:
        standard_cols = tuple(col for col in EVENT_DF_STANDARD_COLUMNS if col in df.columns)
        extra_cols = tuple(col for col in df.columns if col not in standard_cols)
        df = df[list(standard_cols + extra_cols)]

    if sort_by_time:
        sort_cols = [EVENT_TIME]
        if EVENT_ID in df.columns:
            sort_cols.append(EVENT_ID)
        df = df.sort_values(sort_cols, kind='stable')

    if reset_index:
        df = df.reset_index(drop=True)

    return df


def events_from_trigger_dataframe(df: pd.DataFrame, *,
    time_col: str,
    trigger_col: str,
    direction_col: str | None = None,
    event_type: str = 'signal_event',
    reference_price_col: str | None = None,
    entry_price_col: str | None = None,
    signal_value_col: str | None = None,
    strength_col: str | None = None,
    feature_cols: tuple[str, ...] = (),
    event_id_start: int = 0,
) -> pd.DataFrame:
    
    if time_col not in df.columns:
        raise ValueError(f'Missing time column: {time_col!r}')
    if trigger_col not in df.columns:
        raise ValueError(f'Missing trigger column: {trigger_col!r}')
    if direction_col is not None and direction_col not in df.columns:
        raise ValueError(f'Missing direction column: {direction_col!r}')

    missing_features = tuple(col for col in feature_cols if col not in df.columns)
    if missing_features:
        raise ValueError(f'Missing feature columns: {missing_features}')

    trigger = df[trigger_col].fillna(False).astype(bool)
    if not trigger.any():
        return empty_event_dataframe()

    selected_cols = [time_col]
    optional_map = {
        ENTRY_PRICE: entry_price_col,
        REFERENCE_PRICE: reference_price_col,
        SIGNAL_VALUE: signal_value_col,
        STRENGTH: strength_col,
    }
    if direction_col is not None:
        selected_cols.append(direction_col)
    selected_cols.extend(col for col in optional_map.values() if col is not None)
    selected_cols.extend(col for col in feature_cols if col not in selected_cols)

    work = df.loc[trigger, selected_cols].copy()
    rename_map = {time_col: EVENT_TIME}
    rename_map.update({src: dst for dst, src in optional_map.items() if src is not None})
    if direction_col is not None:
        rename_map[direction_col] = DIRECTION
    work = work.rename(columns=rename_map)

    work[EVENT_TIME] = _ensure_datetime(work[EVENT_TIME], EVENT_TIME)
    work[DIRECTION] = pd.NA if direction_col is None else _ensure_direction(work[DIRECTION])
    work[EVENT_TYPE] = event_type
    work[EVENT_ID] = np.arange(event_id_start, event_id_start + len(work), dtype=int)

    return standardize_event_dataframe(work)


def events_from_scenario_inputs(
    inputs_df: pd.DataFrame,
    *,
    event_type: str,
    availability_delay: pd.Timedelta | str,
    direction_col: str | None = None,
    feature_cols: tuple[str, ...] = (),
    event_id_col: str | None = None,
    event_id_start: int = 0,
) -> pd.DataFrame:
    """Convert ScenarioAnalyzer.get_input_results() to one row per event.

    evaluation_bar_time becomes the market lookup key event_bar_time, while
    event_time is when the event becomes knowable. Supply the delay between
    these timestamps: e.g. '1h' for
    open-stamped hourly bars, or '0h' for close-stamped bars. Naive timestamps
    remain naive; their timezone must be defined by the caller's market dataset.

    Missing direction stays unknown. Only selected feature_cols and the input
    classification are copied as context; no entry price or outcome is inferred.
    Generate IDs once and reuse the event dataset across outcome configurations.
    event_id_col preserves existing IDs; generated IDs are local to this dataset.
    This adapter is for fixed-duration bars with a constant availability delay.
    The bar must be the detection/confirmation candle, not a historical extremum.
    Legacy outcome functions that look up market rows using event_time must be
    migrated before consuming this adapter's output.
    """
    if not isinstance(inputs_df, pd.DataFrame):
        raise TypeError('inputs_df must be a pandas DataFrame')
    if not inputs_df.columns.is_unique:
        raise ValueError('inputs_df contains duplicated column names')
    if not isinstance(event_type, str) or not event_type.strip():
        raise ValueError('event_type must be a non-empty string')
    if not isinstance(availability_delay, (str, pd.Timedelta)):
        raise TypeError('availability_delay must be a duration string or pd.Timedelta')
    delay = pd.Timedelta(availability_delay)
    if pd.isna(delay) or delay < pd.Timedelta(0):
        raise ValueError('availability_delay must be finite and non-negative')
    if event_id_col is None and (isinstance(event_id_start, bool) or
                                not isinstance(event_id_start, (int, np.integer)) or event_id_start < 0):
        raise ValueError('event_id_start must be a non-negative integer')

    reserved = {EVENT_ID, EVENT_TIME, EVENT_BAR_TIME, DIRECTION, EVENT_TYPE}
    collisions = reserved.intersection(feature_cols)
    if collisions:
        raise ValueError(f'feature_cols contains reserved columns: {sorted(collisions)}')
    required = ('evaluation_bar_time', 'input_classification')
    missing = [col for col in required if col not in inputs_df.columns]
    extras = list(feature_cols) + [col for col in (direction_col, event_id_col) if col is not None]
    if not inputs_df.empty:
        missing.extend(col for col in extras if col not in inputs_df.columns)
    if missing:
        raise ValueError(f'inputs_df is missing columns: {missing}')

    work = inputs_df.reset_index(drop=True).reindex(columns=list(dict.fromkeys(required + tuple(extras))))
    events = pd.DataFrame(index=work.index)
    events[EVENT_ID] = work[event_id_col] if event_id_col is not None else np.arange(
        event_id_start, event_id_start + len(work), dtype=int)
    events[EVENT_BAR_TIME] = _ensure_datetime(work['evaluation_bar_time'], 'evaluation_bar_time')
    events[EVENT_TIME] = events[EVENT_BAR_TIME] + delay
    events[DIRECTION] = (pd.Series(pd.NA, index=work.index, dtype='Int64')
                         if direction_col is None else _ensure_direction(work[direction_col]))
    events[EVENT_TYPE] = event_type
    events['input_classification'] = work['input_classification']
    for col in feature_cols:
        events[col] = work[col]
    return standardize_event_dataframe(events)


def tag_event_clusters(event_df: pd.DataFrame, *,
    max_gap: pd.Timedelta | str | int,
    time_col: str = EVENT_TIME,
    by_direction: bool = False,
) -> pd.DataFrame:
    df = standardize_event_dataframe(event_df)
    if df.empty:
        out = df.copy()
        out[CLUSTER_ID] = pd.Series(dtype='int64')
        out[CLUSTER_SIZE] = pd.Series(dtype='int64')
        out[EVENT_IN_CLUSTER_RANK] = pd.Series(dtype='int64')
        return out

    gap = pd.to_timedelta(max_gap) if not isinstance(max_gap, int) else None
    keys = [DIRECTION] if by_direction else []

    def _cluster_one(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values(time_col, kind='stable').copy()
        t = pd.to_datetime(group[time_col])
        if gap is None:
            step = t.diff().dt.total_seconds().fillna(np.inf)
            split = step.gt(float(max_gap))
        else:
            split = t.diff().gt(gap).fillna(True)
        group[CLUSTER_ID] = split.cumsum().astype(int) - 1
        group[EVENT_IN_CLUSTER_RANK] = group.groupby(CLUSTER_ID, observed=True).cumcount().astype(int)
        group[CLUSTER_SIZE] = group.groupby(CLUSTER_ID, observed=True)[EVENT_ID].transform('size').astype(int)
        return group

    out = df.groupby(keys, group_keys=False, observed=True).apply(_cluster_one) if keys else _cluster_one(df)
    return standardize_event_dataframe(out.reset_index(drop=True))


def drop_overlapping_events(
    event_df: pd.DataFrame,
    *,
    min_separation: int,
    time_col: str = EVENT_TIME,
    direction_col: str | None = None,
    keep: str = 'first',
) -> pd.DataFrame:
    if min_separation < 0:
        raise ValueError('min_separation must be >= 0')
    if keep not in ('first', 'last'):
        raise ValueError("keep must be either 'first' or 'last'")

    df = standardize_event_dataframe(event_df)
    if df.empty or min_separation == 0:
        return df

    work = df.copy()
    work = work.sort_values([time_col, EVENT_ID], kind='stable').reset_index(drop=True)
    work['__pos__'] = np.arange(len(work), dtype=int)
    work['__dir__'] = 0 if direction_col is None else work[direction_col].astype(int)

    out_parts = []
    for _, group in work.groupby('__dir__', sort=False, observed=True):
        pos = group['__pos__'].to_numpy()
        selected = np.zeros(len(group), dtype=bool)

        if keep == 'first':
            last_kept = -10**18
            for i in range(len(group)):
                if pos[i] - last_kept <= min_separation:
                    continue
                selected[i] = True
                last_kept = pos[i]
        else:
            last_kept = 10**18
            for i in range(len(group) - 1, -1, -1):
                if last_kept - pos[i] <= min_separation:
                    continue
                selected[i] = True
                last_kept = pos[i]

        out_parts.append(group.loc[selected].drop(columns=('__pos__', '__dir__')))

    out = pd.concat(out_parts, axis=0) if out_parts else empty_event_dataframe()
    return standardize_event_dataframe(out)


__all__ = (
    'EVENT_BAR_TIME',
    'events_from_scenario_inputs',
    'CLUSTER_ID',
    'CLUSTER_SIZE',
    'DIRECTION',
    'ENTRY_PRICE',
    'ENTRY_RULE',
    'EVENT_BAR_INDEX',
    'EVENT_DF_REQUIRED_COLUMNS',
    'EVENT_DF_STANDARD_COLUMNS',
    'EVENT_ID',
    'EVENT_IN_CLUSTER_RANK',
    'EVENT_TIME',
    'EVENT_TYPE',
    'REFERENCE_PRICE',
    'SIGNAL_VALUE',
    'STRENGTH',
    'add_event_id_column',
    'drop_overlapping_events', #Drop overlapping event
    'empty_event_dataframe',
    'events_from_trigger_dataframe', #To create event dataframe
    'standardize_event_dataframe',
    'tag_event_clusters', #To spot events in same timeframe
    'validate_event_dataframe',
)