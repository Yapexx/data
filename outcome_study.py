"""Generic TP/SL/horizon studies of an existing event dataset.

No detection or scenario imports. Reuses outcome_eval and event_outcome_features.
The fixed-bin/time-slice approach follows panel_study, without its panel schema.
Naive dates denote UTC; calendar boundaries and reports always use UTC.
"""
from dataclasses import replace
from itertools import product
import json

import numpy as np
import pandas as pd

from crypto_quant_engine.event_study.event_schema import standardize_event_dataframe
from crypto_quant_engine.event_study.outcome_eval import OutcomeConfig, label_event_outcomes
from crypto_quant_engine.analysis.event_outcome_features import (
    GROUPS, KEYS, METRICS, assign_research_populations, build_event_feature_reports,
    resolve_feature_bins,
)

STUDY_COLUMNS = ['common_complete', 'study_era', 'study_period']
COUNT_COLUMNS = ['n_events', 'n_common', 'n_excluded', 'n_ambiguous', 'n_probability',
                 'n_target', 'n_stop', 'n_timeout', 'p_target', 'p_target_lower', 'p_target_upper']
SUMMARY_COLUMNS = GROUPS + COUNT_COLUMNS + [
    f'{metric}_{stat}' for metric in METRICS for stat in ('n', 'mean', 'median')
]
SHARE_COLUMNS = ['common_event_share', 'eligible_event_share', 'target_share', 'target_concentration']


# 1. Configuration loop extracted from wick_outcome_research, without changing it.
def check_configs(configs):
    configs = tuple(configs)
    if not configs or not all(isinstance(c, OutcomeConfig) for c in configs):
        raise ValueError('Supply at least one OutcomeConfig')
    if len({c.config_id for c in configs}) != len(configs):
        raise ValueError('Duplicate outcome configurations would duplicate result rows')
    return configs


def evaluate_outcome_configs(market_df, events_df, configs):
    configs = check_configs(configs)
    labelled = pd.concat([label_event_outcomes(market_df, events_df, c) for c in configs], ignore_index=True)
    definitions = pd.DataFrame([
        dict(outcome_config_id=c.config_id, **json.loads(c.definition())) for c in configs
    ])
    return labelled, definitions


def build_outcome_grid(base_config, *, tp, sl, horizons):
    """Cartesian product of threshold values/horizons in the base config's units."""
    if not isinstance(base_config, OutcomeConfig):
        raise TypeError('base_config must be an OutcomeConfig')
    tp, sl, horizons = tuple(tp), tuple(sl), tuple(horizons)
    if not tp or not sl or not horizons:
        raise ValueError('Every grid axis must be non-empty')
    return check_configs(replace(base_config, tp=t, sl=s, horizon=h)
                         for t, s, h in product(tp, sl, horizons))


def _comparison_configs(configs):
    configs = check_configs(configs)
    def fixed(config):
        return {k: v for k, v in json.loads(config.definition()).items() if k not in ('tp', 'sl', 'horizon')}
    if any(fixed(c) != fixed(configs[0]) for c in configs[1:]):
        raise ValueError('A common-cohort study varies only TP, SL and horizon; other settings must agree')
    return configs


# 2. Select events by availability. Never truncate the OHLC supplied for labelling.
def _utc(value):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError('Period boundaries cannot be missing')
    return stamp.tz_localize('UTC') if stamp.tzinfo is None else stamp.tz_convert('UTC')


def _bounds(period, *, allow_open=False):
    if len(period) != 2:
        raise ValueError('A period requires (start, end)')
    start, end = period
    if not allow_open and (start is None or end is None):
        raise ValueError('Era boundaries must be explicit')
    start = _utc(start) if start is not None else None
    end = _utc(end) if end is not None else None
    if start is not None and end is not None and start >= end:
        raise ValueError('Period start must precede end')
    return start, end


def select_event_period(events_df, event_period=None):
    """Availability-time filter [start, end). Does not recompute event snapshots."""
    events = standardize_event_dataframe(events_df, sort_by_time=False)
    if event_period is None:
        return events.reset_index(drop=True)
    start, end = _bounds(event_period, allow_open=True)
    times = pd.to_datetime(events.event_time, utc=True)
    mask = pd.Series(True, index=events.index)
    if start is not None:
        mask &= times >= start
    if end is not None:
        mask &= times < end
    return events.loc[mask].reset_index(drop=True)


def _calendar(events, event_period, eras, calendar):
    if calendar not in ('M', 'Q', None):
        raise ValueError("calendar must be 'M', 'Q', or None")
    periods = []
    for name, boundaries in (eras or {}).items():
        if not isinstance(name, str) or not name or name in ('unassigned', 'full'):
            raise ValueError('Era names must be non-empty strings other than full/unassigned')
        start, end = _bounds(boundaries)
        periods.append((name, start, end))
    periods.sort(key=lambda item: item[1])
    if any(a[2] > b[1] for a, b in zip(periods, periods[1:])):
        raise ValueError('Eras must not overlap')
    times = pd.to_datetime(events.event_time, utc=True)
    frame = events.copy()
    frame['study_era'] = 'unassigned' if periods else 'full'
    for name, start, end in periods:
        frame.loc[(times >= start) & (times < end), 'study_era'] = name
    frame['study_period'] = times.dt.tz_localize(None).dt.to_period(calendar).astype(str) if calendar else 'full'
    era_definitions = pd.DataFrame(periods, columns=['study_era', 'start', 'end'])
    # Explicit event-selection bounds keep empty calendar buckets visible.
    start, end = _bounds(event_period, allow_open=True) if event_period is not None else (None, None)
    start = start if start is not None else times.min()
    end = end if end is not None else (times.max() + pd.Timedelta('1ns'))
    calendar_rows = []
    if calendar and pd.notna(start) and pd.notna(end):
        for period in pd.period_range(start.tz_localize(None), (end-pd.Timedelta('1ns')).tz_localize(None), freq=calendar):
            calendar_rows.append(dict(study_period=str(period), start=period.start_time.tz_localize('UTC'),
                                      end=(period+1).start_time.tz_localize('UTC')))
    return frame, era_definitions, pd.DataFrame(calendar_rows, columns=['study_period', 'start', 'end'])


# 3. Establish the common set using coverage, never label success or ambiguity.
def common_complete_membership(labelled, events, configs):
    configs = _comparison_configs(configs)
    if events.event_id.isna().any() or not events.event_id.is_unique:
        raise ValueError('Events require unique, non-missing IDs')
    if not labelled.columns.is_unique or not set(KEYS + ['horizon_complete']) <= set(labelled):
        raise ValueError('Missing/duplicated outcome key or coverage columns')
    if labelled[KEYS].isna().any().any() or labelled.duplicated(KEYS).any():
        raise ValueError('Duplicated or missing event/config/direction keys')
    expected = set()
    for event in events.itertuples(index=False):
        directions = (-1, 1) if configs[0].directions == 'both' or pd.isna(event.direction) else (int(event.direction),)
        expected.update((event.event_id, c.config_id, direction) for c in configs for direction in directions)
    actual = set(labelled[KEYS].itertuples(index=False, name=None))
    if actual != expected:
        raise ValueError('Outcomes do not contain exactly the expected event/config/direction rows')
    if labelled.horizon_complete.isna().any() or not pd.api.types.is_bool_dtype(labelled.horizon_complete):
        raise ValueError('horizon_complete must be non-missing boolean')
    complete = labelled.groupby('event_id').horizon_complete.all()
    counts = labelled.groupby('event_id').size()
    membership = events[['event_id', 'event_type', 'event_time']].copy()
    membership['common_complete'] = membership.event_id.map(complete).astype(bool)
    membership['n_outcome_rows'] = membership.event_id.map(counts).astype('int64')
    membership['cohort_exclusion'] = pd.Series(pd.NA, index=membership.index, dtype='string')
    membership.loc[~membership.common_complete, 'cohort_exclusion'] = 'incomplete_under_grid'
    return membership


# 4. Common-denominator summaries and calendar concentration.
def _ratio(numerator, denominator):
    return numerator / denominator if denominator else np.nan


def _summary(group):
    common = group.loc[group.path_eligible]
    binary = group.loc[group.probability_eligible]
    counts = common.outcome_label.value_counts()
    n, target, ambiguous = len(common), int(counts.get('target', 0)), int(counts.get('ambiguous', 0))
    row = dict(n_events=len(group), n_common=n, n_excluded=len(group)-n, n_ambiguous=ambiguous,
               n_probability=len(binary), n_target=target, n_stop=int(counts.get('stop', 0)),
               n_timeout=int(counts.get('timeout', 0)), p_target=_ratio(target, len(binary)),
               p_target_lower=_ratio(target, n), p_target_upper=_ratio(target+ambiguous, n))
    for metric in METRICS:
        values = common[metric].dropna()
        row.update({f'{metric}_n': len(values), f'{metric}_mean': values.mean(), f'{metric}_median': values.median()})
    return row


def summarize_study(research, *, slice_column=None, slice_labels=()):
    """Keep empty periods. Target concentration compares successes with exposure.

    Shares use the same full selected study as reference, per config/direction.
    Bounds cover unresolved OHLC ordering; they are not confidence intervals.
    """
    rows = []
    for key, group in research.groupby(GROUPS, observed=True, sort=True):
        identity = dict(zip(GROUPS, key))
        total = _summary(group)
        if slice_column is None:
            rows.append(dict(identity, **total))
            continue
        for label in slice_labels:
            row = dict(identity, **_summary(group.loc[group[slice_column].eq(label)]))
            row[slice_column] = label
            row['common_event_share'] = _ratio(row['n_common'], total['n_common'])
            row['eligible_event_share'] = _ratio(row['n_probability'], total['n_probability'])
            row['target_share'] = _ratio(row['n_target'], total['n_target'])
            row['target_concentration'] = _ratio(row['target_share'], row['eligible_event_share'])
            rows.append(row)
    columns = SUMMARY_COLUMNS if slice_column is None else [slice_column] + SUMMARY_COLUMNS + SHARE_COLUMNS
    return pd.DataFrame(rows, columns=columns)


# 5. Compose generic stages. The caller detects events once, outside this function.
def run_outcome_study(market_df, events_df, configs, *, features_df=None, feature_columns=(),
                      event_period=None, eras=None, calendar='M', feature_bins=None):
    """Evaluate one instrument/timeframe with a common complete event population.

    Only event selection is date-filtered. Full market history/future remains
    available; event paths may cross era/selection boundaries. This is descriptive
    research, not a purged train/test split. All selected labelled rows are retained.
    """
    configs = _comparison_configs(configs)
    feature_columns = tuple(feature_columns)
    if features_df is None and (feature_columns or feature_bins is not None):
        raise ValueError('Feature columns/bins require features_df')
    events = select_event_period(events_df, event_period)
    for column in ('trading_pair', 'exchange', 'market_type', 'frequency'):
        if column in events and events[column].nunique(dropna=True) > 1:
            raise ValueError(f'One market/timeframe per study; multiple {column} values supplied')
    if set(STUDY_COLUMNS).intersection(events.columns):
        raise ValueError('Event context collides with study metadata')
    sliced_events, era_definitions, calendar_definitions = _calendar(events, event_period, eras, calendar)
    labelled, definitions = evaluate_outcome_configs(market_df, events, configs)
    membership = common_complete_membership(labelled, events, configs)
    working = labelled.merge(membership[['event_id', 'common_complete']], on='event_id', validate='many_to_one')
    working = working.merge(sliced_events[['event_id', 'study_era', 'study_period']], on='event_id', validate='many_to_one')
    research = assign_research_populations(working, cohort_column='common_complete')
    result = dict(events_df=events, labelled_events_df=labelled, configurations_df=definitions,
                  cohort_membership_df=membership, era_definitions_df=era_definitions,
                  calendar_definitions_df=calendar_definitions)
    if features_df is not None:
        # One reference population for the whole selected study, including tails.
        # No era or outcome-specific refits. Caller can supply externally fitted edges.
        if not features_df.columns.is_unique or 'event_id' not in features_df:
            raise ValueError('features_df requires unique columns and event_id')
        if features_df.event_id.isna().any() or not features_df.event_id.is_unique:
            raise ValueError('Feature snapshots require unique, non-missing event IDs')
        features = features_df.loc[features_df.event_id.isin(events.event_id)].copy()
        bins = resolve_feature_bins(features, feature_columns, feature_bins)
        feature_reports = build_event_feature_reports(working, features, feature_columns,
                                                      feature_bins=bins, cohort_column='common_complete')
        result.update(feature_reports)
        result['features_df'] = features
        research = feature_reports['research_df']
        if eras:
            reports_by_era = []
            for era in list(era_definitions.study_era) + ['unassigned']:
                tables = build_event_feature_reports(working.loc[working.study_era.eq(era)], features,
                    feature_columns, feature_bins=bins, cohort_column='common_complete')
                reports_by_era.append({name: tables[name].assign(study_era=era) for name in
                                      ('feature_probability_df', 'feature_path_df', 'feature_coverage_df')})
            for name in ('feature_probability_df', 'feature_path_df', 'feature_coverage_df'):
                populated = [r[name] for r in reports_by_era if not r[name].empty]
                result[f'era_{name}'] = (pd.concat(populated, ignore_index=True) if populated
                                         else reports_by_era[0][name].copy())
    result['research_df'] = research
    result['cohort_summary_df'] = summarize_study(research)
    result['era_summary_df'] = summarize_study(research, slice_column='study_era',
        slice_labels=list(era_definitions.study_era) + ['unassigned'] if eras else [])
    result['calendar_summary_df'] = summarize_study(research, slice_column='study_period',
                                                   slice_labels=calendar_definitions.study_period.tolist())
    return result
