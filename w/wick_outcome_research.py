"""Wick detection -> generic outcomes -> small research summaries.

Proposed location: scripts/wicks_analysis/wick_outcome_research.py
Run from the repository root: python -m scripts.wicks_analysis.wick_outcome_research

Each stage is a separate function. Importing this file runs no study or export.
Detection runs once; all outcome configurations reuse the same event dataframe.
The summaries describe independent events, not a portfolio backtest or net P&L.
"""

from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from crypto_quant_engine.event_study.outcome_eval import OutcomeConfig, label_event_outcomes
from crypto_quant_engine.market_data import MarketData, PROJECT_ROOT
from scripts.export_scenario_events import build_enhanced_wick_events


# 1. Settings. Start with this small local sample, then change the dates/market.
MARKET = dict(
    trading_pair='BTC_USDT', exchange='binance', market_type='s',
    start_time=datetime(2024, 1, 1), end_time=datetime(2024, 1, 15),
)
FREQUENCY = '1h'
BAR_TIMESTAMP = 'open'  # Local cache candle timestamps denote candle opening.
IS_UPPER_WICK = False  # False = lower wicks; both trading directions are tested.

# Edit these definitions; uncomment/add rows to compare configurations.
# Thresholds are ATR multiples here. For 'return', 0.01 means 1%.
OUTCOME_SETTINGS = (
    dict(entry='next_open', threshold_unit='atr', tp=1.0, sl=1.0, horizon=12),
    # dict(entry='next_open', threshold_unit='atr', tp=1.5, sl=1.0, horizon=24),
)
RUN_SUMMARIES = True
PRINT_SUMMARIES = True
SAVE_PICKLE = False
SAVE_CSV = False
OUTPUT_DIRECTORY = PROJECT_ROOT / 'data' / 'wick_outcomes'


# 2. Detection. Reuse the approved exporter and its causal event context.
def detect_wick_events(market_data, frequency=FREQUENCY, *, is_upper=IS_UPPER_WICK,
                       bar_timestamp=BAR_TIMESTAMP):
    if bar_timestamp not in ('open', 'close'):
        raise ValueError("bar_timestamp must be 'open' or 'close'")
    delay = pd.Timedelta(frequency) if bar_timestamp == 'open' else pd.Timedelta(0)
    return build_enhanced_wick_events(
        market_data, frequency, is_upper=is_upper, availability_delay=delay,
    )


# 3. Economic labelling. This function can also evaluate an already-saved events_df.
def build_outcome_configs(frequency=FREQUENCY, bar_timestamp=BAR_TIMESTAMP):
    return tuple(OutcomeConfig(
        **settings, bar_duration=frequency, bar_timestamp=bar_timestamp,
        directions='both', ambiguity='mark', atr_source='event',
    ) for settings in OUTCOME_SETTINGS)


def _check_configs(configs):
    configs = tuple(configs)
    if not configs or not all(isinstance(c, OutcomeConfig) for c in configs):
        raise ValueError('Supply at least one OutcomeConfig')
    if len({c.config_id for c in configs}) != len(configs):
        raise ValueError('Duplicate outcome configurations would duplicate result rows')
    return configs


def evaluate_outcome_configs(market_df, events_df, configs):
    configs = _check_configs(configs)
    labelled = pd.concat(
        [label_event_outcomes(market_df, events_df, c) for c in configs],
        ignore_index=True,
    )
    definitions = pd.DataFrame([
        dict(outcome_config_id=c.config_id, **json.loads(c.definition())) for c in configs
    ])
    return labelled, definitions


# 4. Summaries. Keep complete/partial coverage separate and ambiguity visible.
GROUP_COLUMNS = ['event_type', 'outcome_config_id', 'evaluated_direction']
LABELS = ('target', 'stop', 'ambiguous', 'timeout', 'insufficient_data', 'no_entry')
PATH_METRICS = ('horizon_return', 'mfe_return', 'mae_return', 'mfe_atr', 'mae_atr')


def summarize_outcomes(labelled):
    """Counts and ambiguity fraction within each direction/config/coverage group.

    Partial rows can already have a known target/stop/ambiguous label. They stay
    in the partial group. n_events counts each event once within each group.
    No ambiguous or insufficient-data row is silently counted as a loss.
    """
    columns = GROUP_COLUMNS + ['coverage', 'n_events'] + list(LABELS) + ['ambiguity_rate']
    rows = []
    for key, group in labelled.groupby(GROUP_COLUMNS + ['horizon_complete'], sort=True, observed=True):
        counts = group.outcome_label.value_counts()
        row = dict(zip(GROUP_COLUMNS, key[:-1]))
        row.update(coverage='complete' if key[-1] else 'partial', n_events=len(group))
        row.update({name: int(counts.get(name, 0)) for name in LABELS})
        row['ambiguity_rate'] = row['ambiguous'] / len(group)
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def summarize_complete_paths(labelled):
    """Full-horizon means/medians, including all complete outcome labels.

    Excludes every partial row, even when its first-hit label is already known.
    Ambiguous complete paths still have valid continuous metrics and remain.
    Returns are fractions, MAE is non-positive, and ATR metrics use ATR units.
    Comparing different horizons may change the complete-event cohort.
    """
    columns = GROUP_COLUMNS + ['n_events'] + [
        f'{metric}_{stat}' for metric in PATH_METRICS for stat in ('mean', 'median')
    ]
    rows = []
    complete = labelled.loc[labelled.horizon_complete]
    for key, group in complete.groupby(GROUP_COLUMNS, sort=True, observed=True):
        row = dict(zip(GROUP_COLUMNS, key), n_events=len(group))
        for metric in PATH_METRICS:
            row[f'{metric}_mean'] = group[metric].mean()
            row[f'{metric}_median'] = group[metric].median()
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


# 5. Compose stages. Returned frames are available independently in an IDE.
def run_wick_study(market_data, frequency=FREQUENCY, *, is_upper=IS_UPPER_WICK,
                   bar_timestamp=BAR_TIMESTAMP, configs=None, include_summaries=True):
    configs = build_outcome_configs(frequency, bar_timestamp) if configs is None else tuple(configs)
    configs = _check_configs(configs)
    # Prevent inconsistent settings between the detection and outcome stages.
    for config in configs:
        if (pd.Timedelta(config.bar_duration) != pd.Timedelta(frequency)
                or config.bar_timestamp != bar_timestamp):
            raise ValueError('Outcome timing must match the detection timeframe/timestamp convention')
    events = detect_wick_events(market_data, frequency, is_upper=is_upper, bar_timestamp=bar_timestamp)
    labelled, definitions = evaluate_outcome_configs(market_data.get_market_data(frequency), events, configs)
    results = dict(events_df=events, labelled_events_df=labelled, configurations_df=definitions)
    if include_summaries:
        results['outcome_summary_df'] = summarize_outcomes(labelled)
        results['path_summary_df'] = summarize_complete_paths(labelled)
    return results


# 6. Optional saving and console display. Legacy research files are not imported.
def save_results(results, output_directory, *, save_pickle=False, save_csv=False):
    if not (save_pickle or save_csv):
        return []
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, frame in results.items():
        if save_pickle:
            path = directory / f'{name}.pickle'
            frame.to_pickle(path)
            paths.append(path)
        if save_csv:
            path = directory / f'{name}.csv'
            frame.to_csv(path, index=False)
            paths.append(path)
    return paths


def main():
    market_data = MarketData(**MARKET)
    results = run_wick_study(market_data, FREQUENCY, is_upper=IS_UPPER_WICK,
                             bar_timestamp=BAR_TIMESTAMP, include_summaries=RUN_SUMMARIES)
    print(f"Detected {len(results['events_df']):,} events; produced {len(results['labelled_events_df']):,} outcome rows.")
    if PRINT_SUMMARIES:
        for name in ('configurations_df', 'outcome_summary_df', 'path_summary_df'):
            if name in results:
                print(f'\n{name}:')
                print(results[name].to_string(index=False))
        print('\nAmbiguity rates are within each coverage group. Path summaries use complete horizons only.')
        print('Returns are fractions (0.01 = 1%); horizon returns are not TP/SL trade P&L.')
    # A fresh run directory keeps optional exports from overwriting earlier runs.
    run_name = '_'.join((market_data.exchange, market_data.trading_pair, market_data.market_type,
                         FREQUENCY, 'upper' if IS_UPPER_WICK else 'lower',
                         datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')))
    paths = save_results(results, OUTPUT_DIRECTORY / run_name, save_pickle=SAVE_PICKLE, save_csv=SAVE_CSV)
    if paths:
        print(f'Saved {len(paths)} files to {paths[0].parent}')
    return results


if __name__ == '__main__':
    research = main()
    # Examples: research['events_df'], research['labelled_events_df'],
    # research['outcome_summary_df'], research['path_summary_df'].
