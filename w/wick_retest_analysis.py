import numpy as np
import pandas as pd


def _has_cols(df: pd.DataFrame, cols: tuple[str, ...]) -> bool:
    return all(c in df.columns for c in cols)


def _conditional_probability(condition: pd.Series, event: pd.Series):
    cond = condition.fillna(False).astype(bool)
    condition_count = int(cond.sum())

    if condition_count == 0:
        return np.nan, 0, 0  # prob, event_count, condition_count

    ev = event.fillna(False).astype(bool)
    event_count = int((cond & ev).sum())

    probability = event_count / condition_count

    return condition_count, event_count, probability


def add_derived_columns(df: pd.DataFrame, wick_cluster_threshold: float = 0.5) -> pd.DataFrame:
    '''
    Standardize some helper columns (retest flags, regimes, bins...)
    Returns a copy.
    '''
    df = df.copy()

    # --- output classification flags ---------------------------------------
    if 'output_classification' in df.columns:
        oc = df['output_classification']
        df['is_fast_full_retest'] = oc.eq('FAST_FULL_RETEST')
        df['is_late_full_retest'] = oc.eq('LATE_FULL_RETEST')
        df['is_full_retest'] = df['is_fast_full_retest'] | df['is_late_full_retest']
        df['is_partial_only'] = oc.eq('PARTIAL_RETEST')
        df['is_no_retest'] = oc.eq('NO_RETEST')
    else:
        df['is_fast_full_retest'] = False
        df['is_late_full_retest'] = False
        df['is_full_retest'] = False
        df['is_partial_only'] = False
        df['is_no_retest'] = False

    # --- dominant continuation direction (by MFE) --------------------------
    if _has_cols(df, ('mfe_wick_pct', 'mfe_opposite_pct')):
        def _dominant(row):
            mw = row['mfe_wick_pct']
            mo = row['mfe_opposite_pct']
            if pd.isna(mw) or pd.isna(mo): return 'NONE'
            if mw > mo and mw > 0: return 'WICK'
            if mo > mw and mo > 0: return 'OPPOSITE'
            return 'NONE'
        df['dominant_direction'] = df.apply(_dominant, axis=1)
    else:
        df['dominant_direction'] = 'NONE'

    # --- wick size bins ----------------------------------------------------
    if 'wick_ratio' in df.columns:
        try:
            df['wick_size_bin'] = pd.qcut(
                df['wick_ratio'],
                q=4,
                duplicates='drop'
            )
        except ValueError:
            df['wick_size_bin'] = np.nan

    # --- volume regimes ----------------------------------------------------
    if 'volume_ratio' in df.columns:
        vr = df['volume_ratio']
        df['volume_regime'] = pd.cut(
            vr,
            bins=(0.0, 0.7, 1.2, 2.0, np.inf),
            labels=('weak', 'normal', 'high', 'extreme'),
            include_lowest=True,
        )

    # --- ADX regimes -------------------------------------------------------
    if 'adx' in df.columns:
        adx = df['adx']
        df['adx_regime'] = pd.cut(
            adx,
            bins=(0.0, 20.0, 30.0, np.inf),
            labels=('range', 'soft_trend', 'strong_trend'),
            include_lowest=True,
        )

    # --- streak regimes ----------------------------------------------------
    if 'streak_score' in df.columns:
        ss = df['streak_score']
        df['streak_has_wick'] = ss > 0
        df['streak_cluster'] = ss >= wick_cluster_threshold

    # --- retest speed bins -------------------------------------------------
    if 't_full_retest' in df.columns:
        t = df['t_full_retest']
        df['retest_speed_bin'] = pd.cut(
            t,
            bins=(-np.inf, 1, 3, 10, np.inf),
            labels=('1_bar', '2_3_bars', '4_10_bars', '>10_bars'),
        )

    return df


def conditional_probability_by_group(
    df: pd.DataFrame,
    condition: pd.Series,
    event: pd.Series,
    group_cols: tuple[str, ...],
    *,
    condition_name: str = 'condition',
    event_name: str = 'event'
) -> pd.DataFrame:
    '''
    Compute P(event | condition) within groups, with meaningful naming.

    Output columns:
        group_cols...
        p_<event_name>_given_<condition_name>
        n_<condition_name>
        n_<condition_name>_and_<event_name>
    '''
    if not group_cols:
        p, n_both, n_cond = _conditional_probability(condition, event)
        return pd.DataFrame([{
            f'p_{event_name}_given_{condition_name}': p,
            f'n_{condition_name}': n_cond,
            f'n_{condition_name}_and_{event_name}': n_both
        }])
    

    tmp = df.copy()
    tmp['__cond__'] = condition.fillna(False).astype(bool)
    tmp['__event__'] = event.fillna(False).astype(bool)

    grouped = tmp[tmp['__cond__']].groupby(list(group_cols), dropna=False, observed=True)

    rows = []
    for key, g in grouped:
        key_tuple = key if isinstance(key, tuple) else (key,)

        cond_mask = g['__cond__']
        event_mask = g['__event__']

        p = float((cond_mask & event_mask).sum()) / float(cond_mask.sum()) if cond_mask.sum() else np.nan
        n_cond = int(cond_mask.sum())
        n_both = int((cond_mask & event_mask).sum())

        row = dict(zip(group_cols, key_tuple))

        # dynamically named output columns
        row[f'p_{event_name}_given_{condition_name}'] = p
        row[f'n_{condition_name}'] = n_cond
        row[f'n_{condition_name}_and_{event_name}'] = n_both

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def _feature_vs_outcome_quantiles(
    df: pd.DataFrame,
    feature_col: str,
    outcome_col: str,
    q: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> pd.DataFrame:
    '''
    Simple "feature vs outcome" table using quantile bins on a feature
    and mean of outcome in each bin.
    '''
    if not _has_cols(df, (feature_col, outcome_col)):
        return pd.DataFrame()

    sub = df[[feature_col, outcome_col]].dropna()
    if sub.empty:
        return pd.DataFrame()

    try:
        qs = sub[feature_col].quantile(q).values
        bins = np.unique(qs)
        if len(bins) <= 1:
            return pd.DataFrame()
        labels = tuple(f'Q{i}' for i in range(len(bins) - 1))
        sub['__bin__'] = pd.cut(sub[feature_col], bins=bins, labels=labels, include_lowest=True)
    except ValueError:
        return pd.DataFrame()

    out = (
        sub
        .groupby('__bin__', dropna=False, observed=True)
        .agg(
            n=(outcome_col, 'size'),
            **{
                f"{feature_col}_min": (feature_col, 'min'),
                f"{feature_col}_max": (feature_col, 'max'),
                f"{outcome_col}_mean": (outcome_col, 'mean'),
            }
        )
        .reset_index()
        .rename(columns={'__bin__': 'bin'})
    )
    return out


# ---------------------------------------------------------------------------
# 1. Base frequencies
# ---------------------------------------------------------------------------

def base_retest_table(df: pd.DataFrame) -> pd.DataFrame:
    condition = df['wick_category'].ne('NO_WICK')
    full_retest_event = df['is_full_retest'] == True
    partial_retest_event = df['is_partial_only'] == True
    no_retest_event = df['is_no_retest'] == True
    
    records = (('FULL_RETEST',) + _conditional_probability(condition, full_retest_event),
               ('PARTIAL_RETEST',) + _conditional_probability(condition, partial_retest_event),
               ('NO_RETEST',) + _conditional_probability(condition, no_retest_event))
    
    return pd.DataFrame(records, columns=['event', 'event_count', 'any wicks candle count', 'probability'])


def classification_retest_table(df: pd.DataFrame) -> pd.DataFrame:
    '''
    P(output_classification | input_classification)
    as a table with counts and frequencies.
    '''
    if not _has_cols(df, ('input_classification', 'output_classification')):
        return pd.DataFrame()

    grouped = (
        df
        .groupby(['input_classification', 'output_classification'], dropna=False, observed=True)
        .size()
        .reset_index(name='count')
    )
    total_by_input = (
        grouped
        .groupby('input_classification', observed=True)['count']
        .sum()
        .rename('total_input_occurrences')
        .reset_index()
    )
    out = grouped.merge(total_by_input, on='input_classification')
    out['frequency'] = out['count'] / out['total_input_occurrences']
    return out



# ---------------------------------------------------------------------------
# 2. Feature vs outcome: wick size / volume / ADX / streak
# ---------------------------------------------------------------------------

def wick_size_vs_retest(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Wick size → probability of FAST_FULL_RETEST and mean continuation.
    '''
    if not _has_cols(df, ('wick_ratio', 'output_classification')):
        return pd.DataFrame()

    df = df.copy()
    #df['is_fast_full_retest'] = df['output_classification'].eq('FAST_FULL_RETEST')
    return _feature_vs_outcome_quantiles(df, 'wick_ratio', 'is_full_retest') #'is_fast_full_retest'


def volume_vs_retest(df: pd.DataFrame) -> pd.DataFrame:
    if not _has_cols(df, ('volume_ratio', 'output_classification')):
        return pd.DataFrame()

    df = df.copy()
    df['is_fast_full_retest'] = df['output_classification'].eq('FAST_FULL_RETEST')
    return _feature_vs_outcome_quantiles(df, 'volume_ratio', 'is_fast_full_retest')


def adx_vs_direction_dominance(df: pd.DataFrame) -> pd.DataFrame:
    '''
    For each ADX regime, fraction of WICK vs OPPOSITE dominant moves.
    '''
    if not _has_cols(df, ('adx_regime', 'dominant_direction')):
        return pd.DataFrame()

    ct = (
        df
        .groupby(['adx_regime', 'dominant_direction'], dropna=False, observed=True)
        .size()
        .reset_index(name='count')
    )
    total = (
        ct
        .groupby('adx_regime', observed=True)['count']
        .sum()
        .rename('total')
        .reset_index()
    )
    out = ct.merge(total, on='adx_regime')
    out['frequency'] = out['count'] / out['total']
    return out


def streak_vs_retest(df: pd.DataFrame) -> pd.DataFrame:
    if not _has_cols(df, ('streak_has_wick', 'streak_cluster', 'output_classification')):
        return pd.DataFrame()

    ct = (
        df
        .groupby(['streak_has_wick', 'streak_cluster', 'output_classification'], dropna=False, observed=True)
        .size()
        .reset_index(name='count')
    )
    total = (
        ct
        .groupby(['streak_has_wick', 'streak_cluster'], observed=True)['count']
        .sum()
        .rename('total')
        .reset_index()
    )
    out = ct.merge(total, on=['streak_has_wick', 'streak_cluster'])
    out['frequency'] = out['count'] / out['total']
    return out

def feature_vs_no_retest(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    '''
    Quantile analysis: feature → P(no_retest)
    '''
    if not _has_cols(df, (feature, 'is_no_retest')):
        return pd.DataFrame()

    dfc = df.copy()
    dfc['no_retest'] = dfc['is_no_retest'].astype(int)

    return _feature_vs_outcome_quantiles(
        dfc,
        feature_col=feature,
        outcome_col='no_retest'
    )


# ---------------------------------------------------------------------------
# 3. Conditional probability scans (edges)
# ---------------------------------------------------------------------------

def prob_full_retest_by_wick_and_volume(df: pd.DataFrame) -> pd.DataFrame:
    '''
    P(full_retest | wick_category, volume_regime)
    '''
    needed = ('wick_category', 'volume_regime', 'is_full_retest')
    if not _has_cols(df, needed):
        return pd.DataFrame()

    cond = (df['wick_category'].ne('NO_WICK')) & (df['volume_regime'].notna())
    event = df['is_full_retest']
    return conditional_probability_by_group(df, cond, event, ('wick_category', 'volume_regime'), event_name='full_retest', condition_name='significant_wick')


def prob_breakout_wick_direction(df: pd.DataFrame, mfe_threshold: float = 0.002) -> pd.DataFrame:
    '''
    "Breakout in wick direction" = MFE_wick > mfe_threshold
    conditioned on significant wicks + full_retest.
    '''
    needed = ('wick_category', 'is_full_retest', 'mfe_wick_pct')
    if not _has_cols(df, needed):
        return pd.DataFrame()

    cond = (df['wick_category'].ne('NO_WICK')) & (df['is_full_retest'])
    event = df['mfe_wick_pct'] > mfe_threshold

    group_cols = ('wick_category',)
    if 'adx_regime' in df.columns:
        group_cols = ('wick_category', 'adx_regime')

    return conditional_probability_by_group(df, cond, event, group_cols, condition_name='full_restest', event_name='breakout_wick_dir')


def prob_reversal_opposite_direction(df: pd.DataFrame, mfe_threshold: float = 0.002) -> pd.DataFrame:
    '''
    "Reversal" = opposite MFE > wick MFE AND > threshold,
    conditioned on significant wicks + full_retest.
    '''
    needed = ('wick_category', 'is_full_retest', 'mfe_wick_pct', 'mfe_opposite_pct')
    if not _has_cols(df, needed):
        return pd.DataFrame()

    cond = (df['wick_category'].ne('NO_WICK')) & (df['is_full_retest'])
    event = (df['mfe_opposite_pct'] > df['mfe_wick_pct']) & (df['mfe_opposite_pct'] > mfe_threshold)

    group_cols = ('wick_category',)
    if 'adx_regime' in df.columns:
        group_cols = ('wick_category', 'adx_regime')

    return conditional_probability_by_group(df, cond, event, group_cols, event_name='reversal_opposite_direction', condition_name='full_retested_wick')


def prob_fast_full_retest(df: pd.DataFrame) -> pd.DataFrame:
    '''
    P(FAST_FULL_RETEST | wick_category, adx_regime, volume_regime)
    '''
    needed = ('wick_category', 'is_fast_full_retest')
    if not _has_cols(df, needed):
        return pd.DataFrame()

    cond = df['wick_category'].ne('NO_WICK')
    event = df['is_fast_full_retest']

    group_cols: tuple[str, ...] = ('wick_category',)
    if 'adx_regime' in df.columns and 'volume_regime' in df.columns:
        group_cols = ('wick_category', 'adx_regime', 'volume_regime')
    elif 'adx_regime' in df.columns:
        group_cols = ('wick_category', 'adx_regime')
    elif 'volume_regime' in df.columns:
        group_cols = ('wick_category', 'volume_regime')

    return conditional_probability_by_group(df, cond, event, group_cols, event_name='fast_full_retest', condition_name='significant_wick')


def prob_second_retest_after_fast_first(df: pd.DataFrame, with_adx: bool = False, with_volume: bool = False) -> pd.DataFrame:
    '''
    Computes:
        P(second_retest | fast_full_retest & significant wick)
    And also:
        mean time to second retest
        median time to second retest
    '''

    needed = ('wick_category', 'is_fast_full_retest', 'mfe_wick_pct', 't_mfe_wick')
    if not _has_cols(df, needed):
        return pd.DataFrame()

    cond = (df['wick_category'].ne('NO_WICK')) & (df['is_fast_full_retest'])
    event = df['mfe_wick_pct'] > 0.0

    # Only evaluate second retest timing on rows where event == True
    df_with_event = df[event]

    # Choose grouping structure
    group_cols = ['wick_category']
    if with_adx and 'adx_regime' in df.columns: group_cols.append('adx_regime')
    if with_volume and 'volume_regime' in df.columns: group_cols.append('volume_regime')

    # Probability table
    prob_df = conditional_probability_by_group(
        df,
        cond,
        event,
        group_cols,
        condition_name='fast_full_retest',
        event_name='second_retest'
    )

    # Mean/median time-to-second-retest
    timing = (
        df_with_event
        .groupby(group_cols, dropna=False, observed=True)['t_mfe_wick']
        .agg(mean_time='mean', median_time='median', count='size')
        .reset_index()
    )

    # Merge probability + timing
    return prob_df.merge(timing, on=list(group_cols), how='left')


def prob_no_retest(df: pd.DataFrame, group_cols: tuple[str,...]) -> pd.DataFrame:
    '''
    P(no_retest | significant_wick) stratified by category, adx, volume, streak
    '''
    needed = ('is_no_retest',) + group_cols
    if not _has_cols(df, needed): 
        return pd.DataFrame()

    cond = df['wick_category'].ne('NO_WICK')
    event = df['is_no_retest']

    # group_cols = []
    # if 'wick_category' in df.columns: group_cols.append('wick_category')
    # if 'adx_regime' in df.columns: group_cols.append('adx_regime')
    # if 'volume_regime' in df.columns: group_cols.append('volume_regime')
    # if 'streak_cluster' in df.columns: group_cols.append('streak_cluster')
    # if 'candle_type' in df.columns: group_cols.append('candle_type')

    return conditional_probability_by_group(
        df,
        cond,
        event,
        group_cols,
        condition_name='significant_wick',
        event_name='no_retest'
    )


# ---------------------------------------------------------------------------
# 4. EV tables: rough expected value of "play wick direction" etc.
# ---------------------------------------------------------------------------

def ev_tables(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Simple EV estimates of continuation in wick vs opposite direction.
    Grouped by wick_category + optional ADX/volume regimes.
    '''
    needed = ('wick_category', 'cont_return_wick_pct')
    if not _has_cols(df, needed):
        return pd.DataFrame()

    group_cols: list[str] = ['wick_category']
    if 'adx_regime' in df.columns:
        group_cols.append('adx_regime')
    if 'volume_regime' in df.columns:
        group_cols.append('volume_regime')

    agg_kwargs = dict(
    n=('cont_return_wick_pct', 'size'),
    ev_wick=('cont_return_wick_pct', 'mean'),
)

    if 'cont_return_opposite_pct' in df.columns:
        agg_kwargs['ev_opposite'] = ('cont_return_opposite_pct', 'mean')
    
    # --- MFE / MAE in wick direction ---
    if 'mfe_wick_pct' in df.columns:
        agg_kwargs['mfe_wick_mean'] = ('mfe_wick_pct', 'mean')
    if 'mae_wick_pct' in df.columns:
        agg_kwargs['mae_wick_mean'] = ('mae_wick_pct', 'mean')
    
    # --- MFE / MAE in opposite direction ---
    if 'mfe_opposite_pct' in df.columns:
        agg_kwargs['mfe_opp_mean'] = ('mfe_opposite_pct', 'mean')
    if 'mae_opposite_pct' in df.columns:
        agg_kwargs['mae_opp_mean'] = ('mae_opposite_pct', 'mean')

    out = (
        df
        .groupby(list(group_cols), dropna=False, observed=True)
        .agg(**agg_kwargs)
        .reset_index()
    )
    return out
    

def full_retest_timing_stats(df: pd.DataFrame):
    needed = ('is_full_retest', 't_full_retest')
    if not _has_cols(df, needed):
        return pd.DataFrame()
    dfe = df[df['is_full_retest']]
    if dfe.empty:
        return pd.DataFrame()
    
    return dfe['t_full_retest'].describe(percentiles=(0.1,0.25,0.5,0.75,0.9)).to_frame().T

# ---------------------------------------------------------------------------
# 5. Master driver
# ---------------------------------------------------------------------------

def run_all_wick_retest_analyses(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    '''
    High-level entry point.
    Returns dict[name -> DataFrame] with all the analyses.
    '''
    df = add_derived_columns(df)

    results: dict[str, pd.DataFrame] = {}

    # 1) Base frequency matrix
    results['base_retest'] = base_retest_table(df)
    results['classification_retest'] = classification_retest_table(df)
    
    results['full_retest_timing_stats'] = full_retest_timing_stats(df)

    # 2) Feature vs outcome
    results['wick_size_vs_retest'] = wick_size_vs_retest(df)
    results['volume_vs_retest'] = volume_vs_retest(df)
    results['adx_vs_direction_dominance'] = adx_vs_direction_dominance(df)
    results['streak_vs_retest'] = streak_vs_retest(df)

    # 3) Conditional probabilities
    results['prob_full_retest_by_wick_and_volume'] = prob_full_retest_by_wick_and_volume(df)
    #results['prob_breakout_wick_direction'] = prob_breakout_wick_direction(df)
    #results['prob_reversal_opposite_direction'] = prob_reversal_opposite_direction(df)
    #results['prob_fast_full_retest'] = prob_fast_full_retest(df)
    
    #One I specifically asked for
    results['prob_second_after_fast_first'] = prob_second_retest_after_fast_first(df)
    #No retest, check first features one by one, rajouter la trend et les retracements depuis high lows
    # if 'wick_category' in df.columns: group_cols.append('wick_category')
    # if 'adx_regime' in df.columns: group_cols.append('adx_regime')
    # if 'volume_regime' in df.columns: group_cols.append('volume_regime')
    # if 'streak_cluster' in df.columns: group_cols.append('streak_cluster')
    # if 'candle_type' in df.columns: group_cols.append('candle_type')
    results['no_retest_category'] = prob_no_retest(df, ('wick_category',)) #Change rien ?
    results['no_retest_colour'] = prob_no_retest(df, ('candle_type',))
    results['no_retest_cluster'] = prob_no_retest(df, ('streak_cluster',))
    #Less important ?
    results['no_retest_volume'] = prob_no_retest(df, ('volume_regime',))
    results['adx_regime'] = prob_no_retest(df, ('adx_regime',))
    
    
    results['no_retest_wick_ratio'] = feature_vs_no_retest(df, 'wick_ratio')
    results['no_retest_volume_ratio'] = feature_vs_no_retest(df, 'volume_ratio')
    results['no_retest_adx'] = feature_vs_no_retest(df, 'adx')

    # 4) EV tables
    results['ev_tables'] = ev_tables(df)

    return results
