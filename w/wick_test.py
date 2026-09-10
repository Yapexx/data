import pandas as pd
import numpy as np
import os
from datetime import datetime

from crypto_quant_engine.market_data import MarketData
from crypto_quant_engine.core_engine import ScenarioAnalyzer, EveryBarScenarioAnalyzer, Backtester, PrintNotifier
from crypto_quant_engine.portfolio import PortfolioManager, Portfolio
from crypto_quant_engine.signals.pivot_detector import ZigZagATR, pivot_detector_1h
from crypto_quant_engine.tools import HIGH, LOW, VOLUME, CLOSE, display_df

from crypto_quant_engine.signals.talib_volatility import ATRSignal, EMAVolumeSignal, VWAPSignal
from crypto_quant_engine.signals.talib_momentum import RSISignal
from crypto_quant_engine.signals.talib_trend import ADXSignal
from crypto_quant_engine.signals.talib_exhaustion import ATRCompressionSignal

from crypto_quant_engine.scenarios.wick_retest import BasicWickRetestScenario, EnhancedWickRetestScenario

from crypto_quant_engine.analysis.dataset_builder import add_helper_columns
from crypto_quant_engine.analysis.feature_explorer import FeatureEvaluator, add_quantile_column
from crypto_quant_engine.analysis.confluence_table import ConfluenceTable


from scripts.wicks_analysis.wick_retest_analysis import run_all_wick_retest_analyses, add_derived_columns
from crypto_quant_engine.analysis.dataset_builder import wick_enricher, wick_training_dataset

from crypto_quant_engine.analysis.lifecycle.lifecycle_config import default_lifecycle_config, complete_indicator_path_config
from crypto_quant_engine.analysis.lifecycle.lifecycle_eval import evaluate_lifecycle_batch
from crypto_quant_engine.analysis.lifecycle.lifecycle_report import (
    build_lifecycle_report, lifecycle_conditional_report, layer2_reference_execution_report, layer2_execution_cube,
    depth_to_reference, time_to_reference, overshoot, absorption,
    overshoot_with_time_to_ref,
    candle_color_ret, wick_size_ret, adx_ret, volume_ratio_ret,
    adx_for_absorption, adx_and_t_ref, volume_for_absorption, volume_and_t_ref, volume_adx_color_ret)
from crypto_quant_engine.analysis.lifecycle.indicator_path import compute_indicator_path_features
from crypto_quant_engine.analysis.lifecycle.trade_state_path import build_trade_state_path_df
from crypto_quant_engine.analysis.lifecycle.trade_state_report import prepare_standard_trade_table, run_standard_trade_state_suite

from crypto_quant_engine.strategies.wick_fast_absorption import WickFastAbsorptionStrategy, format_payload
from crypto_quant_engine.analysis.trade_analyzer import TradeAnalyzer, compute_mfe_mae_for_events

available_test = {
    'pivot_detector': 0,
    'scenario': 0,
    'fast_retest_strategy': 1,
    'lifecycle': 0,
    'features_vs_outcome': 0
    }

def mfe_given_mae_table(
    df: pd.DataFrame,
    mae_thresholds=(0.5, 1.0, 1.5, 2.0),
    mfe_targets=(1.0, 2.0, 3.0)
) -> pd.DataFrame:

    rows = []

    for mae_th in mae_thresholds:
        subset = df[df['mae_atr'] < mae_th]
        n = len(subset)

        row = {
            'mae_threshold': mae_th,
            'count': n
        }

        for mfe_th in mfe_targets:
            if n == 0:
                row[f'P(mfe>{mfe_th})'] = float('nan')
            else:
                row[f'P(mfe>{mfe_th})'] = (subset['mfe_atr'] > mfe_th).mean()

        rows.append(row)

    return pd.DataFrame(rows)

if __name__ == '__main__':
    def compute_signal_and_get_series(market_data, signal):
        signal.set_market_data(market_data)
        signal.compute_and_cache_full_period()
        return signal.signal_time_series().get()
    
    trading_pair = 'BTC_USDT'#'LINK_USDC'#'BTC_USDT'#'SOL_USDT'
    frequency = '1h'#'1h'
    exchange = 'binance'#'hyperliquid'
    market_type = 's'#'p'

    #market_data = MarketData(trading_pair, start_time=datetime(2024,1,1), end_time=datetime(2025,12,31), exchange=exchange, market_type=market_type)
    market_data = MarketData(trading_pair, start_time=datetime(2020,1,1), end_time=datetime(2025,12,31), exchange=exchange, market_type=market_type)
    market_data_df = market_data.get_market_data(frequency).copy()
    
    is_upper = False
    wick_name = 'upper' if is_upper else 'lower'
    
    #Use pivot detector to get the context as well
    res_trend_df = None
    pivot_pickle_file_name = f"pivot_{trading_pair}_{frequency}_{exchange}_{market_type}.pickle"
    wick_pickle_file_name = f"{wick_name}_wick_{trading_pair}_{frequency}_{exchange}_{market_type}.pickle"
    lifecycle_file_name = f"{wick_name}_life_cycle_wick_{trading_pair}_{frequency}_{exchange}_{market_type}.pickle"

    
    if available_test['pivot_detector']:
        if not os.path.exists(pivot_pickle_file_name):
            #Ici on veut rajouter les columns du pivot detector
            pivot_detector_1h.set_market_data(market_data)
            pivot_detector_1h.compute_and_cache_full_period()
            res_trend = pivot_detector_1h.value_by_dates
            res_trend_f = {ts: {'macro': zz_val['structure_macro'], 'micro': zz_val['structure_micro'], 'htf_macro_trend': zz_val['macro_trend']} for ts, zz_val in res_trend.items()}
            res_trend_df = pd.DataFrame.from_dict(res_trend_f, orient='index')
            res_trend_df.to_pickle(pivot_pickle_file_name)
        else:
            res_trend_df = pd.read_pickle(pivot_pickle_file_name)
            res_trend_df.index.name = 'datetime'
    
    use_atr = True
    retest_period = 24
    include_following_candle = True
    atr_period, volume_period, adx_period = 14, 14, 14
    
    config = default_lifecycle_config()
    #config['max_horizon'] = retest_period
    
    body_pct = 0.1
    large_body_wick_thresold, body_wick_threshold = 1.2, 0.65
    large_atr_wick_threshold, atr_wick_threshold = 0.6, 0.4
    wick_threshold = atr_wick_threshold if use_atr else body_wick_threshold
    large_wick_threshold = large_atr_wick_threshold if use_atr else large_body_wick_thresold
    partial_retest_tolerance = 0.002 #Price needs to be no less than 0.2%
    
    fast_retest_bars = 3
    streak_window = 3
    streak_decay = 0.8
    
    # Test de la strategy
    if available_test['fast_retest_strategy']:
        stop_atr = 3.0#1.0
        tp_r = 3.0
        fast_retets_strategy = WickFastAbsorptionStrategy(frequency, is_upper, body_pct, atr_wick_threshold, large_atr_wick_threshold, atr_period, volume_period, adx_period, streak_window, streak_decay,
                                                          fast_retest_bars, retest_period-8, stop_atr, tp_r)
        print_notifier = PrintNotifier(payload_data_formatter=format_payload)
        portfolio = Portfolio(trading_pair, 1000000)
        portfolio_manager = PortfolioManager(portfolio)
        backtester = Backtester(market_data, fast_retets_strategy, portfolio_manager, notifier=print_notifier)
        backtester.run()
        
        positions = portfolio_manager.get_closed_postions()
        trade_analyzer = TradeAnalyzer(positions)
        trade_summary = trade_analyzer.summary()
        print(trade_summary)
        print(f"Closed postions: {len(positions)} / total postions: {len(portfolio_manager.positions)}")
        print(f"Oco violations: {len(portfolio_manager.oco_violations)}")
        positions_df = trade_analyzer.to_dataframe()
        #display_df(positions_df, 'tmp')
        
        #Il faut aussi que je regarde l'ecart entre l'entre e t la sortie
        entry_delta_atr = (positions_df['direction'] * (positions_df['wick_price'] - positions_df['entry_price'])) / positions_df['atr']
        d_delta = entry_delta_atr.describe()
        print(d_delta)
        
        # positions_df['entry_advantage_atr'] = entry_delta_atr
        # positions_df['pnl_atr'] = positions_df['net_pnl'] / positions_df['atr']
        # positions_df[['entry_advantage_atr', 'pnl_atr']].corr()
        # positions_df['entry_bucket'] = pd.qcut(positions_df['entry_advantage_atr'], 5)
        # positions_df.groupby('entry_bucket')['pnl_atr'].mean()

        
        #Now add the mfe an mae for each position
        df_mfe_exit = compute_mfe_mae_for_events(market_data_df, positions_df, 
                                                 horizon_type='until_exit', horizon_value_col='exit_time', atr_col='atr')
        positions_df['horizon_5'] = 5 #Check over 10 bars
        positions_df['horizon_10'] = 10
        positions_df['horizon_15'] = 15
        df_mfe_5 = compute_mfe_mae_for_events(market_data_df, positions_df, horizon_type='bars', horizon_value_col='horizon_5', atr_col='atr')
        df_mfe_10 = compute_mfe_mae_for_events(market_data_df, positions_df, horizon_type='bars', horizon_value_col='horizon_10', atr_col='atr')
        df_mfe_15 = compute_mfe_mae_for_events(market_data_df, positions_df, horizon_type='bars', horizon_value_col='horizon_15', atr_col='atr')

        mfe_survived_5 = mfe_given_mae_table(df_mfe_5)
        mfe_survived_10 = mfe_given_mae_table(df_mfe_10)
        mfe_survived_15 = mfe_given_mae_table(df_mfe_15)

        #We can then add it on the mfe and mae    

    if available_test['scenario']:
        df_results = None
        wick_from_file = os.path.exists(wick_pickle_file_name)
        if not wick_from_file:
            wick_scenario = EnhancedWickRetestScenario(frequency, is_upper, partial_retest_tolerance=partial_retest_tolerance, wick_threshold=wick_threshold, large_wick_threshold=large_wick_threshold, use_atr=use_atr, include_following_candle=include_following_candle,
                    atr_period=atr_period, volume_period=volume_period, adx_period=adx_period, fast_retest_bars=fast_retest_bars, streak_window=streak_window, streak_decay=streak_decay)
            
            #En faire aussi un scenario ou l'on test sur toutes les candle et on observe pas de break
            #scenario_analyzer = ScenarioAnalyzer(market_data, wick_scenario, 1, output_window_size=retest_period, default_frequency=frequency)
            scenario_analyzer = EveryBarScenarioAnalyzer(market_data, wick_scenario, 1, output_window_size=retest_period, default_frequency=frequency)
            scenario_analyzer.run()
            
            #Get complete df
            additional_info = {'Time to Retest Full': lambda input, output: output.get('t_full_retest') if output is not None else None,
                               'Time to Retest Partial': lambda input, output: output.get('t_partial_retest')  if output is not None else None} #if output is not None else 0
            frequency_result = scenario_analyzer.classification_frequency().round(2) #additional_info
            all_result = scenario_analyzer.get_all_results()
            #Ici on pourra potentiellement rajouter ATR ou creer une master df avec les returns
            df_results = all_result.sort_values(by='datetime').set_index('datetime', drop=True)
            df_results.to_pickle(wick_pickle_file_name)
        else:
            df_results = pd.read_pickle(wick_pickle_file_name)

        #res_with_add_col = add_derived_columns(df_results)
        
        if df_results is not None:
            df_results = pd.merge(df_results, res_trend_df, on='datetime')
            
        enriched_data = wick_enricher(df_results, wick_for_ratio = 'upper' if is_upper else 'lower')
        feature_output = wick_training_dataset(enriched_data, output_col='is_no_retest', is_upper=is_upper)
        first_index = feature_output[['macro', 'micro', 'htf_macro_trend']].notna().all(axis=1).idxmax()
        feature_output = feature_output.loc[first_index:]
        
        #To add quantile column to check precise combination
        wick_size_q = add_quantile_column(feature_output, 'lower_wick')
        volume_ratio_q = add_quantile_column(feature_output, 'volume_ratio')
        
        # analysis = run_all_wick_retest_analyses(df_results)
        # for key, value in analysis.items():
        #     print(key)
        #     display_df(value, 'tmp')
        #     i = 0
        
    #Life cycle evaluation, need to have results from the scenario with all the meta
    #Required the events_df
    if available_test['lifecycle']:
        #Here need to add all the indicator to the df
        if use_atr and not wick_from_file:
            atr_series = wick_scenario.get_signal('atr').signal_time_series.get()
        else:
            atr_signal = ATRSignal(frequency, atr_period)
            atr_series = compute_signal_and_get_series(market_data, atr_signal)
        market_data_df['atr'] = atr_series
        
        if not wick_from_file:
            ema_volume_series = wick_scenario.get_signal('ema_volume').signal_time_series.get()
            adx_series = wick_scenario.get_signal('adx').signal_time_series.get()
        else:
            ema_volume_series = compute_signal_and_get_series(market_data, EMAVolumeSignal(frequency, volume_period))
            adx_series = compute_signal_and_get_series(market_data, ADXSignal(frequency, adx_period))
        
        volume_ratio_series = market_data_df[VOLUME] / ema_volume_series
        market_data_df['volume_ratio'] = volume_ratio_series
        
        market_data_df['adx'] = adx_series
        
        atr_ratio = ATRCompressionSignal(frequency, 5, atr_period)
        atr_ratio_series = compute_signal_and_get_series(market_data, atr_ratio)
        market_data_df['atr_ratio'] = atr_ratio_series
        
        vwap_period = 25 #Between 20 and 30
        vwap_signal = VWAPSignal(frequency, vwap_period)
        vwap_series = compute_signal_and_get_series(market_data, vwap_signal)
        vwap_dist_series = (market_data_df[CLOSE] - vwap_series) / atr_series
        market_data_df['vwap'] = vwap_series
        market_data_df['vwap_dist'] = vwap_dist_series
        
        rsi_entry_period = 14
        rsi_entry_state_signal = RSISignal(frequency, rsi_entry_period)
        rsi_entry_state_series = compute_signal_and_get_series(market_data, rsi_entry_state_signal)
        market_data_df['rsi_entry'] = rsi_entry_state_series
        
        rsi_trade_period = 7
        rsi_trade_state_signal = RSISignal(frequency, rsi_trade_period)
        rsi_trade_state_series = compute_signal_and_get_series(market_data, rsi_trade_state_signal)
        market_data_df['rsi_trade'] = rsi_trade_state_series
       
        # events_df must contain:
        # event_id, bar_index, direction (+1 / -1)
        
        events_df = feature_output.copy()
        events_df['event_id'] = range(len(events_df))
        events_df['direction'] = 1 if is_upper else -1
        events_df['event_time'] = events_df.index
        events_df['reference_price'] = market_data_df[HIGH if is_upper else LOW]
        events_df['retest'] = ~events_df['is_no_retest']
        
        if not os.path.exists(lifecycle_file_name):
            lifecycle_df = evaluate_lifecycle_batch(market_data_df, events_df, config)
            lifecycle_df.to_pickle(lifecycle_file_name)
        else:
            lifecycle_df = pd.read_pickle(lifecycle_file_name)
            
        #Add volume on retest
        event_idx = market_data_df.index.get_indexer(lifecycle_df['event_time'])
        retest_idx = event_idx + lifecycle_df['t_ref'].fillna(0).astype(int)
        valid = (lifecycle_df['reference_touched'] & (event_idx >= 0) & (retest_idx >= 0) & (retest_idx < len(market_data_df)))    
        out = np.full(len(lifecycle_df), np.nan)
        out[valid.values] = market_data_df['volume_ratio'].values[retest_idx[valid]]
        lifecycle_df['volume_ratio_at_retest'] = out
        
        complete_lifecycle_df = lifecycle_df.merge(events_df, on=['event_time', 'event_id'], how='left')
        meta_cols = ['event_time', 'reference_price', 'retest']
        lifecycle_df = lifecycle_df.merge(events_df[meta_cols], on='event_time', how='left')
        group_cols = () #
        #group_cols = ('retest',)
        
        report = build_lifecycle_report(lifecycle_df, group_cols)
        state_df = report['state_table']
        quantile_df = report['quantiles']
        conditional_report = lifecycle_conditional_report(lifecycle_df)

        layer2 = layer2_reference_execution_report(lifecycle_df)

        depth_return, depth_group = depth_to_reference(lifecycle_df)
        time_return, time_group = time_to_reference(lifecycle_df)
        overshoot_return, overshoot_group = overshoot(lifecycle_df)
        absorption_return, absorption_group = absorption(lifecycle_df)
        ov_x_time_return, ov_x_time_group = overshoot_with_time_to_ref(lifecycle_df)
        
        #Now if I want a complete layer 2 report using more indicator
        #For adx we need to check how it was used before maybe a specifc value ? Based on quantiles 
        wick_size_return, size_group = wick_size_ret(complete_lifecycle_df)
        color_return, color_group = candle_color_ret(complete_lifecycle_df)
        adx_return_True, adx_group = adx_ret(complete_lifecycle_df, reference_touched = True)
        adx_return_False, adx_group = adx_ret(complete_lifecycle_df, reference_touched = False)
        volume_ratio_return_True, _ = volume_ratio_ret(complete_lifecycle_df, reference_touched = True)
        volume_ratio_return_False, _ = volume_ratio_ret(complete_lifecycle_df, reference_touched = False)
        
        adx_for_time, _ = adx_for_absorption(complete_lifecycle_df)
        adx_with_time, ret = adx_and_t_ref(complete_lifecycle_df)
        
        volume_ratio_for_time, _ = volume_for_absorption(complete_lifecycle_df)
        volume_ratio_with_time, _ = volume_and_t_ref(complete_lifecycle_df)
        
        volume_adx_color_return, _ = volume_adx_color_ret(complete_lifecycle_df)
        volume_adx_color_not_touched_return, _ = volume_adx_color_ret(complete_lifecycle_df, reference_touched=False)
        volume_adx_color_touched_return, _ = volume_adx_color_ret(complete_lifecycle_df, reference_touched=True)
        
        #Add indicators path
        indicator_path_config = complete_indicator_path_config()
        indicator_path = compute_indicator_path_features(market_data_df, events_df, lifecycle_df, indicator_path_config, full_window=config['max_horizon'])
            
        #compute trade lifecycle
        trade_state_df = build_trade_state_path_df(market_data_df, events_df, lifecycle_df, indicator_path_config)
        live_trade_analysis = run_standard_trade_state_suite(trade_state_df, layer2_execution_cube)
        
    #Analysis using new framework and the rand
    if available_test['features_vs_outcome']:
        df = feature_output #add_helper_columns(df_results, scenario='wick')
        print("After enrichment:", df.shape)
        
        # Inspect columns that are now available
        print("\nAvailable columns:")
        print(sorted(df.columns))
        
        
        # ============================================================
        # 3. SET UP FEATURE EVALUATOR
        # ============================================================
        
        fe = FeatureEvaluator(df, event_col='is_no_retest') # is_no_retest is_full_retest
        conf = ConfluenceTable(fe)

        # Example list of features relevant for wick behavior
        numeric_features = [
            'lower_wick', #OG
            'wick_body_ratio',
            'wick_range_ratio',
            'volume_ratio',
            'adx'
        ]
        
        categorical_features = [
            'volume_regime',
            'adx_regime',
            #'wick_size_bin',
            'candle_color',
            'streak_cluster',
            'macro',
            'micro',
            'htf_macro_trend',
            'session'
        ]
        
        all_features = numeric_features + categorical_features
        
        
        # ============================================================
        # 4. SINGLE FEATURE EVALUATION
        # ============================================================
        
        print("\n=== Single-feature evaluation ===\n")
        single_eval = fe.auto_single_feature_search(all_features)
        
        print(single_eval.head(20))  # Top 20 strongest single features
        
        
        # ============================================================
        # 5. MULTI-FEATURE CONFLUENCE SEARCH
        # ============================================================
        
        print("\n=== Multi-feature confluence search (pairs) ===\n")
        
        multi_results = fe.auto_multi_feature_search(
            features=[
                #Good 
                #'lower_wick',
                #'candle_color',
                #'volume_regime'
                #'volume_ratio',
                #'adx' # does not add much more
                
                # 'lower_wick',
                # 'candle_color',
                # 'streak_cluster', #Not decisive reduce sample size
                # 'volume_ratio'
                
                'lower_wick',
                'candle_color',
                'macro', #Not decisive reduce sample size
                'volume_ratio'
                
                #'macro',
                #'micro',
                #'htf_macro_trend',
                
                #'volume_ratio',
                #'adx'
            ],
            max_depth=3,         # Search only pairs (safe)
            min_count=80         # Require minimum sample
        )
        
        print(multi_results.head(20))
        
        table = conf.build('wick_ratio_bin', 'volume_regime', metric='p_event')

        
        
        # ============================================================
        # 6. THRESHOLD OPTIMIZATION FOR A KEY FEATURE
        # ============================================================
        
        print("\n=== Threshold optimization for wick_ratio ===\n")
        
        opt = fe.optimize_threshold('wick_ratio')
        print("Best threshold:", opt['best_threshold'])
        print("Lift:", opt['lift'])
        print("Z-stat:", opt['z_stat'])
        print("\nCandidate thresholds:")
        print(opt['candidates'].sort_values('lift', ascending=False).head(10))
        
        
        # ============================================================
        # 7. MONOTONICITY CHECK
        # ============================================================
        
        print("\n=== Monotonicity check for wick_ratio ===\n")
        
        mono = fe.detect_monotonicity('wick_ratio')
        print("Is monotonic:", mono['is_monotonic'])
        print("Direction:", mono['direction'])
        print("Violations:", mono['violations'])
        print("\nBins:")
        print(mono['bins'])
        
        
        # ============================================================
        # 8. FULL DUMP FOR A SINGLE FEATURE (OPTIONAL)
        # ============================================================
        
        print("\n=== Detailed feature breakdown: adx ===\n")
        adx_eval = fe.evaluate_feature('adx')
        
        
        
        # ============================================================
        # 9. INSIGHTS SECTION
        # ============================================================
        
        print("\n=== Insights ===")
        
        print("""
        - Use single-feature evaluation to quickly understand which variables carry predictive power.
        - Use confluence search to find powerful two-feature combinations (wick + volume, wick + ADX, etc.).
        - Use threshold optimizer to define a clean rule boundary for numeric features.
        - Use monotonicity detection to verify if a feature’s relationship to the event makes sense.
        - The enriched dataset includes volume/ADX regimes, streak clusters, wick bins, MFE/MAE metrics, etc.
        - These tools allow you to design a WickScore or feed features into ML classifiers.
        """)

        

    # #Ici calcul de higher et lower avec les retracement Fibo
    # #atr period 14
    # zz = ZigZagATR(
    #     frequency = frequency,
    #     atr_mult=1.5,
    #     use_close=True,                # set False to use (H+L)/2 mid for faster pivots
    #     range_band=(0.30, 0.70),
    #     range_persistence=2,           # require 3 consecutive bars in the band
    #     min_leg_atr=0.5                # ignore flimsy trends
        
    # )
    # zz.set_market_data(market_data)
    # res_trend = zz.compute_with_vectorized_calculation()
    # res_trend_f = {ts: {'macro': zz_val['structure_macro'], 'micro': zz_val['structure_micro'], 'score': zz_val['pivot_score'], 'pivot': zz_val['potential_pivot']} for ts, zz_val in res_trend.items()}
    # res_trend_df = pd.DataFrame.from_dict(res_trend_f, orient='index')
    # #show_df(res_trend_df)

    # #Simple Test wick retest
    # if 1:
    #     large_body_wick_thresold, body_wick_threshold = 1.2, 0.65
    #     large_atr_wick_threshold, atr_wick_threshold = 0.6, 0.4
    #     use_atr = True
    #     include_following_candle = True
    #     retest_period = 14
    #     wick_threshold = atr_wick_threshold if use_atr else body_wick_threshold
    #     large_wick_threshold = large_atr_wick_threshold if use_atr else large_body_wick_thresold
    #     basic_wick_scenario = BasicWickRetestScenario(frequency, is_upper, wick_threshold=wick_threshold, large_wick_threshold=large_wick_threshold, use_atr=use_atr, include_following_candle=include_following_candle )
    #     scenario_analyzer = ScenarioAnalyzer(market_data, basic_wick_scenario, 1, output_window_size=retest_period, default_frequency = frequency)
    #     scenario_analyzer.run()
    
    #     additional_info = {'Time to Retest': lambda input, output: output.get('retest_period') if output is not None else None}
    #     frequency_result = scenario_analyzer.classification_frequency(additional_info)
    #     all_result = scenario_analyzer.get_all_results()
    #     sorted_by_input = all_result.sort_values(by="datetime").set_index('datetime', drop=True)
    #     sorted_by_input = pd.concat([sorted_by_input, res_trend_df], axis=1, join='inner')
    #     # show_df(sorted_by_input)
    #     failed = sorted_by_input[sorted_by_input['output_classification'] == 'NO_RETEST']
    
    #     #Ici comparere en function du time to retest
    #     basic_wick_scenario_no_following = BasicWickRetestScenario(frequency, is_upper, wick_threshold=wick_threshold, large_wick_threshold=large_wick_threshold, use_atr=use_atr, include_following_candle= not include_following_candle )
    #     scenario_analyzer_no_following = ScenarioAnalyzer(market_data, basic_wick_scenario_no_following, 1, output_window_size=retest_period, default_frequency = frequency)
    #     scenario_analyzer_no_following.run()
    #     frequency_result_no_following = scenario_analyzer_no_following.classification_frequency(additional_info)
    #     sorted_by_input_no_following = scenario_analyzer_no_following.get_all_results().sort_values(by="datetime").set_index('datetime', drop=True)
    #     sorted_by_input_no_following = pd.concat([sorted_by_input_no_following, res_trend_df], axis=1, join='inner')
    #     failed_no_following = sorted_by_input_no_following[sorted_by_input_no_following['output_classification'] == 'NO_RETEST']
    
    #     #Maintenant merge pour voir la tete tes candle
    #     mask = (sorted_by_input['output_retest_period'] == 1) & (sorted_by_input_no_following['output_retest_period'].isna())
    #     candidate = sorted_by_input[mask]