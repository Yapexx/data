# ATR pivot detector: behavior and research notes

Implementation revision: 7 September 2026. Original research brief remains in
`pivot_detector.txt`; the backup was not changed. This document describes the
refactored implementation, not the previous parameter guide.

## Research contract

The target is whether a completed bar will eventually be identified as an exact
high/low of a confirmed swing. A pivot's bar `t*` and its confirmation bar `t_c`
are different concepts. An event at `t_c` labels `t*`; it does not automatically
label `t_c` positive. A confirmation bar can independently become a later pivot.

"Exact" means the first occurrence of the highest high or lowest low in the
defined swing interval. It does not mean that every small local extremum is a
pivot, nor that the selected extreme was a tradeable entry after observing its
bar's close. The labels remain dependent on the confirmation parameters.

Input timestamps identify bars (the local files use opening timestamps). Features
and confirmation decisions become available only after that bar closes. The
stored `confirmation_time` / `label_available_time` identify that bar; callers
doing wall-clock execution or splitting inside a bar must account for its duration.

## Extreme tracking and confirmation

1. Track both actual highs and lows during bootstrap, including indicator warmup.
2. A strict improvement replaces a candidate, however small; an equal price keeps
   the first occurrence. There is no minimum price step.
3. Require a reversal of
   `atr_mult * confirm_hysteresis_mult * ATR[current bar]` for `confirm_closes`
   consecutive bars. ATR is evaluated independently on each bar, not frozen at
   the candidate. Missing, nonfinite or nonpositive ATR breaks the streak.
4. A new candidate resets its streak to zero. Its own bar never counts toward
   confirmation. This avoids assuming that the candle's opposite wick followed
   its extreme. A tie does not restart the streak.
5. Bootstrap confirms whichever side first meets the rule. If both qualify on
   the same bar, choose the older candidate; choose the high if their indices tie.
6. After a high, track a low; after a low, track a high. Confirmed pivots strictly
   alternate and have increasing indices. At most one event is emitted per bar.

With `use_close=True`, compare the close with the extreme. With `False`, use the
low against a candidate high, or the high against a candidate low. The latter is
still a completed-bar calculation: `confirm_closes` then counts qualifying wick
bars, not closes. It is not an intrabar execution simulation.

The first pivot searches from the first input bar through its confirmation. Each
subsequent pivot searches from the bar **after** the previous pivot through its
own confirmation, inclusive. The opposite wick of the previous pivot candle is
excluded: OHLC cannot establish its order relative to the pivot extreme. A swing
can therefore miss a genuine same-candle reversal by design. Use finer data if
intrabar ordering is needed; do not invent it from daily/hourly OHLC.

## Confirmation transition and causal features

For each completed bar:

1. Advance the old leg's candidate, reversal counter and per-bar accumulators.
2. Build a new feature dictionary using the pivots/macro state confirmed before
   this bar's transition. Current OHLC, ATR and other indicators are allowed.
3. If confirmation qualifies, append an event, change structural state and
   initialize the new leg from the already observed `t*..t_c` history.
4. Attach event fields to the output. Never revise past feature dictionaries.

Bridge reconstruction retains extrema and close-path movements that occurred
before confirmation. It also reconstructs slope, compression, microcycles,
divergence candidates, range persistence and qualifying reversal counts for the
new leg. These reconstructed features are used on subsequent bars. Reconstruction
does not emit historical events or recursively confirm multiple pivots at `t_c`.
An already qualified opposite candidate can be considered on the next bar, using
that bar's information as well.

`direction`, `last_pivot_*`, `local_high/low`, macro/micro features and scores are
pre-transition fields. `pivot_high/low`, `new_pivot_*`, `confirmation_*`,
`current_local_high/low` and `hh/hl/lh/ll` describe the event/post-transition state.
Do not mix the latter into the specified pre-transition ML feature matrix.
`early_reversal` means a qualifying streak is active, including on the bar that
confirms. `reversal_count` exposes the actual pre-transition count for diagnostics.

Range classification can change using the current bar's retracement before the
snapshot; this is causal and does not apply the new pivot early.

## Parameters

| Parameter | Default | Meaning |
|---|---:|---|
| `atr_mult` | 1.5 | Reversal depth in current ATR units |
| `confirm_closes` | 2 | Consecutive qualifying bars after the candidate bar |
| `confirm_hysteresis_mult` | 1.0 | Additional multiplier on reversal depth; >= 1 |
| `use_close` | True | Close confirmation; False uses opposite wicks |
| `atr_period` | 14 | ATR and RSI period; ADX uses max(2, period // 2) |
| `range_band` | (0.30, 0.70) | Local retracement band eligible for a range label |
| `range_persistence` | 0 | Require this value + 1 consecutive in-band bars |
| `min_leg_atr` | 0.0 | Minimum latest confirmed swing size for a directional macro label |

The shared hourly preset and the research script use `range_persistence=2` and
`min_leg_atr=0.5`. These affect features, not the pivot sequence.

Retired constructor arguments are accepted only for migration, ignored, and emit
a `FutureWarning` when explicitly supplied:

- `min_runext_step_atr`: ignored improvements cannot locate the exact extreme.
- `min_pivot_separation_atr`: legitimate double tops/bottoms can have equal prices.
- `min_bars_between_pivots`: rejecting a fixed historical index can block forever.
- `min_leg_span_atr` and `range_if_small_span`: the old rejection/soft flip broke
  alternation. Reversal depth and consecutive bars now provide the pivot filters.

Do not interpret these deprecated arguments as active controls. In particular,
the old five-bar spacing and two-ATR leg-span requirements are no longer applied.
Adding a duration filter later requires a separately specified swing definition.

## Features

### Leg path and exhaustion

For pivot price `p` on bar `k`, define the observable close-based path at bar `t`:

```text
path = abs(close[k] - p) + sum(abs(close[j] - close[j-1]), j=k+1..t)
displacement = abs(close[t] - p)
leg_inefficiency = 1 - displacement / path
```

Zero path and zero displacement give inefficiency zero. Both origins are the
exact pivot price. This is an extreme-to-close and then close-to-close path, not
the unknowable full intrabar traded path. The numerical result is bounded to
[0, 1]; independent diagnostics also verify the path geometry before clipping.

`leg_exhaust_atr` is the directional pivot-to-running-extreme amplitude / ATR.
`slope_decay` is the preceding bar's amplitude/age divided by the current bar's
amplitude/age, both reconstructed within the same leg. Undefined ratios are
missing, not fabricated zeros. `exhaustion_score` retains the existing formula
`leg_exhaust_atr * (1 + slope_decay)`; it is an uncalibrated heuristic.

`leg_compression` is the fraction of the last up to five bars after the pivot
whose candle range H-L is below 0.5 ATR. Invalid ATR observations are excluded.
This is candle-range compression, not a true-range calculation.

`microcycle_count` counts direction changes between nonzero close-to-close moves
starting at the pivot bar's close. Flat closes do not create turns. Density is
that count / bars since the same pivot. Bootstrap values are missing.

RSI divergence compares successive strict running-extreme improvements within
the same reconstructed leg. A missing comparison RSI gives a missing flag.
ADX factor/regime/rising use the current and preceding causal ADX values.
`volume_z` is retained only as a missing compatibility output; no volume statistic
is implemented or included in `training_fields`.

### Retracement and structure

Both local and macro retracement use 0 at the extreme, 1 at the anchor, and allow
values above 1 for an anchor breach. Undefined/nonpositive spans are missing.
Nearest Fibonacci levels use the clipped [0, 1] retracement and now vary normally.

Macro direction requires two confirmed highs and two confirmed lows:

- HH and HL -> up;
- LH and LL -> down;
- mixed comparisons or ties -> range.

Before enough pivots exist the macro state is missing. `min_leg_atr` can classify
an otherwise directional pattern as range. On a new uptrend the macro anchor is
the older of the last two lows; on a downtrend it is the older of the last two
highs. This anchor precedes the establishing HH/LL. It remains fixed during a
directional macro regime; the macro extreme uses newly confirmed prices. A macro
range clears the anchor until a directional pattern is re-established.

`structure_micro` is the current leg direction with a temporary in-band range
override. `structure_macro` applies the same temporary override to macro state.
Both overrides disappear when price leaves the band. `macro_trend` itself changes
only at a pivot confirmation.

### Research inputs and missing values

`categorical_fields` and `numerical_fields` explicitly define the 31 candidate
inputs. `training_fields` joins them. Absolute prices and timestamps are excluded:
`atr_pct = ATR / close` (a fraction, not multiplied by 100), and
`distance_to_pivot_atr = (close - last_pivot_price) / ATR` provide normalized context.
The upper-wick column is correctly named `wick_upper_norm`.

`features_ready` requires a prior confirmed pivot, a running candidate and valid
ATR. Macro structure, divergence or other optional features can still be missing.
The dataset helper preserves those missing values for a future training-only
preprocessor. It does not fit imputers, categories, quantiles or a model.

The retained age/amplitude normalizations, ADX summaries and Fibonacci buckets
are redundant candidates for descriptive research, not an endorsed final feature
set. In particular `leg_amp_norm` often saturates at 1 with the old two-ATR scale.
Scores remain comparison heuristics, not calibrated pivot probabilities.

## APIs, exports and lifecycle

- `update(...)`: consume one chronological completed bar, cache/return its feature
  and event dictionary. If no timestamp is supplied, use the sequential index.
  Reject invalid OHLC, missing timestamps and duplicate/out-of-order timestamps.
- `get_events_df()`: one row per event with pivot index/time/type/price,
  confirmation index/time and delay. Do not mutate the detector's internal lists.
- `get_result_df()`: all cached feature/event rows plus retrospective labels.
  `bar_time` names the bar identifier in this export, avoiding collision with the
  market files' numeric `timestamp`. The raw update dictionary retains `timestamp`.
- `get_research_dataset()`: return `(X, y, metadata)` for feature-ready, resolved
  rows. Numerical/binary inputs are floats; categorical inputs use pandas string
  dtype. Metadata contains bar index/time, pivot subtype and label availability.
- `reset()`: clear bars, state, bootstrap, labels, events and output cache while
  retaining configuration/market binding. Re-running a batch is deterministic.
- `set_market_data(..., clear_data=True)`: reset detector state. Carrying state to
  a different MarketData object with `clear_data=False` raises an error.

`is_pivot` uses nullable integers: 1 for a confirmed exact pivot, 0 for a resolved
nonpivot, missing for unresolved observations. `pivot_type` contains high/low only
at positives; it is missing for both negatives and unresolved rows.

Labels are resolved conservatively. An event resolves the interval through its
`t*`; all later bars remain unresolved until another event closes the next
interval. `label_available_time` is that resolving confirmation bar, including
for negatives. `pivot_confirmation_time` is populated only for positives.
Resolved labels never change; later exports can resolve additional labels without
altering historical feature rows. This intentionally discards some negatives
that could potentially be recognized sooner.

For chronological training at cutoff T, retain training rows only when their
label availability is at or before the last completed bar at T. Do not train on
labels confirmed later merely because their pivot timestamps precede T.

The existing `analysis/dataset_builder.py` pivot enricher now handles empty-string
legacy labels and preserves nullable unresolved labels. Its general enrichment
pipeline still computes full-sample bins and forward outcomes: use the explicit
dataset split above for ML, not an automatic selection of all enriched columns.

Batch and the detector's Signal append entry point calculate ATR/RSI/ADX over the
full supplied causal prefix using the existing TA-Lib dependency definitions.
This avoids resetting Wilder indicators on a short rolling window. The append
entry point requires initialization followed by exactly one new bar at a time;
duplicate cached appends are no-ops. The lower-level append callback accepts
caller-supplied indicators, whose causal correctness is the caller's responsibility.

All observed bars/features are retained for research. Bridge replay is linear in
its observed interval; a complete run processes those intervals without recursive
event replay. The framework append adapter recomputes full indicator prefixes
(O(history) per append) for correctness. This is suitable for this small research
workflow, not a claim of bounded-memory or optimized tick-scale live operation.

## Verification and measured baseline

The old implementation was run unchanged before editing with the same sample
settings below (old detector source at commit `cd4cc05`). No parameter fitting or
search was performed. Counts changed because the definition changed.

| Measure | Daily before | Daily after | Hourly before | Hourly after |
|---|---:|---:|---:|---:|
| Bars | 701 | 701 | 1,464 | 1,464 |
| Highs | 16 | 30 | 45 | 62 |
| Lows | 15 | 30 | 45 | 63 |
| Pivots / all bars | 4.42% | 8.56% | 6.15% | 8.54% |
| Mean confirmation delay, bars | 6.19 | 4.78 | 6.34 | 5.32 |
| Median confirmation delay, bars | 5 | 4 | 4 | 4 |
| Maximum delay, bars | 16 | 13 | 32 | 21 |
| Consecutive same-type events | 1 | 0 | 8 | 0 |

Samples are Binance BTC/USDT spot:

- Daily: requested 2024-01-01 through 2025-12-31; available through 2025-12-01.
- Hourly: fixed 2024-01-01 00:00 through 2024-03-01 23:00.

After excluding initialization and unresolved observations, the daily dataset has
675 rows / 59 positives (8.74%); hourly has 1,440 rows / 124 positives (8.61%).
The final 10 daily and 8 hourly rows are conservatively unresolved. The initial
bootstrap pivot is excluded from both training datasets because its features
were not ready.

Measured daily pivots now include:

| Extreme bar | Type | Price | Confirmation bar |
|---|---|---:|---|
| 2024-01-11 | high | 48,969.48 | 2024-01-16 |
| 2024-01-23 | low | 38,555.00 | 2024-01-27 |
| 2024-03-14 | high | 73,777.00 | 2024-03-19 |
| 2024-03-20 | low | 60,775.00 | 2024-03-25 |
| 2024-03-27 | high | 71,769.54 | 2024-04-03 |

The March 27 candidate previously lasted 216 bars. It now confirms on April 3.
The small-span direction flip and resulting repeated high in December are gone.
The January bootstrap now selects the wick high, and requires two valid qualifying
bars after ATR becomes available, explaining its changed confirmation date.

Verified on both samples: exact first extrema for every event, alternating types,
ordered indices, qualifying confirmation closes, independent close-path and
microcycle calculations, bounded features, no infinities, batch/direct-update
equivalence, reset/replay, prefix invariance around confirmations and a midpoint,
resolved-label stability, and actual Signal append agreement. Market data had no
missing/duplicate sample bars. Missing macro retracements during a macro range,
unavailable divergence comparisons and warmup values are intentional. Audit output
reports all missing counts; no candidate input was constant in either sample.

The daily calculation took about 0.2 seconds and the hourly calculation about
0.8 seconds in a warmed environment; the full checks took about 5 and 10 seconds
respectively. These are indicative measurements, not performance guarantees.

Nineteen deterministic unit tests additionally cover ties, small improvements,
same-candle ambiguity, missing ATR, confirmation bridge paths, double tops,
retired filters, macro timing, range exit, labels/CSV/enricher, reset, invalid input
and short/empty market data. The original script's two CSV exports also executed
successfully and preserved the 60 positives / 10 unresolved daily labels.

## Running the checks

From the repository root, using an environment with the project dependencies:

```powershell
python -B -m unittest crypto_quant_engine.signals.test_pivot_detector
python -B -m scripts.pivot_test --audit
python -B -m scripts.pivot_test --audit --frequency 1h
```

The audit writes no files by default. To export both feature/label and event CSVs:

```powershell
python -B -m scripts.pivot_test --audit --output-dir pivot_audit_results
```

Existing files with matching names in an explicitly supplied output directory are
overwritten. The original no-argument script workflow remains available.
`python -B -m crypto_quant_engine.signals.pivot_diagnostics` is an equivalent audit
entry point. `--pair`, `--exchange` and `--market-type` reuse other local MarketData
files, with the same fixed audit dates; they do not fetch data.

The environment used for verification was
`C:\Users\yannx\Documents\Workplace\freqtrade\.venv\Scripts\python.exe`.
The default Miniconda interpreter did not have pandas installed.

Next research step: use the existing FeatureEvaluator on the explicit dataset,
inspect high/low and chronological stability, and then decide on an initial
feature subset. Keep untouched history for validation. No RF model, optimized
parameters or trading-performance claims are included in this revision.
