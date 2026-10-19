# Standalone research reviewer

## Purpose

Turn a completed study's saved output into an explainable shortlist of research
questions. This is a new implementation, independent of the legacy ranking/report
code. It reads artifacts only: no market download, signal calculation, forward
labelling, study rerun, ML, automatic feature combination search or strategy mutation.

The central distinction is between **a pattern worth investigating** and **an
executable profit**. Strong movement followed by a flat terminal close can still
be interesting. Conversely, a high MFE does not tell us which stop, target or
trailing rule could capture it. A failed saved exit does not veto further path
research, and good path statistics do not establish that risk management can
rescue a strategy.

The standalone entry point is `scripts/review_study.py`; its reusable analysis
module is `crypto_quant_engine/analysis/research_triage.py`. Existing study runners
and result files are unchanged. Existing fixed-bin assignment is reused, not the
old `indicator_ranking.py`, panel baseline or bootstrap.

## Run

From the repository root, using a NEW output directory:

```powershell
& '..\freqtrade\.venv\Scripts\python.exe' -B -m scripts.review_study reports/all_candle_feature_payoffs/initial --output-dir reports/research_reviews/my_review
```

For standard feature-path exports, the input can be the study directory or the
`feature_path.csv` / `feature_path_df.csv` file itself:

```powershell
& '..\freqtrade\.venv\Scripts\python.exe' -B -m scripts.review_study path/to/study --summary-only --output-dir reports/research_reviews/my_summary_review
```

Without combinations, this writes only two files:

- `REVIEW.md`: shortlist by role, calculations, cautions and next research question.
- `evidence.json`: every evaluated bucket/context, component statistics, yearly
  evidence, all assessment reasons, review policy and input/code fingerprints.

There is no overwrite mode. Inputs are hashed and checked again before export.
Only load trusted local pickles; CSV-only studies are supported too.

Pre-event movement predictors are available through the shared feature layer's
optional `pre_event` group. See
[PRE_EVENT_FEATURES.md](../crypto_quant_engine/analysis/PRE_EVENT_FEATURES.md) for
the exact prior-window/start-ATR definitions and snapshot/report integration.
They must first be added to saved snapshots/research and declared in the study's
feature definitions; the reviewer does not calculate them from summary tables.

## Manually selected feature combinations

Use `--combinations scripts/research_combinations.example.json` to enable
explicit combinations. This requires saved row-level observations; single-feature
summary tables cannot reconstruct intersections. Original indicators, outcomes,
and study files remain unchanged. The reusable grouping implementation is
`analysis/feature_groups.py`, called by `row_profiles(..., combinations=...)`.

```powershell
& '..\freqtrade\.venv\Scripts\python.exe' -B -m scripts.review_study reports/all_candle_feature_payoffs/initial --combinations scripts/research_combinations.example.json --output-dir reports/research_reviews/my_combinations
```

The JSON is an object of manually named definitions. Two modes are supported:

```json
{
  "colour_by_volume": {
    "mode": "bins",
    "features": ["ctx_candle_color", "ctx_volume_ema14_inclusive_ratio"]
  },
  "green_above_normal_volume": {
    "mode": "conditions",
    "conditions": [
      {"feature": "ctx_candle_color", "op": "==", "value": "GREEN"},
      {"feature": "ctx_volume_ema14_inclusive_ratio", "op": "between", "value": [1.25, 2], "closed": "right"}
    ]
  }
}
```

- `bins`: all OBSERVED bucket combinations of just the selected features, using
  their original saved bins. Unobserved Cartesian cells are not synthesized.
  At least two distinct features are required. `max_buckets` defaults to 100
  per named definition; exceeding it raises, rather than silently truncating.
- `conditions`: one AND conjunction. Supported operators: `==`, `!=`, `>`,
  `>=`, `<`, `<=`, `in`, `between`. Multiple conditions on one feature are
  combined before component comparisons. `between` defaults to `(lower, upper]`;
  `closed` may be `right`, `left`, `both` or `neither`. At least two distinct
  features must be involved. Even a zero-match condition is reported.
- A definition may include a plain-English `description`. Names cannot overwrite
  saved columns. Only the declared source features may be used; no automatic
  feature discovery, quantile refitting, nested combination search or expressions.
- Missing components stay unknown, even if another condition is already false.
  Outside-saved-bin observations are also excluded from both arms, including
  exact-condition mode. Missing takes precedence if both occur on a row.
- An exact condition's `not_match` group is reference only, not an extra candidate
  selected without your request. Its rows remain in the exported tables.

Every combination uses the same context, direction, split, completeness and
metric-validity rules as single features. It has MFE, MAE, horizon returns,
saved net payoff bounds (when available), counts, medians and yearly evidence.
Both ATR and return-unit excursions are exported when supplied. Saved outcome
label counts include target, stop, timeout and ambiguous paths on the complete
eligible cohort; these counts are NOT a new unambiguous target-before-stop rate.
No missing metric, payoff or path chronology is invented.

Component comparisons use the SAME jointly-known cohort. For GREEN + high volume,
the report shows GREEN alone and high volume alone. Both parent groups include
the combination, which is labelled explicitly. The incremental screen compares
GREEN + high volume with (a) GREEN without high volume and (b) high volume without
GREEN. Each contrast must pass the original effect, support, yearly persistence,
median and tail-sensitivity checks before a combination is a descriptive lead.
A population difference explained by volume alone remains visible as "needs
verification", not as a new combination lead. This conservative screen is not
causal interaction estimation; correlated/redundant conditions can leave too few
comparison observations. The original single-feature profiles are also retained
on their own eligible populations.

With `--combinations`, three additional tables are written in the NEW output
directory, alongside `REVIEW.md` and `evidence.json`:

- `feature_path.csv`: single-feature and combined-bucket path statistics,
  definitions, outcome counts and reference-only markers. Missing/outside groups
  remain separate rows; detailed coverage counts are in evidence.json.
- `yearly_feature_path.csv`: the same grouped metrics by calendar year.
- `combination_components.csv`: each component-alone and
  component-excluding-combination population's metric counts, means and medians,
  both pooled and yearly.

`REVIEW.md` includes a complete combination inventory, even for conditions that
do not reach the shortlist. `evidence.json` preserves full definitions, all
component comparisons/checks and the configuration file's hash. Exported path
tables can be read by the reviewer's summary-only fallback; it cannot re-verify
the row-level incremental/median/tail checks and therefore never promotes such
summary-only observations to descriptive leads. To reproduce the full review,
use the original saved rows with the same combinations JSON.

## Input selection and scope

When a directory is supplied, prefer row-level `payoffs`, `event_payoffs_df`,
`research_df`, then `research` (each `.pickle` before `.csv`). The exact selected
files appear in the manifest; files from different studies are never searched
or automatically merged. The payoff-containing row table is preferred because
it provides both path context and the economics of an already-tested exit.

An explicit feature allowlist and saved numeric bins are required. They come from
`metadata.json` (including nested `input_metadata`), or the existing
`feature_path[_df]` and `feature_bins[_df]` tables. No numeric column discovery
or fresh quantile fitting occurs. If needed, supply `--features definitions.json`:

```json
{
  "some_numeric_feature": {"description": "A feature known at entry", "edges": [0, 1, 2]},
  "some_category": {"description": "A causal state", "edges": null}
}
```

Only actually present features may be requested. Forward-outcome and known
retrospective pivot-label fields are rejected. Availability metadata is checked
when present; without it the report says provenance is unverified. This does not
re-prove arbitrary upstream formulas causal. Existing composite categorical
features may be reviewed as supplied; new combinations are constructed only when
explicitly requested through `--combinations`.

Contexts remain separate: available market identifiers, event type, outcome
configuration, direction and payoff/cost configuration. Never compare a long
12-hour group with a short 24-hour group or silently pool cost models. Missing
market identifiers are displayed as unspecified rather than guessed.

If a row table has a `split` column, only `train` is reviewed by default.
`--partition` can explicitly select another partition. Reviewing an already seen
validation/test partition is exploratory; it does not create independent evidence.
The reader respects saved full-horizon, cohort, path and validation-eligibility
flags. Unknown booleans, duplicate event/config keys or an empty eligible cohort
fail rather than silently produce zero results.

## How a candidate is assessed

Assessment is at **feature + bucket + study context + role**, not a universal
statement that an entire indicator is good. One bucket may have several roles:

1. **Directional tilt:** terminal mean differs from other buckets. Negative
   means relative underperformance, not that reversing the trade is profitable.
2. **Favorable excursion:** larger MFE; examine the adverse leg and path order.
3. **Two-sided movement:** both favorable and adverse excursions are larger.
   This is movement context, not directional evidence.
4. **Reduced adverse excursion:** MAE is less negative. Check whether favorable
   movement also shrinks; reduced activity is not automatically better execution.
5. **Tested economic improvement:** a supplied net payoff improves even when
   comparing bucket lower bounds with reference upper bounds. Absolute payoff
   may STILL be negative; the report states that separately.

Reference = every OTHER usable bucket for that feature, within the same context.
It excludes the selected bucket and missing/out-of-bin values. This is a
descriptive complement comparison, not the older inclusive baseline and not
volatility matching or causal attribution. Interpret changes from older reports
accordingly. Counts are metric-specific where outcome availability differs.

Defaults, printed in every review:

- At least 200 metric-valid observations in both bucket and reference.
- At least 50 observations in both arms within a supported calendar year.
- At least three supported years; matching difference signs in at least 2/3.
- Mean difference at least 0.05 percentage points for terminal returns, 0.2 ATR
  for excursions (0.05 percentage points if only return units exist), and 0.02
  percentage points for conservative net payoff improvement.
- Mean-versus-median contrast disagreement, absolute mean-versus-median return
  sign disagreement, and extreme-observation sensitivity prevent promotion to
  a descriptive lead. They remain visible as observations needing verification.
- Extreme sensitivity means removing the lowest/highest 1% separately from each
  arm reverses or more than halves the mean difference. This is diagnostic only,
  not a recommendation to remove historical losses or optimize exit rules.

These are transparent **review heuristics**, not fitted thresholds, economic
materiality guarantees or statistical tests. `ReviewPolicy` exposes them to
callers; CLI count/year switches expose the common support controls. Do not
silently lower requirements until a desired candidate appears.

Within a role, display order is yearly sign consistency, supported years,
smaller-arm sample size, then names. No opaque composite score or maximum-return
optimization. At most eight entries per role/status are displayed by default;
every entry remains in evidence.json. Missing/unsupported cells are not promoted.
The reviewer can return no descriptive leads at all.

## Row-level versus summary-only review

With saved observations, the reviewer can calculate exact complement means and
medians, 5th/95th percentiles, trimmed-mean diagnostics and within-year contrasts.
Net execution bounds are used only when supplied; no unambiguous-only scalar
mean is substituted as if it described all trades.

With only `feature_path[_df]`, it can recover reference MEANS using each metric's
own valid counts. It NEVER averages medians to manufacture a reference median.
Optional `yearly_feature_path[_df]` adds yearly mean contrasts; counts and saved
bin boundaries are checked for consistency. Original configuration definitions
are displayed if `configurations[_df]` is available.

Summary-only mode cannot check individual extreme observations, exact reference
medians, chronology of extremes, row-level eligibility or causal availability.
Missing/unexported feature categories cannot be reconstructed from a path table.
Such candidates remain **needs verification**, never row-verified descriptive
leads. Unsupported/ambiguous schemas fail explicitly; this is not a reader for
every arbitrary CSV. It currently consumes calendar-year tables, not arbitrary
era definitions, in table-only mode.

## What a review does NOT establish

- Adjacent events and rolling windows overlap; counts are not independent trials.
- Yearly persistence is not untouched validation or a multiple-testing correction.
- ATR-normalized differences may still reflect volatility/regime composition.
- Full-horizon MFE/MAE may occur after a saved stop/target already exited.
- Endpoint and extreme summaries cannot establish which occurred first or what
  an implementable trailing policy would realize. Even extreme timestamps do not
  reconstruct the entire executable price path.
- No confidence intervals, adjusted p-values, discovered strategy, ideal exit,
  portfolio return or new positive-payoff claim is produced.

The next question attached to a lead is a proposal for a separately approved
experiment, not an automatic rerun. Negative economics of one exit policy and
interesting path structure may coexist; neither proves the other away.

## Verification and example reviews

Focused regression tests are in
`tests/research_infrastructure/test_research_triage.py`. They cover flat-close
movement leads, two-sided risk, negative saved economics, complement medians,
tail/mean-median warnings, yearly support, completeness, missingness, fixed bins,
context separation, CSV-only fallback, unchanged inputs and overwrite refusal.

Verification of the initial single-feature implementation: 70 tests passed across
the reviewer and existing feature-descriptive-review suites. An independent check of the
all-candle example reproduced all 130 profile populations and 1,300 metric/arm
counts, means and medians directly from the saved observations. Input hashes
remained unchanged; no missing-category profile was promoted.

The manual-combinations extension passed 96 tests across the grouping, reviewer
and feature-descriptive-review suites. These additionally cover both grouping
modes, exact bounds, missing/outside values, empty matches, reference-only
complements, inherited component effects, saved outcome counts and both excursion
units, component/yearly exports, unchanged inputs and CSV summary round-tripping,
three-component groups, incremental yearly/tail checks and collision-safe labels.
The 37 existing wick-outcome and all-candle feature/payoff regression tests also
passed. An independent real-data audit reproduced 756 combination/component
metric-arm counts, means and medians; source and reviewer fingerprints matched.

Example final reviews:

- `reports/research_reviews/manual_combinations/REVIEW.md`: both manual modes
  applied to the saved all-candle study, with CSV path and component tables.
  The exact GREEN/1.25-2x volume condition intentionally duplicates one crossed
  bucket to demonstrate both interfaces, not two independent discoveries.
- `reports/research_reviews/all_candle_payoffs/REVIEW.md`: recent all-candle
  path statistics alongside already-tested half-ATR exit economics.
- `reports/research_reviews/wick_outcomes/REVIEW.md`: standard wick research
  artifacts with preserved saved bins, no study rerun.
- `reports/research_reviews/wick_summary_only/REVIEW.md`: reduced-capability
  review using only aggregated path/yearly tables.

Earlier `*_v1` review directories are development snapshots retained for audit;
use the unsuffixed final examples above. No original study report was overwritten.
