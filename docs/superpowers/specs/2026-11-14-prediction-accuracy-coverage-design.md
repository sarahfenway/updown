# Predictions: right first time, for nearly every fault

Date: 2026-11-14
Status: awaiting review

## Goal

Two things, in this order of priority:

1. **Trust.** Any prediction shown to a user must be right first time more
   than 90% of the time. We aim for about 96% (roughly 1 miss in 25).
2. **Coverage.** Nearly every new unplanned fault gets a "fixed by" date,
   including the ones the model expects to last longer than two days and
   planned maintenance that has no stated end date.

"Right first time" means the first date shown for a fault held: the fault was
fixed at or before it, with the existing 30 minutes of grace. Nothing shown
later can turn a miss into a win.

## Where we are today

Live, 14 Nov 2026 (`/stats/`, `/api/incidents/`):

- 85% of predictions are fixed by the predicted time. The model aims at 75%.
- Of 24 open faults, 5 show a date, 11 show "likely to be some time"
  (right 61-82% of the time), and 8 show nothing.
- New faults whose 75% point falls in the open-ended "2+ days" class get no
  prediction at all.
- Anything with "planned" in its text is never predicted, although 80% of it
  is undated "planned maintenance" that behaves like an ordinary fault.

## Decisions made with Sarah

| Question | Decision |
|---|---|
| What matters most | Cover more faults, but never at the cost of accuracy |
| Accuracy bar | More than 90% right first time for everything shown |
| How rare should a miss be | About 1 in 25 (aim 96%), accepting later dates |
| After a miss | Admit it on the fault. No second date. |
| Long-running faults | No "likely to be some time" (it cannot reach 90%) |
| Model choice | Left to the implementer; must fit a 512 MB machine |

## Evidence

Backtest on the public export (`/api/incidents.tsv`, 21,254 incidents since
April 2023). Trained on everything resolved before 14 Nov 2025, tested on the
7,651 unplanned faults that started in the following 12 months.

First prediction, by the probability the bound aims for:

| Aim | Faults given a date | Right first time | Typical date | 1 in 10 dates beyond |
|---|---|---|---|---|
| 0.75 (today), no tail table | 91.6% | 84.5% | 0.75 d ahead | under 2 d |
| 0.75 with tail table | 100% | 84.4% | 0.75 d ahead | 1.75 d |
| 0.90 with tail table | 100% | 93.2% | 1.25 d ahead | 4.0 d |
| 0.93 with tail table | 99.8% | 95.1% | 1.75 d ahead | 5.5 d |
| **0.95 with tail table** | **99.4%** | **96.3%** | **2.5 d ahead** | **7.25 d** |

A different model family (a discrete-time survival model) was tested against
the current 9-class classifier plus a tail table. At an equal hit rate it was
no tighter, and it was less well calibrated. The classifier stays.

Other findings that shaped the design:

- TfL's own stated end dates held only 86% of the time (31 of 36 parseable
  cases), which is below the bar. The date is already in the fault text and
  in the API's `end_time`, so dated planned work gets no prediction.
- "Still open a day later" is true 53-85% of the time depending on age. No
  version of the "some time" claim reaches 90%.
- For a fault that has already overrun, a 95% date is two to three weeks
  away. Sarah chose to show no second date instead.

Limits of the backtest: it is a re-implementation from the public export,
without network flags, report counts or the historical-baseline blend. It
compares options reliably but does not forecast exact live numbers. The
retrain's own cross-validation and the live display rule are what enforce
the bar.

## Design

### 1. Training (`ml/train.py`)

- `PREDICTION_COVERAGE_TARGET` rises from 0.75 to 0.95.
- **Exclusion rule narrows.** Drop a row only when it has a stated end:
  `information` is true, or the text contains "until ". Undated planned
  maintenance is kept and trained on like any other fault. `train.py`
  computes this itself from `information` and `text`; the API's
  `is_planned_work` field stays as it is but is no longer used.
- **Tail table.** From the training rows, count resolved faults by block
  offset for every offset from `MAX_OFFSET_CLASS` (8) up to a 120-block
  horizon (30 days), per category (`faulty_lift`, `staff_issue`,
  `planned_maintenance`, `other`) and for all categories together. Faults
  that lasted beyond the horizon go in an overflow count so their share of
  the tail is never assigned to a date.
- **Bundle.** The table ships in the uploaded file as `tail_distribution`.
  `metadata.feature_version` becomes 4.
- **Cross-validation scores the real thing.** Each fold builds its own tail
  table from its training part, decodes exactly as the app does (section 2),
  and scores against the raw, uncapped block offset. It reports the share of
  faults given a date, right-first-time overall, and right-first-time per
  category.
- **Gates.** The retrain refuses to upload when overall right-first-time
  (with grace) is below 0.93, or when any category with at least 100
  cross-validation samples is below 0.90. `retrain.yml` passes these as
  defaults. The existing no-regression gate is unchanged.

### 2. Prediction (`incidents/ml.py`)

- `PREDICTION_COVERAGE_TARGET` is 0.95 here too, kept in sync as today.
- **Extended decode.** `_decode_offset_bound` builds one cumulative curve:
  - for offsets below 8, the blended class probabilities, as now;
  - from offset 8 onward, the probability of resolving before 8 plus the
    model's "2+ days" probability multiplied by the tail table's cumulative
    share at that offset. The category's table is used when it holds at
    least 30 faults, otherwise the all-categories table.
  The bound is the first offset where the curve reaches 0.95. If it is not
  reached within the horizon there is no prediction.
- **Day-level bounds.** A bound 8 or more blocks out is rounded up to the
  last block of that service day (the overnight block ending 05:00). The
  label names the day, not a five-hour block: "by end of Tuesday", or
  "by end of Sat 21 Nov" beyond six days.
- **Planned work.** With a version 4 bundle, only faults with a stated end
  (`information`, or "until " in the text) are skipped. A version 3 bundle
  keeps the old rule, because that model never saw planned maintenance.
- **No tail table in the bundle** (version 3): decode behaves as today at
  the new target, so fewer faults get a date until the retrain lands.
- **Upload validation** accepts version 4 and rejects a version 4 bundle
  that has no `tail_distribution`.
- `predict_remaining_duration` is deleted (section 4).

### 3. When a prediction is shown (`incidents/views.py`)

The existing mechanism stays: predictions are grouped by confidence bucket
and a bucket is shown only when its recent record is good enough, using the
last 30 days and falling back to 90 then 180. Two constants change:

- minimum samples: 15 becomes 30;
- the test becomes "observed right-first-time rate is at least 90%",
  replacing "Wilson lower bound at least 55%".

At a 0.95 aim every new prediction lands in the 90-100% bucket, so this is
in effect one rule for the whole site. Predictions made by the old model sit
in the 70-90% buckets, have an 85% record, and drop out on their own.

Removed because they contradict the 90% rule:

- the station override that let a 40-50% bucket show at 65% accuracy;
- the "likely to be some time" policy, its age bands and its replay.

### 4. After a miss

When an open fault's predicted time has passed (beyond the 30 minutes of
grace) and its prediction was one the site would show:

- the fault shows **"Taking longer than we predicted"**, with a tooltip
  saying what was predicted and the recent right-first-time rate;
- no new date is offered, ever, for that fault.

Supporting changes:

- Revised predictions are removed: `refresh_revised_predictions`,
  `predict_remaining_duration`, `prediction_is_revised` and the "now
  expected by" wording. The two `revised_*` columns stay in the database
  unread and unwritten; dropping them is a separate cleanup so this change
  needs no migration.
- **A first prediction is final.** Today a prediction is recomputed whenever
  TfL edits the fault text, which silently replaces the first promise.
  `refresh_prediction` will run only when the fault has no prediction yet.
- `predict_durations` never stores a first prediction whose time has already
  passed, so a backfill cannot make the site claim a prediction it never
  showed.

### 5. API (`/api/incidents/`)

- The `prediction` object keeps its shape. `revised` is always false and
  `kind: "some_time"` is no longer emitted.
- A past-due prediction is still omitted, as now. Each issue gains a boolean
  `prediction_overdue`, so the iOS app can show the same admission. Older
  app versions ignore the field.
- `?predictions=all` is unchanged: it is an explicit opt-in and marks what
  it adds as `reliable: false`.

### 6. Pages

- `home.html` and `beta.html`: day-level wording, the overdue message, and
  removal of the revised and long-running branches.
- `stats.html`: the headline becomes "Right first time". The explanatory
  text states the 90% rule, the 30-sample minimum and the 96% aim. The
  Long-Running Faults section and the station-override paragraph go.
  "Hidden: predicted time has passed" becomes "Taking longer than
  predicted". The existing worst-case line, which counts open overdue
  predictions as misses, stays.

### 7. Memory and speed

The tail table is about 600 integers. The model keeps 9 classes, so its size
and load time are unchanged. Removing the long-running replay removes an
hourly scan of 180 days of incidents. No new runtime dependencies.

## Rollout

1. Merge and deploy. Until a new model is uploaded the old one keeps
   working, at the stricter aim and with fewer dates.
2. Run the retrain workflow with `dry_run` and read the cross-validation
   figures. Expect about 96% overall. If lifts fall below 90%, raise the aim
   in both files before going further.
3. Run the retrain for real.
4. Run `predict_durations` (without `--overwrite`) so open faults that have
   no prediction get one.
5. Expect two to three days with few predictions on the site, until 30
   new-style predictions have resolved and the 90-100% bucket has a record.
6. Watch `/stats/`: right-first-time should settle near 96%.

## Testing

Test-first, in the existing suites.

- `ml/test_train.py`: the narrowed exclusion rule; tail table contents,
  overflow and per-category fallback; cross-validation scoring against
  uncapped offsets; both gates refusing an upload.
- `tests/test_incidents.py`:
  - extended decode: bound inside the classifier range; bound in the tail;
    no bound within the horizon; category fallback below 30 faults; day
    rounding; a version 3 bundle behaving as before;
  - planned-work rule per bundle version;
  - display rule at 29 and 30 samples, and at 89% and 90%;
  - overdue state: shown only when the bucket is shown, never with a date;
  - a text edit does not replace an existing prediction;
  - `predict_durations` skipping past-due bounds;
  - API: `prediction_overdue`, no `some_time`, `revised` false;
  - upload validation for version 4 with and without a tail table.

## Not in scope

- Dated planned work (TfL's dates are below the bar and already in the text).
- Clock-time precision instead of blocks and days.
- Any estimate for a fault after its first prediction has been missed.
- Dropping the `revised_*` columns.
- New data sources that could make dates sooner at the same accuracy.

## Risks

- **Dates get later.** A lift usually fixed within eight hours will often
  say "by end of Thursday". That is the accepted price of 96%.
- **Live accuracy may differ from the backtest.** The training gates stop a
  weak model shipping, and the display rule hides predictions if the live
  record drops below 90%.
- **A visible gap after launch** of two to three days (rollout step 5).
