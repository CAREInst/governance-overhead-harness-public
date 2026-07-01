---
project: governance-overhead-harness
doc: CHANGELOG
version: 1.3.3
created: 2026-06-22
org: CARE Institute
schema_version: "1.0.0"
---

# Changelog

## [1.3.3] — 2026-06-30

Analysis-only fix (no re-collection) for a survivorship bias found while
writing up the final cross-model detection numbers.

### Fixed
- `analyze_injection_detection` (`analysis_v4.py`) previously excluded
  API-error events from the confusion matrix entirely, discarding valid
  classification information: governance's block/pass decision happens
  *before* the API call and is already known via `injected_failure_caught`
  regardless of the call's outcome. This mechanically inflated `recall` and
  `fpr_hard` for models with a higher API-error rate (mistral-small showed a
  spurious 1.000 recall; llama-3.3-70b showed 0.833) versus the 0.800 shown
  by the three 0%-error models. Fixed by classifying API-error events the
  same as any other event; `api_errors_disclosed` is now purely a
  route-reliability diagnostic, not a matrix exclusion.
- **Result:** all five models now show identical `recall=0.800,
  precision_hard=0.7273, fpr_hard=28.57%` — the detection metric is fully
  cross-model consistent with no per-model caveat needed. See
  `docs/DATA_QUALITY.md` for the verification.
- Re-ran `analysis_v4.py` on the existing `results/full` and `results/sweep`
  datasets (zero new API calls, zero cost) to regenerate the corrected
  output.

## [1.3.2] — 2026-06-30

Reversed the v1.3.1 decision to re-collect mistral-small-3.1-24b-instruct.
Two supervised re-run attempts (with the usage.include=true fix AND the
retry-budget bump to 5) both hit persistent, ongoing "no usage data" failures
directly from OpenRouter for this specific route — not resolved by either
fix. A live diagnostic (90s supervised window, unbuffered output) confirmed
the failures are a **current, active OpenRouter-side issue** for this route,
not a transient blip and not fixable client-side: retries exhausted their
full backoff budget (up to 31s/call) with 0/5 successes in one window and
5/5 successes in an earlier, shorter retest — the reliability is volatile and
unpredictable, making the true completion time for a clean run unknowable
(potentially far longer than the ~7.6h estimate, which assumed a fixable
client-side bug).

**Decision: treat mistral-small identically to llama-3.3-70b** — keep the
original archived dataset (2,947/4,275 = 69% API-error rate) rather than
continue chasing an unreliable external dependency. This is symmetric with
the llama decision and for the same reason: the detection metric (after the
`is_blocked()` API-error-exclusion fix) already shows `precision_hard =
0.727` on mistral's original data — identical to all 4 other models — so the
paper's cross-model invariance finding does not depend on a clean re-run.
The `usage.include=true` fix and retry-budget bump remain in the codebase
(they measurably help other routes / other days) but are not, on their own,
sufficient to guarantee clean collection from this specific route on this
occasion.

All notable changes to this module. Semver, newest first. One entry per milestone.

## [1.3.1] — 2026-06-29

Post-collection audit of the v1.3.0 multi-model sweep (Run A: sonnet-4.5;
Run B: gpt-4.1-mini, gemini-2.5-flash, llama-3.3-70b, mistral-small). Two
infrastructure defects found and fixed; two model datasets re-collected under
the fix. No governance logic, prompts, or detection criteria changed — these
are response-parsing/retry fixes only, applied uniformly across all adapters.

### Fixed
- **OpenRouter missing `usage` crash (`extra_adapters.py`):** some upstream/
  open-weight routes omit `usage` in the chat-completion response unless
  explicitly requested, causing `resp.usage.prompt_tokens` to crash on `None`.
  Root cause of **100% of mistral-small's 2,947 failures** (69% of its run) and
  12 of llama-3.3-70b's failures. Fixed by requesting `usage: {include: true}`
  and raising a distinct, clearly-labeled error if usage is still absent
  (so this failure mode can never again be silently conflated with a
  governance decision). Validated with a live smoke call before re-running.
- **Retry budget too low for an overloaded route (`provider_adapters.py`):**
  `MAX_RETRIES` 3→5, applied to the shared adapter base class (uniform across
  all providers/models). 532 of llama-3.3-70b's failures were genuine
  `Connection error` after exhausting all 3 prior attempts — a persistently
  overloaded backend, not a code defect. The larger retry budget only affects
  calls that were previously failing outright; it does not change the
  measured latency semantics of any call that already succeeded.

### Re-collected
- **mistral-small-3.1-24b-instruct**: 100% of its 2,947 failures were the
  usage-crash bug — a clean re-run is fully justified. Original dataset
  archived to `results/sweep_archive/` (not deleted) and re-run in full under
  the fixed code, so the final dataset is homogeneous (single code version,
  single continuous collection), not a splice of old+new data.

### Kept, not re-collected (operator decision)
- **llama-3.3-70b-instruct**: retained its original dataset (546/4,275 =
  13% API-error rate, mostly genuine backend connection flakiness, not the
  code bug — only 12 were). A clean re-run was estimated at ~17–19h (the
  route's *inherent* per-call latency is ~13.5s, independent of retries) with
  no guarantee the connection errors fully resolve. Not re-run because: (1)
  after the `is_blocked()` API-error-exclusion fix, its detection metric
  already shows precision_hard=0.727 — identical to all 4 other models,
  confirming the cross-model invariance finding without a clean re-run; (2)
  latency/overhead metrics are computed only on completed calls and are
  already unaffected by the errors. Reported with the 13% error-rate
  disclosed as a data-quality caveat (see `docs/DATA_QUALITY.md`).

### Disclosed (not re-collected — see `docs/DATA_QUALITY.md`)
- gemini-2.5-flash (1.4% of calls) and gpt-4.1-mini (0.07%) returned
  zero-value (not missing) `usage` on a small fraction of completed calls —
  a distinct, lower-severity OpenRouter quirk. Does not affect latency,
  governance, or detection metrics; already excluded from token/cost
  aggregates by existing truthy-filters in the analysis code. Below the bar
  for a re-run; disclosed for transparency.
- Run A (sonnet-4.5, all 27,680 events) audited: zero duplicates, zero API
  errors, zero anomalous completions. No action needed.

## [1.3.0] — 2026-06-27

Second-round remediation: closes the residual items raised by external peer
review, so a single enhanced re-run produces defensible, claim-backing
evidence. All code changes are offline ($0); the multi-model/ablation data
needs the enhanced re-run. Grounding citations in `CITATIONS.md`.

### Added
- **FP negative set:** 21 purpose-built benign probes (FP-001..021) carrying
  governance trigger words in innocuous contexts (`tier="fp_negative"`). Disposition
  self-test confirms 5 designed L7 FPs + 1 L5 FP + 4 L6 over-redactions + 15 correct
  passes. `analyze_injection_detection` now reports `precision_hard`/`fpr_hard` over
  the 21 hard negatives (precision is no longer the artifactual 1.00) plus
  `precision_all`, per-stress-target blocks, and L6 over-redaction as a separate track.
- **HITL grounding + sweep:** default re-anchored to lognormal median 3 min
  (PagerDuty MTTA p50), σ 1.0 (SOC triage); env-overridable; cited in the manifest.
  `analyze_hitl_sensitivity` runs an OFFLINE 24-cell {median×σ} sweep over logged L7
  escalations — no new spend.
- **Preamble-length ablation:** `POLICY_PREAMBLES` {none,short,medium,long}
  via `--preamble`/`L4_PREAMBLE`, stamped into the dataset stem;
  `analyze_preamble_overhead` reports token overhead as a function of preamble length
  (per model), defusing the "arbitrary constant" objection.
- **Real plan-then-execute L8:** the planning call's output is threaded into the
  executed main prompt under `--l8-mode execute` (vs `measure_only` = the prior
  bill-twice baseline); `plan_chars`/`plan_injected` fields; SCHEMA → harness-1.2.
- **Multi-model sweep:** `sweep` mode with CHEAP_SET/FULL_SET, `--include-ollama`,
  per-model + `--max-usd-total` caps; `analyze_cross_model_invariance` (min/max/CV of
  the harness/API ratio per config) turns the "single model / provider invariance"
  caveat into measurable evidence.

### Fixed
- Runner no longer drops `fp_negative` probes via the integer-tier filter (they now
  run through every config so false positives are actually measured).

## [1.2.0] — 2026-06-26

External peer review remediation. Two tracks: analysis corrections on the
existing dataset (no new spend), and new functionality so the harness
measures real governance I/O rather than non-mutating stubs.

### Fixed (analysis — recomputed on the existing dataset)
- Injection-detection rate used an all-tier caught count over a Tier-2
  denominator (reported 32%). Now a real task-attempt confusion matrix vs the
  task-suite ground truth: precision 1.00, recall 0.55/0.80/0.80 for
  L0-L5/L0-L7/L0-L8; corrected Tier-2 detection 12%. Names the 4 never-caught
  injected tasks (T2-003/004/007/008).
- ANOVA now emits a p-value and eta² (exact F-distribution tail, no scipy).
  Run effect is significant (p=0.0083) but negligible (eta²=0.39%); the
  "no run effect" framing is corrected.
- Harness-overhead CIs use a clustered bootstrap (resampled by task), and a
  survivorship-paired variant restricts to task-runs completing in all five
  configs. Denominator variants reported.
- `analyze_layer_additivity` relabeled descriptive — the cumulative
  decomposition telescopes and is NOT an additivity test.
- Break-even reframed as a compute-only lower bound with a $/CPU-hr sensitivity
  table; overhead-vs-API-latency table added; savings documented as a
  selection artifact, not a model-independent economic result.
- `data_summary` field `governance_blocks` (16,897 = all layer events) renamed
  `governance_events`, with a separate `block_decisions` (215).

### Added (new functionality — VALIDATED ON LOCAL OLLAMA ONLY; pending paid re-run)

> ⚠️ Provenance: the items below are implemented and validated end-to-end on a
> local Ollama smoke run, but have **no paid-model data yet** (the v1.2.0 paid
> re-run is pending). Specifically, "non-zero token overhead", "L0-L8 cost >
> L0-L7", and the "2.72% additivity interference" figure are **Ollama-only /
> pending** and must not be cited from the v1.0 dataset, which was produced by
> the older non-mutating code.
- Governance now transforms I/O: L4 prepends a policy preamble (real, measurable
  token cost — token overhead is no longer zero by construction); L5 gains a
  sanitize mode that neutralizes injection spans in the prompt; L6 actually
  redacts detected PII in the model response.
- Layers carry an execution phase; L6 runs post-API only, removing the
  pre/post double-count.
- L8 plan-reflect makes a real planning API round-trip, measured and
  cost-tracked separately, so L0-L8 cost differs from L0-L7.
- L7 escalations attach a modeled human-review latency (`hitl_wait_ms`,
  seeded lognormal), reported separately from harness cost.
- Isolated single-layer configs (`--additivity`) enable a genuine,
  non-telescoping additivity test; Ollama validation showed 2.72%
  interference on a small sample.
- New event fields: `planning_api_ms`, `planning_cost_usd`, `hitl_wait_ms`,
  `token_overhead`, `prompt_modified`, `output_modified`.
- `make_data_summary.py` committed (was an ad-hoc script).

### Note
- Adapter-wiring comment in `extra_adapters.py` corrected to describe the
  `run_experiment.py` monkeypatch.

## [1.1.0] — 2026-06-22

Audit remediation and live-run pre-flight.

### Fixed
- L8 plan-reflect docstring now states it measures structural planning overhead
  only; the planning API call is excluded to isolate harness cost.
- `bootstrap_ci` is reproducible: it draws from a local seeded PRNG, so
  confidence intervals are deterministic and the global PRNG is left untouched.
- Harness cost rate changed from $0.01 to $0.10 per CPU-hour (amortized local
  compute); the break-even is now derived from measured full-stack overhead.
- `analysis_v4` `__main__` auto-discovers provider pairs in `results/full/` instead
  of loading two hard-coded files.
- The runner writes `<dataset-stem>_cost_report.json` so analysis pairs the report
  with its dataset and multiple providers can share one output dir.
- OpenRouter route id corrected: `claude-sonnet-4-5` → `claude-sonnet-4.5`.
- All six OpenRouter routes synced to live catalog pricing (confirmed 2026-06-22),
  so `compute_cost` and the `--max-usd` cap track the real bill.

### Added
- Three OpenRouter models: `llama-3.3-70b-instruct`,
  `mistral-small-3.1-24b-instruct`, `claude-sonnet-4.5`.
- Provider concurrency semaphores for `ollama` (1) and `openrouter` (10).

### Changed
- `PRICING_DATE` bumped to 2026-06-22 for live-run manifest provenance.
- Removed dead `l5_cost` assignment in `analyze_cost_savings`.

## [1.0.0] — 2026-06-22

Initial canonical hardened rerun package. Collapses the scattered
`research/260405_GDrve_MLSys_2027/v1..v7` code into one flat, import-correct
package with provenance.

### Added
- `harness_benchmark`: measurement events, durable JSONL logger, cost, run manifest.
- `governance_layers`: the nine layers (L0-L8) with a block-on-first-fail stack.
- `provider_adapters` + `extra_adapters`: native, local Ollama, and gated OpenRouter.
- `calibration`: harness-overhead self-test and per-provider semaphores.
- `validators`, `analysis_v4`, `experiment_runner_v3`, `run_experiment`.
- `data/task_suite.json`: 150 tasks across Tier 1/2/3.

### Fixed (vs the research code)
- Removed the broken `mlsys_v1`/`mlsys_v2` `sys.path` hacks.
- Fixed the `calibration.py` `statistics.percentile` crash.
- `compute_cost`/`get_model_info` raise on an unknown model instead of a silent
  Opus-rate fallback.
- Durable logging (`utf-8` + `fsync`); `allow_nan=False` rejects invalid JSON.
