---
project: governance-overhead-harness
doc: CHANGELOG
version: 1.1.0
created: 2026-06-22
org: CARE Institute
schema_version: "1.0.0"
---

# Changelog

All notable changes to this module. Semver, newest first. One entry per milestone.

## [1.1.0] — 2026-06-22

Audit remediation and live-run pre-flight.

### Fixed
- L8 plan-reflect docstring now states it measures structural planning overhead
  only; the planning API call is excluded to isolate harness cost (C1).
- `bootstrap_ci` is reproducible: it draws from a local seeded PRNG, so
  confidence intervals are deterministic and the global PRNG is left untouched (C2).
- Harness cost rate changed from $0.01 to $0.10 per CPU-hour (amortized local
  compute); the break-even is now derived from measured full-stack overhead (H2).
- `analysis_v4` `__main__` auto-discovers provider pairs in `results/full/` instead
  of loading two hard-coded files (H3).
- The runner writes `<dataset-stem>_cost_report.json` so analysis pairs the report
  with its dataset and multiple providers can share one output dir (H3).
- OpenRouter route id corrected: `claude-sonnet-4-5` → `claude-sonnet-4.5`.
- All six OpenRouter routes synced to live catalog pricing (confirmed 2026-06-22),
  so `compute_cost` and the `--max-usd` cap track the real bill.

### Added
- Three OpenRouter models: `llama-3.3-70b-instruct`,
  `mistral-small-3.1-24b-instruct`, `claude-sonnet-4.5` (H1).
- Provider concurrency semaphores for `ollama` (1) and `openrouter` (10) (M3).

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
- `keystore`: SQLCipher-encrypted credential store (kept outside the repo).
- `data/task_suite.json`: 150 tasks across Tier 1/2/3.

### Fixed (vs the research code)
- Removed the broken `mlsys_v1`/`mlsys_v2` `sys.path` hacks.
- Fixed the `calibration.py` `statistics.percentile` crash.
- `compute_cost`/`get_model_info` raise on an unknown model instead of a silent
  Opus-rate fallback.
- Durable logging (`utf-8` + `fsync`); `allow_nan=False` rejects invalid JSON.
