# OPERATORS MANUAL — governance-overhead rerun

This package reruns the MLSys 2027 "Cost of Governance" experiment: 150 tasks
(Tier 1/2/3) x 5 layer configs (L0, L0-L3, L0-L5, L0-L7, L0-L8) x N runs,
measuring per-layer governance latency/cost vs. live model calls.

Always start with **PREFLIGHT.md**.

## Smoke run (local, zero spend) — do this first

```bash
cd code
python3 run_experiment.py smoke --model ollama/<model> --tasks 3
```

- Runs the full 5-config governance sweep on a few Tier-1 tasks against a local
  Ollama model. No external send, no cost.
- Output: `results/smoke/<provider>_<model>.jsonl` + `manifest.json`.
- Must print `SMOKE: PASS` (events > 0, api errors 0, cost $0) before going live.

## Live run (external, spends money) — operator-gated

Prerequisites: smoke passed, `OPENROUTER_API_KEY` set in your environment,
and your explicit go.

```bash
# Reproduce the published 3 providers (run each; pin the model id you intend):
python3 run_experiment.py full --model openrouter/anthropic/claude-haiku-4.5 \
        --runs 5 --max-usd 5 --all-tiers --allow-external
python3 run_experiment.py full --model openrouter/openai/gpt-4.1-mini \
        --runs 5 --max-usd 5 --all-tiers --allow-external
python3 run_experiment.py full --model openrouter/google/gemini-2.5-flash \
        --runs 5 --max-usd 5 --all-tiers --allow-external
```

- `--all-tiers` runs all 150 tasks (Tier 1/2/3). Drop it for the published 100
  (Tier 1+2 only).
- `--max-usd` is a **hard stop**: the run saves partial results and stops when
  the cap is hit.
- The gate (`EXPERIMENT_ALLOW_EXTERNAL`) is opened in-process only after the key
  loads and `--allow-external` is passed; nothing leaks on a local run.
- Determinism: `--seed` (default 42) + `temperature=0` are pinned and recorded
  in the manifest.

## Outputs & provenance

Each run writes, in `results/<smoke|full>/`:
- `<provider>_<model>.jsonl` — one MeasurementEvent per line.
- `manifest.json` — git commit, config hash, seed, pinned pricing date, model
  ids, and the dataset's **SHA-256 + record count** (the integrity anchors a
  reviewer checks).
- `cost_report.json` (full runs).

## Analysis

```bash
python3 analysis_v4.py ../results/full        # statistical analysis over the run dir
```

## Monitoring & troubleshooting

- Progress + cost print as the run proceeds; a `heartbeat.json` updates every 50
  calls.
- `compute_cost` / `get_model_info` **raise** on an unrecognized model id — pin
  the exact id (or register it with `extra_adapters.register_openrouter_model`).
- `NaN`/`Infinity` in an event **raises** at serialization — by design, so the
  dataset stays valid JSON.
- Re-running overwrites by appending to the same dated file; start a clean
  `results/` for a fresh dataset.

## What stays manual (peer-review integrity)

- Pinning the exact OpenRouter route ids + pricing for the final table.
- Reconciling the rerun's per-layer numbers against the published v3/v4 data.
- The decision to publish/push results.
