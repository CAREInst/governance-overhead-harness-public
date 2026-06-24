---
project: governance-overhead-harness
doc: README
version: 1.1.0
created: 2026-06-22
org: CARE Institute
schema_version: "1.0.0"
---

# governance-overhead-harness

This package reruns the MLSys 2027 "Cost of Governance" experiment. It is built
to a peer-review standard. It is self-contained and hardened.

It replaces the older, scattered `research/260405_GDrve_MLSys_2027/v1..v7` code.
That code is now one flat package. Imports are fixed. Provenance is tracked.

## What it measures

The harness measures the cost of an agent governance stack. It runs the same
tasks twice: with the layers off, and with the layers on. For each call it
records latency, tokens, and dollars. It then reports the harness overhead as a
share of the API latency.

## Layout

```
governance-overhead-harness/
├── code/
│   ├── harness_benchmark.py     events, logger, cost, run manifest
│   ├── governance_layers.py     the nine layers (L0-L8)
│   ├── provider_adapters.py     native cloud adapters
│   ├── extra_adapters.py        local Ollama + gated OpenRouter
│   ├── calibration.py           overhead self-test + semaphores
│   ├── validators.py            output validators
│   ├── analysis_v4.py           statistics + paper tables
│   ├── experiment_runner_v3.py  reference runner
│   └── run_experiment.py        CLI: smoke + full
├── data/task_suite.json         150 tasks (Tier 1/2/3 x 50)
├── results/                     run outputs (gitignored; tracked via manifest)
├── ARCHITECTURE.md              how the parts fit
├── CHANGELOG.md                 version history
├── MANIFEST.yaml                components + license
├── PREFLIGHT.md                 go/no-go checklist
└── OPERATORS_MANUAL.md          run procedure
```

## Quick start

Run the self-test. It needs no network and no spend.

```bash
cd code
python3 -c "import calibration; print(calibration.measure_harness_overhead(200)['conclusion'])"
```

Run a local smoke. It uses a local Ollama model. It sends nothing out and costs
nothing.

```bash
cd code
python3 run_experiment.py smoke --model ollama/<your-model> --tasks 3
```

## Live run

A full run sends data to OpenRouter and spends money. It stays shut off until
you opt in. You must pass `--allow-external` and load an API key. The gate is
shut by default. See `PREFLIGHT.md` and `OPERATORS_MANUAL.md` first.

```bash
cd code
python3 run_experiment.py full --model openrouter/anthropic/claude-sonnet-4.5 \
        --runs 5 --max-usd 40 --allow-external
```

## Reproduction target

- Full 150-task suite (Tier 1/2/3).
- Determinism pinned: temperature 0 plus a seed.
- One gateway: OpenRouter. External send is off by default.
- Published subset: Tier 1+2, which is 100 tasks.

## Provenance

Every event carries a schema version. Every run writes a manifest. The manifest
holds the git commit, the config hash, the seed, the pricing date, and the
dataset checksum. So any result ties back to exact code and inputs.

## License

Apache-2.0. See `MANIFEST.yaml`.
