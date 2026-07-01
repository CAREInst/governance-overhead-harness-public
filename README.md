---
project: governance-overhead-harness
doc: README
version: 1.3.3
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

This is governance-*orchestration* overhead: metering, attribution, policy
bookkeeping, and pre/post checks around the model call. It is not the cost of
a safety-classifier's own inference (an extra model call to judge the first
one). The two are not comparable; see `CITATIONS.md` before citing the headline
number against a guardrail system that runs extra inferences.

The suite also measures detection quality: 20 injected failures (ground truth)
and 21 purpose-built benign probes that carry governance trigger words
(`delete`, `production`, `override`, ...) in innocuous contexts, so precision
is measured against real false-positive pressure, not an easy negative set.

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
│   ├── run_experiment.py        CLI: smoke + full + sweep
│   └── make_data_summary.py     compact per-config JSONL digest
├── data/task_suite.json         171 tasks (150 Tier 1/2/3 x 50 + 21 false-positive probes)
├── results/                     run outputs (gitignored; tracked via manifest)
├── ARCHITECTURE.md              how the parts fit
├── CHANGELOG.md                 version history
├── CITATIONS.md                 grounding for modeled parameters + prior-art positioning
├── DATA_QUALITY.md              per-model data-quality audit and fixes
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
you opt in. You must pass `--allow-external` and set `OPENROUTER_API_KEY`. The
gate is shut by default. See `PREFLIGHT.md` and `OPERATORS_MANUAL.md` first.

```bash
cd code
export OPENROUTER_API_KEY=sk-or-...
python3 run_experiment.py full --model openrouter/anthropic/claude-sonnet-4.5 \
        --runs 5 --max-usd 40 --all-tiers --allow-external
```

Three knobs are recorded in the run manifest for provenance, and can be varied
to reproduce the ablations in the paper:

| Flag | Values | What it varies |
|---|---|---|
| `--preamble` | `none`, `short`, `medium`, `long` | L4 policy-preamble length — token overhead as a function of length, not one fixed point |
| `--l8-mode` | `execute` (default), `measure_only` | whether L8's plan is threaded into the executed prompt, or only its round-trip cost is measured |
| `--additivity` | flag | also runs isolated single-layer configs, for a real (non-telescoping) additivity test |

## Multi-model sweep

`sweep` mode runs the same suite across several models, for cross-model
invariance evidence. A free local Ollama leg can run alongside the paid ones:

```bash
cd code
export OPENROUTER_API_KEY=sk-or-...
python3 run_experiment.py sweep --model-set cheap --include-ollama \
        --runs 5 --all-tiers --max-usd-total 25 --allow-external
```

`--max-usd-total` is a hard cap across every model in the sweep; `--max-usd`
still caps each model individually. `--models <id> <id> ...` overrides
`--model-set` with an explicit list. See `DATA_QUALITY.md` for what to expect
if a specific route is unreliable that day — this happened during our own
collection and is documented there rather than silently retried away.

## Reproduction target

- Full 150-task suite (Tier 1/2/3), plus 21 purpose-built false-positive
  probes (`fp_negative`) for precision/recall measurement.
- Determinism pinned: temperature 0 plus a seed (live-API responses are not
  bitwise-reproducible; see `DATA_QUALITY.md`).
- One gateway: OpenRouter. External send is off by default.
- Published subset: Tier 1+2, which is 100 tasks.
- Cross-model invariance: the paper's headline detection metric
  (`precision_hard`) was reproduced identically across five models spanning
  three providers via `sweep` mode. See `CHANGELOG.md` and `DATA_QUALITY.md`
  for the exact set and each model's data-quality notes.

## Provenance

Every event carries a schema version. Every run writes a manifest. The manifest
holds the git commit, the config hash, the seed, the pricing date, and the
dataset checksum. So any result ties back to exact code and inputs.

## License

Apache-2.0. See `MANIFEST.yaml`.
