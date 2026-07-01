---
project: governance-overhead-harness
doc: ARCHITECTURE
version: 1.1.0
created: 2026-06-22
org: CARE Institute
schema_version: "1.0.0"
---

# Architecture

## Overview

This module measures the cost of agent governance. It runs the same tasks with
governance off and on. For each call it records latency, tokens, and dollars.
It then reports the harness overhead as a share of the API latency.

The package is flat. Every file in `code/` imports its peers by name. There is
no nested package and no path hack. One CLI drives the whole flow.

## Component map

```
run_experiment.py        CLI entry: smoke (local) and full (gated external)
        │
        ▼
experiment_runner_v3.py  LiveExperimentRunner: the per-call measurement loop
        │
        ├── governance_layers.py   nine layers (L0-L8), block-on-first-fail
        ├── provider_adapters.py   native cloud adapters
        ├── extra_adapters.py      local Ollama + gated OpenRouter
        ├── calibration.py         overhead self-test + semaphores
        ├── validators.py          output checks after a run
        └── harness_benchmark.py   events, JSONL logger, cost, run manifest
        │
        ▼
analysis_v4.py           statistics + paper tables from the JSONL
make_data_summary.py     compact per-config digest of the JSONL
```

## Data flow

A run moves through a fixed path. The steps are simple.

1. The CLI loads tasks from `data/task_suite.json`.
2. For each task it picks a layer config. There are five configs: L0, L0-L3,
   L0-L5, L0-L7, and L0-L8.
3. It runs the governance stack on the prompt. This is the pre-API step.
4. If no layer blocks, it calls the model through an adapter.
5. It computes the cost from pinned pricing.
6. It runs the post-API output filter if that layer is on.
7. It logs one event per call to a JSONL file.

Each event holds the layer config, the harness latency, the API latency, the
token counts, and the cost. The harness latency is the time spent in the stack.
The API latency is the time spent waiting on the model.

## The governance stack

`run_governance_stack` runs the enabled layers in order. The layers are L0
through L8. L0 is a bare passthrough and sets the baseline. The other layers
add real checks: tool permission, token counting, logging, input validation,
injection detection, output filtering, a human-in-the-loop gate, and a
plan-reflect step.

Each layer returns a decision: pass, block, or modify. The stack stops on the
first block. A blocked call never reaches the model, so its API cost is zero.
This is how governance can save money when an attack is caught early.

Note on L8: it measures the structure cost of planning only. The extra
planning API call is out of scope, so the harness cost stays clean.

## Adapters and the external-send gate

The adapter factory picks a backend from the model prefix. An `ollama/` prefix
runs a local model with no network egress and no cost. An `openrouter/` prefix
uses one external gateway. Any other name uses a native adapter.

The OpenRouter path is gated. It refuses to send unless the operator opens the
gate. The gate opens only when two things are true: the run uses
`--allow-external`, and an API key is present in the `OPENROUTER_API_KEY`
environment variable. A local smoke never touches the gate.

## Cost and the budget cap

`compute_cost` prices each call from a pinned table. An unknown model raises an
error rather than charge a default rate. The OpenRouter routes carry live
catalog prices, so the recorded cost matches the real bill.

A `CostTracker` sums the spend and enforces `--max-usd`. When the cap is hit,
the runner skips the rest of the work. The cap is both a safety net and a stop.

## Provenance

Every event carries a schema version. Every run writes a manifest next to its
dataset. The manifest records the git commit, the config hash, the seed, the
pricing date, the model ids, and the dataset checksum with a record count. A
reader can therefore tie any result back to exact code and inputs.

## Analysis

`analysis_v4` reads the JSONL after a run. It scans `results/full/` and pairs
each `*_cost_report.json` with its dataset by stem. It then computes the
overhead ratios, paired token tests, layer additivity, run consistency, cost
savings, and the injection break-even. It prints two paper tables and saves a
full result file.
