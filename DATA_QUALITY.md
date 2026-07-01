---
project: governance-overhead-harness
doc: DATA_QUALITY
created: 2026-06-29
org: CARE Institute
---

# Data Quality Notes — Multi-Model Sweep (v1.3.3)

Disclosed anomalies found during a full post-collection audit of the 5-model
sweep (Run A: sonnet-4.5; Run B: gpt-4.1-mini, gemini-2.5-flash, llama-3.3-70b,
mistral-small). See `CHANGELOG.md` v1.3.1–v1.3.3 for the code fixes and the
final collection decisions.

## FIXED: `recall`/`fpr_hard` survivorship bias in `analyze_injection_detection`

An earlier version of this analysis excluded API-error events from the
confusion matrix entirely (`continue`, before classification). This
mechanically inflated **recall** and **fpr_hard** for models with a higher
API-error rate — e.g. mistral-small (69% error rate) showed an apparent
"perfect recall" of 1.000 and llama-3.3-70b (13%) showed 0.833, both
above the 0.800 shown by the three 0%-error models.

**Root cause:** governance's block/pass decision is made *before* the API
call and is therefore fully known regardless of whether the subsequent API
call succeeds — `injected_failure_caught` is already correctly `False` on an
API-error event (we only reach that code path when governance did *not*
block the call). Excluding these events discarded valid classification
information rather than using it.

**Fix (analysis-code only — no re-collection needed):** API-error events are
no longer skipped; they are classified via the same `is_blocked()` logic as
every other event (already correct), with `api_errors_disclosed` retained
purely as a route-reliability diagnostic, not a matrix exclusion. Verified
directly against the raw data before applying the fix: for mistral-small,
100% of its "missing" positive-task attempts were the same 4 structurally
never-caught tasks (T2-003/004/007/008) with `injected_failure_caught=False`
— unambiguously false negatives.

**Result — full cross-model consistency, no caveat needed:**

| Model | API-error rate | tp | fn | recall | precision_hard | fpr_hard |
|---|---|---|---|---|---|---|
| sonnet-4.5 | 0% | 80 | 20 | 0.800 | 0.7273 | 28.57% |
| gemini-2.5-flash | 0% | 80 | 20 | 0.800 | 0.7273 | 28.57% |
| gpt-4.1-mini | 0% | 80 | 20 | 0.800 | 0.7273 | 28.57% |
| llama-3.3-70b | 13% | 80 | 20 | 0.800 | 0.7273 | 28.57% |
| mistral-small | 69% | 80 | 20 | 0.800 | 0.7273 | 28.57% |

**All five detection metrics are now identical across all five models**,
confirming the underlying detection behavior is fully deterministic and
provider-invariant — this can be cited directly in the paper without any
per-model asterisk. The elevated API-error rates for llama/mistral remain
disclosed as a route-reliability data point (relevant to *latency* sample
size, not detection accuracy).

## Audit method

For every dataset: checked for duplicate `(task_id, run_number, layer_config)`
keys, empty/zero-token completions, incompletion-reason breakdown (governance
block vs. API error), token-count outliers, timestamp gaps >5 min, and cost
sanity (negative/anomalous values).

## Findings

| Model | API errors | Zero-usage completions | Duplicates | Action |
|---|---|---|---|---|
| claude-sonnet-4.5 (Run A) | 0 | 0 | 0 | none needed |
| gpt-4.1-mini | 0 | 3 / 4,275 (0.07%) | 0 | disclosed only |
| gemini-2.5-flash | 0 | 62 / 4,275 (1.4%) | 0 | disclosed only |
| llama-3.3-70b | 546 / 4,275 (13%) | 0 | 0 | **disclosed, kept** (see below) |
| mistral-small | 2,947 / 4,275 (69%) | 0 | 0 | **disclosed, kept** (see below) |

## mistral-small-3.1-24b-instruct — two re-run attempts, both reverted

Root cause (confirmed): OpenRouter omits `usage` in the chat-completion
response for this route unless `usage: {include: true}` is requested — fixed
in `extra_adapters.py` and validated with a live smoke call (real usage
parsed correctly). A first re-run attempt then hit 59/60 failures with a
*different* symptom: usage was requested but OpenRouter still returned none.

This was initially treated as a transient blip (an immediate 5-call retest
came back 5/5 clean) and addressed by adding "no usage data" to the
retryable-error set in `provider_adapters.py`. A second, supervised re-run
attempt — run with unbuffered output and watched live — showed the failures
recurring persistently: a 90-second window produced repeated retry cycles
(up to 5 attempts, ~31s backoff each) with 0 successes, followed by a
separate 5-call check with 5/5 failures. The reliability of this route's
usage-accounting is **volatile on an unpredictable timescale** — neither a
one-shot nor a several-retry strategy reliably gets through it, and there is
no way to bound how long a full clean run would take.

**Decision: keep the original archived dataset**, for the same reason as
llama below — `precision_hard = 0.727` on the ORIGINAL flawed data (after
`is_blocked()` excludes API-error calls from the confusion matrix) is already
identical to the other 4 models, so the cross-model invariance finding does
not require a clean mistral dataset. The two code fixes
(`usage.include=true`, retry-budget bump 3→5) remain in the codebase — they
are correct, uniform improvements that may help on other days or other
routes — but this specific route's live behavior on 2026-06-29/30 was not
resolved by either.

## llama-3.3-70b — disclosed API-error rate, not re-collected

13% of calls failed (532 genuine connection errors after 3 retries + 12
usage-crashes). Root cause is mostly a persistently overloaded OpenRouter
backend (inherent ~13.5s/call, independent of retries), not a code defect. A
clean re-run was estimated at ~17–19h with no guarantee the connection
flakiness fully resolves. Decision: **keep the existing dataset**, because:

1. After the `is_blocked()` fix (analysis_v4.py) excludes API-error calls from
   the confusion matrix, llama's detection metric already shows
   `precision_hard = 0.727` — identical to all 4 other models. The
   cross-model invariance finding does not depend on a clean re-run.
2. Latency/overhead metrics are computed only on completed calls; the error
   rate does not bias them, only reduces llama's effective sample size
   (3,449 completed calls — still ample for stable per-config means).

Report llama's data with this 13% API-error rate disclosed as a per-route
reliability caveat, distinct from — and much higher than — the other four
models' near-zero error rates.

Across every model, the **280 governance-layer block decisions per config are
identical** — confirming pre-API governance (L4/L5/L7) is deterministic on the
governed prompt text, independent of which model answers.

## Zero-usage completions (gemini-2.5-flash, gpt-4.1-mini) — not re-collected

A small fraction of completed calls returned `usage.prompt_tokens = 0,
usage.completion_tokens = 0` from OpenRouter without erroring (distinct from
the `usage = None` crash fixed in v1.3.1 — this is a *populated-but-zero*
report, not a missing one). Effects:

- **Latency/overhead metrics: unaffected.** Wall-clock timing is measured
  independently of the usage payload.
- **Governance/detection metrics: unaffected.** Blocks are decided before the
  API call.
- **Token-overhead aggregate: self-limiting.** `analyze_preamble_overhead`'s
  `mean_in()` and `make_data_summary.py`'s token aggregates use a truthy
  filter (`if e.get("input_tokens")`), which already excludes these zero
  records rather than treating them as valid zero-token observations.
- **Per-task L0 baseline: guarded.** `run_experiment.py` only records an L0
  baseline `if event.input_tokens > 0`; when it's not set, `token_overhead`
  for that task's later configs defaults to 0 (a minor undercount for that
  specific task, not an inflated/corrupted value).
- **Cost: negligible undercount.** These calls are billed as $0 in our
  tracker; true OpenRouter-side cost for ~65 calls total is immaterial
  against the ~$34 total spend.

Given the low prevalence (≤1.4%) and zero effect on the paper's primary
metrics, these were disclosed rather than re-collected.

## Reproducibility note

`sweep --models` writes append-only per-model files. Before any re-collection,
the prior (defective) dataset for that model must be moved out of the target
directory — otherwise the logger appends new data onto the old, producing a
non-homogeneous mixed-code-version file. `results/sweep_archive/` preserves
every superseded dataset for audit traceability; nothing is deleted.
