---
project: governance-overhead-harness
doc: CITATIONS
created: 2026-06-27
org: CARE Institute
---

# Citations & Positioning

Grounding for the modeled/assumed parameters and the prior-art positioning, so
reviewers see the choices are sourced, not arbitrary.

## Critical positioning (avoid "too good to be true")

The harness measures **governance-orchestration overhead** (metering, attribution,
policy bookkeeping, pre/post checks) — explicitly distinct from **safety-classifier
inference overhead**. The headline ~0.0078% is NOT comparable apples-to-apples
with guardrail systems that run extra model inferences (e.g. NeMo Guardrails
~+58% end-to-end latency). State this distinction prominently or the figure reads
as implausible.

## Guardrail / governance overhead baselines (the cost envelope)

- **NVIDIA NeMo Guardrails** — end-to-end latency 0.91s → 1.44s (~+58%),
  throughput 112.9 → 98.7 tok/s, policy detection 75% → 98.9%.
  https://developer.nvidia.com/blog/measuring-the-effectiveness-and-performance-of-ai-guardrails-in-generative-ai-applications/
- **LlamaFirewall (arXiv 2505.03574)** — PromptGuard 2: AUC 0.98, Recall@1%FPR
  97.5%, 19–92 ms; CodeShield ~60 ms; AgentDojo ASR 17.63% → 1.75%.
  https://arxiv.org/abs/2505.03574
- **Adversarial Prompt Evaluation (arXiv 2502.15427)** — 15 guardrails; in- vs
  out-of-distribution collapse (Llama-Guard 2 precision 94.6% → 45.3%); motivates
  the hard-negative / OOD split. https://arxiv.org/abs/2502.15427
- **HiddenLayer — Evaluating Prompt Injection Datasets** — label noise/staleness;
  justifies DEFINING our own fp_negative set rather than reusing a public corpus.
  https://www.hiddenlayer.com/research/evaluating-prompt-injection-datasets

## HITL (L7) modeled human-review latency — lognormal, default median 3 min, σ 1.0

Lognormal is justified: right-skew/heavy-tail is universal across human-review
reference classes, and PR-latency studies log-transform latency for exactly this.
Default anchored to the fast acknowledge/triage tier; sensitivity swept over
median {1,3,5,15,60,480} min × σ {0.5,1,1.5,2}.

- **PagerDuty MTTA** — median (p50) acknowledge 2.82 min, 56% within 4 min; p50
  recommended over mean (PRIMARY default-median anchor).
  https://www.pagerduty.com/resources/digital-operations/learn/reduce-mtta/
- **SOC '3-minute rule'** — ~90% of alerts triaged in ~5 min, novel 20+ min
  (justifies σ≈1.0 tail). https://www.socinvestigation.com/the-3-minute-alert-rule-how-fast-socs-actually-work/
- **BleepingComputer** — SOC triage ~7 min/alert mean, long tail.
  https://www.bleepingcomputer.com/news/security/why-more-analysts-wont-solve-your-socs-alert-problem/
- **Zhang et al., Empirical Software Engineering 2022** — PR latency heavily
  right-skewed, log-transformed (lognormal family justification).
  https://arxiv.org/abs/2108.09946
- **DORA — Streamlining Change Approval** + **Octopus — CABs Don't Work** — CAB
  adds up to ~168 h; the worst-case batched-governance upper sweep cell.
  https://dora.dev/capabilities/streamlining-change-approval/ ·
  https://octopus.com/blog/change-advisory-boards-dont-work
- **GetStream — Scaling Content Moderation** — ~30 s/item; sub-minute floor cell.
  https://getstream.io/blog/scaling-content-moderation/

## PII / L6 evaluation methodology

- **Microsoft Presidio** — precision/recall + F-β (β=2, recall-weighted), TP/FP/FN
  per entity; report L6 as best-effort with an over-redaction rate, not folded
  into FPR. https://microsoft.github.io/presidio/evaluation/
