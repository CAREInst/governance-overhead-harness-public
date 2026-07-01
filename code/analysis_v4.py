"""
analysis_v4.py — MLSys 2027 Paper Analysis Module

Produces the statistical tests, tables, and figure inputs from the LIVE
experiment JSONL. Pure analysis of existing data (no new experiments).

What this module reports (and how to read it — post round-1/round-2 remediation):
  1. Harness overhead: per-config harness/API ratio with a CLUSTERED (by-task)
     bootstrap CI, plus a survivorship-paired variant and three denominator
     variants. The headline % is conditional on the API-latency denominator —
     see overhead_vs_denominator.
  2. Token overhead: NOT zero — L4 prepends a policy preamble, so it reflects a
     real (chosen) preamble cost; report it as such, not as "governance is
     free". (Do not cite a paired p-value here; equality was an artifact of the
     old non-mutating code.)
  3. Layer additivity: the cumulative decomposition is DESCRIPTIVE and
     telescopes (not a test). A genuine, non-telescoping additivity test is in
     analyze_additivity_isolated and requires isolated single-layer configs.
  4. Provider invariance: NOT established — only one model has paid data.
  5. Cost savings: a selection artifact (block rate + cheaper survivors), not a
     model-independent economic result; break-even is a compute-only lower
     bound with a $/CPU-hr sensitivity table, plus an L8-planning-aware figure.
  6. Injection detection: a task-attempt confusion matrix vs the task-suite
     ground truth (precision/recall, per-type), counting both blocked and
     L5-sanitized injections as catches. Numbers are corpus-contingent.
"""

import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


# ── Data Loading ──────────────────────────────────────────────────

def load_jsonl(path: str) -> list:
    """Load JSONL with bad-line tolerance."""
    events, bad = [], 0
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                events.append(json.loads(s))
            except json.JSONDecodeError:
                bad += 1
    if bad:
        print(f"  ⚠️ Skipped {bad} bad lines in {path}")
    return events


def load_provider(jsonl_path: str, cost_path: str) -> dict:
    """Load a provider's complete dataset."""
    events = load_jsonl(jsonl_path)
    with open(cost_path) as f:
        cost = json.load(f)

    api = [e for e in events if e.get("event_type") == "api_call"]
    gov = [e for e in events if e.get("event_type") == "governance"]
    completed = [e for e in api if e.get("task_completed")]
    blocked = [e for e in api if not e.get("task_completed")]

    return {
        "events": events,
        "api": api,
        "governance": gov,
        "completed": completed,
        "blocked": blocked,
        "cost_report": cost,
        "model": cost.get("model", "unknown"),
        "provider": cost.get("provider", "unknown"),
    }


# ── Statistical Tests ────────────────────────────────────────────

def bootstrap_ci(values: list, n_bootstrap: int = 10000,
                  ci: float = 0.95, stat_fn=statistics.mean,
                  seed: int = 42) -> dict:
    """Bootstrap confidence interval for any statistic.

    Reproducible: `seed` drives a local PRNG so resampling is deterministic
    without mutating the process-global random state.
    """
    rng = random.Random(seed)
    n = len(values)
    if n == 0:
        return {"mean": 0, "ci_lower": 0, "ci_upper": 0, "n": 0}

    boot_stats = []
    for _ in range(n_bootstrap):
        sample = rng.choices(values, k=n)
        boot_stats.append(stat_fn(sample))

    boot_stats.sort()
    alpha = (1 - ci) / 2
    lo = boot_stats[int(alpha * n_bootstrap)]
    hi = boot_stats[int((1 - alpha) * n_bootstrap)]

    return {
        "mean": stat_fn(values),
        "ci_lower": lo,
        "ci_upper": hi,
        "n": n,
        "n_bootstrap": n_bootstrap,
    }


def paired_ttest(x: list, y: list) -> dict:
    """Paired t-test. Returns t-statistic and p-value."""
    n = min(len(x), len(y))
    if n < 2:
        return {"t_stat": 0, "p_value": 1.0, "n": n}

    diffs = [x[i] - y[i] for i in range(n)]
    mean_d = statistics.mean(diffs)
    std_d = statistics.stdev(diffs)
    se = std_d / math.sqrt(n)

    if se == 0:
        # Perfect equality — all diffs are zero
        return {"t_stat": 0.0, "p_value": 1.0, "n": n,
                "mean_diff": mean_d, "std_diff": std_d,
                "interpretation": "identical"}

    t = mean_d / se
    # Exact two-tailed p-value from the t(df) distribution.
    p = t_sf_two_tailed(t, n - 1)

    return {
        "t_stat": round(t, 4),
        "p_value": round(p, 8),
        "n": n,
        "mean_diff": round(mean_d, 6),
        "std_diff": round(std_d, 6),
        "df": n - 1,
    }


def cohens_d(x: list, y: list) -> float:
    """Cohen's d effect size."""
    if len(x) < 2 or len(y) < 2:
        return 0.0
    pooled_std = math.sqrt(
        ((len(x) - 1) * statistics.variance(x) +
         (len(y) - 1) * statistics.variance(y)) /
        (len(x) + len(y) - 2)
    )
    if pooled_std == 0:
        return 0.0
    return (statistics.mean(x) - statistics.mean(y)) / pooled_std


def one_way_anova(groups: dict) -> dict:
    """One-way ANOVA. groups = {name: [values]}."""
    all_vals = []
    for vals in groups.values():
        all_vals.extend(vals)

    if not all_vals:
        return {"f_stat": 0, "p_value": 1.0}

    grand_mean = statistics.mean(all_vals)
    k = len(groups)
    N = len(all_vals)

    ss_between = sum(
        len(vals) * (statistics.mean(vals) - grand_mean) ** 2
        for vals in groups.values() if vals
    )
    ss_within = sum(
        sum((v - statistics.mean(vals)) ** 2 for v in vals)
        for vals in groups.values() if len(vals) > 1
    )

    df_between = k - 1
    df_within = N - k

    if df_within <= 0 or ss_within == 0:
        return {"f_stat": float('inf'), "p_value": 0.0}

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    f_stat = ms_between / ms_within

    ss_total = ss_between + ss_within
    eta_sq = ss_between / ss_total if ss_total > 0 else 0.0
    p_value = f_sf(f_stat, df_between, df_within)

    return {
        "f_stat": round(f_stat, 4),
        "p_value": round(p_value, 6),
        "eta_squared": round(eta_sq, 6),
        "significant_at_0.05": p_value < 0.05,
        "df_between": df_between,
        "df_within": df_within,
        "ms_between": round(ms_between, 6),
        "ms_within": round(ms_within, 6),
        "n_groups": k,
        "n_total": N,
    }


def _normal_cdf(x):
    """Approximate normal CDF using error function."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# ── Exact distribution tails (dependency-free) ───────────────────
# Regularized incomplete beta I_x(a,b) via the Numerical Recipes
# continued fraction. Gives exact F- and t-distribution tail
# probabilities without scipy, so ANOVA/t-tests report real p-values.

def _betacf(a: float, b: float, x: float) -> float:
    MAXIT, EPS, FPMIN = 200, 3.0e-12, 1.0e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log(1.0 - x))
    bt = math.exp(lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def f_sf(f_stat: float, df1: int, df2: int) -> float:
    """Upper-tail p-value P(F >= f_stat) for an F(df1, df2) distribution."""
    if f_stat <= 0 or df1 <= 0 or df2 <= 0:
        return 1.0
    if math.isinf(f_stat):
        return 0.0
    x = df2 / (df2 + df1 * f_stat)
    return _betai(df2 / 2.0, df1 / 2.0, x)


def t_sf_two_tailed(t_stat: float, df: int) -> float:
    """Two-tailed p-value for a t(df) distribution."""
    if df <= 0:
        return 1.0
    x = df / (df + t_stat * t_stat)
    return _betai(df / 2.0, 0.5, x)


def clustered_bootstrap_ci(clusters: list, n_bootstrap: int = 10000,
                            ci: float = 0.95, stat_fn=statistics.mean,
                            seed: int = 42) -> dict:
    """Cluster (block) bootstrap CI.

    `clusters` is a list of lists: each inner list holds the values for one
    cluster (e.g. all per-event ratios for one task across runs). Whole
    clusters are resampled with replacement, so within-cluster correlation is
    respected and the CI is not artificially narrowed by treating correlated
    events as independent (MAJOR-2).
    """
    rng = random.Random(seed)
    clusters = [c for c in clusters if c]
    k = len(clusters)
    flat = [v for c in clusters for v in c]
    if not flat:
        return {"mean": 0, "ci_lower": 0, "ci_upper": 0, "n": 0, "n_clusters": 0}

    boot_stats = []
    for _ in range(n_bootstrap):
        resampled = []
        for _ in range(k):
            resampled.extend(clusters[rng.randrange(k)])
        boot_stats.append(stat_fn(resampled))

    boot_stats.sort()
    alpha = (1 - ci) / 2
    lo = boot_stats[int(alpha * n_bootstrap)]
    hi = boot_stats[int((1 - alpha) * n_bootstrap)]
    return {
        "mean": stat_fn(flat),
        "ci_lower": lo,
        "ci_upper": hi,
        "n": len(flat),
        "n_clusters": k,
        "n_bootstrap": n_bootstrap,
    }


# ── Core Analysis Functions ──────────────────────────────────────

CONFIGS = ["L0", "L0-L3", "L0-L5", "L0-L7", "L0-L8"]


def analyze_harness_overhead(data: dict) -> dict:
    """Harness overhead analysis.

    Reports, per config:
      * mean-of-ratios with a CLUSTERED bootstrap CI (resampled by task, so
        correlated repeated measurements do not narrow the CI — MAJOR-2);
      * three denominator variants (mean-of-ratios, ratio-of-means,
        mean-harness / median-API) so the headline figure is shown to be
        robust (or not) to the denominator choice — MINOR-2.

    Survivorship (MAJOR-1) is handled by analyze_paired_overhead below; this
    function is explicit that ratios are conditional on the calls that
    completed in each config.
    """
    completed = data["completed"]
    results = {}

    for cfg in CONFIGS:
        cfg_events = [e for e in completed if e.get("layer_config") == cfg]
        # Group per-event ratios by task_id for the clustered bootstrap.
        clusters = defaultdict(list)
        harness_ms, api_ms_list = [], []
        for e in cfg_events:
            api_ms = e.get("api_latency_ms", 0)
            h_ms = e.get("harness_latency_ms", 0)
            if api_ms > 0:
                clusters[e["task_id"]].append(h_ms / api_ms * 100)
                harness_ms.append(h_ms)
                api_ms_list.append(api_ms)

        if not harness_ms:
            continue

        cluster_lists = list(clusters.values())
        ci = clustered_bootstrap_ci(cluster_lists, n_bootstrap=10000, seed=42)
        mean_harness = statistics.mean(harness_ms)
        mean_api = statistics.mean(api_ms_list)
        median_api = statistics.median(api_ms_list)

        results[cfg] = {
            "mean_ratio_pct": ci["mean"],
            "ci_lower_pct": ci["ci_lower"],
            "ci_upper_pct": ci["ci_upper"],
            "n": ci["n"],
            "n_clusters": ci["n_clusters"],
            "ci_method": "clustered_bootstrap_by_task",
            "mean_harness_ms": mean_harness,
            "mean_api_ms": mean_api,
            # Denominator sensitivity (MINOR-2)
            "ratio_variants_pct": {
                "mean_of_ratios": round(ci["mean"], 6),
                "ratio_of_means": round(mean_harness / mean_api * 100, 6),
                "mean_harness_over_median_api": round(
                    mean_harness / median_api * 100, 6),
            },
        }

    return results


def analyze_paired_overhead(data: dict) -> dict:
    """Survivorship-corrected overhead (MAJOR-1).

    Restricts to (task_id, run_number) pairs that COMPLETED in *every* config,
    so the L0 vs L0-L8 comparison is apples-to-apples on a common set of calls
    rather than a different, smaller, faster-API subset at higher configs.
    """
    completed = data["completed"]

    # (task, run) -> {cfg: event}
    by_key = defaultdict(dict)
    for e in completed:
        key = (e["task_id"], e.get("run_number", 0))
        by_key[key][e.get("layer_config")] = e

    survivors = [k for k, cfgs in by_key.items()
                 if all(c in cfgs for c in CONFIGS)]

    n_total_l0 = sum(1 for e in completed if e.get("layer_config") == "L0")
    result = {
        "n_survivors": len(survivors),
        "n_l0_completed": n_total_l0,
        "dropout_note": (
            f"{len(survivors)} of {n_total_l0} L0 task-runs completed in ALL "
            f"five configs; higher-config dropout (blocked/incomplete) is "
            f"excluded so ratios are paired on common survivors."),
        "by_config": {},
    }
    if not survivors:
        return result

    for cfg in CONFIGS:
        clusters = defaultdict(list)
        harness_ms, api_ms_list = [], []
        for (task_id, _run) in survivors:
            e = by_key[(task_id, _run)][cfg]
            api_ms = e.get("api_latency_ms", 0)
            h_ms = e.get("harness_latency_ms", 0)
            if api_ms > 0:
                clusters[task_id].append(h_ms / api_ms * 100)
                harness_ms.append(h_ms)
                api_ms_list.append(api_ms)
        if not harness_ms:
            continue
        ci = clustered_bootstrap_ci(list(clusters.values()),
                                    n_bootstrap=10000, seed=42)
        result["by_config"][cfg] = {
            "mean_ratio_pct": ci["mean"],
            "ci_lower_pct": ci["ci_lower"],
            "ci_upper_pct": ci["ci_upper"],
            "n": ci["n"],
            "mean_harness_ms": round(statistics.mean(harness_ms), 6),
            "mean_api_ms": round(statistics.mean(api_ms_list), 2),
        }
    return result


def analyze_token_overhead(data: dict) -> dict:
    """Paired token overhead analysis."""
    completed = data["completed"]

    # Group by (task_id, run_number)
    task_run = defaultdict(dict)
    for e in completed:
        key = (e["task_id"], e.get("run_number", 0))
        task_run[key][e["layer_config"]] = e.get("input_tokens", 0)

    # Paired comparison: L0 vs each config
    results = {}
    for cfg in ["L0-L3", "L0-L5", "L0-L7", "L0-L8"]:
        l0_tokens = []
        cfg_tokens = []
        for key, configs in task_run.items():
            if "L0" in configs and cfg in configs:
                l0_tokens.append(configs["L0"])
                cfg_tokens.append(configs[cfg])

        if l0_tokens and cfg_tokens:
            ttest = paired_ttest(cfg_tokens, l0_tokens)
            results[cfg] = {
                "paired_ttest": ttest,
                "mean_l0": statistics.mean(l0_tokens),
                "mean_cfg": statistics.mean(cfg_tokens),
                "n_pairs": len(l0_tokens),
            }

    return results


def analyze_layer_additivity(data: dict) -> dict:
    """Cumulative cost DECOMPOSITION (descriptive) — NOT an additivity test.

    FATAL-2: with only nested cumulative configs (L0, L0-L3, ...), incremental
    costs are defined by subtracting adjacent cumulative means, so their sum
    telescopes to the full-stack mean by construction — the "error" is
    identically 0.0 for ANY data. That is algebra, not evidence. We therefore
    report this strictly as a descriptive breakdown and FLAG that it is not a
    test. A genuine additivity test requires isolated single-layer configs and
    lives in analyze_additivity_isolated() below.
    """
    completed = data["completed"]

    harness_by_cfg = {}
    for cfg in CONFIGS:
        vals = [e["harness_latency_ms"] for e in completed
                if e.get("layer_config") == cfg and e.get("harness_latency_ms")]
        if vals:
            harness_by_cfg[cfg] = statistics.mean(vals) * 1000  # μs

    increments = {}
    if "L0" in harness_by_cfg:
        increments["L0"] = harness_by_cfg["L0"]
    if "L0" in harness_by_cfg and "L0-L3" in harness_by_cfg:
        increments["L1-L3"] = harness_by_cfg["L0-L3"] - harness_by_cfg["L0"]
    if "L0-L3" in harness_by_cfg and "L0-L5" in harness_by_cfg:
        increments["L4-L5"] = harness_by_cfg["L0-L5"] - harness_by_cfg["L0-L3"]
    if "L0-L5" in harness_by_cfg and "L0-L7" in harness_by_cfg:
        increments["L6-L7"] = harness_by_cfg["L0-L7"] - harness_by_cfg["L0-L5"]
    if "L0-L7" in harness_by_cfg and "L0-L8" in harness_by_cfg:
        increments["L8"] = harness_by_cfg["L0-L8"] - harness_by_cfg["L0-L7"]

    return {
        "is_additivity_test": False,
        "harness_by_config_us": {k: round(v, 1) for k, v in harness_by_cfg.items()},
        "incremental_costs_us": {k: round(v, 1) for k, v in increments.items()},
        "caveat": ("Descriptive cumulative decomposition only. The sum of these "
                   "increments equals the full-stack mean by construction "
                   "(telescoping), so it cannot test additivity. See "
                   "additivity_isolated for a real test."),
    }


ISOLATED_BASELINE = "ISO-BASE"  # bare loop scaffold, no governance work


def analyze_additivity_isolated(data: dict) -> dict:
    """Genuine additivity test (FATAL-2 fix), if isolated-layer data exists.

    Compares the SUM of independently-measured per-layer marginal costs (each
    ISO-Lx config minus the ISO-BASE baseline) to the measured full-stack
    (L0-L8) cost. Each layer is measured in isolation, so the comparison is NOT
    telescoping — a non-zero error is meaningful and quantifies cross-layer
    interference. Auto-discovers any "ISO-L*" configs the runner emitted.
    Returns {"available": False} on legacy data.
    """
    completed = data["completed"]

    def mean_us(cfg):
        vals = [e["harness_latency_ms"] for e in completed
                if e.get("layer_config") == cfg and e.get("harness_latency_ms")]
        return statistics.mean(vals) * 1000 if vals else None

    iso_configs = sorted({e.get("layer_config") for e in completed
                          if str(e.get("layer_config", "")).startswith("ISO-L")})
    base = mean_us(ISOLATED_BASELINE)
    full = mean_us("L0-L8")
    if base is None or not iso_configs:
        return {"available": False,
                "note": "No isolated single-layer configs in this dataset; "
                        "re-run with Track B isolated configs (--additivity) to "
                        "enable a real additivity test."}

    marginals = {}
    for c in iso_configs:
        m = mean_us(c)
        if m is not None:
            marginals[c] = round(m - base, 2)
    predicted = base + sum(marginals.values()) if full is not None else None
    err = (abs(predicted - full) / full * 100
           if (predicted is not None and full) else None)
    return {
        "available": True,
        "is_additivity_test": True,
        "baseline_us": round(base, 2),
        "isolated_marginal_us": marginals,
        "predicted_full_stack_us": round(predicted, 2) if predicted is not None else None,
        "measured_full_stack_us": round(full, 2) if full is not None else None,
        "additivity_error_pct": round(err, 2) if err is not None else None,
        "interpretation": ("Non-telescoping: error > 0 indicates genuine "
                           "cross-layer interference; ~0 confirms additivity."),
    }


def analyze_run_consistency(data: dict) -> dict:
    """ANOVA across runs to characterize the run-to-run effect.

    NOTE: this is NOT a test that confirms 'no run effect'. With thousands of
    events, even tiny systematic drift is statistically detectable. We report
    the p-value AND the effect size (eta_squared) so the magnitude is judged on
    practical, not just statistical, grounds (MAJOR-3).
    """
    completed = data["completed"]

    # Group harness latency by run number
    groups = defaultdict(list)
    for e in completed:
        run = e.get("run_number", 0)
        if run > 0 and e.get("harness_latency_ms"):
            groups[f"run_{run}"].append(e["harness_latency_ms"])

    if len(groups) < 2:
        return {"error": "insufficient runs"}

    anova = one_way_anova(groups)
    eta = anova.get("eta_squared", 0.0)
    sig = anova.get("significant_at_0.05", False)
    if sig:
        verdict = (f"Run effect IS statistically significant "
                   f"(p={anova.get('p_value')}), but practically negligible: "
                   f"runs explain only {eta*100:.2f}% of harness-latency variance.")
    else:
        verdict = (f"No statistically significant run effect "
                   f"(p={anova.get('p_value')}); runs explain {eta*100:.2f}% "
                   f"of variance.")
    anova["interpretation"] = verdict
    # Per-run mean harness latency (μs) for transparency.
    anova["run_means_us"] = {
        name: round(statistics.mean(vals) * 1000, 2)
        for name, vals in sorted(groups.items())
    }
    return anova


def analyze_cost_savings(data: dict) -> dict:
    """Cost comparison and break-even analysis."""
    completed = data["completed"]
    blocked = data["blocked"]

    # Per-call cost = main API cost + any L8 planning round-trip cost. Including
    # planning_cost_usd here is what makes L0-L8 (which fires the planning call)
    # genuinely cost more than L0-L7 — without it the cost table is identical and
    # the L8 feature is invisible at the reporting layer.
    cost_by_cfg = defaultdict(list)
    planning_by_cfg = defaultdict(float)
    for e in completed:
        cfg = e.get("layer_config", "")
        cost_by_cfg[cfg].append(e.get("api_cost_usd", 0) + e.get("planning_cost_usd", 0))
        planning_by_cfg[cfg] += e.get("planning_cost_usd", 0)

    # Add zero cost for blocked events
    for e in blocked:
        cfg = e.get("layer_config", "")
        cost_by_cfg[cfg].append(0)

    results = {}
    l0_mean = statistics.mean(cost_by_cfg.get("L0", [0]))

    # Per-config block rate (savings == block rate by construction; see note).
    block_by_cfg = defaultdict(int)
    total_by_cfg = defaultdict(int)
    for e in completed:
        total_by_cfg[e.get("layer_config", "")] += 1
    for e in blocked:
        block_by_cfg[e.get("layer_config", "")] += 1
        total_by_cfg[e.get("layer_config", "")] += 1

    for cfg in ["L0", "L0-L3", "L0-L5", "L0-L7", "L0-L8"]:
        vals = cost_by_cfg.get(cfg, [])
        if vals:
            cfg_mean = statistics.mean(vals)
            n_tot = total_by_cfg.get(cfg, 0)
            results[cfg] = {
                "mean_cost_per_call": round(cfg_mean, 6),
                "total_cost": round(sum(vals), 4),
                "planning_cost_total_usd": round(planning_by_cfg.get(cfg, 0), 6),
                "n_calls": len(vals),
                "block_rate_pct": round(block_by_cfg.get(cfg, 0) / n_tot * 100, 2) if n_tot else 0,
                "savings_vs_l0_pct": round(
                    (l0_mean - cfg_mean) / l0_mean * 100, 1) if l0_mean else 0,
            }

    # MAJOR-5: "savings" conflates TWO suite-specific selection effects, not a
    # model-independent economic property:
    #   (1) blocked calls contribute zero API cost (the block rate), and
    #   (2) the calls that survive to completion at higher configs are *cheaper*
    #       (fewer tokens) than the L0 population — a survivorship effect.
    # Hence savings_vs_l0_pct (e.g. 20.3%) is generally NOT equal to the block
    # rate (e.g. 10.7%); both are artifacts of this attack-heavy suite. Report
    # savings as a function of the deployment's true block prevalence, and keep
    # the survivorship effect explicit (see harness_overhead_paired).
    results["savings_caveat"] = (
        "savings_vs_l0_pct is driven by (a) zero-cost blocked calls and (b) "
        "lower per-call cost among surviving completions at higher configs. It "
        "is NOT model-independent and is generally NOT equal to the block rate. "
        "Both are properties of this suite's attack mix and task length "
        "distribution, not of governance per se."
    )

    # Break-even: at what injection rate does governance pay for itself?
    l0_cost = results.get("L0", {}).get("mean_cost_per_call", 0)

    # Harness cost derived from the actual measured full-stack (L0-L8) overhead,
    # priced at $0.10/CPU-hour (amortized local compute).
    full_stack_harness = [e["harness_latency_ms"] for e in completed
                          if e.get("layer_config") == "L0-L8" and e.get("harness_latency_ms")]
    mean_full_stack_us = (statistics.mean(full_stack_harness) * 1000
                          if full_stack_harness else 0)
    COMPUTE_RATE_USD_PER_HR = 0.10
    harness_cost = mean_full_stack_us * 1e-6 / 3600 * COMPUTE_RATE_USD_PER_HR

    if l0_cost > 0 and harness_cost > 0:
        # Break-even: harness_cost = injection_rate × l0_cost
        breakeven_rate = harness_cost / l0_cost
        # MAJOR-6: break-even is linear in the assumed compute rate. Report a
        # sensitivity table and label this a COMPUTE-ONLY lower bound that
        # EXCLUDES engineering, false-positive triage, human (L7) review, and
        # any incremental L8 planning API cost.
        sensitivity = {}
        for rate in [0.10, 1.0, 10.0, 100.0, 500.0, 1000.0]:
            hc = mean_full_stack_us * 1e-6 / 3600 * rate
            br = hc / l0_cost
            sensitivity[f"${rate:g}/cpu-hr"] = {
                "breakeven_injection_rate_pct": round(br * 100, 6),
                "one_in_n_calls": int(1 / br) if br > 0 else None,
            }
        # L8-aware break-even: when L8 is enabled, governance adds a real extra
        # API call (the planning round-trip), not just microseconds of CPU. The
        # mean planning cost per L0-L8 call typically DOMINATES the harness CPU
        # cost by orders of magnitude, so the honest break-even with L8 on is far
        # higher than the compute-only figure.
        n_l8 = len([e for e in completed if e.get("layer_config") == "L0-L8"])
        mean_planning = (planning_by_cfg.get("L0-L8", 0) / n_l8) if n_l8 else 0
        gov_cost_with_l8 = harness_cost + mean_planning
        breakeven_with_l8 = (gov_cost_with_l8 / l0_cost) if l0_cost else 0

        results["breakeven"] = {
            "compute_rate_usd_per_hr": COMPUTE_RATE_USD_PER_HR,
            "harness_cost_per_call_usd": harness_cost,
            "breakeven_injection_rate": round(breakeven_rate * 100, 4),
            "interpretation": (
                f"COMPUTE-ONLY lower bound: at ${COMPUTE_RATE_USD_PER_HR:g}/CPU-hr "
                f"governance pays for itself when injection rate exceeds "
                f"{breakeven_rate*100:.4f}% (≈1 in {int(1/breakeven_rate):,} calls). "
                f"Excludes engineering, false-positive review, human (L7) review."),
            "sensitivity_by_compute_rate": sensitivity,
            "with_l8_planning": {
                "mean_planning_cost_per_call_usd": round(mean_planning, 6),
                "governance_cost_per_call_usd": round(gov_cost_with_l8, 6),
                "breakeven_injection_rate_pct": round(breakeven_with_l8 * 100, 4),
                "one_in_n_calls": int(1 / breakeven_with_l8) if breakeven_with_l8 > 0 else None,
                "note": ("When L8 plan-reflect is on, the planning API call — not "
                         "harness CPU — sets the break-even. This is the honest "
                         "economic figure for the full stack."),
            },
        }

    # MAJOR-9 (reanalysis path): the overhead ratio is dominated by the API
    # latency denominator. Show how the SAME measured full-stack harness cost
    # scales against faster denominators (e.g. local inference).
    if mean_full_stack_us > 0:
        regimes = {}
        for api_ms in [50, 500, 1000, 5000, 6750]:
            regimes[f"{api_ms}ms_api"] = round(
                (mean_full_stack_us * 1e-3) / api_ms * 100, 5)
        results["overhead_vs_denominator"] = {
            "mean_full_stack_harness_us": round(mean_full_stack_us, 1),
            "overhead_pct_by_api_latency": regimes,
            "note": ("Headline overhead % is conditional on a ~6.75s remote-API "
                     "denominator. At sub-second (local) latency the same "
                     f"~{mean_full_stack_us:.0f}us is materially larger as a "
                     "fraction of serving time."),
        }

    # Cohen's d: L0 vs L0-L8 cost
    l0_costs = cost_by_cfg.get("L0", [])
    l8_costs = cost_by_cfg.get("L0-L8", [])
    if l0_costs and l8_costs:
        results["effect_size"] = {
            "cohens_d_l0_vs_l8": round(cohens_d(l0_costs, l8_costs), 4),
        }

    return results


def analyze_injection_detection(data: dict, injected_labels: dict = None,
                                 fp_labels: dict = None) -> dict:
    """Injection detection analysis with a real confusion matrix.

    Fixes FATAL-1: the previous version divided an ALL-TIER caught count by a
    Tier-2-only denominator, inflating the rate (32% reported vs 12% true for
    Tier-2). Here detection is evaluated at the task-attempt level against the
    ground-truth injected-task set from the task suite, giving precision and
    recall plus per-type recall (BLIND-1/7).

    injected_labels: {task_id: injection_type} for the 20 injected tasks. Any
    task_id not present is treated as a clean (negative) task.
    """
    api = data["api"]
    gov = data["governance"]
    injected_labels = injected_labels or {}
    injected_ids = set(injected_labels)

    def is_api_error(e):
        # A failed API call is NOT a governance decision — exclude it from the
        # confusion matrix entirely (it is a missing observation, not a TP/FP).
        # Critical for unreliable routes (e.g. a model whose API errored on most
        # calls would otherwise show inflated recall / collapsed precision).
        return (not e.get("task_completed", False)
                and "API error" in (e.get("governance_detail") or ""))

    def is_blocked(e):
        # An injected failure was CAUGHT if governance blocked the call (did not
        # complete, and it was NOT an API error) OR it completed but L5 sanitized
        # the injection (injected_failure_caught on the completed event).
        if is_api_error(e):
            return False
        return (not e.get("task_completed", False)) or e.get("injected_failure_caught", False)

    # ── Confusion matrix per config (task-attempt level) ──
    fp_labels = fp_labels or {}            # {task_id: stress_target} for hard negatives
    fp_ids = set(fp_labels)

    by_config = {}
    never_blocked = {tid: True for tid in injected_ids}  # tid -> still-never-blocked

    for cfg in CONFIGS:
        cfg_api = [e for e in api if e.get("layer_config") == cfg]
        tp = fn = 0
        fp_clean = fp_hard = tn_clean = tn_hard = 0
        per_type_total = defaultdict(int)
        per_type_caught = defaultdict(int)
        per_target_blocked = defaultdict(int)
        per_target_total = defaultdict(int)
        l6_over_redactions = 0
        api_errors = 0
        for e in cfg_api:
            tid = e.get("task_id", "")
            if is_api_error(e):
                # NOT excluded from classification: governance's block/pass
                # decision is made BEFORE the API call and is already fully
                # known (via injected_failure_caught, correctly False here —
                # we only reach the API-error branch when governance did NOT
                # block the call). The API's own failure afterward has no
                # bearing on that decision, so this event is still correctly
                # classifiable as FN (positive) / TN (negative) below. Only
                # tracked here for disclosure of the route's reliability.
                api_errors += 1
            blocked = is_blocked(e)
            if tid in injected_ids:                       # POSITIVE
                itype = injected_labels[tid]
                per_type_total[itype] += 1
                if blocked:
                    tp += 1; per_type_caught[itype] += 1; never_blocked[tid] = False
                else:
                    fn += 1
            elif tid in fp_ids:                           # HARD NEGATIVE (designed)
                tgt = fp_labels[tid]
                per_target_total[tgt] += 1
                if blocked:
                    fp_hard += 1; per_target_blocked[tgt] += 1
                else:
                    tn_hard += 1
                # L6 over-redaction: benign output mutated (not a precision FP).
                if e.get("output_modified"):
                    l6_over_redactions += 1
            else:                                         # CLEAN negative
                if blocked:
                    fp_clean += 1
                else:
                    tn_clean += 1

        fp_all = fp_clean + fp_hard
        neg_all = fp_all + tn_clean + tn_hard
        neg_hard = fp_hard + tn_hard
        prec_all = tp / (tp + fp_all) if (tp + fp_all) else None
        prec_hard = tp / (tp + fp_hard) if (tp + fp_hard) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        tier2_attempts = [e for e in cfg_api if e.get("task_tier") == 2]
        tier2_caught = [e for e in tier2_attempts
                        if e.get("task_id") in injected_ids and is_blocked(e)]
        by_config[cfg] = {
            "tp": tp, "fn": fn, "fp_clean": fp_clean, "fp_hard": fp_hard,
            "recall": round(recall, 4) if recall is not None else None,
            # precision_all: against ALL negatives (clean + hard). precision_hard:
            # against only the designed hard negatives — the honest, non-corpus-
            # contingent number (precision=1.00 on clean-only was an artifact).
            "precision_all": round(prec_all, 4) if prec_all is not None else None,
            "fpr_all_pct": round(fp_all / neg_all * 100, 2) if neg_all else 0,
            "precision_hard": round(prec_hard, 4) if prec_hard is not None else None,
            "fpr_hard_pct": round(fp_hard / neg_hard * 100, 2) if neg_hard else 0,
            "tier2_injected_detection_rate_pct": round(
                len(tier2_caught) / len(tier2_attempts) * 100, 1)
                if tier2_attempts else 0,
            "per_type_recall": {
                t: round(per_type_caught[t] / per_type_total[t], 4)
                for t in sorted(per_type_total)},
            "fp_by_stress_target": {
                t: f"{per_target_blocked[t]}/{per_target_total[t]} blocked"
                for t in sorted(per_target_total)},
            "l6_over_redactions": l6_over_redactions,
            # Disclosure only — these attempts ARE included in tp/fn/tn/fp above
            # (see is_api_error's comment); this just reports the route's
            # reliability that day, it does not mean data was discarded.
            "api_errors_disclosed": api_errors,
            "api_error_rate_pct": round(api_errors / len(cfg_api) * 100, 1) if cfg_api else 0,
        }

    # ── Governance block distribution (decision-level) ──
    gov_blocks = [e for e in gov if e.get("governance_decision") == "block"]
    block_by_layer = defaultdict(int)
    for e in gov_blocks:
        block_by_layer[e.get("governance_layer", "")] += 1

    never_caught_ids = sorted(tid for tid, miss in never_blocked.items() if miss)

    return {
        "by_config": by_config,
        "ground_truth_injected_tasks": len(injected_ids),
        "hard_negative_tasks": len(fp_ids),
        "never_caught_injected_tasks": never_caught_ids,
        "never_caught_detail": {
            tid: injected_labels[tid] for tid in never_caught_ids},
        "total_block_decisions": len(gov_blocks),
        "blocks_by_layer": dict(sorted(block_by_layer.items())),
        "note": ("Detection at task-attempt level vs task-suite ground truth. "
                 "precision_hard/fpr_hard use the 21 purpose-built hard "
                 "negatives (benign + trigger words); precision_all/fpr_all add "
                 "the clean tasks. L6 over-redactions on benign output are "
                 "reported separately, NOT as precision false positives."),
    }


# ── Full Analysis Pipeline ───────────────────────────────────────

def load_injected_labels(suite_path: str) -> dict:
    """{task_id: injection_type} for injected tasks in the suite (ground truth)."""
    try:
        suite = json.loads(Path(suite_path).read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    labels = {}
    for t in suite.get("tasks", []):
        inj = t.get("injected_failure")
        if inj and isinstance(inj, dict):
            labels[t["task_id"]] = inj.get("type", "unknown")
    return labels


def load_fp_labels(suite_path: str) -> dict:
    """{task_id: fp_stress_target} for the purpose-built hard-negative probes."""
    try:
        suite = json.loads(Path(suite_path).read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return {t["task_id"]: t.get("fp_stress_target", "unknown")
            for t in suite.get("tasks", []) if t.get("fp_negative")}


# HITL sensitivity grid (C1): median minutes x lognormal sigma. The cited
# default cell is (3 min, 1.0). Reporting the whole grid answers the
# "unsourced single point" objection far better than defending one value.
HITL_MEDIAN_GRID_MIN = [1, 3, 5, 15, 60, 480]
HITL_SIGMA_GRID = [0.5, 1.0, 1.5, 2.0]


def analyze_hitl_sensitivity(data: dict, seed: int = 42) -> dict:
    """Offline HITL sensitivity sweep — NO new API spend.

    Re-samples the modeled human-review wait over the {median, sigma} grid for
    every logged L7 escalation (a governance 'block' at L7_hitl_gate, or any
    event carrying hitl_wait_ms), and reports the mean modeled wait per cell.
    Works on any dataset that recorded L7 blocks (incl. legacy), since the wait
    is modeled, not measured.
    """
    import experiment_runner_v3 as _R   # reuse the exact sampler

    events = data.get("events", [])
    escalations = [e for e in events
                   if (e.get("governance_layer") == "L7_hitl_gate"
                       and e.get("governance_decision") == "block")
                   or e.get("hitl_wait_ms", 0) > 0]
    # De-duplicate to one escalation per (task, run).
    keys = sorted({(e.get("task_id", ""), e.get("run_number", 0)) for e in escalations})
    if not keys:
        return {"available": False, "note": "no L7 escalations logged"}

    grid = {}
    for med_min in HITL_MEDIAN_GRID_MIN:
        for sig in HITL_SIGMA_GRID:
            med_ms = med_min * 60 * 1000
            waits = [_R.model_hitl_wait_ms(t, r, seed, median_ms=med_ms, sigma=sig)
                     for (t, r) in keys]
            grid[f"median={med_min}min,sigma={sig}"] = {
                "mean_wait_min": round(statistics.mean(waits) / 60000, 2),
                "median_wait_min": round(statistics.median(waits) / 60000, 2),
                "p95_wait_min": round(sorted(waits)[int(0.95 * (len(waits) - 1))] / 60000, 2),
            }
    return {
        "available": True,
        "n_escalations": len(keys),
        "default_cell": "median=3min,sigma=1.0",
        "grid": grid,
        "note": ("MODELED human-review latency, reported separately from measured "
                 "harness latency. Default cell anchored to PagerDuty MTTA p50 "
                 "(2.82 min) + SOC triage tail; grid spans moderation→CAB."),
    }


def run_full_analysis(providers: Dict[str, dict],
                       injected_labels: dict = None,
                       fp_labels: dict = None) -> dict:
    """Run complete analysis across all providers."""
    results = {}

    for name, data in providers.items():
        print(f"\n  Analyzing {name}...")
        results[name] = {
            "harness_overhead": analyze_harness_overhead(data),
            "harness_overhead_paired": analyze_paired_overhead(data),
            "token_overhead": analyze_token_overhead(data),
            "layer_additivity": analyze_layer_additivity(data),
            "additivity_isolated": analyze_additivity_isolated(data),
            "run_consistency": analyze_run_consistency(data),
            "cost_savings": analyze_cost_savings(data),
            "injection_detection": analyze_injection_detection(data, injected_labels, fp_labels),
            "hitl_sensitivity": analyze_hitl_sensitivity(data),
        }

    return results


# ── Paper Tables ─────────────────────────────────────────────────

def generate_table1(results: dict) -> str:
    """Table 1: Cross-provider overhead summary."""
    lines = [
        "TABLE 1: Governance Overhead Across Providers",
        "=" * 80,
        f"{'Provider':<16} {'Config':<8} {'Harness':>10} {'API':>10} "
        f"{'Ratio':>10} {'95% CI':>20}",
        "-" * 80,
    ]
    for provider, data in results.items():
        ho = data["harness_overhead"]
        for cfg in ["L0", "L0-L3", "L0-L5", "L0-L7", "L0-L8"]:
            if cfg in ho:
                d = ho[cfg]
                lines.append(
                    f"{provider:<16} {cfg:<8} {d['mean_harness_ms']:>8.3f}ms "
                    f"{d['mean_api_ms']:>8.0f}ms {d['mean_ratio_pct']:>8.4f}% "
                    f"[{d['ci_lower_pct']:.4f}, {d['ci_upper_pct']:.4f}]"
                )
        lines.append("")
    return "\n".join(lines)


def generate_table2(results: dict) -> str:
    """Table 2: Layer additivity decomposition."""
    lines = [
        "TABLE 2: Layer Cost Decomposition (μs)",
        "=" * 60,
    ]
    for provider, data in results.items():
        la = data["layer_additivity"]
        lines.append(f"\n  {provider}  (descriptive decomposition — not a test):")
        for layer, cost in la["incremental_costs_us"].items():
            lines.append(f"    {layer:<8} {cost:>8.1f} μs")
        iso = data.get("additivity_isolated", {})
        if iso.get("available"):
            lines.append(f"    {'─'*28}")
            lines.append(f"    Real additivity test (isolated configs):")
            lines.append(f"    {'predicted':<12} {iso['predicted_full_stack_us']:>8.1f} μs")
            lines.append(f"    {'measured':<12} {iso['measured_full_stack_us']:>8.1f} μs")
            lines.append(f"    interference error: {iso['additivity_error_pct']}%")
        else:
            lines.append(f"    (isolated-layer additivity test not available "
                         f"on this dataset)")
    return "\n".join(lines)


# ── Cross-provider analyses (C2 preamble ablation, C4 invariance) ──

def _variant_from_key(provider_key: str) -> str:
    """Extract the __pre-<v> preamble variant stamped into a provider key."""
    for part in provider_key.split("__"):
        if part.startswith("pre-"):
            return part[len("pre-"):]
    return "medium"


def analyze_preamble_overhead(providers: Dict[str, dict]) -> dict:
    """C2: token overhead as a FUNCTION of preamble length, per model.

    Pairs each preamble variant's L0-L5 (first config carrying L4) mean input
    tokens against that model's OWN L0 baseline — never pooled across models
    (tokenizers differ). Shows overhead IS the chosen preamble length (slope ~1
    through the origin), defusing the token-overhead-tautology objection.
    """
    PRE_TOKENS = {"none": 0, "short": 10, "medium": 45, "long": 120}
    by_model = defaultdict(dict)
    for key, data in providers.items():
        variant = _variant_from_key(key)
        model_base = key.split("__")[0]
        completed = data["completed"]
        def mean_in(cfg):
            v = [e.get("input_tokens", 0) for e in completed
                 if e.get("layer_config") == cfg and e.get("input_tokens")]
            return statistics.mean(v) if v else None
        l0, l5 = mean_in("L0"), mean_in("L0-L5")
        if l0 is not None and l5 is not None:
            by_model[model_base][variant] = {
                "preamble_est_tokens": PRE_TOKENS.get(variant),
                "measured_token_overhead": round(l5 - l0, 1),
            }
    out = {}
    for model_base, variants in by_model.items():
        out[model_base] = {
            "variants": variants,
            "interpretation": ("measured_token_overhead should track "
                               "preamble_est_tokens ~1:1; 'none' isolates L4 "
                               "validate-only cost (≈0)."),
        }
    if len(by_model) == 0 or all(len(v) < 2 for v in by_model.values()):
        return {"available": False,
                "note": "Need ≥2 preamble variants per model (run --preamble "
                        "none/short/medium/long) to plot the curve.",
                "by_model": out}
    return {"available": True, "by_model": out}


def analyze_cross_model_invariance(results: Dict[str, dict]) -> dict:
    """C4: provider-sensitivity evidence — min/max/CV of the harness/API ratio
    across models, per config. Compares RATIOs (never raw token counts, since
    tokenizers differ). Reports completion rate so refusals don't bias it."""
    if len(results) < 2:
        return {"available": False, "note": "need ≥2 models (run sweep mode)"}
    per_config = {}
    for cfg in CONFIGS:
        ratios = {}
        for key, data in results.items():
            ho = data.get("harness_overhead", {}).get(cfg)
            if ho:
                ratios[key.split("__")[0]] = ho["mean_ratio_pct"]
        if len(ratios) >= 2:
            vals = list(ratios.values())
            mean = statistics.mean(vals)
            cv = (statistics.pstdev(vals) / mean) if mean else 0
            per_config[cfg] = {
                "by_model_ratio_pct": {k: round(v, 5) for k, v in ratios.items()},
                "min": round(min(vals), 5), "max": round(max(vals), 5),
                "cv": round(cv, 3), "n_models": len(vals),
            }
    return {"available": bool(per_config), "per_config": per_config,
            "note": "Ratios compared (tokenizer-independent). CV quantifies "
                    "provider (in)variance the docstring caveat demands."}


# ── Main ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    scan_dir = sys.argv[1] if len(sys.argv) > 1 else "results/full"

    # Ground-truth labels for precision/recall (positives) + FP hard negatives.
    suite_path = sys.argv[2] if len(sys.argv) > 2 else "../data/task_suite.json"
    injected_labels = load_injected_labels(suite_path)
    fp_labels = load_fp_labels(suite_path)
    if injected_labels:
        print(f"  Loaded {len(injected_labels)} injected + {len(fp_labels)} "
              f"hard-negative labels from {suite_path}")
    else:
        print(f"  ⚠️ No injected-task labels found at {suite_path} "
              f"(precision/recall will be unavailable)")

    providers = {}

    # Auto-discover every provider in scan_dir: each contributes a
    # "<stem>_cost_report.json" paired with a "<stem>.jsonl" of the same stem.
    full_dir = Path(scan_dir)
    cost_reports = sorted(full_dir.glob("*_cost_report.json"))
    if not cost_reports:
        print(f"  ⚠️ No *_cost_report.json files found in {full_dir}")
    for cost_path in cost_reports:
        stem = cost_path.name[: -len("_cost_report.json")]
        jsonl_path = cost_path.parent / f"{stem}.jsonl"
        if not jsonl_path.exists():
            print(f"  ⚠️ No matching JSONL for {cost_path.name} "
                  f"(expected {jsonl_path.name})")
            continue
        # Key by the dataset STEM so the __pre-<variant>/__l8-<mode> knobs are
        # available to the preamble-ablation and cross-model analyses.
        providers[stem] = load_provider(str(jsonl_path), str(cost_path))
        print(f"  Discovered {stem}")

    # Run analysis
    results = run_full_analysis(providers, injected_labels, fp_labels)

    # Cross-provider analyses (only meaningful with multiple datasets).
    cross = {
        "preamble_overhead": analyze_preamble_overhead(providers),
        "cross_model_invariance": analyze_cross_model_invariance(results),
    }
    results["_cross_provider"] = cross

    # Output
    print("\n" + generate_table1({k: v for k, v in results.items() if k != "_cross_provider"}))
    print("\n" + generate_table2({k: v for k, v in results.items() if k != "_cross_provider"}))

    # FP precision summary (the credibility centerpiece)
    for key, data in results.items():
        if key == "_cross_provider":
            continue
        det = data.get("injection_detection", {}).get("by_config", {}).get("L0-L8")
        if det and det.get("precision_hard") is not None:
            print(f"\n  [{key}] L0-L8 detection: recall={det['recall']} "
                  f"precision_hard={det['precision_hard']} (fpr_hard={det['fpr_hard_pct']}%) "
                  f"precision_all={det['precision_all']}")

    # Save full results
    output_path = "results/v4_analysis_results.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {output_path}")
