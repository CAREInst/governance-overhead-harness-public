"""
make_data_summary.py — compact per-config summary of a run's JSONL.

Emits results/full/data_summary.json: a small, reviewer-friendly digest of a
35MB event log (per-config latencies, tokens, costs, run-to-run variance, and a
corrected event/decision breakdown). No new experiments — pure summarization.

MINOR-3 fix: the run_meta field that counts governance records is named
`governance_events` (every per-layer check, ~16.9k), distinct from
`block_decisions` (governance decisions == "block", 215). The old name
`governance_blocks` conflated the two and was off by ~75x.

Usage:
    python3 make_data_summary.py [results/full]
"""

import json
import statistics
import collections
import sys
from pathlib import Path


def load_events(path: Path) -> list:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                try:
                    out.append(json.loads(s))
                except json.JSONDecodeError:
                    pass
    return out


def summarize(jsonl_path: Path) -> dict:
    events = load_events(jsonl_path)
    api = [e for e in events if e.get("event_type") == "api_call"]
    gov = [e for e in events if e.get("event_type") == "governance"]
    completed = [e for e in api if e.get("task_completed")]
    blocked = [e for e in api if not e.get("task_completed")]

    decision_counts = collections.Counter(
        e.get("governance_decision", "") for e in gov)
    block_decisions = decision_counts.get("block", 0)

    by_cfg = collections.defaultdict(list)
    for e in completed:
        by_cfg[e["layer_config"]].append(e)

    summary = {
        "run_meta": {
            "dataset": jsonl_path.name,
            "total_events": len(events),
            "api_call_events": len(api),
            "completed": len(completed),
            "blocked_api_calls": len(blocked),
            # Corrected naming (MINOR-3):
            "governance_events": len(gov),          # every per-layer check
            "block_decisions": block_decisions,     # governance_decision == "block"
            "governance_decision_counts": dict(decision_counts),
            "api_errors": sum(
                1 for e in events
                if "API error" in (e.get("governance_detail") or "")),
            "total_spend_usd": round(
                max((e.get("cumulative_cost_usd", 0) for e in api), default=0), 4),
        },
        "per_config": {},
    }

    for cfg in ["L0", "L0-L3", "L0-L5", "L0-L7", "L0-L8"]:
        evs = by_cfg.get(cfg, [])
        if not evs:
            continue
        api_lats = [e["api_latency_ms"] for e in evs if e.get("api_latency_ms")]
        har_lats = [e["harness_latency_ms"] for e in evs if e.get("harness_latency_ms")]
        tokens = [e["total_tokens"] for e in evs if e.get("total_tokens")]
        # Per-call cost includes the L8 planning round-trip (planning_cost_usd).
        costs = [e.get("api_cost_usd", 0) + e.get("planning_cost_usd", 0) for e in evs]
        planning_total = sum(e.get("planning_cost_usd", 0) for e in evs)
        tok_overhead = [e.get("token_overhead", 0) for e in evs]
        by_run = collections.defaultdict(list)
        for e in evs:
            by_run[e["run_number"]].append(e["harness_latency_ms"])
        run_means_us = [round(statistics.mean(v) * 1000, 1)
                        for v in (by_run[k] for k in sorted(by_run))]
        summary["per_config"][cfg] = {
            "n": len(evs),
            "api_latency_ms": {
                "mean": round(statistics.mean(api_lats), 1),
                "median": round(statistics.median(api_lats), 1),
                "stdev": round(statistics.stdev(api_lats), 1) if len(api_lats) > 1 else 0,
            },
            "harness_latency_us": {
                "mean": round(statistics.mean(har_lats) * 1000, 2),
                "median": round(statistics.median(har_lats) * 1000, 2),
                "stdev": round(statistics.stdev(har_lats) * 1000, 2) if len(har_lats) > 1 else 0,
            },
            "tokens_mean": round(statistics.mean(tokens), 1) if tokens else 0,
            "token_overhead_mean": round(statistics.mean(tok_overhead), 2) if tok_overhead else 0,
            "cost_usd_mean": round(statistics.mean(costs), 6) if costs else 0,
            "planning_cost_total_usd": round(planning_total, 6),
            "run_means_us": run_means_us,
            "run_stdev_us": round(statistics.stdev(run_means_us), 2) if len(run_means_us) > 1 else 0,
            "tier_counts": dict(collections.Counter(
                e.get("task_id", "").split("-")[0] for e in evs)),
        }
    return summary


if __name__ == "__main__":
    scan_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "results/full")
    jsonls = sorted(scan_dir.glob("*.jsonl"))
    if not jsonls:
        print(f"  ⚠️ No .jsonl in {scan_dir}")
        sys.exit(1)
    # Largest JSONL is the dataset (heartbeat/manifest are tiny json, not jsonl).
    jsonl_path = max(jsonls, key=lambda p: p.stat().st_size)
    summary = summarize(jsonl_path)
    out = scan_dir / "data_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    rm = summary["run_meta"]
    print(f"  Wrote {out}")
    print(f"  governance_events={rm['governance_events']}  "
          f"block_decisions={rm['block_decisions']}  "
          f"(old 'governance_blocks' conflated these)")
