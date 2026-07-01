"""
harness_benchmark.py — Core Measurement Harness for MLSys 2027

Every measurement carries a financial unit (USD). Every event is
JSONL-serializable. Every cost is computed from pinned provider
pricing (see PRICING_DATE).

Five measurement points per API call:
  1. Pre-API: start timer, record input
  2. Post-API: record usage, stop timer, compute cost
  3. Pre-tool: record tool name and input size
  4. Post-tool: record result size and execution time
  5. Governance: record each layer's decision with execution time
"""

import json, time, os, uuid, asyncio
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── Provider Pricing ──────────────────────────────────────────────
# Native table below pinned April 2026; OpenRouter routes (extra_adapters.py)
# were re-confirmed against the live catalog on 2026-06-22. PRICING_DATE stamps
# the run manifest and reflects the live-run (OpenRouter) pricing date.

PRICING_DATE = "2026-06-22"
# harness-1.1: added Track-B fields (planning_api_ms, planning_cost_usd,
# hitl_wait_ms, token_overhead, prompt_modified, injection_sanitized,
# output_modified) and the v1.2.0 governance knobs in the run manifest.
# harness-1.2: real plan-then-execute L8 (plan_chars, plan_injected); the plan
# is threaded into the executed main prompt under L8_MODE=execute.
SCHEMA_VERSION = "harness-1.2"

PRICING = {
    # Anthropic
    "claude-haiku-4-5-20251001":     {"input": 0.80,  "output": 4.00,   "provider": "anthropic", "tier": "budget"},
    "claude-sonnet-4-20250514":      {"input": 3.00,  "output": 15.00,  "provider": "anthropic", "tier": "mid"},
    "claude-opus-4-20250514":        {"input": 15.00, "output": 75.00,  "provider": "anthropic", "tier": "frontier"},
    # OpenAI
    "gpt-4.1-mini":                  {"input": 0.40,  "output": 1.60,   "provider": "openai",    "tier": "budget"},
    "gpt-4.1":                       {"input": 2.00,  "output": 8.00,   "provider": "openai",    "tier": "mid"},
    # Google
    "gemini-3-flash":                {"input": 0.15,  "output": 0.60,   "provider": "google",    "tier": "budget"},
    "gemini-3-pro":                  {"input": 1.25,  "output": 5.00,   "provider": "google",    "tier": "mid"},
    # NVIDIA (via Bedrock)
    "nemotron-3-super":              {"input": 0.15,  "output": 0.60,   "provider": "nvidia",    "tier": "mid"},
    # DeepSeek (OpenAI-compatible API)
    "deepseek-v4":                   {"input": 0.30,  "output": 0.50,   "provider": "deepseek",  "tier": "budget"},
    # xAI Grok (OpenAI-compatible API)
    "grok-4.1-fast":                 {"input": 0.20,  "output": 0.50,   "provider": "xai",       "tier": "budget"},
}

def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Compute USD cost from token counts and pinned pricing."""
    if model not in PRICING:
        raise ValueError(f"compute_cost: model {model!r} not in PRICING; pin it. Known: {sorted(PRICING)}")
    p = PRICING[model]
    return round(
        (input_tokens / 1_000_000) * p["input"] +
        (output_tokens / 1_000_000) * p["output"], 8)

def get_model_info(model: str) -> dict:
    """Return provider, tier, and pricing for a model."""
    if model not in PRICING:
        raise ValueError(f"get_model_info: unknown model {model!r}. Known: {sorted(PRICING)}")
    return PRICING[model]


# ── Measurement Event ─────────────────────────────────────────────

@dataclass
class MeasurementEvent:
    """One instrumented event with financial unit (USD).

    Every field that can carry a dollar value does. This is the
    unit of measure for long-term audit and business planning.
    """
    event_id: str = ""
    experiment_id: str = ""
    task_id: str = ""
    run_number: int = 0
    model: str = ""
    provider: str = ""
    tier: str = ""
    layer_config: str = ""
    event_type: str = ""         # api_call | tool_call | governance | task_complete
    schema_version: str = ""
    timestamp: str = ""

    # Token metrics
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    # Timing metrics (ms)
    api_latency_ms: float = 0.0
    harness_latency_ms: float = 0.0
    total_latency_ms: float = 0.0

    # Track-B metrics: real plan-reflect round-trip + modeled human review.
    # planning_* capture the L8 planning API call (excluded from harness_latency
    # so the two costs stay separable). hitl_wait_ms is the MODELED human
    # decision latency for L7 escalations — reported separately from harness
    # cost, never folded into it.
    planning_api_ms: float = 0.0
    planning_cost_usd: float = 0.0
    plan_chars: int = 0              # length of the L8 plan text
    plan_injected: bool = False      # plan was threaded into the executed main prompt
    hitl_wait_ms: float = 0.0

    # Financial metrics (USD)
    api_cost_usd: float = 0.0
    harness_cost_usd: float = 0.0
    cumulative_cost_usd: float = 0.0

    # Governance metrics
    governance_layer: str = ""
    governance_decision: str = ""    # pass | block | modify
    governance_detail: str = ""

    # I/O-mutation tracking (Track B): governance genuinely transforms I/O.
    token_overhead: int = 0          # input-token delta vs the L0 baseline
    prompt_modified: bool = False    # any pre-API layer rewrote the prompt (incl. L4 preamble)
    injection_sanitized: bool = False  # L5 specifically neutralized an injection span
    output_modified: bool = False    # L6 redacted the model response

    # Task metrics
    task_completed: bool = False
    task_tier: int = 0
    injected_failure: str = ""
    injected_failure_caught: bool = False

    # Iteration tracking
    iteration: int = 0
    api_calls_in_task: int = 0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        if not self.event_id:
            self.event_id = uuid.uuid4().hex[:12]
        if not self.schema_version:
            self.schema_version = SCHEMA_VERSION
        self.total_tokens = self.input_tokens + self.output_tokens
        self.total_latency_ms = round(self.api_latency_ms + self.harness_latency_ms, 4)

    def to_json(self) -> str:
        d = asdict(self)
        # allow_nan=False: NaN/Infinity are invalid JSON and would corrupt the
        # dataset for strict readers (R jsonlite, pandas). Fail loudly instead.
        return json.dumps(d, default=str, allow_nan=False)

    @classmethod
    def from_json(cls, line: str) -> 'MeasurementEvent':
        return cls(**json.loads(line))


# ── Measurement Logger ────────────────────────────────────────────

class MeasurementLogger:
    """Append-only JSONL logger. Thread-safe via single-writer pattern."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._count = 0

    def log(self, event: MeasurementEvent):
        line = event.to_json()
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())   # durability: a crash must not lose logged events
        self._count += 1

    @property
    def event_count(self) -> int:
        return self._count

    def read_all(self) -> list:
        """Read all events from the log."""
        events = []
        if self.path.exists():
            with open(self.path) as f:
                for line in f:
                    if line.strip():
                        events.append(json.loads(line))
        return events


# ── Cost Tracker (Hard Stop) ──────────────────────────────────────

class CostTracker:
    """Real-time cost accumulator with hard stop at max_usd.

    This prevents runaway experiment costs. When the cap is hit,
    can_continue() returns False and the experiment saves partial
    results.
    """

    def __init__(self, max_usd: float = 30.0):
        self.max_usd = max_usd
        self.total_usd = 0.0
        self.call_count = 0
        self._events: list = []

    def record(self, cost_usd: float, model: str = "", task_id: str = ""):
        self.total_usd = round(self.total_usd + cost_usd, 8)
        self.call_count += 1
        self._events.append({
            "call": self.call_count,
            "cost": cost_usd,
            "cumulative": self.total_usd,
            "model": model,
            "task_id": task_id,
            "ts": datetime.now(timezone.utc).isoformat(),
        })

    def can_continue(self) -> bool:
        return self.total_usd < self.max_usd

    @property
    def remaining_usd(self) -> float:
        return round(self.max_usd - self.total_usd, 6)

    def summary(self) -> dict:
        return {
            "total_cost_usd": round(self.total_usd, 6),
            "max_usd": self.max_usd,
            "remaining_usd": self.remaining_usd,
            "api_calls": self.call_count,
            "cost_per_call_avg": round(self.total_usd / max(self.call_count, 1), 8),
        }


# ── Layer Configuration ───────────────────────────────────────────

LAYER_NAMES = [
    "L0_bare_loop",
    "L1_tool_dispatch",
    "L2_context_mgmt",
    "L3_observability",
    "L4_input_validation",
    "L5_injection_detection",
    "L6_output_filtering",
    "L7_hitl_gate",
    "L8_plan_reflect",
]

def make_layer_configs() -> list:
    """Generate 9 incremental layer configurations.

    L0 alone, L0+L1, L0+L1+L2, ..., all 9 layers.
    Returns list of (config_name, enabled_layers_set) tuples.
    """
    configs = []
    for i in range(len(LAYER_NAMES)):
        enabled = LAYER_NAMES[:i+1]
        name = "+".join(enabled)
        configs.append((name, set(enabled)))
    return configs

def config_short_name(config_name: str) -> str:
    """L0_bare_loop+L1_tool_dispatch → L0-L1"""
    parts = config_name.split("+")
    if len(parts) == 1:
        return parts[0].split("_")[0]
    return f"{parts[0].split('_')[0]}-{parts[-1].split('_')[0]}"


# ── Manifest ──────────────────────────────────────────────────────

def write_manifest(path: str, version: str, files: dict,
                    cost_tracker: CostTracker = None,
                    test_results: dict = None,
                    notes: str = ""):
    """Write a manifest.json summarizing the version's outputs."""
    manifest = {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pricing_date": PRICING_DATE,
        "models": list(PRICING.keys()),
        "files": files,
        "total_cost_usd": cost_tracker.total_usd if cost_tracker else 0.0,
        "api_calls": cost_tracker.call_count if cost_tracker else 0,
        "test_results": test_results or {},
        "notes": notes,
    }
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return manifest


# ── Run manifest: provenance + dataset integrity (peer review) ────
import hashlib as _hashlib
import subprocess as _subprocess


def _git_commit(repo_dir: str) -> str:
    try:
        r = _subprocess.run(["git", "-C", repo_dir, "rev-parse", "HEAD"],
                            capture_output=True, text=True)
        return r.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def file_sha256(path: str) -> str:
    h = _hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_run_manifest(path, experiment_id, models, seed, config, repo_dir,
                       dataset_path=None, record_count=None):
    """Write a provenance manifest for one run (links to events via experiment_id)."""
    cfg_blob = json.dumps(config, sort_keys=True, default=str)
    manifest = {
        "experiment_id": experiment_id,
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(repo_dir),
        "pricing_date": PRICING_DATE,
        "models": models,
        "seed": seed,
        "config": config,
        "config_hash": _hashlib.sha256(cfg_blob.encode()).hexdigest()[:16],
        "dataset_path": str(dataset_path) if dataset_path else None,
        "record_count": record_count,
        "dataset_sha256": file_sha256(dataset_path) if dataset_path and os.path.exists(dataset_path) else None,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    return manifest
