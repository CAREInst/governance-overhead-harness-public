"""
calibration.py — Measurement Overhead Self-Test + Calibration Runner

Before any experiment, validate that the measurement harness itself
introduces near-zero overhead. Then run a sequential calibration to
establish baseline API latency per provider.
"""

import time, json, asyncio, os
from harness_benchmark import (MeasurementEvent, MeasurementLogger,
                                CostTracker, compute_cost, PRICING)
from governance_layers import run_governance_stack, LAYER_REGISTRY

# ── Measurement Overhead Self-Test ────────────────────────────────

def measure_harness_overhead(n_iterations: int = 1000) -> dict:
    """Measure the overhead of the measurement harness itself.

    Runs governance layers on synthetic input n_iterations times,
    reports mean/p95/max overhead in microseconds.
    """
    text = "What is the weather in Tokyo today? Please tell me the temperature."

    # L0-L3: lightweight layers
    light_times = []
    for _ in range(n_iterations):
        start = time.monotonic()
        run_governance_stack(text, {"L0_bare_loop", "L1_tool_dispatch",
                                     "L2_context_mgmt", "L3_observability"},
                              {"tool_name": "get_weather", "permission": "read"})
        light_times.append((time.monotonic() - start) * 1e6)  # microseconds

    # L0-L8: all layers
    full_times = []
    for _ in range(n_iterations):
        start = time.monotonic()
        run_governance_stack(text, set(LAYER_REGISTRY.keys()),
                              {"tool_name": "get_weather", "permission": "read"})
        full_times.append((time.monotonic() - start) * 1e6)

    # Event serialization
    serial_times = []
    for _ in range(n_iterations):
        e = MeasurementEvent(experiment_id="cal", task_id="T0", model="test",
                              input_tokens=500, output_tokens=200)
        start = time.monotonic()
        _ = e.to_json()
        serial_times.append((time.monotonic() - start) * 1e6)

    from statistics import mean

    def percentile(data, p):
        s = sorted(data)
        if not s:
            return 0.0
        k = (len(s) - 1) * (p / 100)
        lo = int(k); hi = min(lo + 1, len(s) - 1)
        return s[lo] + (s[hi] - s[lo]) * (k - lo)
    return {
        "n_iterations": n_iterations,
        "light_stack_L0_L3": {
            "mean_us": round(mean(light_times), 2),
            "p95_us": round(percentile(light_times, 95), 2),
            "max_us": round(max(light_times), 2),
        },
        "full_stack_L0_L8": {
            "mean_us": round(mean(full_times), 2),
            "p95_us": round(percentile(full_times, 95), 2),
            "max_us": round(max(full_times), 2),
        },
        "event_serialization": {
            "mean_us": round(mean(serial_times), 2),
            "p95_us": round(percentile(serial_times, 95), 2),
            "max_us": round(max(serial_times), 2),
        },
        "conclusion": "PASS" if mean(full_times) < 1000 else "WARNING: harness overhead > 1ms",
    }


# ── Provider Semaphore Configuration ──────────────────────────────

PROVIDER_SEMAPHORES = {
    "anthropic": 8,
    "openai": 15,
    "google": 5,
    "nvidia": 5,
    "deepseek": 10,
    "xai": 10,
    "ollama": 1,
    "openrouter": 10,
}

def get_semaphore(provider: str) -> int:
    """Return the recommended concurrent call limit for a provider."""
    return PROVIDER_SEMAPHORES.get(provider, 5)


# ── Calibration Suite ─────────────────────────────────────────────

CALIBRATION_TASKS = [
    {"task_id": "CAL-001", "prompt": "What is 2 + 2?"},
    {"task_id": "CAL-002", "prompt": "Name three primary colors."},
    {"task_id": "CAL-003", "prompt": "What is the capital of France?"},
    {"task_id": "CAL-004", "prompt": "How many days are in a week?"},
    {"task_id": "CAL-005", "prompt": "What is H2O?"},
    {"task_id": "CAL-006", "prompt": "Name a mammal."},
    {"task_id": "CAL-007", "prompt": "What comes after Tuesday?"},
    {"task_id": "CAL-008", "prompt": "Is the Earth round?"},
    {"task_id": "CAL-009", "prompt": "What is 10 × 5?"},
    {"task_id": "CAL-010", "prompt": "Name a programming language."},
]
