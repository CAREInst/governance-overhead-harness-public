"""
governance_layers.py — 9 Toggleable Governance Layers

Each layer is a function: layer_fn(input_data, context) → (output, decision, latency_ms)
Layers are independently enableable via a set of layer names.
"""

import re, time, json
from dataclasses import dataclass
from typing import Optional

@dataclass
class LayerResult:
    """Result of a governance layer execution."""
    layer: str
    decision: str          # pass | block | modify
    detail: str = ""
    latency_ms: float = 0.0
    modified_output: str = ""


# ── Layer Implementations ─────────────────────────────────────────

def l0_bare_loop(text: str, ctx: dict = None) -> LayerResult:
    """L0: Passthrough — zero overhead baseline."""
    return LayerResult("L0_bare_loop", "pass", "baseline", 0.0)

def l1_tool_dispatch(text: str, ctx: dict = None) -> LayerResult:
    """L1: Tool dispatch permission check."""
    start = time.monotonic()
    tool_name = ctx.get("tool_name", "") if ctx else ""
    permission = ctx.get("permission", "read") if ctx else "read"
    required = {"bash": "execute", "write_file": "write",
                "edit_file": "write", "read_file": "read",
                "get_weather": "read", "calculate": "read"}
    req = required.get(tool_name, "read")
    levels = {"read": 0, "write": 1, "execute": 2}
    granted = levels.get(permission, 0) >= levels.get(req, 0)
    ms = (time.monotonic() - start) * 1000
    return LayerResult("L1_tool_dispatch",
                        "pass" if granted else "block",
                        f"perm={permission} req={req}", ms)

def l2_context_mgmt(text: str, ctx: dict = None) -> LayerResult:
    """L2: Context management — token counting."""
    start = time.monotonic()
    est_tokens = len(text) // 4 if text else 0
    ms = (time.monotonic() - start) * 1000
    return LayerResult("L2_context_mgmt", "pass",
                        f"est_tokens={est_tokens}", ms)

def l3_observability(text: str, ctx: dict = None) -> LayerResult:
    """L3: Observability — structured event logging."""
    start = time.monotonic()
    event = json.dumps({"type": "trace", "ts": time.time(),
                         "text_len": len(text) if text else 0})
    ms = (time.monotonic() - start) * 1000
    return LayerResult("L3_observability", "pass",
                        f"logged {len(event)} bytes", ms)

def l4_input_validation(text: str, ctx: dict = None) -> LayerResult:
    """L4: Input validation — empty, oversize, null bytes."""
    start = time.monotonic()
    if not text or not text.strip():
        ms = (time.monotonic() - start) * 1000
        return LayerResult("L4_input_validation", "block", "empty input", ms)
    if len(text) > 50_000:
        ms = (time.monotonic() - start) * 1000
        return LayerResult("L4_input_validation", "block",
                            f"oversize: {len(text)}", ms)
    if '\x00' in text:
        ms = (time.monotonic() - start) * 1000
        return LayerResult("L4_input_validation", "block", "null byte", ms)
    ms = (time.monotonic() - start) * 1000
    return LayerResult("L4_input_validation", "pass", "valid", ms)

# Injection detection patterns
INJECTION_PATTERNS = [
    (r'ignore\s+(all\s+)?previous\s+instructions', "instruction_override", 0.95),
    (r'you\s+are\s+now\s+(?:DAN|evil|unfiltered)', "role_hijack", 0.90),
    (r'<\|(?:system|im_start|endoftext)\|>', "special_token_injection", 0.95),
    (r'system\s*:\s*you\s+are', "system_prompt_injection", 0.85),
    (r'forget\s+(?:everything|all|your)\s+(?:instructions|rules)', "memory_wipe", 0.90),
    (r'(?:print|reveal|show|output)\s+(?:your|the)\s+system\s+prompt', "prompt_leak", 0.85),
    (r'base64\s*[:=]\s*[A-Za-z0-9+/]{20,}', "encoding_attack", 0.80),
    (r'(?:sudo|admin|root)\s+(?:mode|access|override)', "privilege_escalation", 0.80),
    (r'\{\{.*\}\}.*\{\{', "template_injection", 0.75),
    (r'(?:IGNORE|OVERRIDE|BYPASS)\s+(?:SAFETY|FILTER|RULES)', "caps_override", 0.85),
]
COMPILED_PATTERNS = [(re.compile(p, re.I), name, conf)
                      for p, name, conf in INJECTION_PATTERNS]

def l5_injection_detection(text: str, ctx: dict = None) -> LayerResult:
    """L5: Injection detection — 10 regex patterns."""
    start = time.monotonic()
    if not text:
        ms = (time.monotonic() - start) * 1000
        return LayerResult("L5_injection_detection", "pass", "empty", ms)
    for pattern, name, confidence in COMPILED_PATTERNS:
        if pattern.search(text):
            ms = (time.monotonic() - start) * 1000
            return LayerResult("L5_injection_detection", "block",
                                f"{name} (conf={confidence})", ms)
    ms = (time.monotonic() - start) * 1000
    return LayerResult("L5_injection_detection", "pass", "clean", ms)

PII_PATTERNS = [
    (r'\b\d{3}-\d{2}-\d{4}\b', "ssn"),
    (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', "email"),
    (r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', "phone"),
    (r'\b(?:\d{4}[-\s]?){3}\d{4}\b', "credit_card"),
    (r'(?:sk-|api[_-]?key[=:\s])\S{10,}', "api_key"),
]
COMPILED_PII = [(re.compile(p), name) for p, name in PII_PATTERNS]

def l6_output_filtering(text: str, ctx: dict = None) -> LayerResult:
    """L6: Output filtering — PII detection."""
    start = time.monotonic()
    if not text:
        ms = (time.monotonic() - start) * 1000
        return LayerResult("L6_output_filtering", "pass", "empty", ms)
    for pattern, name in COMPILED_PII:
        if pattern.search(text):
            ms = (time.monotonic() - start) * 1000
            return LayerResult("L6_output_filtering", "modify",
                                f"PII detected: {name}", ms)
    ms = (time.monotonic() - start) * 1000
    return LayerResult("L6_output_filtering", "pass", "clean", ms)

RISK_KEYWORDS_HIGH = {"delete", "drop", "remove", "deploy", "config",
                       "production", "migrate", "destroy", "truncate"}
RISK_KEYWORDS_LOW = {"read", "list", "get", "search", "test", "check",
                      "print", "show", "describe", "count"}

def l7_hitl_gate(text: str, ctx: dict = None) -> LayerResult:
    """L7: HITL gate — risk assessment on tool calls."""
    start = time.monotonic()
    if not text:
        ms = (time.monotonic() - start) * 1000
        return LayerResult("L7_hitl_gate", "pass", "no text", ms)
    words = set(text.lower().split())
    high = words & RISK_KEYWORDS_HIGH
    low = words & RISK_KEYWORDS_LOW
    if high and not low:
        ms = (time.monotonic() - start) * 1000
        return LayerResult("L7_hitl_gate", "block",
                            f"HIGH risk: {high}", ms)
    elif high and low:
        ms = (time.monotonic() - start) * 1000
        return LayerResult("L7_hitl_gate", "pass",
                            f"MEDIUM risk: {high} mitigated by {low}", ms)
    ms = (time.monotonic() - start) * 1000
    return LayerResult("L7_hitl_gate", "pass", "LOW risk", ms)

def l8_plan_reflect(text: str, ctx: dict = None) -> LayerResult:
    """L8: Plan-reflect stub — plan generation overhead simulation.

    MEASUREMENT NOTE: This layer measures structural planning overhead only
    (data structure construction, ~1μs). The planning API call itself is
    excluded from measurement to isolate harness cost. See paper Section 3.2.
    """
    start = time.monotonic()
    plan = {"goal": text[:100] if text else "",
            "steps": [{"step": 1, "action": "analyze"},
                      {"step": 2, "action": "execute"},
                      {"step": 3, "action": "verify"}],
            "status": "generated"}
    plan_json = json.dumps(plan)
    ms = (time.monotonic() - start) * 1000
    return LayerResult("L8_plan_reflect", "pass",
                        f"plan {len(plan_json)} bytes", ms)


# ── Layer Registry ────────────────────────────────────────────────

LAYER_REGISTRY = {
    "L0_bare_loop": l0_bare_loop,
    "L1_tool_dispatch": l1_tool_dispatch,
    "L2_context_mgmt": l2_context_mgmt,
    "L3_observability": l3_observability,
    "L4_input_validation": l4_input_validation,
    "L5_injection_detection": l5_injection_detection,
    "L6_output_filtering": l6_output_filtering,
    "L7_hitl_gate": l7_hitl_gate,
    "L8_plan_reflect": l8_plan_reflect,
}

def run_governance_stack(text: str, enabled_layers: set,
                          ctx: dict = None) -> list:
    """Run all enabled governance layers in order.

    Returns list of LayerResult for each enabled layer.
    """
    results = []
    for layer_name in [
        "L0_bare_loop", "L1_tool_dispatch", "L2_context_mgmt",
        "L3_observability", "L4_input_validation",
        "L5_injection_detection", "L6_output_filtering",
        "L7_hitl_gate", "L8_plan_reflect"
    ]:
        if layer_name in enabled_layers:
            fn = LAYER_REGISTRY[layer_name]
            result = fn(text, ctx)
            results.append(result)
            if result.decision == "block":
                break  # Stop on first block
    return results
