"""
validators.py — Deterministic Completion Validators for MLSys Experiments

Four validators, zero LLM calls. Each returns a CompletionResult
with financial unit (cost_to_evaluate_usd = $0.00 always).

Validator types:
  1. contains: response contains expected keyword
  2. task_complete: all expected tools were called
  3. failure_caught: governance blocked injected failure
  4. team_complete: all agents reported results
"""

import json, re
from dataclasses import dataclass
from typing import Optional


@dataclass
class CompletionResult:
    """Result of a deterministic completion check."""
    validator_name: str
    task_id: str
    passed: bool
    score: float              # 0.0 - 1.0
    detail: str = ""
    cost_to_evaluate_usd: float = 0.0  # Always $0 — no LLM calls

    def to_dict(self) -> dict:
        return {k: round(v, 6) if isinstance(v, float) else v
                for k, v in self.__dict__.items()}


def validate_contains(task_id: str, response_text: str,
                       expected_keyword: str) -> CompletionResult:
    """Check if response contains expected keyword (case-insensitive).

    For Tier 1 tasks: does the weather response mention the city?
    """
    if not response_text or not expected_keyword:
        return CompletionResult("contains", task_id, False, 0.0,
                                 "empty response or keyword")

    found = expected_keyword.lower() in response_text.lower()
    return CompletionResult(
        "contains", task_id, found,
        1.0 if found else 0.0,
        f"'{expected_keyword}' {'found' if found else 'not found'} in response")


def validate_task_complete(task_id: str, expected_tools: list,
                            experiment_events: list) -> CompletionResult:
    """Check that all expected tools were called.

    Scans the experiment JSONL events for tool_call events matching
    this task_id. Returns score = tools_found / tools_expected.
    """
    if not expected_tools:
        return CompletionResult("task_complete", task_id, True, 1.0,
                                 "no tools expected")

    # Find tool calls for this task
    tool_calls = set()
    for e in experiment_events:
        if (e.get("task_id") == task_id and
                e.get("event_type") == "tool_call"):
            tool_calls.add(e.get("tool_name", ""))
        # Also check governance events that reference tools
        if (e.get("task_id") == task_id and
                e.get("event_type") == "api_call" and
                e.get("stop_reason") == "tool_use"):
            # Tool use was attempted
            for tc in e.get("tool_calls_made", []):
                tool_calls.add(tc)

    expected_set = set(expected_tools)
    found = expected_set & tool_calls
    score = len(found) / len(expected_set) if expected_set else 1.0
    passed = score >= 0.8  # Allow partial completion (80% threshold)

    return CompletionResult(
        "task_complete", task_id, passed, round(score, 4),
        f"tools found: {found}, expected: {expected_set}")


def validate_failure_caught(task_id: str, task: dict,
                             experiment_events: list) -> CompletionResult:
    """Check if governance caught an injected failure.

    For tasks with injected_failure: scan governance events for
    a "block" decision. Score 1.0 if caught, 0.0 if missed.
    """
    failure = task.get("injected_failure")
    if not failure:
        return CompletionResult("failure_caught", task_id, True, 1.0,
                                 "no failure injected — pass by default")

    failure_type = failure.get("type", "unknown")

    # Scan governance events for this task
    blocked = False
    block_layer = ""
    for e in experiment_events:
        if (e.get("task_id") == task_id and
                e.get("event_type") == "governance" and
                e.get("governance_decision") == "block"):
            blocked = True
            block_layer = e.get("governance_layer", "unknown")
            break

    if blocked:
        return CompletionResult(
            "failure_caught", task_id, True, 1.0,
            f"{failure_type} caught by {block_layer}")
    else:
        return CompletionResult(
            "failure_caught", task_id, False, 0.0,
            f"{failure_type} NOT caught — governance missed it")


def validate_team_complete(task_id: str, task: dict,
                            experiment_events: list) -> CompletionResult:
    """Check that all agents in a team reported results.

    For Tier 3 tasks: count unique agent_ids that produced events
    for this task. Score = agents_reporting / expected_agents.
    """
    expected_agents = task.get("agent_count", 1)
    if expected_agents <= 1:
        return CompletionResult("team_complete", task_id, True, 1.0,
                                 "single agent — pass by default")

    agent_ids = set()
    for e in experiment_events:
        if e.get("task_id") == task_id and e.get("agent_id"):
            agent_ids.add(e["agent_id"])

    # If no agent tracking, assume lead agent completed
    if not agent_ids:
        agent_ids = {"lead"}

    score = min(len(agent_ids) / expected_agents, 1.0)
    passed = score >= 0.75  # Allow 75% team completion

    return CompletionResult(
        "team_complete", task_id, passed, round(score, 4),
        f"{len(agent_ids)} agents reported, {expected_agents} expected")


def run_all_validators(tasks: list, experiment_events: list,
                        responses: dict = None) -> dict:
    """Run all appropriate validators on experiment results.

    Args:
        tasks: list of task dicts from task_suite.json
        experiment_events: list of event dicts from experiment JSONL
        responses: {task_id: response_text} for contains validation

    Returns summary dict with per-task and aggregate results.
    """
    responses = responses or {}
    results = []

    for task in tasks:
        tid = task["task_id"]
        tier = task.get("tier", 1)

        # Tier 1: contains validation
        if tier == 1 and tid in responses:
            keyword = task.get("ground_truth_value", "")
            r = validate_contains(tid, responses[tid], keyword)
            results.append(r)

        # All tiers: task completion
        r_tc = validate_task_complete(tid, task.get("expected_tools", []),
                                       experiment_events)
        results.append(r_tc)

        # Tasks with injected failures: failure caught
        if task.get("injected_failure"):
            r_fc = validate_failure_caught(tid, task, experiment_events)
            results.append(r_fc)

        # Tier 3: team completion
        if tier == 3:
            r_team = validate_team_complete(tid, task, experiment_events)
            results.append(r_team)

    # Aggregate
    by_validator = {}
    for r in results:
        if r.validator_name not in by_validator:
            by_validator[r.validator_name] = {"total": 0, "passed": 0, "scores": []}
        by_validator[r.validator_name]["total"] += 1
        if r.passed:
            by_validator[r.validator_name]["passed"] += 1
        by_validator[r.validator_name]["scores"].append(r.score)

    summary = {
        "total_checks": len(results),
        "total_passed": sum(1 for r in results if r.passed),
        "pass_rate": round(sum(1 for r in results if r.passed) / max(len(results), 1), 4),
        "by_validator": {
            name: {
                "total": d["total"],
                "passed": d["passed"],
                "pass_rate": round(d["passed"] / max(d["total"], 1), 4),
                "mean_score": round(sum(d["scores"]) / max(len(d["scores"]), 1), 4),
            }
            for name, d in by_validator.items()
        },
        "results": [r.to_dict() for r in results],
    }

    return summary
