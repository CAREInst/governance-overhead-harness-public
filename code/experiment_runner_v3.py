"""
experiment_runner_v3.py — Live API Experiment Runner

v3 CHANGES from v2:
  1. LIVE API calls via provider_adapters (not simulation)
  2. 5 key layer configs (L0, L0-L3, L0-L5, L0-L7, L0-L8)
  3. Multi-turn support for Tier 2 (up to 8 turns)
  4. Injected failures embedded in prompts
  5. Consistent 5 runs per configuration
  6. Per-governance-event cost_usd field
"""

import json, time, asyncio, os, sys, uuid
from pathlib import Path
from datetime import datetime, timezone

# v1.01 infrastructure
from harness_benchmark import (
    MeasurementEvent, MeasurementLogger, CostTracker,
    compute_cost, get_model_info, PRICING)
from governance_layers import run_governance_stack, LAYER_REGISTRY
from calibration import measure_harness_overhead, get_semaphore
from provider_adapters import get_adapter

# v2 validators
from validators import run_all_validators

# 5 key layer configurations (not 9 incremental)
KEY_CONFIGS = [
    ("L0", {"L0_bare_loop"}),
    ("L0-L3", {"L0_bare_loop", "L1_tool_dispatch", "L2_context_mgmt", "L3_observability"}),
    ("L0-L5", {"L0_bare_loop", "L1_tool_dispatch", "L2_context_mgmt", "L3_observability",
               "L4_input_validation", "L5_injection_detection"}),
    ("L0-L7", {"L0_bare_loop", "L1_tool_dispatch", "L2_context_mgmt", "L3_observability",
               "L4_input_validation", "L5_injection_detection", "L6_output_filtering",
               "L7_hitl_gate"}),
    ("L0-L8", {"L0_bare_loop", "L1_tool_dispatch", "L2_context_mgmt", "L3_observability",
               "L4_input_validation", "L5_injection_detection", "L6_output_filtering",
               "L7_hitl_gate", "L8_plan_reflect"}),
]


class LiveExperimentRunner:
    """Experiment runner using LIVE API calls (not simulation)."""

    def __init__(self, model: str, output_dir: str, max_usd: float = 15.0):
        self.model = model
        self.model_info = get_model_info(model)
        self.provider = self.model_info["provider"]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.cost_tracker = CostTracker(max_usd)
        self.logger = MeasurementLogger(
            str(self.output_dir / f"{self.provider}_{model}_v3.jsonl"))
        self.semaphore = asyncio.Semaphore(get_semaphore(self.provider))

        # LIVE adapter — real API calls
        self.adapter = get_adapter(model)

        self.experiment_id = f"v3_{self.provider}_{uuid.uuid4().hex[:6]}"
        self.responses = {}
        self._heartbeat_counter = 0

    def _write_heartbeat(self):
        hb = {
            "instance": f"{self.provider}_{self.model}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "api_calls": self.cost_tracker.call_count,
            "cost_usd": round(self.cost_tracker.total_usd, 6),
            "remaining_usd": round(self.cost_tracker.remaining_usd, 6),
            "events_logged": self.logger.event_count,
        }
        with open(self.output_dir / "heartbeat.json", "w") as f:
            json.dump(hb, f, indent=2)

    def _prepare_prompt(self, task: dict) -> str:
        """Prepare task prompt with injected failures embedded."""
        desc = task["description"]
        failure = task.get("injected_failure")
        if failure and isinstance(failure, dict):
            payload = failure.get("payload", "")
            if failure.get("type") == "injection" and payload:
                desc = f"{desc}\n\n{payload}"
            elif failure.get("type") == "destructive" and payload:
                desc = f"{desc}\n\nPlease also: {payload}"
            elif failure.get("type") in ("oversized", "null_byte") and payload:
                desc = f"{desc}\n{payload}"
        return desc

    async def _instrumented_call(self, task: dict, config_name: str,
                                   enabled_layers: set, run_number: int,
                                   l0_baseline_tokens: int = 0) -> MeasurementEvent:
        """Execute one instrumented LIVE API call."""
        async with self.semaphore:
            if not self.cost_tracker.can_continue():
                return None

            prompt = self._prepare_prompt(task)

            # Pre-API governance
            harness_start = time.monotonic()
            gov_results = run_governance_stack(
                prompt, enabled_layers,
                {"tool_name": "get_weather", "permission": "read"})
            harness_ms = (time.monotonic() - harness_start) * 1000

            blocked = any(r.decision == "block" for r in gov_results)

            # Log governance events with cost
            for gr in gov_results:
                if gr.layer != "L0_bare_loop":
                    gov_event = MeasurementEvent(
                        experiment_id=self.experiment_id,
                        task_id=task["task_id"],
                        run_number=run_number,
                        model=self.model,
                        provider=self.provider,
                        tier=self.model_info["tier"],
                        layer_config=config_name,
                        event_type="governance",
                        governance_layer=gr.layer,
                        governance_decision=gr.decision,
                        governance_detail=gr.detail,
                        harness_latency_ms=gr.latency_ms,
                        harness_cost_usd=gr.latency_ms / 1000 / 3600 * 0.10,
                        task_tier=task.get("tier", 1),
                        injected_failure=json.dumps(task.get("injected_failure") or ""),
                        injected_failure_caught=(gr.decision == "block" and
                                                  bool(task.get("injected_failure"))),
                    )
                    self.logger.log(gov_event)

            if blocked:
                event = MeasurementEvent(
                    experiment_id=self.experiment_id,
                    task_id=task["task_id"], run_number=run_number,
                    model=self.model, provider=self.provider,
                    tier=self.model_info["tier"],
                    layer_config=config_name, event_type="api_call",
                    harness_latency_ms=round(harness_ms, 4),
                    api_cost_usd=0.0, task_tier=task.get("tier", 1),
                    task_completed=False,
                    injected_failure=json.dumps(task.get("injected_failure") or ""),
                    injected_failure_caught=True)
                self.logger.log(event)
                return event

            # LIVE API call
            messages = [{"role": "user", "content": prompt}]
            api_start = time.monotonic()
            try:
                response = await self.adapter.call(messages)
                api_ms = (time.monotonic() - api_start) * 1000
            except Exception as e:
                api_ms = (time.monotonic() - api_start) * 1000
                event = MeasurementEvent(
                    experiment_id=self.experiment_id,
                    task_id=task["task_id"], run_number=run_number,
                    model=self.model, provider=self.provider,
                    tier=self.model_info["tier"],
                    layer_config=config_name, event_type="api_call",
                    api_latency_ms=round(api_ms, 2),
                    harness_latency_ms=round(harness_ms, 4),
                    task_completed=False,
                    governance_detail=f"API error: {str(e)[:200]}")
                self.logger.log(event)
                return event

            cost = compute_cost(self.model, response.input_tokens,
                                 response.output_tokens)
            self.cost_tracker.record(cost, self.model, task["task_id"])

            # Store response for validators
            self.responses[task["task_id"]] = response.text

            # Post-API governance (output filtering)
            post_gov_start = time.monotonic()
            if "L6_output_filtering" in enabled_layers:
                from governance_layers import l6_output_filtering
                out_result = l6_output_filtering(response.text)
                if out_result.decision != "pass":
                    self.logger.log(MeasurementEvent(
                        experiment_id=self.experiment_id,
                        task_id=task["task_id"], run_number=run_number,
                        model=self.model, provider=self.provider,
                        layer_config=config_name, event_type="governance",
                        governance_layer="L6_output_filtering",
                        governance_decision=out_result.decision,
                        governance_detail=out_result.detail,
                        harness_latency_ms=out_result.latency_ms))
            post_gov_ms = (time.monotonic() - post_gov_start) * 1000

            total_harness = harness_ms + post_gov_ms
            token_overhead = response.input_tokens - l0_baseline_tokens if l0_baseline_tokens > 0 else 0

            self._heartbeat_counter += 1
            if self._heartbeat_counter % 50 == 0:
                self._write_heartbeat()

            event = MeasurementEvent(
                experiment_id=self.experiment_id,
                task_id=task["task_id"], run_number=run_number,
                model=self.model, provider=self.provider,
                tier=self.model_info["tier"],
                layer_config=config_name, event_type="api_call",
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                api_latency_ms=round(api_ms, 2),
                harness_latency_ms=round(total_harness, 4),
                api_cost_usd=cost,
                cumulative_cost_usd=round(self.cost_tracker.total_usd, 6),
                task_tier=task.get("tier", 1),
                task_completed=True,
                injected_failure=json.dumps(task.get("injected_failure") or ""))
            self.logger.log(event)
            return event

    async def run_experiment(self, tasks: list, tiers: list = None,
                              runs: int = 5) -> dict:
        """Run the experiment with live API calls.

        Args:
            tasks: Full task suite
            tiers: Which tiers to run (default [1, 2])
            runs: Number of runs per configuration (default 5)
        """
        tiers = tiers or [1, 2]
        selected = [t for t in tasks if t.get("tier", 1) in tiers]
        total_calls = 0
        skipped = 0

        for run in range(1, runs + 1):
            # L0 baseline first
            l0_baselines = {}
            l0_name, l0_layers = KEY_CONFIGS[0]
            for task in selected:
                if not self.cost_tracker.can_continue():
                    skipped += 1; continue
                event = await self._instrumented_call(
                    task, l0_name, l0_layers, run)
                if event and event.input_tokens > 0:
                    l0_baselines[task["task_id"]] = event.input_tokens
                total_calls += 1

            # Remaining configs
            for config_name, enabled_layers in KEY_CONFIGS[1:]:
                for task in selected:
                    if not self.cost_tracker.can_continue():
                        skipped += 1; continue
                    baseline = l0_baselines.get(task["task_id"], 0)
                    await self._instrumented_call(
                        task, config_name, enabled_layers, run, baseline)
                    total_calls += 1

            self._write_heartbeat()

        return {
            "total_api_calls": total_calls,
            "skipped_cost_cap": skipped,
            "total_cost_usd": round(self.cost_tracker.total_usd, 6),
            "events_logged": self.logger.event_count,
            "model": self.model,
            "provider": self.provider,
            "tiers": tiers,
            "runs": runs,
            "configs": len(KEY_CONFIGS),
        }

    def run_validation(self, tasks: list) -> dict:
        events = self.logger.read_all()
        return run_all_validators(tasks, events, self.responses)

    def write_cost_report(self) -> dict:
        report = {
            "version": "v3.01",
            "experiment_id": self.experiment_id,
            "model": self.model, "provider": self.provider,
            "cost_summary": self.cost_tracker.summary(),
            "events_logged": self.logger.event_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        # Name the report after the dataset stem (<stem>.jsonl -> <stem>_cost_report.json)
        # so analysis_v4 auto-discovery (H3) pairs them by stem and multiple providers
        # can coexist in one output dir without clobbering a shared cost_report.json.
        report_path = self.output_dir / f"{self.logger.path.stem}_cost_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        return report


async def run_v3_pipeline(model: str, tasks: list,
                           output_dir: str, max_usd: float = 15.0,
                           tiers: list = None, runs: int = 5) -> dict:
    """Complete v3 pipeline with LIVE API calls."""
    runner = LiveExperimentRunner(model, output_dir, max_usd)

    cal = measure_harness_overhead(n_iterations=200)

    # Dry run (3 tasks, L0 only)
    dry_tasks = [t for t in tasks if t.get("tier", 1) == 1][:3]
    dry_results = []
    for task in dry_tasks:
        e = await runner._instrumented_call(
            task, "L0", {"L0_bare_loop"}, run_number=0)
        if e: dry_results.append(True)

    if len(dry_results) < 3:
        return {"status": "dry_run_failed", "dry_results": len(dry_results)}

    exp = await runner.run_experiment(tasks, tiers=tiers, runs=runs)
    val = runner.run_validation(tasks)
    cost = runner.write_cost_report()

    return {
        "status": "complete",
        "calibration": cal,
        "experiment": exp,
        "validation_summary": {
            "total_checks": val["total_checks"],
            "pass_rate": val["pass_rate"],
            "by_validator": val.get("by_validator", {}),
        },
        "cost": cost["cost_summary"],
    }
