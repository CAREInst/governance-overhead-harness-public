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

import json, time, asyncio, os, sys, uuid, math, hashlib, random
from pathlib import Path
from datetime import datetime, timezone

# v1.01 infrastructure
from harness_benchmark import (
    MeasurementEvent, MeasurementLogger, CostTracker,
    compute_cost, get_model_info, PRICING)
from governance_layers import run_governance_stack, LAYER_REGISTRY, LAYER_ORDER
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


def make_isolated_configs() -> list:
    """Isolated single-layer configs for a genuine additivity test (FATAL-2).

    ISO-BASE is the bare loop; each ISO-Lx is the bare loop plus exactly one
    governance layer. The marginal cost of layer x measured in isolation is
    (ISO-Lx − ISO-BASE); summing those independent marginals and comparing to
    the measured full stack is a real, non-telescoping additivity test.
    """
    configs = [("ISO-BASE", {"L0_bare_loop"})]
    for layer in LAYER_ORDER:
        if layer == "L0_bare_loop":
            continue
        short = layer.split("_")[0]            # "L4_input_validation" -> "L4"
        configs.append((f"ISO-{short}", {"L0_bare_loop", layer}))
    return configs


# ── Modeled human-in-the-loop review latency (MAJOR-10) ───────────
# L7 escalations route to a human. We MODEL the human decision latency with a
# lognormal and report it SEPARATELY from harness latency (never folded in).
#
# Distribution family: lognormal — right-skew/heavy-tail is empirically
# universal across human-review reference classes (SOC triage, incident MTTA,
# PR review, CAB), and PR-latency studies log-transform latency for exactly
# this reason. DEFAULT median anchored to PagerDuty incident MTTA (p50 = 2.82
# min; p50 recommended over mean), with SOC alert triage (~90% closed in ~5
# min, 20+ min tail) corroborating sigma ≈ 1.0. L7 models a FAST automated
# acknowledge/triage gate, NOT a deliberate code-review/CAB process (those are
# the upper-tail reference classes in the sensitivity sweep).
# Overridable at call time via env so the offline sweep can drive the grid.
HITL_MEDIAN_MS = int(os.environ.get("HITL_MEDIAN_MS", 3 * 60 * 1000))  # 3 min (PagerDuty MTTA p50)
HITL_SIGMA = float(os.environ.get("HITL_SIGMA", 1.0))                  # SOC-triage-tier tail
HITL_ANCHOR = ("lognormal; median=PagerDuty MTTA p50 2.82min, sigma~1.0 from "
               "SOC triage (~90% <5min, 20+min tail); MODELED acknowledge/triage "
               "gate, not code-review/CAB; report as a {median,sigma} sweep")


def model_hitl_wait_ms(task_id: str, run_number: int, seed: int = 42,
                       median_ms: float = None, sigma: float = None) -> float:
    """Deterministically sample a modeled human-review latency (ms).

    Seeded by (seed, task, run) only — NOT config — so the same task escalated
    under different layer configs yields the same human wait (it is the same
    human reviewing the same request). The run index varies the draw across the
    repeated runs so the modeled latency is distributional, not a constant.

    NOTE: this is a MODELED scalar, not a measurement. The distribution params
    (HITL_MEDIAN_MS, HITL_SIGMA) are an assumption recorded in the run manifest;
    the paper must treat human-review latency as modeled/unvalidated, reported
    separately from (never summed into) measured harness latency.
    """
    key = f"{seed}:{task_id}:{run_number}".encode()
    h = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
    rng = random.Random(h)
    mu = math.log(median_ms if median_ms is not None else HITL_MEDIAN_MS)
    return round(math.exp(rng.gauss(mu, sigma if sigma is not None else HITL_SIGMA)), 1)


# Cheap planning prompt for the real L8 plan-reflect round-trip (B3).
PLANNING_INSTRUCTION = (
    "Produce a concise 3-step plan (one short line each) to accomplish the "
    "following task. Output only the plan.\n\nTask: ")


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

        # Track-B/C behavior switches.
        self.l5_mode = os.environ.get("L5_MODE", "block")   # "block" | "sanitize"
        self.l8_mode = os.environ.get("L8_MODE", "execute")  # "execute" | "measure_only"
        self.model_hitl = True      # MAJOR-10: model human-review latency on L7 escalation
        try:
            self.seed = int(os.environ.get("EXPERIMENT_SEED", "42"))
        except ValueError:
            self.seed = 42

    def governance_config(self) -> dict:
        """Provenance of the v1.2.0 governance knobs, for the run manifest.

        Records exactly which controllable surface produced a dataset so the
        config_hash can distinguish materially different runs (e.g. L5 block vs
        sanitize, a different preamble, different HITL assumptions).
        """
        import governance_layers as _G
        preamble = _G.active_preamble()
        return {
            "l5_mode": self.l5_mode,
            "l8_mode": self.l8_mode,
            "model_hitl": self.model_hitl,
            "hitl_median_ms": HITL_MEDIAN_MS,
            "hitl_sigma": HITL_SIGMA,
            "hitl_distribution": "lognormal(mu=ln(median), sigma); MODELED, not measured",
            "hitl_anchor": HITL_ANCHOR,
            "l4_preamble_variant": os.environ.get("L4_PREAMBLE", "medium"),
            "policy_preamble_chars": len(preamble),
            "policy_preamble_est_tokens": len(preamble) // 4,
            "policy_preamble_sha256": hashlib.sha256(preamble.encode()).hexdigest()[:16],
            "planning_instruction_sha256": hashlib.sha256(
                PLANNING_INSTRUCTION.encode()).hexdigest()[:16],
        }

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

    async def _planning_call(self, task_prompt: str):
        """L8 real plan-reflect: a planning API round-trip (B3 + C3).

        Returns (plan_text, latency_ms, cost_usd). The planning cost is tracked
        SEPARATELY (planning_cost_usd) so L8's round-trip cost is visible. Under
        L8_MODE=execute the returned plan_text is threaded into the executed
        main prompt (plan-then-execute) so L8 has a genuine effect on the main
        call — not a fire-and-forget extra bill.
        """
        messages = [{"role": "user", "content": PLANNING_INSTRUCTION + task_prompt[:500]}]
        start = time.monotonic()
        resp = await self.adapter.call(messages)
        ms = (time.monotonic() - start) * 1000
        cost = compute_cost(self.model, resp.input_tokens, resp.output_tokens)
        self.cost_tracker.record(cost, self.model, "L8_planning")
        return (resp.text or "").strip(), round(ms, 2), cost

    async def _instrumented_call(self, task: dict, config_name: str,
                                   enabled_layers: set, run_number: int,
                                   l0_baseline_tokens: int = 0) -> MeasurementEvent:
        """Execute one instrumented LIVE API call with real governance I/O."""
        async with self.semaphore:
            if not self.cost_tracker.can_continue():
                return None

            prompt = self._prepare_prompt(task)
            ctx = {"tool_name": "get_weather", "permission": "read",
                   "l5_mode": self.l5_mode}

            # ── Pre-API governance (prompt phase). Mutations are threaded. ──
            harness_start = time.monotonic()
            pre = run_governance_stack(prompt, enabled_layers, ctx, phase="pre")
            harness_ms = (time.monotonic() - harness_start) * 1000

            effective_prompt = pre.final_text
            prompt_modified = effective_prompt != prompt

            has_injection = bool(task.get("injected_failure"))
            # A governance layer CATCHES an injected failure if it blocks it OR if
            # L5 sanitizes (neutralizes) it (decision == "modify"). L4 preamble and
            # L6 PII redaction are also "modify" but are NOT injection catches, so
            # only L5's modify counts here.
            def catches_injection(gr):
                return (gr.decision == "block"
                        or (gr.layer == "L5_injection_detection"
                            and gr.decision == "modify"))
            injection_sanitized = any(
                gr.layer == "L5_injection_detection" and gr.decision == "modify"
                for gr in pre.results)

            # L7 escalation → model human-review latency (reported separately).
            # Seeded by (seed, task, run) — NOT config: the same task escalated in
            # L0-L7 vs L0-L8 represents the same human reviewing the same request,
            # so the modeled wait must not differ by config. run varies the draw.
            hitl_wait = 0.0
            for gr in pre.results:
                if gr.layer == "L7_hitl_gate" and gr.decision == "block" and self.model_hitl:
                    hitl_wait = model_hitl_wait_ms(task["task_id"], run_number, self.seed)

            # Log pre-API governance events.
            for gr in pre.results:
                if gr.layer == "L0_bare_loop":
                    continue
                self.logger.log(MeasurementEvent(
                    experiment_id=self.experiment_id,
                    task_id=task["task_id"], run_number=run_number,
                    model=self.model, provider=self.provider,
                    tier=self.model_info["tier"], layer_config=config_name,
                    event_type="governance",
                    governance_layer=gr.layer, governance_decision=gr.decision,
                    governance_detail=gr.detail,
                    harness_latency_ms=gr.latency_ms,
                    harness_cost_usd=gr.latency_ms / 1000 / 3600 * 0.10,
                    hitl_wait_ms=(hitl_wait if gr.layer == "L7_hitl_gate" else 0.0),
                    prompt_modified=(gr.decision == "modify"),
                    task_tier=task.get("tier", 1),
                    injected_failure=json.dumps(task.get("injected_failure") or ""),
                    injected_failure_caught=(catches_injection(gr) and has_injection),
                ))

            if pre.blocked:
                event = MeasurementEvent(
                    experiment_id=self.experiment_id,
                    task_id=task["task_id"], run_number=run_number,
                    model=self.model, provider=self.provider,
                    tier=self.model_info["tier"],
                    layer_config=config_name, event_type="api_call",
                    harness_latency_ms=round(harness_ms, 4),
                    hitl_wait_ms=hitl_wait,
                    api_cost_usd=0.0, task_tier=task.get("tier", 1),
                    task_completed=False,
                    injected_failure=json.dumps(task.get("injected_failure") or ""),
                    injected_failure_caught=has_injection)
                self.logger.log(event)
                return event

            # ── L8 real plan-reflect: planning API round-trip (B3 + C3) ──
            planning_ms, planning_cost = 0.0, 0.0
            plan_text, plan_injected = "", False
            if "L8_plan_reflect" in enabled_layers:
                if not self.cost_tracker.can_continue():
                    return None
                try:
                    plan_text, planning_ms, planning_cost = await self._planning_call(effective_prompt)
                except Exception:
                    plan_text, planning_ms, planning_cost = "", 0.0, 0.0

            # ── Main LIVE API call ──
            # plan-then-execute: under L8_MODE=execute the plan (generated FROM
            # the governed prompt) is appended to the EXECUTED prompt, so it adds
            # real main-call input tokens and can affect the response.
            # measure_only reproduces the prior bill-twice-no-effect baseline.
            main_content = effective_prompt
            if plan_text and self.l8_mode == "execute":
                main_content = (f"{effective_prompt}\n\n[PLAN] Follow this plan to "
                                f"complete the task:\n{plan_text}\n")
                plan_injected = True
            messages = [{"role": "user", "content": main_content}]
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
                    planning_api_ms=planning_ms, planning_cost_usd=planning_cost,
                    task_completed=False,
                    governance_detail=f"API error: {str(e)[:200]}")
                self.logger.log(event)
                return event

            cost = compute_cost(self.model, response.input_tokens,
                                 response.output_tokens)
            self.cost_tracker.record(cost, self.model, task["task_id"])

            # ── Post-API governance (output phase): L6 redaction ──
            post_gov_start = time.monotonic()
            post = run_governance_stack(response.text, enabled_layers, ctx, phase="post")
            post_gov_ms = (time.monotonic() - post_gov_start) * 1000
            final_output = post.final_text
            output_modified = final_output != response.text
            # Log EVERY post-phase layer result (pass and modify), matching the
            # pre-phase coverage, so "L6 ran clean" is distinguishable from "L6
            # never ran" — otherwise the L6 denominator is biased.
            for gr in post.results:
                self.logger.log(MeasurementEvent(
                    experiment_id=self.experiment_id,
                    task_id=task["task_id"], run_number=run_number,
                    model=self.model, provider=self.provider,
                    tier=self.model_info["tier"], layer_config=config_name,
                    event_type="governance",
                    governance_layer=gr.layer,
                    governance_decision=gr.decision,
                    governance_detail=gr.detail,
                    harness_latency_ms=gr.latency_ms,
                    harness_cost_usd=gr.latency_ms / 1000 / 3600 * 0.10,
                    output_modified=(gr.decision == "modify")))

            # Store the GOVERNED (redacted) response for validators.
            self.responses[task["task_id"]] = final_output

            total_harness = harness_ms + post_gov_ms
            token_overhead = (response.input_tokens - l0_baseline_tokens
                              if l0_baseline_tokens > 0 else 0)

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
                planning_api_ms=planning_ms,
                planning_cost_usd=planning_cost,
                plan_chars=len(plan_text),
                plan_injected=plan_injected,
                api_cost_usd=cost,
                cumulative_cost_usd=round(self.cost_tracker.total_usd, 6),
                token_overhead=token_overhead,
                prompt_modified=prompt_modified,
                injection_sanitized=injection_sanitized,
                output_modified=output_modified,
                task_tier=task.get("tier", 1),
                task_completed=True,
                injected_failure=json.dumps(task.get("injected_failure") or ""),
                # A completed call still COUNTS as catching the injection if L5
                # sanitized it (neutralized, then allowed through).
                injected_failure_caught=(injection_sanitized and has_injection))
            self.logger.log(event)
            return event

    async def run_experiment(self, tasks: list, tiers: list = None,
                              runs: int = 5, include_isolated: bool = False,
                              isolated_tiers: list = None) -> dict:
        """Run the experiment with live API calls.

        Args:
            tasks: Full task suite
            tiers: Which tiers to run (default [1, 2])
            runs: Number of runs per configuration (default 5)
            include_isolated: also run isolated single-layer configs for a real
                additivity test (B5/FATAL-2). To bound cost these run on a
                smaller task set (isolated_tiers, default [1]) — harness timing
                per layer is task-independent, so a subset suffices.
            isolated_tiers: tiers used for the isolated additivity sweep.
        """
        tiers = tiers or [1, 2]
        # Include fp_negative probes (string tier) alongside the integer tiers —
        # they are benign true-negatives that must run through every config to
        # measure false positives.
        selected = [t for t in tasks
                    if t.get("tier", 1) in tiers or t.get("fp_negative")]
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

        # ── Isolated single-layer additivity sweep (B5/FATAL-2) ──
        if include_isolated:
            iso_selected = [t for t in tasks
                            if t.get("tier", 1) in (isolated_tiers or [1])]
            for run in range(1, runs + 1):
                for config_name, enabled_layers in make_isolated_configs():
                    for task in iso_selected:
                        if not self.cost_tracker.can_continue():
                            skipped += 1; continue
                        await self._instrumented_call(
                            task, config_name, enabled_layers, run)
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
            "configs": len(KEY_CONFIGS) + (len(make_isolated_configs()) if include_isolated else 0),
            "isolated_included": include_isolated,
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
