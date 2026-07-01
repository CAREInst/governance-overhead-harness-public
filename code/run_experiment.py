"""
run_experiment.py — clean runner for the governance-overhead rerun.

Modes
  smoke   Local Ollama. Tiny subset. Zero external send, zero cost.
          Validates the full harness + governance sweep end-to-end before any
          spend. This is the gate that must pass before a live run.
  full    150 tasks x 5 layer configs x N runs through the OpenRouter gateway
          (+ 21 fp_negative probes for false-positive measurement). Requires
          BOTH --allow-external AND an OPENROUTER_API_KEY in the environment.
  sweep   Same suite across MULTIPLE models (cross-model invariance evidence),
          with per-model and sweep-total cost caps.

Governance knobs (recorded in the run manifest for provenance):
  --preamble {none,short,medium,long}   L4 policy-preamble length (token-overhead ablation)
  --l8-mode  {execute,measure_only}     L8 plan-then-execute vs bill-twice baseline
  --additivity                          isolated single-layer configs (real additivity test)

Examples
  python3 run_experiment.py smoke --model ollama/llama3.2 --tasks 3 --additivity
  python3 run_experiment.py full  --model openrouter/openai/gpt-4.1-mini --runs 5 --max-usd 5 --allow-external
  python3 run_experiment.py sweep --model-set cheap --include-ollama --runs 5 --max-usd-total 20 --allow-external
"""

import argparse
import asyncio
import json
import os
import pathlib
import sys
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(HERE))

import experiment_runner_v3 as R          # noqa: E402
import extra_adapters                     # noqa: E402
import harness_benchmark as H             # noqa: E402

# Route every adapter lookup through the prefix-aware factory.
R.get_adapter = extra_adapters.get_adapter

# Multi-model sweep sets (all priced in extra_adapters / PRICING).
CHEAP_SET = [
    "openrouter/openai/gpt-4.1-mini",
    "openrouter/google/gemini-2.5-flash",
    "openrouter/meta-llama/llama-3.3-70b-instruct",
    "openrouter/mistralai/mistral-small-3.1-24b-instruct",
]
FULL_SET = CHEAP_SET + [
    "openrouter/anthropic/claude-haiku-4.5",
    "openrouter/anthropic/claude-sonnet-4.5",
]


class Runner(R.LiveExperimentRunner):
    """LiveExperimentRunner with a slash-safe, knob-stamped dataset filename."""

    def __init__(self, model: str, output_dir: str, max_usd: float):
        super().__init__(model, output_dir, max_usd)
        safe = model.replace("/", "_")
        # Stamp the preamble variant and L8 mode into the stem so each variant
        # auto-discovers as its own provider in analysis (and its paired cost
        # report inherits the suffix via logger.path.stem).
        pre = os.environ.get("L4_PREAMBLE", "medium")
        l8 = os.environ.get("L8_MODE", "execute")
        stem = f"{self.provider}_{safe}__pre-{pre}__l8-{l8}"
        self.logger = H.MeasurementLogger(str(self.output_dir / f"{stem}.jsonl"))


def load_tasks(tiers, limit=None, include_fp=False):
    suite = json.loads((PKG / "data" / "task_suite.json").read_text())
    tasks = [t for t in suite["tasks"] if t.get("tier") in tiers]
    if limit:
        tasks = tasks[:limit]
    if include_fp:
        # fp_negative probes are benign true-negatives; they run through every
        # config so we can measure real false-positive/precision behavior.
        tasks = tasks + [t for t in suite["tasks"] if t.get("fp_negative")]
    return tasks


def ollama_up(model: str) -> bool:
    """One quick local call to confirm Ollama is serving the model."""
    name = model.split("/", 1)[1] if "/" in model else model
    try:
        body = json.dumps({"model": name, "messages": [{"role": "user", "content": "ping"}],
                           "stream": False, "options": {"temperature": 0, "seed": 1}}).encode()
        req = urllib.request.Request(f"{extra_adapters.OLLAMA_HOST}/api/chat", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            json.loads(r.read())
        return True
    except Exception as e:
        print(f"  Ollama check failed for {name}: {str(e)[:160]}")
        return False


def write_manifest(runner, model, seed, config, out_dir, name="manifest.json"):
    return H.write_run_manifest(
        str(out_dir / name), runner.experiment_id, [model], seed, config, str(PKG),
        dataset_path=str(runner.logger.path), record_count=runner.logger.event_count)


def _open_external_gate(args) -> int:
    """Shared full/sweep preflight: check the key and open the external-send gate.

    Returns 0 on success, or a nonzero refusal code.
    """
    if not args.allow_external:
        print("REFUSED: external modes need --allow-external (operator confirmation) "
              "and an OPENROUTER_API_KEY in the environment. External send stays gated.")
        return 2
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("REFUSED: set OPENROUTER_API_KEY in your environment for a live run.")
        return 5
    os.environ["EXPERIMENT_ALLOW_EXTERNAL"] = "1"   # open the gate only now
    return 0


def _apply_knobs(args):
    """Push the Track-C knobs into the environment before any governed call."""
    os.environ["L4_PREAMBLE"] = getattr(args, "preamble", "medium")
    os.environ["L8_MODE"] = getattr(args, "l8_mode", "execute")


async def _run_one_model(model, args, out, tiers, max_usd) -> dict:
    """Run the full suite (+fp +isolated) for a single model into out/."""
    tasks = load_tasks(tiers, include_fp=not args.no_fp)
    runner = Runner(model, str(out), max_usd=max_usd)
    extra = []
    if args.additivity:
        extra.append("isolated additivity")
    if not args.no_fp:
        extra.append("fp-negative set")
    tail = (" + " + " + ".join(extra)) if extra else ""
    print(f"  {model}: {len(tasks)} tasks x 5 configs x {args.runs} runs "
          f"(pre={os.environ['L4_PREAMBLE']}, l8={os.environ['L8_MODE']}, cap ${max_usd}){tail}")
    result = await runner.run_experiment(tasks, tiers=tiers, runs=args.runs,
                                         include_isolated=args.additivity)
    runner.write_cost_report()
    stem = runner.logger.path.stem
    man = write_manifest(runner, model, args.seed,
                         {"mode": args.mode, "tasks": len(tasks), "configs": 5,
                          "runs": args.runs, "tiers": tiers,
                          "additivity": args.additivity, "fp_set": not args.no_fp,
                          "governance": runner.governance_config()},
                         out, name=f"{stem}_manifest.json")
    print(f"    events: {man['record_count']} | cost: ${result['total_cost_usd']} | "
          f"sha256={man['dataset_sha256'][:12]}...")
    return result


async def run(args) -> int:
    os.environ["EXPERIMENT_SEED"] = str(args.seed)
    _apply_knobs(args)

    if args.mode == "smoke":
        extra_adapters.register_local_model(args.model)
        if not ollama_up(args.model):
            print("SMOKE ABORTED: Ollama is not serving the model. "
                  "Start `ollama serve` and `ollama pull <model>` first.")
            return 3
        tiers, runs = [1], 1
        tasks = load_tasks([1], limit=args.tasks, include_fp=not args.no_fp)
        out = PKG / "results" / "smoke"
        out.mkdir(parents=True, exist_ok=True)
        runner = Runner(args.model, str(out), max_usd=1.0)   # >0 so the zero-cost cap is not pre-tripped
        print(f"SMOKE: {len(tasks)} task(s) x 5 configs x 1 run on {args.model} (local, $0)")
        result = await runner.run_experiment(tasks, tiers=tiers, runs=runs,
                                             include_isolated=args.additivity)
        man = write_manifest(runner, args.model, args.seed,
                             {"mode": "smoke", "tasks": len(tasks), "configs": 5, "runs": runs,
                              "governance": runner.governance_config()}, out)
        events = runner.logger.read_all()
        bad = [e for e in events if "API error" in (e.get("governance_detail") or "")]
        print(f"  events: {len(events)} | api errors: {len(bad)} | cost: ${result['total_cost_usd']}")
        print(f"  dataset: {runner.logger.path.name} sha256={man['dataset_sha256'][:12]}... "
              f"records={man['record_count']}")
        ok = len(events) > 0 and not bad and result["total_cost_usd"] == 0.0
        print("SMOKE:", "PASS" if ok else "FAIL")
        return 0 if ok else 4

    tiers = [1, 2, 3] if args.all_tiers else [1, 2]

    if args.mode == "full":
        rc = _open_external_gate(args)
        if rc:
            return rc
        out = PKG / "results" / "full"
        out.mkdir(parents=True, exist_ok=True)
        print(f"FULL: single model {args.model} (EXTERNAL)")
        await _run_one_model(args.model, args, out, tiers, args.max_usd)
        return 0

    # sweep mode — multiple models, hard total cap
    if args.mode == "sweep":
        models = list(args.models) if args.models else (
            FULL_SET if args.model_set == "full" else CHEAP_SET)
        out = PKG / "results" / "sweep"
        out.mkdir(parents=True, exist_ok=True)

        ollama_models = []
        if args.include_ollama:
            ollama_models = [args.ollama_model]
            extra_adapters.register_local_model(args.ollama_model)

        # Ollama legs are free + deterministic; run them first, no gate/budget.
        for om in ollama_models:
            if not ollama_up(om):
                print(f"  (skip) Ollama not serving {om}")
                continue
            print(f"SWEEP[ollama]: {om} ($0, local)")
            await _run_one_model(om, args, out, tiers, max_usd=1.0)

        if models:
            rc = _open_external_gate(args)
            if rc:
                return rc
        print(f"SWEEP: {len(models)} external model(s), total cap ${args.max_usd_total}")
        spent_total = 0.0
        for m in models:
            remaining = args.max_usd_total - spent_total
            if remaining <= 0.01:
                print(f"  (stop) sweep-total cap ${args.max_usd_total} reached before {m}")
                break
            per_cap = min(args.max_usd, remaining)
            res = await _run_one_model(m, args, out, tiers, max_usd=per_cap)
            spent_total += res.get("total_cost_usd", 0.0)
            print(f"  sweep-total spent so far: ${spent_total:.4f} / ${args.max_usd_total}")
        print(f"SWEEP DONE. total external spend: ${spent_total:.4f}")
        return 0

    return 1


def main():
    ap = argparse.ArgumentParser(description="governance-overhead rerun runner")
    ap.add_argument("mode", choices=["smoke", "full", "sweep"])
    ap.add_argument("--model", default="ollama/llama3.2")
    ap.add_argument("--tasks", type=int, default=3, help="smoke: number of tier-1 tasks")
    ap.add_argument("--runs", type=int, default=5, help="runs per config")
    ap.add_argument("--max-usd", type=float, default=15.0, help="full/per-model hard cost cap")
    ap.add_argument("--max-usd-total", type=float, default=10.0, help="sweep: hard cap across all models")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--all-tiers", action="store_true", help="include Tier 3 (150 tasks)")
    ap.add_argument("--additivity", action="store_true",
                    help="also run isolated single-layer configs (real additivity test)")
    ap.add_argument("--no-fp", action="store_true", help="exclude the fp_negative probe set")
    ap.add_argument("--preamble", choices=["none", "short", "medium", "long"], default="medium",
                    help="L4 policy-preamble length (token-overhead ablation)")
    ap.add_argument("--l8-mode", choices=["execute", "measure_only"], default="execute",
                    help="L8 plan-then-execute (default) vs bill-twice baseline")
    ap.add_argument("--model-set", choices=["cheap", "full"], default="cheap", help="sweep: which model set")
    ap.add_argument("--models", nargs="+", help="sweep: explicit model list (overrides --model-set)")
    ap.add_argument("--include-ollama", action="store_true", help="sweep: prepend a free local Ollama leg")
    ap.add_argument("--ollama-model", default="ollama/llama3.2", help="sweep: which Ollama model")
    ap.add_argument("--allow-external", action="store_true", help="open the external-send gate")
    args = ap.parse_args()
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
