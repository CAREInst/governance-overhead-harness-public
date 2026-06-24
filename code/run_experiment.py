"""
run_experiment.py — clean runner for the governance-overhead rerun.

Modes
  smoke   Local Ollama. Tiny subset. Zero external send, zero cost.
          Validates the full harness + governance sweep end-to-end before any
          spend. This is the gate that must pass before a live run.
  full    150 tasks x 5 layer configs x N runs through the OpenRouter gateway.
          Requires BOTH --allow-external AND an OPENROUTER_API_KEY in the environment. The
          external-send gate stays shut otherwise.

Every run writes a provenance manifest next to its dataset: git commit, config
hash, pinned pricing date, seed, and the dataset's SHA-256 + record count.

Examples
  python3 run_experiment.py smoke --model ollama/llama3.2 --tasks 3
  python3 run_experiment.py full  --model openrouter/openai/gpt-4.1-mini \\
          --runs 5 --max-usd 5 --allow-external
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


class Runner(R.LiveExperimentRunner):
    """LiveExperimentRunner with a slash-safe dataset filename."""

    def __init__(self, model: str, output_dir: str, max_usd: float):
        super().__init__(model, output_dir, max_usd)
        safe = model.replace("/", "_")
        self.logger = H.MeasurementLogger(str(self.output_dir / f"{self.provider}_{safe}.jsonl"))


def load_tasks(tiers, limit=None):
    suite = json.loads((PKG / "data" / "task_suite.json").read_text())
    tasks = [t for t in suite["tasks"] if t.get("tier") in tiers]
    return tasks[:limit] if limit else tasks


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


def write_manifest(runner, model, seed, config, out_dir):
    return H.write_run_manifest(
        str(out_dir / "manifest.json"), runner.experiment_id, [model], seed, config, str(PKG),
        dataset_path=str(runner.logger.path), record_count=runner.logger.event_count)


async def run(args) -> int:
    os.environ["EXPERIMENT_SEED"] = str(args.seed)

    if args.mode == "smoke":
        extra_adapters.register_local_model(args.model)
        if not ollama_up(args.model):
            print("SMOKE ABORTED: Ollama is not serving the model. "
                  "Start `ollama serve` and `ollama pull <model>` first.")
            return 3
        tiers, runs, tasks = [1], 1, load_tasks([1], limit=args.tasks)
        out = PKG / "results" / "smoke"
        out.mkdir(parents=True, exist_ok=True)
        runner = Runner(args.model, str(out), max_usd=1.0)   # >0 so the zero-cost cap is not pre-tripped
        print(f"SMOKE: {len(tasks)} task(s) x 5 configs x 1 run on {args.model} (local, $0)")
        result = await runner.run_experiment(tasks, tiers=tiers, runs=runs)
        man = write_manifest(runner, args.model, args.seed,
                             {"mode": "smoke", "tasks": len(tasks), "configs": 5, "runs": runs}, out)
        events = runner.logger.read_all()
        bad = [e for e in events if "API error" in (e.get("governance_detail") or "")]
        print(f"  events: {len(events)} | api errors: {len(bad)} | cost: ${result['total_cost_usd']}")
        print(f"  dataset: {runner.logger.path.name} sha256={man['dataset_sha256'][:12]}... "
              f"records={man['record_count']}")
        ok = len(events) > 0 and not bad and result["total_cost_usd"] == 0.0
        print("SMOKE:", "PASS" if ok else "FAIL")
        return 0 if ok else 4

    # full mode
    if not args.allow_external:
        print("REFUSED: full mode needs --allow-external (operator confirmation) "
              "External send stays gated.")
        return 2
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("REFUSED: set OPENROUTER_API_KEY in your environment for a live run.")
        return 5
    os.environ["EXPERIMENT_ALLOW_EXTERNAL"] = "1"   # open the gate only now
    tiers = [1, 2, 3] if args.all_tiers else [1, 2]
    tasks = load_tasks(tiers)
    out = PKG / "results" / "full"
    out.mkdir(parents=True, exist_ok=True)
    runner = Runner(args.model, str(out), max_usd=args.max_usd)
    print(f"FULL: {len(tasks)} tasks x 5 configs x {args.runs} runs on {args.model} "
          f"(EXTERNAL, cap ${args.max_usd})")
    result = await runner.run_experiment(tasks, tiers=tiers, runs=args.runs)
    runner.write_cost_report()
    man = write_manifest(runner, args.model, args.seed,
                        {"mode": "full", "tasks": len(tasks), "configs": 5,
                         "runs": args.runs, "tiers": tiers}, out)
    print(f"  events: {man['record_count']} | cost: ${result['total_cost_usd']} | "
          f"dataset sha256={man['dataset_sha256'][:12]}...")
    return 0


def main():
    ap = argparse.ArgumentParser(description="governance-overhead rerun runner")
    ap.add_argument("mode", choices=["smoke", "full"])
    ap.add_argument("--model", default="ollama/llama3.2")
    ap.add_argument("--tasks", type=int, default=3, help="smoke: number of tier-1 tasks")
    ap.add_argument("--runs", type=int, default=5, help="full: runs per config")
    ap.add_argument("--max-usd", type=float, default=15.0, help="full: hard cost cap")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--all-tiers", action="store_true", help="full: include Tier 3 (150 tasks)")
    ap.add_argument("--allow-external", action="store_true", help="full: open the external-send gate")
    args = ap.parse_args()
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
