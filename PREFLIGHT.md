# PREFLIGHT — governance-overhead rerun

Work top to bottom. Do **not** run `full` until every box is checked and you
have explicitly decided to spend.

## A. Environment

```bash
cd code
python3 --version                 # 3.12+
python3 -c "import harness_benchmark, governance_layers, provider_adapters, \
            calibration, validators, extra_adapters, experiment_runner_v3; print('imports OK')"
python3 -c "import calibration; print(calibration.measure_harness_overhead(200)['conclusion'])"  # PASS
```

- [ ] All modules import (flat package).
- [ ] Harness overhead self-test prints `PASS`.

## B. Local model (for the smoke)

```bash
ollama serve >/dev/null 2>&1 &     # if not already running
ollama list                        # pick a model you have, e.g. llama3.2, qwen3
```

- [ ] `ollama list` shows at least one model; note its name as `<m>`.

## C. Credentials (only needed before a LIVE run)

Provide your OpenRouter API key via the environment (never commit it):

- [ ] `export OPENROUTER_API_KEY=sk-or-...` set in the run shell.

## D. External-send gate (verify it is shut)

```bash
python3 -c "import extra_adapters as e; print('external allowed:', e.external_send_allowed())"  # False
```

- [ ] Gate reports **False** by default. (A `full` run opens it only after
      `--allow-external` + a loaded key.)

## E. Smoke — local, zero spend (the GO/NO-GO gate)

```bash
python3 run_experiment.py smoke --model ollama/<m> --tasks 3
```

- [ ] Prints `SMOKE: PASS`, `api errors: 0`, `cost: $0.0`.
- [ ] `results/smoke/manifest.json` exists with a `dataset_sha256` and
      `git_commit`.

## F. Decision to go live

A `full` run sends data to OpenRouter and spends money. Only proceed when:

- [ ] Smoke passed.
- [ ] You confirm the reproduction target (default: full 150 tasks, Tier 1/2/3).
- [ ] `--max-usd` cap set to a number you accept.
- [ ] You (the operator) explicitly say go.

Then see **OPERATORS_MANUAL.md → Live run**.
