"""
extra_adapters.py — local + gated-external adapters for the rerun.

Two additions to the published provider set:
  - OllamaAdapter:     local model via Ollama. Zero external send, zero cost,
                       deterministic (temperature=0 + seed). Used for the smoke.
  - OpenRouterAdapter: single external gateway (OpenAI-compatible). GATED — it
                       refuses to call unless external send is explicitly
                       enabled, so a local run can never leak to the network.

Model-string convention (the prefix selects the adapter):
  ollama/<name>                     -> OllamaAdapter        (e.g. ollama/llama3.2)
  openrouter/<vendor>/<model>       -> OpenRouterAdapter    (e.g. openrouter/openai/gpt-4.1-mini)
  <native key>                      -> the original adapter (Anthropic/OpenAI/...)

The clean runner imports get_adapter from THIS module so the prefixes work.
"""

import json
import os
import urllib.request

import harness_benchmark as _H
from provider_adapters import (ProviderAdapter, UnifiedResponse,
                               get_adapter as _base_get_adapter)

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def experiment_seed() -> int:
    """The pinned seed for deterministic generation (override via EXPERIMENT_SEED)."""
    try:
        return int(os.environ.get("EXPERIMENT_SEED", "42"))
    except ValueError:
        return 42


def external_send_allowed() -> bool:
    """True only when the operator has explicitly opened the external-send gate."""
    return os.environ.get("EXPERIMENT_ALLOW_EXTERNAL", "").strip().lower() in ("1", "true", "yes")


class OllamaAdapter(ProviderAdapter):
    """Local Ollama model. No network egress, no cost, deterministic."""

    def __init__(self, model: str):
        super().__init__(model, "ollama")
        self._name = model.split("/", 1)[1] if "/" in model else model

    async def _call_impl(self, messages: list, tools: list = None) -> UnifiedResponse:
        import asyncio

        def _do() -> dict:
            body = json.dumps({
                "model": self._name,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0, "seed": experiment_seed()},
            }).encode("utf-8")
            req = urllib.request.Request(
                f"{OLLAMA_HOST}/api/chat", data=body,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as resp:
                return json.loads(resp.read())

        data = await asyncio.to_thread(_do)
        text = (data.get("message") or {}).get("content", "")
        return UnifiedResponse(
            text=text,
            input_tokens=int(data.get("prompt_eval_count", 0)),
            output_tokens=int(data.get("eval_count", 0)),
            provider="ollama", model=self.model,
            stop_reason=data.get("done_reason", "stop"))


class OpenRouterAdapter(ProviderAdapter):
    """Single external gateway via OpenRouter (OpenAI-compatible). Gated."""

    def __init__(self, model: str):
        super().__init__(model, "openrouter")
        self._route = model.split("/", 1)[1] if "/" in model else model
        self._client = None

    def _get_client(self):
        if not self._client:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=os.environ.get("OPENROUTER_API_KEY", ""),
                base_url="https://openrouter.ai/api/v1")
        return self._client

    async def _call_impl(self, messages: list, tools: list = None) -> UnifiedResponse:
        if not external_send_allowed():
            raise RuntimeError(
                "External send BLOCKED: OpenRouter call refused because "
                "EXPERIMENT_ALLOW_EXTERNAL is not set. Open the gate in the "
                "preflight before any live-provider run.")
        import asyncio
        oai = [{"role": m["role"],
                "content": m["content"] if isinstance(m.get("content"), str)
                else json.dumps(m["content"])} for m in messages]

        def _do():
            return self._get_client().chat.completions.create(
                model=self._route, messages=oai, max_completion_tokens=1024,
                temperature=0, seed=experiment_seed())

        resp = await asyncio.to_thread(_do)
        ch = resp.choices[0]
        return UnifiedResponse(
            text=ch.message.content or "",
            input_tokens=resp.usage.prompt_tokens,
            output_tokens=resp.usage.completion_tokens,
            provider="openrouter", model=self.model,
            stop_reason=ch.finish_reason)


def register_local_model(model: str) -> None:
    """Register an Ollama model as zero-cost so compute_cost accepts it."""
    _H.PRICING.setdefault(
        model, {"input": 0.0, "output": 0.0, "provider": "ollama", "tier": "local"})


def register_openrouter_model(model: str, input_usd: float, output_usd: float,
                              tier: str = "external") -> None:
    """Pin per-token pricing for an OpenRouter route (USD per 1M tokens)."""
    _H.PRICING.setdefault(
        model, {"input": input_usd, "output": output_usd,
                "provider": "openrouter", "tier": tier})


# OpenRouter route pricing (USD per 1M tokens), confirmed against the live
# OpenRouter catalog on 2026-06-22 — these are current list prices, not the
# published-v3 pins, so compute_cost (and the --max-usd cap it drives) tracks
# the real bill. Re-confirm route ids/prices before any later rerun.
register_openrouter_model("openrouter/anthropic/claude-haiku-4.5", 1.00, 5.00, "budget")
register_openrouter_model("openrouter/openai/gpt-4.1-mini", 0.40, 1.60, "budget")
register_openrouter_model("openrouter/google/gemini-2.5-flash", 0.30, 2.50, "budget")
register_openrouter_model("openrouter/meta-llama/llama-3.3-70b-instruct", 0.10, 0.32, "open-weight")
register_openrouter_model("openrouter/mistralai/mistral-small-3.1-24b-instruct", 0.351, 0.555, "open-weight")
register_openrouter_model("openrouter/anthropic/claude-sonnet-4.5", 3.00, 15.00, "mid")


def get_adapter(model: str) -> ProviderAdapter:
    """Prefix-aware factory: ollama/* local, openrouter/* gated, else native."""
    if model.startswith("ollama/"):
        return OllamaAdapter(model)
    if model.startswith("openrouter/"):
        return OpenRouterAdapter(model)
    return _base_get_adapter(model)
