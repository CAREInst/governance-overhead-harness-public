"""
provider_adapters.py — Unified Provider Interface for 4 Providers

Normalizes Anthropic, OpenAI, Google, and NVIDIA/Bedrock responses
into a common format for measurement. Each adapter handles rate
limiting with exponential backoff.
"""

import json, time, asyncio, os
from dataclasses import dataclass
from typing import Optional, Protocol

@dataclass
class UnifiedResponse:
    """Provider-agnostic response format."""
    text: str = ""
    tool_calls: list = None
    input_tokens: int = 0
    output_tokens: int = 0
    provider: str = ""
    model: str = ""
    stop_reason: str = ""
    raw_response: dict = None

    def __post_init__(self):
        if self.tool_calls is None:
            self.tool_calls = []
        if self.raw_response is None:
            self.raw_response = {}

    def to_dict(self) -> dict:
        return {
            "text": self.text, "tool_calls": self.tool_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "provider": self.provider, "model": self.model,
            "stop_reason": self.stop_reason,
        }


class ProviderAdapter:
    """Base class for provider adapters."""

    # Retry budget applies UNIFORMLY to every adapter/model — it only changes
    # the tail of calls that were failing outright (transient connection/rate
    # errors); a call that already succeeds within the old budget is unaffected,
    # so this does not alter latency semantics for any previously-collected
    # data. Bumped 3->5 after the 2026-06-29 sweep showed a persistently
    # overloaded OpenRouter backend for meta-llama/llama-3.3-70b-instruct
    # (532 "Connection error" failures even after 3 attempts).
    MAX_RETRIES = 5
    BASE_DELAY = 1.0

    def __init__(self, model: str, provider: str):
        self.model = model
        self.provider = provider

    async def call(self, messages: list, tools: list = None) -> UnifiedResponse:
        """Make an API call with exponential backoff on rate limits.

        v3.02 FIX: Surfaces real error messages instead of swallowing them.
        Retries on rate limits AND transient network errors.
        """
        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                return await self._call_impl(messages, tools)
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                # Retry on rate limits, timeouts, and transient network errors
                retryable = any(k in err_str for k in [
                    "rate", "429", "timeout", "connection",
                    "temporarily", "503", "502", "overloaded",
                    # OpenRouter's usage-accounting metadata can be missing on a
                    # transient basis (confirmed 2026-06-29: 59/60 calls failed
                    # this way in a live run, then 5/5 succeeded on immediate
                    # retest) — treat as retryable rather than a permanent error.
                    "no usage data"])
                if retryable and attempt < self.MAX_RETRIES - 1:
                    delay = self.BASE_DELAY * (2 ** attempt)
                    print(f"  ⚠️ {self.provider} retry {attempt+1}/{self.MAX_RETRIES}: {str(e)[:100]}")
                    await asyncio.sleep(delay)
                else:
                    # Non-retryable error — surface the real message
                    raise RuntimeError(
                        f"{self.provider} API error (attempt {attempt+1}/{self.MAX_RETRIES}): {e}"
                    ) from e
        raise RuntimeError(
            f"Max retries ({self.MAX_RETRIES}) exceeded for {self.provider}. "
            f"Last error: {last_error}")

    async def _call_impl(self, messages: list, tools: list = None) -> UnifiedResponse:
        raise NotImplementedError


class AnthropicAdapter(ProviderAdapter):
    """Adapter for Anthropic Claude models."""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        super().__init__(model, "anthropic")
        self._client = None

    def _get_client(self):
        if not self._client:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    async def _call_impl(self, messages: list, tools: list = None) -> UnifiedResponse:
        client = self._get_client()
        kwargs = {"model": self.model, "max_tokens": 1024, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        response = await asyncio.to_thread(client.messages.create, **kwargs)
        text = ""
        tool_calls = []
        for block in response.content:
            if hasattr(block, "text"):
                text += block.text
            elif block.type == "tool_use":
                tool_calls.append({"name": block.name, "input": block.input,
                                    "id": block.id})
        return UnifiedResponse(
            text=text, tool_calls=tool_calls,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            provider="anthropic", model=self.model,
            stop_reason=response.stop_reason)


class OpenAIAdapter(ProviderAdapter):
    """Adapter for OpenAI GPT models."""

    def __init__(self, model: str = "gpt-4.1"):
        super().__init__(model, "openai")
        self._client = None

    def _get_client(self):
        if not self._client:
            from openai import OpenAI
            self._client = OpenAI()
        return self._client

    async def _call_impl(self, messages: list, tools: list = None) -> UnifiedResponse:
        client = self._get_client()
        # Convert Anthropic-style messages to OpenAI format
        oai_messages = []
        for m in messages:
            if isinstance(m.get("content"), str):
                oai_messages.append({"role": m["role"], "content": m["content"]})
            else:
                oai_messages.append({"role": m["role"],
                                      "content": json.dumps(m["content"])})
        kwargs = {"model": self.model, "max_completion_tokens": 1024, "messages": oai_messages}
        if tools:
            oai_tools = [{"type": "function", "function": {
                "name": t["name"], "description": t.get("description", ""),
                "parameters": t.get("input_schema", t.get("parameters", {}))
            }} for t in tools]
            kwargs["tools"] = oai_tools
        response = await asyncio.to_thread(client.chat.completions.create, **kwargs)
        choice = response.choices[0]
        text = choice.message.content or ""
        tool_calls = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                tool_calls.append({"name": tc.function.name,
                                    "input": json.loads(tc.function.arguments),
                                    "id": tc.id})
        return UnifiedResponse(
            text=text, tool_calls=tool_calls,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            provider="openai", model=self.model,
            stop_reason=choice.finish_reason)


class GoogleAdapter(ProviderAdapter):
    """Adapter for Google Gemini models.

    Maps internal model keys (gemini-3-flash, gemini-3-pro) to
    Google's actual API model identifiers.
    """

    # Internal key → Google API model identifier
    MODEL_MAP = {
        "gemini-3-flash": "gemini-2.5-flash",
        "gemini-3-pro": "gemini-2.5-pro",
    }

    def __init__(self, model: str = "gemini-3-pro"):
        super().__init__(model, "google")
        self._model_obj = None
        self._api_model = self.MODEL_MAP.get(model, model)

    def _get_model(self):
        if not self._model_obj:
            import google.generativeai as genai
            genai.configure(api_key=os.environ.get("GOOGLE_API_KEY", ""))
            self._model_obj = genai.GenerativeModel(self._api_model)
        return self._model_obj

    async def _call_impl(self, messages: list, tools: list = None) -> UnifiedResponse:
        model = self._get_model()
        # Convert to Gemini format — use only the last user message
        prompt = ""
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, str):
                prompt += f"{content}\n"
            else:
                prompt += f"{json.dumps(content)}\n"
        prompt = prompt.strip()
        response = await asyncio.to_thread(model.generate_content, prompt)
        text = ""
        try:
            if response and response.candidates:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "text") and part.text:
                        text += part.text
            elif response and hasattr(response, "text") and response.text:
                text = response.text
        except (ValueError, IndexError, AttributeError) as e:
            text = f"[Response blocked or empty: {str(e)[:100]}]"
        # Token counting (Gemini provides usage_metadata)
        usage = getattr(response, "usage_metadata", None)
        input_tokens = getattr(usage, "prompt_token_count", 0) if usage else len(prompt) // 4
        output_tokens = getattr(usage, "candidates_token_count", 0) if usage else len(text) // 4
        return UnifiedResponse(
            text=text, tool_calls=[],
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            provider="google", model=self.model,
            stop_reason="end_turn")


class NvidiaBedrockAdapter(ProviderAdapter):
    """Adapter for NVIDIA Nemotron 3 Super via AWS Bedrock."""

    def __init__(self, model: str = "nemotron-3-super"):
        super().__init__(model, "nvidia")
        self._client = None

    def _get_client(self):
        if not self._client:
            import boto3
            self._client = boto3.client("bedrock-runtime",
                                         region_name="us-east-1")
        return self._client

    async def _call_impl(self, messages: list, tools: list = None) -> UnifiedResponse:
        client = self._get_client()
        body = {
            "messages": [{"role": m["role"],
                          "content": [{"text": m["content"] if isinstance(m["content"], str) else json.dumps(m["content"])}]}
                         for m in messages],
            "inferenceConfig": {"maxTokens": 1024},
        }
        response = await asyncio.to_thread(
            client.converse,
            modelId="nvidia.nemotron-3-super-v1:0",
            **body)
        text = ""
        for block in response.get("output", {}).get("message", {}).get("content", []):
            if "text" in block:
                text += block["text"]
        usage = response.get("usage", {})
        return UnifiedResponse(
            text=text, tool_calls=[],
            input_tokens=usage.get("inputTokens", 0),
            output_tokens=usage.get("outputTokens", 0),
            provider="nvidia", model=self.model,
            stop_reason=response.get("stopReason", "end_turn"))


class DeepSeekAdapter(ProviderAdapter):
    """Adapter for DeepSeek V4 — OpenAI-compatible API."""

    def __init__(self, model: str = "deepseek-v4"):
        super().__init__(model, "deepseek")
        self._client = None

    def _get_client(self):
        if not self._client:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
                base_url="https://api.deepseek.com")
        return self._client

    async def _call_impl(self, messages: list, tools: list = None) -> UnifiedResponse:
        client = self._get_client()
        oai_messages = []
        for m in messages:
            if isinstance(m.get("content"), str):
                oai_messages.append({"role": m["role"], "content": m["content"]})
            else:
                oai_messages.append({"role": m["role"],
                                      "content": json.dumps(m["content"])})
        kwargs = {"model": "deepseek-chat", "max_completion_tokens": 1024,
                  "messages": oai_messages}
        if tools:
            oai_tools = [{"type": "function", "function": {
                "name": t["name"], "description": t.get("description", ""),
                "parameters": t.get("input_schema", t.get("parameters", {}))
            }} for t in tools]
            kwargs["tools"] = oai_tools
        response = await asyncio.to_thread(client.chat.completions.create, **kwargs)
        choice = response.choices[0]
        text = choice.message.content or ""
        tool_calls = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                tool_calls.append({"name": tc.function.name,
                                    "input": json.loads(tc.function.arguments),
                                    "id": tc.id})
        return UnifiedResponse(
            text=text, tool_calls=tool_calls,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            provider="deepseek", model=self.model,
            stop_reason=choice.finish_reason)


class GrokAdapter(ProviderAdapter):
    """Adapter for xAI Grok 4.1 Fast — OpenAI-compatible API."""

    def __init__(self, model: str = "grok-4.1-fast"):
        super().__init__(model, "xai")
        self._client = None

    def _get_client(self):
        if not self._client:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=os.environ.get("XAI_API_KEY", ""),
                base_url="https://api.x.ai/v1")
        return self._client

    async def _call_impl(self, messages: list, tools: list = None) -> UnifiedResponse:
        client = self._get_client()
        oai_messages = []
        for m in messages:
            if isinstance(m.get("content"), str):
                oai_messages.append({"role": m["role"], "content": m["content"]})
            else:
                oai_messages.append({"role": m["role"],
                                      "content": json.dumps(m["content"])})
        kwargs = {"model": "grok-4.1-fast", "max_completion_tokens": 1024,
                  "messages": oai_messages}
        if tools:
            oai_tools = [{"type": "function", "function": {
                "name": t["name"], "description": t.get("description", ""),
                "parameters": t.get("input_schema", t.get("parameters", {}))
            }} for t in tools]
            kwargs["tools"] = oai_tools
        response = await asyncio.to_thread(client.chat.completions.create, **kwargs)
        choice = response.choices[0]
        text = choice.message.content or ""
        tool_calls = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                tool_calls.append({"name": tc.function.name,
                                    "input": json.loads(tc.function.arguments),
                                    "id": tc.id})
        return UnifiedResponse(
            text=text, tool_calls=tool_calls,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            provider="xai", model=self.model,
            stop_reason=choice.finish_reason)


# ── Factory ───────────────────────────────────────────────────────

ADAPTER_MAP = {
    "claude-haiku-4-5-20251001": AnthropicAdapter,
    "claude-sonnet-4-20250514": AnthropicAdapter,
    "claude-opus-4-20250514": AnthropicAdapter,
    "gpt-4.1-mini": OpenAIAdapter,
    "gpt-4.1": OpenAIAdapter,
    "gemini-3-flash": GoogleAdapter,
    "gemini-3-pro": GoogleAdapter,
    "nemotron-3-super": NvidiaBedrockAdapter,
    "deepseek-v4": DeepSeekAdapter,
    "grok-4.1-fast": GrokAdapter,
}

def get_adapter(model: str) -> ProviderAdapter:
    """Factory: return the correct adapter for a model string."""
    cls = ADAPTER_MAP.get(model)
    if not cls:
        raise ValueError(f"Unknown model: {model}")
    return cls(model)
