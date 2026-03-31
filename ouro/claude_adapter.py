# OUROBOROS — ouro/claude_adapter.py
# Unified adapter supporting both Anthropic (Claude) and Ollama (local) backends
# behind the same client.chat.completions.create() interface.

from __future__ import annotations
import logging
from typing import Any

logger = logging.getLogger(__name__)


class _Message:
    def __init__(self, content: str):
        self.content = content


class _Choice:
    def __init__(self, content: str):
        self.message = _Message(content)


class _Response:
    def __init__(self, content: str):
        self.choices = [_Choice(content)]


class _Completions:
    def __init__(self, client: Any, model: str, provider: str):
        self._client = client
        self._model = model
        self._provider = provider

    def create(
        self,
        model: str | None = None,
        messages: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        **kwargs,
    ) -> _Response:
        used_model = model or self._model
        messages = messages or []

        if self._provider == "ollama":
            # Ollama exposes an OpenAI-compatible endpoint
            response = self._client.chat.completions.create(
                model=used_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = response.choices[0].message.content or ""
            return _Response(text)
        else:
            # Anthropic SDK
            import anthropic
            try:
                response = self._client.messages.create(
                    model=used_model,
                    max_tokens=max_tokens,
                    messages=messages,
                )
                text = next(
                    (b.text for b in response.content if b.type == "text"), ""
                )
                return _Response(text)
            except anthropic.APIError as exc:
                logger.error("[claude_adapter] API error: %s", exc)
                raise


class _Chat:
    def __init__(self, client: Any, model: str, provider: str):
        self.completions = _Completions(client, model, provider)


class ClaudeAdapter:
    """
    Drop-in LLM client supporting two providers:
      - provider="claude"  → Anthropic SDK (requires ANTHROPIC_API_KEY)
      - provider="ollama"  → OpenAI-compatible local Ollama endpoint
    """

    def __init__(self, api_key: str, model: str, provider: str = "claude", base_url: str = ""):
        self._provider = provider

        if provider == "ollama":
            from openai import OpenAI
            self._client = OpenAI(
                api_key="ollama",  # Ollama ignores the key but SDK requires one
                base_url=base_url or "http://localhost:11434/v1",
            )
        else:
            import anthropic
            self._client = anthropic.Anthropic(api_key=api_key)

        self.chat = _Chat(self._client, model, provider)
