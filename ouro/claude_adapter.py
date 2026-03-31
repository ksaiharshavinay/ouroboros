# OUROBOROS — ouro/claude_adapter.py
# Thin adapter that wraps the Anthropic SDK behind the same
# client.chat.completions.create() interface the rest of the code expects.

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
    def __init__(self, client: Any, model: str):
        self._client = client
        self._model = model

    def create(
        self,
        model: str | None = None,
        messages: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        **kwargs,
    ) -> _Response:
        import anthropic

        used_model = model or self._model
        try:
            response = self._client.messages.create(
                model=used_model,
                max_tokens=max_tokens,
                messages=messages or [],
            )
            text = next(
                (b.text for b in response.content if b.type == "text"), ""
            )
            return _Response(text)
        except anthropic.APIError as exc:
            logger.error("[claude_adapter] API error: %s", exc)
            raise


class _Chat:
    def __init__(self, client: Any, model: str):
        self.completions = _Completions(client, model)


class ClaudeAdapter:
    """Drop-in replacement for the OpenAI client used in this project."""

    def __init__(self, api_key: str, model: str = "claude-haiku-4-5"):
        import anthropic
        self._client = anthropic.Anthropic(api_key=api_key)
        self.chat = _Chat(self._client, model)
