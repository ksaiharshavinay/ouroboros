# OUROBOROS — ouro/ollama_adapter.py
# Drop-in adapter that talks to a locally running Ollama instance
# via its OpenAI-compatible API (http://localhost:11434/v1).
#
# Copyright (c) 2026 Sai Harsha Vinaykudikala. All rights reserved.
# PATENT PENDING

from __future__ import annotations
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Default Ollama base URL
OLLAMA_BASE_URL = "http://localhost:11434/v1"


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
        used_model = model or self._model
        try:
            response = self._client.chat.completions.create(
                model=used_model,
                messages=messages or [],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content or ""
            return _Response(content)
        except Exception as exc:
            logger.error("[ollama_adapter] API error: %s", exc)
            raise


class _Chat:
    def __init__(self, client: Any, model: str):
        self.completions = _Completions(client, model)


class OllamaAdapter:
    """
    Drop-in replacement for ClaudeAdapter that routes requests to a
    locally running Ollama instance via its OpenAI-compatible REST API.

    Usage:
        from ouro.ollama_adapter import OllamaAdapter
        client = OllamaAdapter(model="llama3.2")
        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": "Hello!"}]
        )
        print(resp.choices[0].message.content)
    """

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = OLLAMA_BASE_URL,
    ):
        from openai import OpenAI  # openai>=1.0 is already a project dependency

        # Ollama's OpenAI-compatible endpoint requires a dummy api_key
        self._client = OpenAI(base_url=base_url, api_key="ollama")
        self._model = model
        self.chat = _Chat(self._client, model)

        logger.info("[ollama_adapter] Initialised — model=%s  base_url=%s", model, base_url)
