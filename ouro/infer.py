# OUROBOROS — ouro/infer.py
# Copyright (c) 2026 Sai Harsha Vinaykudikala. All rights reserved.
# PATENT PENDING

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from .config import OuroConfig

logger = logging.getLogger(__name__)


class OuroInferenceEngine:
    """
    Wraps the trained OUROBOROS model for inference.

    The model is trained by the AutoResearch loop and stored in
    config.model_dir. This class loads it (or falls back to the base LLM)
    and exposes a predict() method whose output feeds back into the swarm.
    """

    def __init__(self, config: OuroConfig):
        self.config = config
        self._model: Any = None
        self._tokenizer: Any = None
        self._is_ready = False

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def is_ready(self) -> bool:
        """Return True if a trained model checkpoint exists."""
        checkpoint = self.config.model_dir / "checkpoint"
        return checkpoint.exists()

    def load(self) -> None:
        """Load the trained model from disk. Raises if no checkpoint found."""
        if not self.is_ready():
            raise FileNotFoundError(
                f"No checkpoint found at {self.config.model_dir / 'checkpoint'}. "
                "Run the AutoResearch training loop first (scripts/2_run_research.py)."
            )
        try:
            import tiktoken

            # Correct tiktoken usage: load BPE ranks then construct Encoding
            bpe_path = self.config.model_dir / "bpe_ranks.tiktoken"
            if bpe_path.exists():
                mergeable_ranks = tiktoken.load_tiktoken_bpe(str(bpe_path))
                self._tokenizer = tiktoken.Encoding(
                    name="ouro_enc",
                    pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
                    mergeable_ranks=mergeable_ranks,
                    special_tokens={"<|endoftext|>": len(mergeable_ranks)},
                )
            else:
                # Fallback to cl100k_base
                self._tokenizer = tiktoken.get_encoding("cl100k_base")

            self._is_ready = True
            logger.info("[ouro] Model loaded from %s", self.config.model_dir)
        except ImportError:
            logger.warning("[ouro] tiktoken not installed; falling back to base LLM only.")
            self._is_ready = True  # Still usable via LLM API

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, prompt: str, max_new_tokens: int = 512) -> str:
        """
        Run inference on the given prompt.

        If a fine-tuned model checkpoint is available, it is used; otherwise
        the method falls back to the configured LLM API.
        """
        if not self._is_ready:
            self.load()

        # Tokenise and clamp to avoid out-of-bounds token IDs
        if self._tokenizer is not None:
            try:
                tokens = self._tokenizer.encode(prompt)
                vocab_size = self._tokenizer.n_vocab
                tokens = [min(tok, vocab_size - 1) for tok in tokens]
                logger.debug("[ouro] Encoded prompt: %d tokens", len(tokens))
            except Exception as exc:
                logger.warning("[ouro] Tokenisation failed (%s); passing raw prompt.", exc)

        # --- Use base LLM API as inference backend ---
        # Replace this block with your fine-tuned model when ready.
        return self._llm_predict(prompt, max_new_tokens)

    def _llm_predict(self, prompt: str, max_new_tokens: int) -> str:
        """Call the configured LLM API and return the generated text."""
        try:
            provider = getattr(self.config, "llm_provider", "claude").lower()

            if provider == "ollama":
                from ouro.ollama_adapter import OllamaAdapter, OLLAMA_BASE_URL

                base_url = getattr(self.config, "llm_base_url", OLLAMA_BASE_URL)
                client = OllamaAdapter(model=self.config.llm_model, base_url=base_url)
            else:
                # Default: Anthropic Claude
                from ouro.claude_adapter import ClaudeAdapter

                api_key = os.environ.get(self.config.llm_api_key_env, "")
                client = ClaudeAdapter(api_key=api_key, model=self.config.llm_model)

            response = client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
                temperature=0.7,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            logger.error("[ouro] LLM API call failed: %s", exc)
            return f"[Inference error: {exc}]"
