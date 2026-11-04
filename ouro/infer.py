# OUROBOROS — ouro/infer.py
# Copyright (c) 2026 Kudikala Sai Harsha Vinay. All rights reserved.
# PATENT PENDING

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

from .config import OuroConfig

logger = logging.getLogger(__name__)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class OuroInferenceEngine:
    """
    Wraps the trained OUROBOROS model for inference.

    Checkpoint contract (Phase 1, §4.2 S1.3): the AutoResearch loop writes
    config.model_dir / "checkpoint"/ — a directory containing a real LoRA
    adapter (adapter_model.safetensors + adapter_config.json + tokenizer
    files) and checkpoint_metadata.json with the adapter's SHA-256 and the
    base model id.

    Behaviour:
      * Checkpoint directory with adapter weights present → load base model
        + adapter locally (HuggingFace) and generate with it. The adapter
        hash actually loaded is recorded in `self.checkpoint_sha256` and
        surfaced in predict_with_metadata().
      * No adapter checkpoint → fall back to the configured LLM API
        (unchanged from before).
      * Legacy pre-Phase-1 checkpoints (a lone JSON file named
        "checkpoint", no weights) count as "exists" for is_ready() but
        cannot be loaded locally; they fall back to the API path.
    """

    def __init__(self, config: OuroConfig):
        self.config = config
        self._model: Any = None
        self._tokenizer: Any = None
        self._is_ready = False
        self._backend: str = "api"  # "local" once a LoRA adapter is loaded
        self.checkpoint_sha256: str | None = None
        self.checkpoint_metadata: dict | None = None

    # ------------------------------------------------------------------
    # Checkpoint discovery
    # ------------------------------------------------------------------

    @property
    def checkpoint_dir(self) -> Path:
        return self.config.model_dir / "checkpoint"

    def has_adapter_checkpoint(self) -> bool:
        """True iff a real, loadable LoRA adapter checkpoint exists."""
        return (
            self.checkpoint_dir.is_dir()
            and (self.checkpoint_dir / "adapter_model.safetensors").exists()
            and (self.checkpoint_dir / "adapter_config.json").exists()
        )

    def is_ready(self) -> bool:
        """Return True if any checkpoint (adapter dir or legacy file) exists."""
        return self.checkpoint_dir.exists()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_local(self) -> None:
        """Load base model + this cycle's LoRA adapter from the checkpoint."""
        # Integrity first (cheap, no ML imports): the adapter on disk must
        # hash to the value recorded when it was trained.
        meta_path = self.checkpoint_dir / "checkpoint_metadata.json"
        metadata: dict = {}
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text())
        self.checkpoint_metadata = metadata

        base_model = (
            metadata.get("base_model")
            or getattr(self.config, "train_base_model", None)
            or "Qwen/Qwen2.5-0.5B-Instruct"
        )
        adapter_file = self.checkpoint_dir / "adapter_model.safetensors"
        actual_hash = _sha256_file(adapter_file)
        recorded_hash = metadata.get("adapter_sha256")
        if recorded_hash and recorded_hash != actual_hash:
            raise RuntimeError(
                "Checkpoint adapter hash mismatch: metadata records "
                f"{recorded_hash[:16]}… but the file on disk hashes to "
                f"{actual_hash[:16]}…. Refusing to run against an "
                "unverifiable checkpoint."
            )
        self.checkpoint_sha256 = actual_hash

        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_dir)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.float32)
        self._model = PeftModel.from_pretrained(base, str(self.checkpoint_dir))
        self._model.eval()
        self._backend = "local"
        logger.info(
            "[ouro] Loaded local model: base=%s adapter=%s sha256=%s",
            base_model, self.checkpoint_dir, actual_hash[:16],
        )

    def load(self) -> None:
        """Load the trained model from disk. Raises if no checkpoint found."""
        if not self.is_ready():
            raise FileNotFoundError(
                f"No checkpoint found at {self.checkpoint_dir}. "
                "Run the AutoResearch training loop first (scripts/2_run_research.py)."
            )
        if self.has_adapter_checkpoint():
            self._load_local()
        else:
            logger.warning(
                "[ouro] Checkpoint at %s has no adapter weights (legacy "
                "metadata-only format); using the LLM API backend.",
                self.checkpoint_dir,
            )
            self._backend = "api"
        self._is_ready = True

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _local_predict(self, prompt: str, max_new_tokens: int, context: str | None = None) -> str:
        import torch

        text = f"{context}\n\n{prompt}" if context else prompt
        inputs = self._tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        generated = output[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()

    def predict_with_metadata(
        self, prompt: str, max_new_tokens: int = 512, context: str | None = None
    ) -> dict:
        """predict() plus the provenance record for the run."""
        text = self.predict(prompt, max_new_tokens=max_new_tokens, context=context)
        return {
            "text": text,
            "backend": self._backend,
            "checkpoint_dir": str(self.checkpoint_dir) if self._backend == "local" else None,
            "checkpoint_sha256": self.checkpoint_sha256 if self._backend == "local" else None,
            "base_model": (self.checkpoint_metadata or {}).get("base_model")
            if self._backend == "local"
            else getattr(self.config, "llm_model", None),
        }

    def predict(self, prompt: str, max_new_tokens: int = 512, context: str | None = None) -> str:
        """
        Run inference on the given prompt.

        If a fine-tuned adapter checkpoint is available it is loaded and
        used for generation; otherwise the configured LLM API is used.
        """
        if not self._is_ready:
            self.load()

        if self._backend == "local" and self._model is not None:
            return self._local_predict(prompt, max_new_tokens, context=context)

        return self._llm_predict(prompt, max_new_tokens, context=context)

    def _llm_predict(self, prompt: str, max_new_tokens: int, context: str | None = None) -> str:
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

            messages: list[dict] = []
            if context:
                messages.append({"role": "system", "content": context})
            messages.append({"role": "user", "content": prompt})

            response = client.chat.completions.create(
                model=self.config.llm_model,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=0.7,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            logger.error("[ouro] LLM API call failed: %s", exc)
            return f"[Inference error: {exc}]"
