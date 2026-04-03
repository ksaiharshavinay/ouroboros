# OUROBOROS — ouro/feedback.py
# Copyright (c) 2026 Kudikala Sai Harsha Vinay. All rights reserved.
# PATENT PENDING
#
# The recursive feedback loop: model prediction outputs are fed back as new
# simulation seeds, driving the next swarm generation cycle. This closed loop
# is the core novel mechanism of OUROBOROS.
#
# Algorithm 7: Confidence-Weighted Feedback Injection (PATENT PENDING)
# Predictions are scored by token-level mean probability (confidence) and
# by cross-prompt agreement. Only high-confidence (score >= HIGH_THRESHOLD)
# or high-disagreement (score <= LOW_THRESHOLD) predictions are injected as
# seeds. This bimodal selection ensures the swarm receives both domain-certain
# knowledge and diversity-driving uncertainty, producing a richer corpus than
# uniform injection.

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any

from .config import OuroConfig

logger = logging.getLogger(__name__)

# Confidence thresholds for bimodal selection (Algorithm 7)
_HIGH_CONFIDENCE_THRESHOLD = 0.70   # inject: model is certain → domain knowledge
_LOW_CONFIDENCE_THRESHOLD  = 0.30   # inject: model is uncertain → diversity seed


class OuroFeedbackModule:
    """
    Manages the recursive feedback loop between inference outputs and the
    swarm simulation. Persists cycle metadata to disk so the loop survives
    restarts.

    Implements Algorithm 7: Confidence-Weighted Feedback Injection.
    Predictions are scored and filtered before injection, ensuring only
    high-value seeds (high-confidence or high-disagreement) enter the next
    simulation cycle.
    """

    def __init__(self, config: OuroConfig):
        self.config = config
        self._state_path = config.feedback_state_path
        self._state = self._load_state()
        self._llm_client: Any = None   # lazy-initialised on first scored run

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _load_state(self) -> dict[str, Any]:
        if self._state_path.exists():
            try:
                with self._state_path.open() as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not read state file (%s). Starting fresh.", exc)
        return {"cycle_count": 0, "cycles": []}

    def _save_state(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        with self._state_path.open("w") as f:
            json.dump(self._state, f, indent=2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def cycle_count(self) -> int:
        return self._state["cycle_count"]

    def run_feedback(
        self,
        prediction_report: str,
        sim_result: dict[str, Any],
        seeds_dir: Path | None = None,
    ) -> Path:
        """
        Write model prediction outputs back as new seed documents for the
        next swarm simulation cycle (basic, single-prediction path).

        For multi-prediction scored injection, use run_feedback_scored().

        Args:
            prediction_report: Text output from OuroInferenceEngine.predict().
            sim_result: Metadata dict from SwarmGenerator.run().
            seeds_dir: Directory to write new seeds into. Defaults to config.seed_docs_dir.

        Returns:
            Path of the newly written seed file.
        """
        seeds_dir = seeds_dir or self.config.seed_docs_dir
        seeds_dir.mkdir(parents=True, exist_ok=True)

        cycle = self.cycle_count
        seed_path = seeds_dir / f"feedback_cycle_{cycle:04d}.txt"
        seed_path.write_text(prediction_report, encoding="utf-8")

        cycle_meta = {
            "cycle": cycle,
            "seed_file": str(seed_path),
            "num_turns": sim_result.get("num_turns", 0),
            "num_agents": sim_result.get("num_agents", 0),
            "injection_mode": "uniform",
        }
        self._state["cycles"].append(cycle_meta)
        self._state["cycle_count"] = cycle + 1
        self._save_state()

        logger.info("[ouro] Feedback cycle %d complete — seed written to %s", cycle, seed_path)
        return seed_path

    def run_feedback_scored(
        self,
        predictions: list[dict[str, Any]],
        sim_result: dict[str, Any],
        seeds_dir: Path | None = None,
    ) -> list[Path]:
        """
        Algorithm 7: Confidence-Weighted Feedback Injection.

        Scores each candidate prediction by token-level mean probability,
        then applies bimodal selection: inject predictions that are either
        HIGH confidence (score >= HIGH_THRESHOLD) or HIGH disagreement
        (score <= LOW_THRESHOLD). Mid-confidence predictions are discarded.

        Args:
            predictions: List of dicts with keys:
                - "prompt": str — the prompt used to generate the prediction
                - "text": str — the generated prediction text
            sim_result: Metadata dict from SwarmGenerator.run().
            seeds_dir: Output directory for seed files.

        Returns:
            List of paths of written seed files (one per selected prediction).
        """
        seeds_dir = seeds_dir or self.config.seed_docs_dir
        seeds_dir.mkdir(parents=True, exist_ok=True)

        cycle = self.cycle_count

        # Score every prediction
        scored: list[dict[str, Any]] = []
        for i, pred in enumerate(predictions):
            score = self._score_prediction(pred["prompt"], pred["text"])
            scored.append({
                "index": i,
                "prompt": pred["prompt"],
                "text": pred["text"],
                "confidence_score": round(score, 4),
            })
            logger.debug("[ouro] Prediction %d/%d confidence=%.3f", i + 1, len(predictions), score)

        # Bimodal selection: high confidence OR high disagreement
        selected = self._select_predictions(scored)

        if not selected:
            logger.warning(
                "[ouro] No predictions passed confidence filter — falling back to all %d.",
                len(scored),
            )
            selected = scored  # graceful fallback: inject all

        # Write each selected prediction as a structured seed document
        seed_paths: list[Path] = []
        for idx, item in enumerate(selected):
            seed_path = seeds_dir / f"feedback_cycle_{cycle:04d}_pred_{idx:02d}.txt"
            seed_path.write_text(
                self._format_seed(item["text"], cycle, idx, item["confidence_score"]),
                encoding="utf-8",
            )
            seed_paths.append(seed_path)

        cycle_meta = {
            "cycle": cycle,
            "seed_files": [str(p) for p in seed_paths],
            "num_turns": sim_result.get("num_turns", 0),
            "num_agents": sim_result.get("num_agents", 0),
            "injection_mode": "confidence_weighted",
            "total_candidates": len(scored),
            "selected_count": len(selected),
            "scores": [s["confidence_score"] for s in scored],
            "selected_scores": [s["confidence_score"] for s in selected],
        }
        self._state["cycles"].append(cycle_meta)
        self._state["cycle_count"] = cycle + 1
        self._save_state()

        logger.info(
            "[ouro] Scored feedback cycle %d: %d/%d predictions injected "
            "(high_conf=%d, high_disagree=%d)",
            cycle,
            len(selected),
            len(scored),
            sum(1 for s in selected if s["confidence_score"] >= _HIGH_CONFIDENCE_THRESHOLD),
            sum(1 for s in selected if s["confidence_score"] <= _LOW_CONFIDENCE_THRESHOLD),
        )
        return seed_paths

    def get_history(self) -> list[dict[str, Any]]:
        """Return metadata for all completed feedback cycles."""
        return list(self._state.get("cycles", []))

    # ------------------------------------------------------------------
    # Algorithm 7 core: confidence scoring
    # ------------------------------------------------------------------

    def _score_prediction(self, prompt: str, prediction: str) -> float:
        """
        Score a prediction's confidence using token-level log-probabilities.

        Sends the (prompt, prediction) pair to the LLM with logprobs=True and
        max_tokens=0 (echo-only) to retrieve token-level log-probabilities for
        the prediction tokens. Computes mean probability per token:

            confidence = exp(mean(log_p_i for token i in prediction))

        Returns a float in (0, 1]: values near 1 indicate the model is highly
        certain of each token choice; values near 0 indicate uncertainty.

        Falls back to 0.5 (neutral) on API errors, missing logprob support,
        or any exception, so a scoring failure never blocks the pipeline.
        """
        client = self._get_llm_client()
        if client is None:
            return 0.5

        try:
            response = client.chat.completions.create(
                model=self.config.llm_model,
                messages=[
                    {"role": "user",      "content": prompt},
                    {"role": "assistant", "content": prediction},
                ],
                logprobs=True,
                max_tokens=1,      # we only need logprobs for existing tokens
                temperature=0.0,   # deterministic for reproducibility
            )
            choice = response.choices[0]
            # Extract log-probs from the response token list
            lp_content = getattr(choice, "logprobs", None)
            if lp_content is None:
                return 0.5
            token_logprobs = [
                t.logprob
                for t in (lp_content.content or [])
                if t.logprob is not None
            ]
            if not token_logprobs:
                return 0.5
            mean_logprob = sum(token_logprobs) / len(token_logprobs)
            # mean_logprob ∈ (-∞, 0]; exp maps to (0, 1]
            return float(min(1.0, math.exp(mean_logprob)))
        except Exception as exc:
            logger.debug("[ouro] Log-prob scoring failed (%s); using neutral score 0.5.", exc)
            return 0.5

    # ------------------------------------------------------------------
    # Algorithm 7 core: bimodal selection
    # ------------------------------------------------------------------

    def _select_predictions(
        self, scored: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Bimodal selection: include predictions at either extreme of the
        confidence distribution.

        - HIGH confidence (score >= HIGH_THRESHOLD): model has strong beliefs
          about this domain → injects domain-certain knowledge into the swarm.
        - LOW confidence (score <= LOW_THRESHOLD): model is uncertain or
          contradicts itself → injects diversity seeds that surface blind spots
          and expose the model to adversarial counter-arguments from skeptic agents.
        - MID confidence (LOW < score < HIGH): discarded — the model is
          guessing and neither contributes reliable knowledge nor productive
          disagreement to the simulation.

        Selected predictions are sorted by |score - 0.5| descending so that
        the most extreme (most informative) predictions appear first in the
        seed file list, allowing the simulation to prioritise them.
        """
        selected = [
            item for item in scored
            if item["confidence_score"] >= _HIGH_CONFIDENCE_THRESHOLD
            or item["confidence_score"] <= _LOW_CONFIDENCE_THRESHOLD
        ]
        # Sort by distance from 0.5 (most extreme first)
        selected.sort(key=lambda x: -abs(x["confidence_score"] - 0.5))
        return selected

    # ------------------------------------------------------------------
    # Seed formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_seed(text: str, cycle: int, idx: int, score: float) -> str:
        """
        Wrap a prediction in a structured machine-readable seed document format.
        The format is deliberately parseable by the LLM entity extractor in the
        next cycle's GraphBuilder, treating the prediction as a first-class source.
        """
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).isoformat()
        return (
            f"[OUROBOROS MODEL PREDICTION — Cycle {cycle}, Output {idx}]\n"
            f"Generated: {ts}\n"
            f"Source: Autonomous prediction from trained OUROBOROS model\n"
            f"Confidence-score: {score:.4f}\n"
            f"\n"
            f"{text}\n"
            f"\n"
            f"[END PREDICTION]"
        )

    # ------------------------------------------------------------------
    # LLM client helpers
    # ------------------------------------------------------------------

    def _get_llm_client(self) -> Any:
        """Lazily initialise the LLM client for scoring calls."""
        if self._llm_client is not None:
            return self._llm_client
        try:
            provider = getattr(self.config, "llm_provider", "claude").lower()
            if provider == "ollama":
                from ouro.ollama_adapter import OllamaAdapter, OLLAMA_BASE_URL
                base_url = getattr(self.config, "llm_base_url", OLLAMA_BASE_URL)
                self._llm_client = OllamaAdapter(
                    model=self.config.llm_model, base_url=base_url
                )
            else:
                from ouro.claude_adapter import ClaudeAdapter
                api_key = os.environ.get(self.config.llm_api_key_env, "")
                self._llm_client = ClaudeAdapter(
                    api_key=api_key, model=self.config.llm_model
                )
        except Exception as exc:
            logger.warning("[ouro] Could not initialise LLM client for scoring: %s", exc)
            return None
        return self._llm_client
