# OUROBOROS — swarm/report.py
# Copyright (c) 2026 Kudikala Sai Harsha Vinay. All rights reserved.
# PATENT PENDING

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Analyses agent interaction turns using a ReACT-style prompting strategy
    and produces a prediction report. The report text becomes the seed for
    the next feedback cycle.
    """

    def __init__(self, llm_client: Any, model: str = "gpt-4o-mini"):
        self._llm = llm_client
        self._model = model

    def generate(self, turns: list[dict[str, Any]], top_entities: list[dict[str, Any]]) -> str:
        """
        Generate a prediction report from simulation turns.

        Args:
            turns: List of agent interaction dicts (each with at least 'content').
            top_entities: Top entities from GraphBuilder.get_top_entities().

        Returns:
            A text report summarising trends and predictions.
        """
        sample = self._sample_turns(turns, max_turns=30)
        entity_str = ", ".join(e.get("name", "") for e in top_entities[:10])

        prompt = (
            "You are analysing a social simulation. "
            "Below are representative agent interactions from the simulation.\n\n"
            f"Key entities: {entity_str}\n\n"
            "Interactions:\n" +
            "\n".join(f"- {t.get('content', '')}" for t in sample) +
            "\n\nUsing a step-by-step ReACT approach:\n"
            "1. Observe: What are the dominant themes?\n"
            "2. Reason: What social dynamics are emerging?\n"
            "3. Predict: What will happen in the next cycle?\n"
            "Write a structured prediction report (3-5 paragraphs)."
        )
        try:
            response = self._llm.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1024,
            )
            return response.choices[0].message.content or "[Empty report]"
        except Exception as exc:
            logger.error("[report] LLM call failed (%s); returning fallback summary.", exc)
            themes = list({t.get("content", "")[:60] for t in sample})[:5]
            return (
                "Report generation failed. Observed themes:\n"
                + "\n".join(f"- {th}" for th in themes)
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sample_turns(self, turns: list[dict[str, Any]], max_turns: int = 30) -> list[dict[str, Any]]:
        """
        Return a deduplicated sample: first 15 turns + up to 15 high-influence turns,
        with no duplicates (by turn identity, not content).
        """
        seen_ids: set[int] = set()
        result: list[dict[str, Any]] = []

        def add(t: dict[str, Any]) -> None:
            tid = id(t)
            if tid not in seen_ids:
                seen_ids.add(tid)
                result.append(t)

        for t in turns[:15]:
            add(t)

        high_influence = sorted(
            turns, key=lambda t: t.get("influence", 0), reverse=True
        )[:15]
        for t in high_influence:
            add(t)
            if len(result) >= max_turns:
                break

        return result
