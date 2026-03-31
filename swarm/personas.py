# OUROBOROS — swarm/personas.py
# Copyright (c) 2026 Sai Harsha Vinaykudikala. All rights reserved.
# PATENT PENDING

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


class PersonaGenerator:
    """
    Generates N agent personas for the OASIS swarm simulation.
    Each persona has a name, background, and a set of beliefs seeded from
    the knowledge graph's top entities.
    """

    DEFAULT_PERSONA_TEMPLATE = {
        "name": "Agent",
        "background": "A curious observer.",
        "beliefs": [],
        "influence_score": 0.5,
    }

    def __init__(self, llm_client: Any, model: str = "gpt-4o-mini"):
        self._llm = llm_client
        self._model = model

    def generate(self, n: int, top_entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Generate n agent personas informed by top_entities from the knowledge graph.

        Returns:
            List of persona dicts compatible with the OASIS agent schema.
        """
        entity_names = [e.get("name", "") for e in top_entities[:20]]
        prompt = (
            f"Generate {n} diverse social-media agent personas for a simulation. "
            "Each persona should be a JSON object with keys: "
            "name, background (1-2 sentences), beliefs (list of 3-5 short strings), "
            "influence_score (float 0-1). "
            f"Ground the beliefs in these topics: {', '.join(entity_names)}. "
            f"Return ONLY a JSON array of {n} objects."
        )
        try:
            response = self._llm.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,
                max_tokens=min(4096, n * 120),
            )
            raw = response.choices[0].message.content or "[]"
            # Strip ```json ... ``` fences robustly
            raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
            raw = re.sub(r"\s*```$", "", raw.strip())
            personas = json.loads(raw)
            if not isinstance(personas, list):
                raise ValueError("Expected a JSON array of personas.")
            return personas[:n]
        except Exception as exc:
            logger.error("[personas] Generation failed (%s); using defaults.", exc)
            return [
                {**self.DEFAULT_PERSONA_TEMPLATE, "name": f"Agent_{i}"}
                for i in range(n)
            ]

    @staticmethod
    def to_oasis_format(personas: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Convert our persona schema to the format expected by OASIS's
        generate_twitter_agent_graph / generate_reddit_agent_graph.
        """
        oasis_personas = []
        for p in personas:
            oasis_personas.append({
                "user_profile": (
                    f"{p.get('name', 'Agent')}: {p.get('background', '')} "
                    f"Beliefs: {'; '.join(p.get('beliefs', []))}"
                ),
                "activity_level": p.get("influence_score", 0.5),
            })
        return oasis_personas
