# OUROBOROS — swarm/personas.py
# Copyright (c) 2026 Kudikala Sai Harsha Vinay. All rights reserved.
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
        Generates in batches of 5 for reliability with local LLMs.

        Returns:
            List of persona dicts compatible with the OASIS agent schema.
        """
        entity_names = [e.get("name", "") for e in top_entities[:20]]
        topic_str = ", ".join(entity_names) if entity_names else "AI, technology, society, research"

        personas: list[dict[str, Any]] = []
        batch_size = 5
        batches = max(1, (n + batch_size - 1) // batch_size)

        for batch_idx in range(batches):
            remaining = n - len(personas)
            if remaining <= 0:
                break
            count = min(batch_size, remaining)
            prompt = (
                f"Generate {count} diverse agent personas for a social simulation.\n"
                "Rules:\n"
                "- Output ONLY a valid JSON array, nothing else.\n"
                "- No markdown, no code fences, no explanation.\n"
                f"- Each object must have: name (string), background (1-2 sentences relevant to the topics), "
                f"beliefs (array of 3-5 short strings grounded in the topics), influence_score (float 0.0-1.0).\n"
                f"- Ground ALL personas specifically in these topics: {topic_str}.\n"
                f"- Make each persona distinct: vary their stance (advocate, sceptic, researcher, policymaker, journalist).\n"
                f"Format: [{{\"name\":\"...\",\"background\":\"...\",\"beliefs\":[...],\"influence_score\":0.7}}]\n"
            )
            try:
                response = self._llm.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.8,
                    max_tokens=count * 200,
                )
                raw = response.choices[0].message.content or "[]"
                batch = self._parse_persona_response(raw, count, topic_str)
                personas.extend(batch)
                logger.info("[personas] Batch %d/%d: got %d personas.", batch_idx + 1, batches, len(batch))
            except Exception as exc:
                logger.error("[personas] Batch %d generation error (%s).", batch_idx + 1, exc)
                personas.extend(self._topic_defaults(count, len(personas), topic_str))

        return personas[:n]

    def _parse_persona_response(
        self, raw: str, expected: int, topic_str: str
    ) -> list[dict[str, Any]]:
        """Robustly parse persona JSON, with fallback to per-object extraction."""
        # Strip code fences
        text = re.sub(r"```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
        text = re.sub(r"```", "", text).strip()

        # 1. Full array parse
        try:
            result = json.loads(text)
            if isinstance(result, list) and result:
                return [p for p in result if isinstance(p, dict) and p.get("name")]
        except json.JSONDecodeError:
            pass

        # 2. Extract first [...] block
        array_match = re.search(r"\[.*?\]", text, re.DOTALL)
        if array_match:
            try:
                result = json.loads(array_match.group())
                if isinstance(result, list) and result:
                    return [p for p in result if isinstance(p, dict) and p.get("name")]
            except json.JSONDecodeError:
                pass

        # 3. Extract individual objects
        objects = re.findall(r'\{[^{}]*"name"\s*:\s*"[^"]+[^{}]*\}', text)
        personas = []
        for obj_str in objects:
            try:
                obj = json.loads(obj_str)
                if isinstance(obj, dict) and obj.get("name"):
                    personas.append(obj)
            except json.JSONDecodeError:
                continue
        if personas:
            return personas

        # 4. Topic-aware fallback
        logger.warning("[personas] Parse failed; generating topic-aware defaults.")
        return self._topic_defaults(expected, 0, topic_str)

    def _topic_defaults(
        self, n: int, offset: int, topic_str: str
    ) -> list[dict[str, Any]]:
        """Generate named, topic-grounded fallback personas."""
        topics = [t.strip() for t in topic_str.split(",") if t.strip()]
        roles = ["researcher", "policymaker", "journalist", "advocate", "sceptic",
                 "engineer", "ethicist", "entrepreneur", "regulator", "analyst"]
        defaults = []
        for i in range(n):
            role = roles[(offset + i) % len(roles)]
            focus = topics[(offset + i) % len(topics)] if topics else "technology"
            defaults.append({
                "name": f"{focus.title().replace(' ', '')}_{role.title()}_{offset + i}",
                "background": (
                    f"A {role} specialising in {focus}. "
                    f"Works at the intersection of {focus} and societal impact."
                ),
                "beliefs": [
                    f"{focus} will reshape industry within 5 years.",
                    f"Governance frameworks for {focus} are critically underdeveloped.",
                    f"Transparency in {focus} is a prerequisite for public trust.",
                ],
                "influence_score": round(0.3 + (((offset + i) % 7) * 0.1), 1),
            })
        return defaults

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
