# OUROBOROS — swarm/graph.py
# Copyright (c) 2026 Kudikala Sai Harsha Vinay. All rights reserved.
# PATENT PENDING
#
# KuzuDB (MIT licensed) — graph persistence for agent knowledge networks.

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class GraphBuilder:
    """
    Builds and queries a KuzuDB knowledge graph from seed documents.
    Each node represents a named entity; edges represent co-occurrence
    or explicit relationships extracted by the LLM.
    """

    def __init__(self, db_path: Path, llm_client: Any, model: str = "gpt-4o-mini"):
        self._db_path = db_path
        self._llm = llm_client
        self._model = model
        self._db: Any = None
        self._conn: Any = None
        self._cycle: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self, cycle: int = 0) -> None:
        """Open (or create) the KuzuDB database."""
        import kuzu

        self._cycle = cycle
        self._db_path.mkdir(parents=True, exist_ok=True)
        # Kuzu >= 0.9 uses single-file databases: the path must be a file,
        # not a directory (0.7.x accepted a directory). Keep the configured
        # directory as the container and put the DB file inside it.
        self._db = kuzu.Database(str(self._db_path / "kuzu.db"))
        self._conn = kuzu.Connection(self._db)
        self._init_schema()

    def close(self) -> None:
        """Close the database connection and release handles."""
        try:
            if self._conn is not None:
                self._conn.close()
        except Exception:
            pass
        finally:
            self._conn = None

        try:
            if self._db is not None:
                self._db.close()
        except Exception:
            pass
        finally:
            self._db = None

    def _init_schema(self) -> None:
        self._conn.execute(
            "CREATE NODE TABLE IF NOT EXISTS Entity(id STRING PRIMARY KEY, name STRING, type STRING)"
        )
        self._conn.execute(
            "CREATE REL TABLE IF NOT EXISTS Related(FROM Entity TO Entity, weight DOUBLE)"
        )

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def build_from_docs(self, docs: list[Path]) -> None:
        """Extract entities from docs and insert them into the graph."""
        for doc in docs:
            text = doc.read_text(encoding="utf-8", errors="replace")
            entities = self._llm_extract(text)
            self._insert_entities(entities)

    def _llm_extract(self, text: str) -> list[dict[str, str]]:
        """Ask the LLM to extract named entities from text. Retries once on parse failure."""
        prompt = (
            "Extract named entities (people, organisations, concepts) from the text below.\n"
            "Rules:\n"
            "- Output ONLY a valid JSON array, nothing else.\n"
            "- No markdown, no code fences, no explanation before or after.\n"
            "- Format: [{\"name\": \"EntityName\", \"type\": \"person|organisation|concept\"}]\n"
            "- If no entities found, output: []\n\n"
            "Text:\n" + text[:3000]
        )
        for attempt in range(2):
            try:
                response = self._llm.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0 if attempt == 1 else 0.2,
                    max_tokens=512,
                )
                raw = response.choices[0].message.content or "[]"
                entities = self._parse_entity_response(raw)
                if entities is not None:
                    return entities
                if attempt == 0:
                    logger.warning("[graph] Entity parse failed; retrying at temp=0.")
                else:
                    logger.error("[graph] Entity extraction failed after retry; using empty list.")
            except Exception as exc:
                if attempt == 0:
                    logger.warning("[graph] Entity extraction error (%s); retrying.", exc)
                else:
                    logger.error("[graph] Entity extraction error after retry: %s", exc)
        return []

    @staticmethod
    def _parse_entity_response(raw: str) -> list[dict[str, str]] | None:
        """
        Robustly parse an entity JSON array from a raw LLM response.

        Handles:
        - Markdown code fences (```json ... ```)
        - Leading/trailing prose around the JSON array
        - Individual objects extracted via regex when array parse fails
        Returns None if no valid entities could be extracted.
        """
        # 1. Strip code fences
        text = re.sub(r"```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
        text = re.sub(r"```", "", text).strip()

        # 2. Try parsing the whole thing as JSON
        try:
            result = json.loads(text)
            if isinstance(result, list):
                return [e for e in result if isinstance(e, dict) and e.get("name")]
        except json.JSONDecodeError:
            pass

        # 3. Find the first [...] block anywhere in the response
        array_match = re.search(r"\[.*?\]", text, re.DOTALL)
        if array_match:
            try:
                result = json.loads(array_match.group())
                if isinstance(result, list):
                    return [e for e in result if isinstance(e, dict) and e.get("name")]
            except json.JSONDecodeError:
                pass

        # 4. Fallback: extract individual {"name": ..., "type": ...} objects via regex
        objects = re.findall(r'\{[^{}]*"name"\s*:\s*"[^"]+[^{}]*\}', text)
        entities = []
        for obj_str in objects:
            try:
                obj = json.loads(obj_str)
                if isinstance(obj, dict) and obj.get("name"):
                    entities.append(obj)
            except json.JSONDecodeError:
                continue
        if entities:
            return entities

        return None

    def _insert_entities(self, entities: list[dict[str, str]]) -> None:
        """Insert entities using CREATE + skip on duplicate (safe across KuzuDB versions)."""
        for ent in entities:
            name = ent.get("name", "").strip()
            etype = ent.get("type", "unknown").strip()
            if not name:
                continue
            # Stable, cycle-namespaced ID using content hash
            raw_id = f"c{self._cycle}_{name.lower()}"
            eid = "e_" + hashlib.md5(raw_id.encode()).hexdigest()[:12]
            try:
                self._conn.execute(
                    "CREATE (e:Entity {id: $id, name: $name, type: $type})",
                    parameters={"id": eid, "name": name, "type": etype},
                )
            except Exception:
                pass  # Skip duplicates

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_top_entities(self, n: int = 10) -> list[dict[str, Any]]:
        """Return the top-n entities ordered by connection degree."""
        try:
            result = self._conn.execute(
                "MATCH (e:Entity)-[r:Related]-() "
                "RETURN e.name AS name, e.type AS type, COUNT(r) AS degree "
                "ORDER BY degree DESC "
                f"LIMIT {n}"
            )
            rows = []
            while result.has_next():
                row = result.get_next()
                rows.append({"name": row[0], "type": row[1], "degree": row[2]})
            return rows
        except Exception as exc:
            logger.error("[graph] get_top_entities failed: %s", exc)
            return []
