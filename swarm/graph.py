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
            "CREATE NODE TABLE IF NOT EXISTS Source(id STRING PRIMARY KEY, kind STRING, ref STRING, cycle INT64)"
        )
        # Phase 2 (Finding 3): Related edges now carry provenance — the
        # kind of relation, the source unit that produced it, and the
        # cycle. Kuzu's CREATE ... IF NOT EXISTS silently keeps an older
        # table's narrower schema, so an existing pre-Phase-2 Related
        # table (weight only) is detected and recreated. The graph is a
        # derived cache — it is rebuilt from the source documents on
        # every run — so recreating the edge table loses nothing that
        # cannot be re-extracted.
        self._conn.execute(
            "CREATE REL TABLE IF NOT EXISTS Related(FROM Entity TO Entity, "
            "weight DOUBLE, kind STRING, source STRING, cycle INT64)"
        )
        if not self._related_has_provenance_props():
            logger.warning(
                "[graph] Existing Related table predates the Phase 2 "
                "provenance schema; recreating it."
            )
            self._conn.execute("DROP TABLE Related")
            self._conn.execute(
                "CREATE REL TABLE Related(FROM Entity TO Entity, "
                "weight DOUBLE, kind STRING, source STRING, cycle INT64)"
            )
        self._conn.execute(
            "CREATE REL TABLE IF NOT EXISTS DerivedFrom(FROM Entity TO Source, kind STRING, cycle INT64)"
        )

    def _related_has_provenance_props(self) -> bool:
        """True iff the Related table has the Phase 2 provenance columns."""
        try:
            result = self._conn.execute("CALL table_info('Related') RETURN *")
            props = set()
            while result.has_next():
                props.add(result.get_next()[1])
            return {"kind", "source", "cycle"} <= props
        except Exception:
            # Cannot introspect (should not happen on kuzu 0.11); the
            # CREATE above just ran with the full schema, so assume it.
            return True

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def build_from_docs(self, docs: list[Path]) -> None:
        """Extract entities from docs and insert them into the graph.

        Beyond the entity nodes, each document also produces typed,
        provenance-carrying edges: entity->source (DerivedFrom) for every
        entity, and entity--entity co-occurrence (Related) for every pair
        of entities extracted from the same document.
        """
        for doc in docs:
            text = doc.read_text(encoding="utf-8", errors="replace")
            entities = self._llm_extract(text)
            self._insert_entities(entities)
            self._insert_unit_edges(entities, source_kind="seed_doc", source_ref=doc.name)

    def build_from_turns(self, turns: list[dict[str, Any]], source_ref: str) -> None:
        """
        Extract entities from simulation turns and insert them with the
        same edge model as seed documents (Phase 2, Finding 3): each turn
        is a source unit, so the graph records which turn produced which
        entity and which entities co-occurred within a turn.

        Per-turn failures are logged and skipped — graph enrichment must
        never fail the simulation that produced the turns.
        """
        for idx, turn in enumerate(turns):
            content = (turn.get("content") or "").strip()
            if not content:
                continue
            ref = (
                f"{source_ref}:r{turn.get('round', 0)}"
                f":a{turn.get('agent_id', 0)}:t{idx}"
            )
            try:
                entities = self._llm_extract(content)
                self._insert_entities(entities)
                self._insert_unit_edges(entities, source_kind="turn", source_ref=ref)
            except Exception as exc:
                logger.warning("[graph] Turn %d graph enrichment failed: %s", idx, exc)

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

    def _entity_id(self, name: str) -> str:
        """Stable, cycle-namespaced entity ID (content hash of cycle + name)."""
        raw_id = f"c{self._cycle}_{name.strip().lower()}"
        return "e_" + hashlib.md5(raw_id.encode()).hexdigest()[:12]

    @staticmethod
    def _source_id(source_kind: str, source_ref: str) -> str:
        """Stable source-node ID (content hash of kind + reference)."""
        raw_id = f"{source_kind}:{source_ref}"
        return "s_" + hashlib.md5(raw_id.encode()).hexdigest()[:12]

    def _insert_entities(self, entities: list[dict[str, str]]) -> None:
        """Insert entities using CREATE + skip on duplicate (safe across KuzuDB versions)."""
        for ent in entities:
            name = ent.get("name", "").strip()
            etype = ent.get("type", "unknown").strip()
            if not name:
                continue
            eid = self._entity_id(name)
            try:
                self._conn.execute(
                    "CREATE (e:Entity {id: $id, name: $name, type: $type})",
                    parameters={"id": eid, "name": name, "type": etype},
                )
            except Exception:
                pass  # Skip duplicates

    def _insert_unit_edges(
        self, entities: list[dict[str, str]], source_kind: str, source_ref: str
    ) -> None:
        """
        Insert the edges one source unit (a seed document or one
        simulation turn) gives rise to:

        * DerivedFrom: entity -> Source, one per entity (provenance:
          this unit produced this entity).
        * Related: entity -- entity, one per unordered entity pair
          (co-occurrence within this unit), carrying kind, source and
          cycle so an edge can always be traced back to its origin.

        No-ops when the graph is not open (e.g. extraction-only use).
        """
        if self._conn is None:
            return
        # Dedupe by normalised name, preserving first-seen type/order
        unique: dict[str, dict[str, str]] = {}
        for ent in entities:
            name = ent.get("name", "").strip()
            if name:
                unique.setdefault(name.lower(), ent)
        names = list(unique.values())
        if not names:
            return

        sid = self._source_id(source_kind, source_ref)
        try:
            self._conn.execute(
                "CREATE (s:Source {id: $id, kind: $kind, ref: $ref, cycle: $cycle})",
                parameters={
                    "id": sid, "kind": source_kind,
                    "ref": source_ref, "cycle": self._cycle,
                },
            )
        except Exception:
            pass  # Source already recorded (same unit re-processed)

        for ent in names:
            eid = self._entity_id(ent["name"])
            try:
                self._conn.execute(
                    "MATCH (e:Entity {id: $eid}), (s:Source {id: $sid}) "
                    "CREATE (e)-[:DerivedFrom {kind: $kind, cycle: $cycle}]->(s)",
                    parameters={
                        "eid": eid, "sid": sid,
                        "kind": "mention", "cycle": self._cycle,
                    },
                )
            except Exception as exc:
                logger.debug("[graph] DerivedFrom insert failed: %s", exc)

        # Co-occurrence: one edge per unordered pair, direction fixed by
        # entity id so a pair is stored once per source unit.
        ordered = sorted(names, key=lambda e: self._entity_id(e["name"]))
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                a_id = self._entity_id(ordered[i]["name"])
                b_id = self._entity_id(ordered[j]["name"])
                try:
                    self._conn.execute(
                        "MATCH (a:Entity {id: $a}), (b:Entity {id: $b}) "
                        "CREATE (a)-[:Related {weight: 1.0, kind: 'co_occurrence', "
                        "source: $source, cycle: $cycle}]->(b)",
                        parameters={
                            "a": a_id, "b": b_id,
                            "source": source_ref, "cycle": self._cycle,
                        },
                    )
                except Exception as exc:
                    logger.debug("[graph] Related insert failed: %s", exc)

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

    def get_edges_for_entity(self, name: str) -> list[dict[str, Any]]:
        """
        Return all co-occurrence edges touching an entity (matched by
        name, across cycles — entity nodes are cycle-namespaced, edges
        are the cross-cycle record). Each edge reports the other entity,
        its kind, the source unit that produced it, and the cycle.
        """
        if self._conn is None:
            return []
        try:
            result = self._conn.execute(
                "MATCH (e:Entity)-[r:Related]-(o:Entity) "
                "WHERE e.name = $name "
                "RETURN o.name AS other, r.kind AS kind, r.source AS source, "
                "r.cycle AS cycle, r.weight AS weight",
                parameters={"name": name},
            )
            rows = []
            while result.has_next():
                row = result.get_next()
                rows.append({
                    "entity": name,
                    "other_entity": row[0],
                    "kind": row[1],
                    "source": row[2],
                    "cycle": row[3],
                    "weight": row[4],
                })
            return rows
        except Exception as exc:
            logger.error("[graph] get_edges_for_entity failed: %s", exc)
            return []

    def get_entity_provenance(self, name: str) -> list[dict[str, Any]]:
        """
        Return the provenance of an entity: every source unit (seed
        document or simulation turn) it was extracted from, with cycle.
        """
        if self._conn is None:
            return []
        try:
            result = self._conn.execute(
                "MATCH (e:Entity)-[r:DerivedFrom]->(s:Source) "
                "WHERE e.name = $name "
                "RETURN s.kind AS source_kind, s.ref AS source_ref, r.cycle AS cycle",
                parameters={"name": name},
            )
            rows = []
            while result.has_next():
                row = result.get_next()
                rows.append({
                    "entity": name,
                    "source_kind": row[0],
                    "source_ref": row[1],
                    "cycle": row[2],
                })
            return rows
        except Exception as exc:
            logger.error("[graph] get_entity_provenance failed: %s", exc)
            return []
