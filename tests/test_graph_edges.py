"""Tests for the Phase 2 graph edges (Finding 3): typed, provenance-
carrying edges in the Kuzu knowledge graph.

Covers: co-occurrence edges within a source document, entity->source
provenance, the same edge model for simulation turns, the new query
APIs, get_top_entities returning real degrees now that edges exist, and
the pre-Phase-2 schema migration (weight-only Related table).
"""
import json

import pytest
from unittest.mock import MagicMock

from swarm.graph import GraphBuilder

kuzu = pytest.importorskip("kuzu", reason="kuzu not installed")


def make_llm_sequence(entity_lists: list[list[dict]]):
    """Mock LLM whose successive chat calls return the given entity lists."""
    llm = MagicMock()
    responses = []
    for entities in entity_lists:
        choice = MagicMock()
        choice.message.content = json.dumps(entities)
        responses.append(MagicMock(choices=[choice]))
    llm.chat.completions.create.side_effect = responses
    return llm


def _open(tmp_path, llm, cycle: int = 0) -> GraphBuilder:
    gb = GraphBuilder(db_path=tmp_path / "kuzu", llm_client=llm)
    gb.open(cycle=cycle)
    return gb


# ---------------------------------------------------------------------------
# Co-occurrence edges + provenance from documents
# ---------------------------------------------------------------------------

def test_doc_cooccurrence_edges_and_provenance(tmp_path):
    doc1 = tmp_path / "doc1.txt"
    doc2 = tmp_path / "doc2.txt"
    doc1.write_text("Alpha Beta Gamma")
    doc2.write_text("Beta Gamma Delta")
    llm = make_llm_sequence([
        [{"name": "Alpha", "type": "concept"},
         {"name": "Beta", "type": "concept"},
         {"name": "Gamma", "type": "concept"}],
        [{"name": "Beta", "type": "concept"},
         {"name": "Gamma", "type": "concept"},
         {"name": "Delta", "type": "concept"}],
    ])
    gb = _open(tmp_path, llm)
    try:
        gb.build_from_docs([doc1, doc2])

        # Beta co-occurs with Alpha + Gamma in doc1, Gamma + Delta in doc2
        edges = gb.get_edges_for_entity("Beta")
        by_other = {}
        for e in edges:
            by_other.setdefault(e["other_entity"], []).append(e)
        assert set(by_other) == {"Alpha", "Gamma", "Delta"}
        assert len(by_other["Gamma"]) == 2, "Beta-Gamma pair occurs in both docs"
        sources = {e["source"] for e in by_other["Gamma"]}
        assert sources == {"doc1.txt", "doc2.txt"}
        assert all(e["kind"] == "co_occurrence" for e in edges)
        assert all(e["cycle"] == 0 for e in edges)

        # Alpha's only neighbour is Beta (from doc1)
        alpha_edges = gb.get_edges_for_entity("Alpha")
        assert {e["other_entity"] for e in alpha_edges} == {"Beta", "Gamma"}

        # Provenance: which document produced which entity
        prov_alpha = gb.get_entity_provenance("Alpha")
        assert [(p["source_kind"], p["source_ref"]) for p in prov_alpha] == [
            ("seed_doc", "doc1.txt")
        ]
        prov_delta = gb.get_entity_provenance("Delta")
        assert [(p["source_kind"], p["source_ref"]) for p in prov_delta] == [
            ("seed_doc", "doc2.txt")
        ]
        prov_beta = gb.get_entity_provenance("Beta")
        assert {p["source_ref"] for p in prov_beta} == {"doc1.txt", "doc2.txt"}
    finally:
        gb.close()


def test_get_top_entities_returns_ranked_entities(tmp_path):
    """Finding 3 regression: with edges inserted, degree ranking works."""
    doc = tmp_path / "doc.txt"
    doc.write_text("Hub A B C")
    llm = make_llm_sequence([
        [{"name": n, "type": "concept"} for n in ("Hub", "A", "B", "C")],
    ])
    gb = _open(tmp_path, llm)
    try:
        gb.build_from_docs([doc])
        top = gb.get_top_entities(n=10)
        assert top, "top entities must no longer be empty"
        # Complete co-occurrence among 4 entities: every degree is 3
        assert {row["name"] for row in top} == {"Hub", "A", "B", "C"}
        assert all(row["degree"] == 3 for row in top)
    finally:
        gb.close()


# ---------------------------------------------------------------------------
# The same edge model for simulation turns
# ---------------------------------------------------------------------------

def test_turn_edges_and_provenance(tmp_path):
    turns = [
        {"round": 1, "agent_id": 3, "content": "Solar winds rising"},
        {"round": 2, "agent_id": 4, "content": "Solar markets falling"},
    ]
    llm = make_llm_sequence([
        [{"name": "Solar", "type": "concept"}, {"name": "Winds", "type": "concept"}],
        [{"name": "Solar", "type": "concept"}, {"name": "Markets", "type": "concept"}],
    ])
    gb = _open(tmp_path, llm, cycle=1)
    try:
        gb.build_from_turns(turns, source_ref="corpus_cycle_0001")
        edges = gb.get_edges_for_entity("Solar")
        assert {e["other_entity"] for e in edges} == {"Winds", "Markets"}
        prov = gb.get_entity_provenance("Solar")
        assert len(prov) == 2
        assert all(p["source_kind"] == "turn" for p in prov)
        assert any("corpus_cycle_0001" in p["source_ref"] and ":a3:" in p["source_ref"]
                   for p in prov)
        assert all(p["cycle"] == 1 for p in prov)
    finally:
        gb.close()


def test_build_from_turns_skips_empty_and_survives_bad_turns(tmp_path):
    turns = [
        {"round": 0, "agent_id": 0, "content": ""},
        {"round": 0, "agent_id": 1, "content": "Lone entity here"},
    ]
    llm = make_llm_sequence([[{"name": "Lone", "type": "concept"}]])
    gb = _open(tmp_path, llm)
    try:
        gb.build_from_turns(turns, source_ref="corpus_cycle_0000")
        assert gb.get_entity_provenance("Lone"), "single-entity turn still gets provenance"
        assert gb.get_edges_for_entity("Lone") == [], "no pair -> no co-occurrence edge"
    finally:
        gb.close()


# ---------------------------------------------------------------------------
# Schema migration for pre-Phase-2 databases
# ---------------------------------------------------------------------------

def test_old_schema_related_table_is_migrated(tmp_path):
    import kuzu

    db_dir = tmp_path / "kuzu"
    db_dir.mkdir(parents=True)
    # Build a database with the pre-Phase-2 (weight-only) Related schema
    db = kuzu.Database(str(db_dir / "kuzu.db"))
    conn = kuzu.Connection(db)
    conn.execute(
        "CREATE NODE TABLE Entity(id STRING PRIMARY KEY, name STRING, type STRING)"
    )
    conn.execute("CREATE REL TABLE Related(FROM Entity TO Entity, weight DOUBLE)")
    conn.close()
    db.close()

    gb = _open(tmp_path, make_llm_sequence([
        [{"name": "X", "type": "concept"}, {"name": "Y", "type": "concept"}],
    ]))
    try:
        doc = tmp_path / "d.txt"
        doc.write_text("X Y")
        gb.build_from_docs([doc])
        edges = gb.get_edges_for_entity("X")
        assert len(edges) == 1 and edges[0]["other_entity"] == "Y"
    finally:
        gb.close()


def test_edge_queries_on_unopened_graph_return_empty(tmp_path):
    gb = GraphBuilder(db_path=tmp_path / "kuzu", llm_client=MagicMock())
    assert gb.get_edges_for_entity("Anything") == []
    assert gb.get_entity_provenance("Anything") == []
