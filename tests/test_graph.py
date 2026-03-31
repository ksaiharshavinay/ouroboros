"""Tests for GraphBuilder."""
import hashlib
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from swarm.graph import GraphBuilder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_llm(response: str = '[{"name": "Alice", "type": "person"}]'):
    llm = MagicMock()
    choice = MagicMock()
    choice.message.content = response
    llm.chat.completions.create.return_value = MagicMock(choices=[choice])
    return llm


# ---------------------------------------------------------------------------
# JSON fence stripping
# ---------------------------------------------------------------------------

def test_json_fence_stripping():
    llm = make_llm('```json\n[{"name": "Bob", "type": "org"}]\n```')
    gb = GraphBuilder(db_path=Path("/tmp/test_kuzu"), llm_client=llm)
    result = gb._llm_extract("some text")
    assert result[0]["name"] == "Bob"


def test_json_fence_stripping_extra_whitespace():
    llm = make_llm('```json  \n[{"name": "Carol", "type": "concept"}]\n  ```')
    gb = GraphBuilder(db_path=Path("/tmp/test_kuzu"), llm_client=llm)
    result = gb._llm_extract("text")
    assert result[0]["name"] == "Carol"


def test_json_fence_no_language_tag():
    """Fences without the 'json' language tag should still be stripped."""
    llm = make_llm('```\n[{"name": "Dave", "type": "person"}]\n```')
    gb = GraphBuilder(db_path=Path("/tmp/test_kuzu"), llm_client=llm)
    result = gb._llm_extract("text")
    assert result[0]["name"] == "Dave"


def test_json_fence_uppercase_tag():
    llm = make_llm('```JSON\n[{"name": "Eve", "type": "concept"}]\n```')
    gb = GraphBuilder(db_path=Path("/tmp/test_kuzu"), llm_client=llm)
    result = gb._llm_extract("text")
    assert result[0]["name"] == "Eve"


def test_no_fence_plain_json():
    """Plain JSON with no fences must parse correctly."""
    llm = make_llm('[{"name": "Frank", "type": "org"}]')
    gb = GraphBuilder(db_path=Path("/tmp/test_kuzu"), llm_client=llm)
    result = gb._llm_extract("text")
    assert result[0]["name"] == "Frank"


# ---------------------------------------------------------------------------
# LLM extraction error paths
# ---------------------------------------------------------------------------

def test_llm_returns_invalid_json_falls_back():
    llm = make_llm("NOT JSON AT ALL")
    gb = GraphBuilder(db_path=Path("/tmp/test_kuzu"), llm_client=llm)
    result = gb._llm_extract("text")
    assert result == []


def test_llm_returns_empty_array():
    llm = make_llm("[]")
    gb = GraphBuilder(db_path=Path("/tmp/test_kuzu"), llm_client=llm)
    result = gb._llm_extract("text")
    assert result == []


def test_llm_raises_exception_returns_empty():
    llm = MagicMock()
    llm.chat.completions.create.side_effect = Exception("API down")
    gb = GraphBuilder(db_path=Path("/tmp/test_kuzu"), llm_client=llm)
    result = gb._llm_extract("text")
    assert result == []


def test_retry_uses_temperature_zero():
    """On second attempt (retry), the LLM must be called with temperature=0.0."""
    llm = MagicMock()
    c1 = MagicMock()
    c1.message.content = "INVALID"
    c2 = MagicMock()
    c2.message.content = '[{"name": "Retry", "type": "person"}]'
    llm.chat.completions.create.side_effect = [
        MagicMock(choices=[c1]),
        MagicMock(choices=[c2]),
    ]
    gb = GraphBuilder(db_path=Path("/tmp/test_kuzu"), llm_client=llm)
    result = gb._llm_extract("text")
    second_call_kwargs = llm.chat.completions.create.call_args_list[1][1]
    assert second_call_kwargs.get("temperature") == 0.0
    assert result[0]["name"] == "Retry"


# ---------------------------------------------------------------------------
# Entity ID namespacing
# ---------------------------------------------------------------------------

def test_entity_id_is_cycle_namespaced():
    """Entity IDs from different cycles must not collide."""
    name = "TestEntity"
    id_c0 = "e_" + hashlib.md5(f"c0_{name.lower()}".encode()).hexdigest()[:12]
    id_c1 = "e_" + hashlib.md5(f"c1_{name.lower()}".encode()).hexdigest()[:12]
    assert id_c0 != id_c1


def test_entity_id_same_cycle_same_name_is_stable():
    """Same entity in the same cycle should always produce the same ID."""
    name = "StableEntity"
    id_a = "e_" + hashlib.md5(f"c0_{name.lower()}".encode()).hexdigest()[:12]
    id_b = "e_" + hashlib.md5(f"c0_{name.lower()}".encode()).hexdigest()[:12]
    assert id_a == id_b


def test_entity_id_is_case_insensitive():
    """'Alice' and 'alice' must produce the same ID within the same cycle."""
    id_upper = "e_" + hashlib.md5(f"c0_alice".encode()).hexdigest()[:12]
    id_lower = "e_" + hashlib.md5(f"c0_alice".encode()).hexdigest()[:12]
    assert id_upper == id_lower


# ---------------------------------------------------------------------------
# close() safety
# ---------------------------------------------------------------------------

def test_close_is_idempotent():
    """Calling close() twice must not raise."""
    gb = GraphBuilder(db_path=Path("/tmp/test_kuzu2"), llm_client=MagicMock())
    gb.close()
    gb.close()


def test_close_sets_handles_to_none():
    """After close(), both _conn and _db must be None."""
    gb = GraphBuilder(db_path=Path("/tmp/test_kuzu3"), llm_client=MagicMock())
    gb._conn = MagicMock()
    gb._db = MagicMock()
    gb.close()
    assert gb._conn is None
    assert gb._db is None


# ---------------------------------------------------------------------------
# build_from_docs
# ---------------------------------------------------------------------------

def test_build_from_docs_calls_extract_per_doc(tmp_path):
    """build_from_docs must call the LLM once per document."""
    doc1 = tmp_path / "doc1.txt"
    doc2 = tmp_path / "doc2.txt"
    doc1.write_text("Document one content")
    doc2.write_text("Document two content")

    llm = make_llm("[]")
    gb = GraphBuilder(db_path=tmp_path / "kuzu", llm_client=llm)
    gb._insert_entities = MagicMock()
    gb.build_from_docs([doc1, doc2])
    assert llm.chat.completions.create.call_count == 2


def test_build_from_docs_empty_list_does_nothing(tmp_path):
    llm = make_llm("[]")
    gb = GraphBuilder(db_path=tmp_path / "kuzu", llm_client=llm)
    gb._insert_entities = MagicMock()
    gb.build_from_docs([])
    llm.chat.completions.create.assert_not_called()
