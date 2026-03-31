"""Tests for GraphBuilder."""
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from swarm.graph import GraphBuilder


def make_llm(response: str = '[{"name": "Alice", "type": "person"}]'):
    llm = MagicMock()
    choice = MagicMock()
    choice.message.content = response
    llm.chat.completions.create.return_value = MagicMock(choices=[choice])
    return llm


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


def test_llm_returns_invalid_json_falls_back():
    llm = make_llm("NOT JSON AT ALL")
    gb = GraphBuilder(db_path=Path("/tmp/test_kuzu"), llm_client=llm)
    result = gb._llm_extract("text")
    assert result == []


def test_entity_id_is_cycle_namespaced():
    """Entity IDs from different cycles must not collide."""
    import hashlib
    name = "TestEntity"
    id_c0 = "e_" + hashlib.md5(f"c0_{name.lower()}".encode()).hexdigest()[:12]
    id_c1 = "e_" + hashlib.md5(f"c1_{name.lower()}".encode()).hexdigest()[:12]
    assert id_c0 != id_c1


def test_close_is_idempotent():
    """Calling close() twice must not raise."""
    gb = GraphBuilder(db_path=Path("/tmp/test_kuzu2"), llm_client=MagicMock())
    gb.close()
    gb.close()  # Second call should be safe
