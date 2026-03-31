"""Tests for OuroborosSwarmGenerator — pure helper methods."""
from __future__ import annotations

import json
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ouro.config import OuroConfig
from swarm.generator import OuroborosSwarmGenerator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_config(tmp_path):
    return OuroConfig(
        seed_docs_dir=tmp_path / "seeds",
        corpus_dir=tmp_path / "corpus",
        model_dir=tmp_path / "model",
        feedback_state_path=tmp_path / "state.json",
        kuzu_db_path=tmp_path / "kuzu",
    )


@pytest.fixture
def generator(tmp_config):
    with patch("ouro.claude_adapter.ClaudeAdapter"), \
         patch("ouro.ollama_adapter.OllamaAdapter"):
        with patch.object(OuroborosSwarmGenerator, "_make_llm_client", return_value=MagicMock()):
            gen = OuroborosSwarmGenerator(tmp_config)
    return gen


# ---------------------------------------------------------------------------
# name property
# ---------------------------------------------------------------------------

def test_name_contains_ouroboros(generator):
    assert "Ouroboros" in generator.name or "ouroboros" in generator.name.lower()


def test_name_is_string(generator):
    assert isinstance(generator.name, str)


# ---------------------------------------------------------------------------
# _export_corpus
# ---------------------------------------------------------------------------

def test_export_corpus_creates_file(generator, tmp_path):
    turns = [{"content": "hello", "round": 0, "agent_id": 0}]
    path = generator._export_corpus(turns, cycle=0)
    assert path.exists()


def test_export_corpus_file_contains_jsonl(generator):
    turns = [{"a": 1}, {"a": 2}]
    path = generator._export_corpus(turns, cycle=1)
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["a"] == 1


def test_export_corpus_cycle_in_filename(generator):
    path = generator._export_corpus([], cycle=42)
    assert "0042" in path.name


def test_export_corpus_empty_turns(generator):
    path = generator._export_corpus([], cycle=0)
    assert path.exists()
    assert path.read_text().strip() == ""


# ---------------------------------------------------------------------------
# _stub_simulation
# ---------------------------------------------------------------------------

def test_stub_simulation_returns_list(generator):
    result = generator._stub_simulation(n_agents=5, rounds=3)
    assert isinstance(result, list)


def test_stub_simulation_turn_count(generator):
    result = generator._stub_simulation(n_agents=5, rounds=3)
    # min(5, 10) * 3 rounds = 15 turns
    assert len(result) == 15


def test_stub_simulation_caps_at_10_agents(generator):
    result = generator._stub_simulation(n_agents=100, rounds=1)
    agent_ids = {t["agent_id"] for t in result}
    assert max(agent_ids) < 10


def test_stub_simulation_turn_has_required_keys(generator):
    turns = generator._stub_simulation(n_agents=2, rounds=1)
    for t in turns:
        assert "round" in t
        assert "agent_id" in t
        assert "content" in t


# ---------------------------------------------------------------------------
# _run_async
# ---------------------------------------------------------------------------

def test_run_async_executes_coroutine(generator):
    async def coro():
        return 42

    result = generator._run_async(coro())
    assert result == 42


def test_run_async_returns_correct_value(generator):
    async def coro():
        return [1, 2, 3]

    result = generator._run_async(coro())
    assert result == [1, 2, 3]


# ---------------------------------------------------------------------------
# _run_oasis — camel-oasis absent (stub path)
# ---------------------------------------------------------------------------

def test_run_oasis_falls_back_to_stub_when_no_camel(generator):
    """When camel-oasis is not installed, _run_oasis should return stub turns."""
    import builtins
    real_import = builtins.__import__

    def block_oasis(name, *args, **kwargs):
        if name == "oasis":
            raise ImportError("camel-oasis not installed")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=block_oasis):
        turns = asyncio.run(generator._run_oasis([], rounds=2))

    assert isinstance(turns, list)


# ---------------------------------------------------------------------------
# get_report / get_top_entities (last-run accessors)
# ---------------------------------------------------------------------------

def test_get_report_initially_empty(generator):
    assert generator.get_report() == ""


def test_last_turns_initially_empty(generator):
    assert generator._last_turns == []
