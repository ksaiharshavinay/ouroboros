"""Tests for ReportGenerator."""
import pytest
from unittest.mock import MagicMock

from swarm.report import ReportGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_llm(content: str = "A great prediction report."):
    llm = MagicMock()
    choice = MagicMock()
    choice.message.content = content
    llm.chat.completions.create.return_value = MagicMock(choices=[choice])
    return llm


def make_turns(n: int, influence: float = 0.5):
    return [{"content": f"Turn {i}", "influence": influence} for i in range(n)]


# ---------------------------------------------------------------------------
# generate()
# ---------------------------------------------------------------------------

def test_generate_returns_string():
    gen = ReportGenerator(make_llm())
    result = gen.generate(make_turns(5), [])
    assert isinstance(result, str)
    assert len(result) > 0


def test_generate_includes_entity_names_in_prompt():
    """Top entity names should appear in the prompt sent to the LLM."""
    llm = make_llm()
    gen = ReportGenerator(llm)
    entities = [{"name": "QuantumAI", "type": "concept", "degree": 10}]
    gen.generate(make_turns(3), entities)
    prompt = llm.chat.completions.create.call_args[1]["messages"][0]["content"]
    assert "QuantumAI" in prompt


def test_generate_with_empty_turns():
    """generate() must not crash when given zero turns."""
    gen = ReportGenerator(make_llm())
    result = gen.generate([], [])
    assert isinstance(result, str)


def test_llm_failure_returns_fallback():
    llm = MagicMock()
    llm.chat.completions.create.side_effect = Exception("API down")
    gen = ReportGenerator(llm)
    result = gen.generate(make_turns(3), [])
    assert "error" in result.lower() or "theme" in result.lower() or "fail" in result.lower()


# ---------------------------------------------------------------------------
# _sample_turns()
# ---------------------------------------------------------------------------

def test_no_duplicate_turns_in_sample():
    """High-influence turns that also appear in the first 15 must not be duplicated."""
    gen = ReportGenerator(make_llm())
    turns = make_turns(20, influence=1.0)  # All high influence
    sample = gen._sample_turns(turns, max_turns=30)
    seen = set(id(t) for t in sample)
    assert len(sample) == len(seen)


def test_sample_max_capped():
    gen = ReportGenerator(make_llm())
    turns = make_turns(100)
    sample = gen._sample_turns(turns, max_turns=20)
    assert len(sample) <= 20


def test_sample_high_influence_turns_included():
    """High-influence turns beyond the first 15 should be pulled in."""
    gen = ReportGenerator(make_llm())
    # First 15 turns have low influence; turns 15-29 have high influence
    low = [{"content": f"low {i}", "influence": 0.0} for i in range(15)]
    high = [{"content": f"high {i}", "influence": 1.0} for i in range(15)]
    turns = low + high
    sample = gen._sample_turns(turns, max_turns=30)
    contents = {t["content"] for t in sample}
    assert any("high" in c for c in contents)


def test_sample_empty_turns_returns_empty():
    gen = ReportGenerator(make_llm())
    assert gen._sample_turns([], max_turns=30) == []
