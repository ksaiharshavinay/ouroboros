"""Tests for ReportGenerator."""
import pytest
from unittest.mock import MagicMock

from swarm.report import ReportGenerator


def make_llm(content: str = "A great prediction report."):
    llm = MagicMock()
    choice = MagicMock()
    choice.message.content = content
    llm.chat.completions.create.return_value = MagicMock(choices=[choice])
    return llm


def make_turns(n: int, influence: float = 0.5):
    return [{"content": f"Turn {i}", "influence": influence} for i in range(n)]


def test_generate_returns_string():
    gen = ReportGenerator(make_llm())
    result = gen.generate(make_turns(5), [])
    assert isinstance(result, str)
    assert len(result) > 0


def test_no_duplicate_turns_in_sample():
    """High-influence turns that also appear in the first 15 must not be duplicated."""
    gen = ReportGenerator(make_llm())
    turns = make_turns(20, influence=1.0)  # All high influence
    sample = gen._sample_turns(turns, max_turns=30)
    seen = set(id(t) for t in sample)
    assert len(sample) == len(seen)  # No duplicates by identity


def test_llm_failure_returns_fallback():
    llm = MagicMock()
    llm.chat.completions.create.side_effect = Exception("API down")
    gen = ReportGenerator(llm)
    result = gen.generate(make_turns(3), [])
    assert "error" in result.lower() or "theme" in result.lower() or "fail" in result.lower()


def test_sample_max_capped():
    gen = ReportGenerator(make_llm())
    turns = make_turns(100)
    sample = gen._sample_turns(turns, max_turns=20)
    assert len(sample) <= 20
