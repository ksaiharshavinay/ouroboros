"""Tests for PersonaGenerator."""
import json
import pytest
from unittest.mock import MagicMock

from swarm.personas import PersonaGenerator


def make_llm(content: str):
    llm = MagicMock()
    choice = MagicMock()
    choice.message.content = content
    llm.chat.completions.create.return_value = MagicMock(choices=[choice])
    return llm


VALID_PERSONAS = json.dumps([
    {"name": "Alice", "background": "A researcher.", "beliefs": ["AI is powerful"], "influence_score": 0.8},
    {"name": "Bob", "background": "A sceptic.", "beliefs": ["AI is overhyped"], "influence_score": 0.3},
])


def test_generate_returns_correct_count():
    llm = make_llm(VALID_PERSONAS)
    gen = PersonaGenerator(llm)
    result = gen.generate(2, [])
    assert len(result) == 2


def test_generate_parses_names():
    llm = make_llm(VALID_PERSONAS)
    gen = PersonaGenerator(llm)
    result = gen.generate(2, [])
    assert result[0]["name"] == "Alice"


def test_fence_stripping():
    fenced = f"```json\n{VALID_PERSONAS}\n```"
    llm = make_llm(fenced)
    gen = PersonaGenerator(llm)
    result = gen.generate(2, [])
    assert len(result) == 2


def test_fallback_on_invalid_json():
    llm = make_llm("NOT JSON")
    gen = PersonaGenerator(llm)
    result = gen.generate(3, [])
    assert len(result) == 3
    assert all("Agent_" in p["name"] for p in result)


def test_to_oasis_format():
    personas = [
        {"name": "Alice", "background": "Researcher.", "beliefs": ["AI is great"], "influence_score": 0.9}
    ]
    oasis = PersonaGenerator.to_oasis_format(personas)
    assert len(oasis) == 1
    assert "user_profile" in oasis[0]
    assert "Alice" in oasis[0]["user_profile"]
    assert oasis[0]["activity_level"] == 0.9
