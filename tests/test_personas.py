"""Tests for PersonaGenerator."""
import json
import pytest
from unittest.mock import MagicMock

from swarm.personas import PersonaGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Basic generation
# ---------------------------------------------------------------------------

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
    assert result[1]["name"] == "Bob"


def test_generate_truncates_to_n():
    """If LLM returns more personas than requested, only n are returned."""
    llm = make_llm(VALID_PERSONAS)
    gen = PersonaGenerator(llm)
    result = gen.generate(1, [])
    assert len(result) == 1


def test_generate_returns_list():
    llm = make_llm(VALID_PERSONAS)
    gen = PersonaGenerator(llm)
    assert isinstance(gen.generate(2, []), list)


def test_generate_uses_entity_names_in_prompt():
    """Top entity names should be referenced in the prompt sent to the LLM."""
    llm = make_llm(VALID_PERSONAS)
    gen = PersonaGenerator(llm)
    entities = [{"name": "BlockchainTopic", "type": "concept", "degree": 5}]
    gen.generate(2, entities)
    call_args = llm.chat.completions.create.call_args
    prompt_content = call_args[1]["messages"][0]["content"]
    assert "BlockchainTopic" in prompt_content


# ---------------------------------------------------------------------------
# JSON fence stripping
# ---------------------------------------------------------------------------

def test_fence_stripping():
    fenced = f"```json\n{VALID_PERSONAS}\n```"
    llm = make_llm(fenced)
    gen = PersonaGenerator(llm)
    result = gen.generate(2, [])
    assert len(result) == 2


def test_fence_stripping_no_language_tag():
    fenced = f"```\n{VALID_PERSONAS}\n```"
    llm = make_llm(fenced)
    gen = PersonaGenerator(llm)
    result = gen.generate(2, [])
    assert len(result) == 2


def test_fence_stripping_uppercase_tag():
    fenced = f"```JSON\n{VALID_PERSONAS}\n```"
    llm = make_llm(fenced)
    gen = PersonaGenerator(llm)
    result = gen.generate(2, [])
    assert result[0]["name"] == "Alice"


# ---------------------------------------------------------------------------
# Error / fallback paths
# ---------------------------------------------------------------------------

def test_fallback_on_invalid_json():
    llm = make_llm("NOT JSON")
    gen = PersonaGenerator(llm)
    result = gen.generate(3, [])
    assert len(result) == 3
    assert all("Agent_" in p["name"] for p in result)


def test_fallback_on_llm_exception():
    llm = MagicMock()
    llm.chat.completions.create.side_effect = Exception("API error")
    gen = PersonaGenerator(llm)
    result = gen.generate(2, [])
    assert len(result) == 2
    assert all("Agent_" in p["name"] for p in result)


def test_fallback_when_response_is_object_not_array():
    """If LLM returns a JSON object instead of array, fallback kicks in."""
    llm = make_llm('{"name": "single", "background": "oops"}')
    gen = PersonaGenerator(llm)
    result = gen.generate(2, [])
    assert len(result) == 2
    assert all("Agent_" in p["name"] for p in result)


# ---------------------------------------------------------------------------
# to_oasis_format
# ---------------------------------------------------------------------------

def test_to_oasis_format():
    personas = [
        {"name": "Alice", "background": "Researcher.", "beliefs": ["AI is great"], "influence_score": 0.9}
    ]
    oasis = PersonaGenerator.to_oasis_format(personas)
    assert len(oasis) == 1
    assert "user_profile" in oasis[0]
    assert "Alice" in oasis[0]["user_profile"]
    assert oasis[0]["activity_level"] == 0.9


def test_to_oasis_format_missing_optional_keys():
    """to_oasis_format must not crash on personas with missing optional fields."""
    personas = [{"name": "Minimal"}]
    oasis = PersonaGenerator.to_oasis_format(personas)
    assert len(oasis) == 1
    assert "user_profile" in oasis[0]


def test_to_oasis_format_default_activity_level():
    personas = [{"name": "X", "background": "bg", "beliefs": []}]
    oasis = PersonaGenerator.to_oasis_format(personas)
    assert oasis[0]["activity_level"] == 0.5
