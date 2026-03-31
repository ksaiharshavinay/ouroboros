"""Tests for OuroFeedbackModule."""
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from ouro.config import OuroConfig
from ouro.feedback import OuroFeedbackModule


@pytest.fixture
def tmp_config(tmp_path):
    cfg = OuroConfig(
        seed_docs_dir=tmp_path / "seeds",
        corpus_dir=tmp_path / "corpus",
        model_dir=tmp_path / "model",
        feedback_state_path=tmp_path / "state.json",
        kuzu_db_path=tmp_path / "kuzu",
    )
    return cfg


def test_initial_cycle_count(tmp_config):
    fb = OuroFeedbackModule(tmp_config)
    assert fb.cycle_count == 0


def test_run_feedback_increments_cycle(tmp_config):
    fb = OuroFeedbackModule(tmp_config)
    fb.run_feedback("prediction text", {"num_turns": 10, "num_agents": 5})
    assert fb.cycle_count == 1


def test_run_feedback_writes_seed_file(tmp_config):
    fb = OuroFeedbackModule(tmp_config)
    seed_path = fb.run_feedback("hello world", {"num_turns": 1, "num_agents": 1})
    assert seed_path.exists()
    assert seed_path.read_text() == "hello world"


def test_cycle_count_persists(tmp_config):
    fb1 = OuroFeedbackModule(tmp_config)
    fb1.run_feedback("cycle 0", {"num_turns": 1, "num_agents": 1})
    # New instance reads from disk
    fb2 = OuroFeedbackModule(tmp_config)
    assert fb2.cycle_count == 1


def test_corrupted_state_file(tmp_config):
    tmp_config.feedback_state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_config.feedback_state_path.write_text("INVALID JSON")
    fb = OuroFeedbackModule(tmp_config)
    assert fb.cycle_count == 0  # Falls back to fresh state


def test_get_history(tmp_config):
    fb = OuroFeedbackModule(tmp_config)
    fb.run_feedback("p1", {"num_turns": 5, "num_agents": 10})
    fb.run_feedback("p2", {"num_turns": 8, "num_agents": 10})
    history = fb.get_history()
    assert len(history) == 2
    assert history[0]["cycle"] == 0
    assert history[1]["cycle"] == 1


def test_multi_cycle_metadata_isolation(tmp_config):
    fb = OuroFeedbackModule(tmp_config)
    fb.run_feedback("c0", {"num_turns": 3, "num_agents": 2})
    fb.run_feedback("c1", {"num_turns": 7, "num_agents": 4})
    h = fb.get_history()
    assert h[0]["num_turns"] == 3
    assert h[1]["num_turns"] == 7
