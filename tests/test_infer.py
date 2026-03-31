"""Tests for OuroInferenceEngine."""
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from ouro.config import OuroConfig
from ouro.infer import OuroInferenceEngine


@pytest.fixture
def tmp_config(tmp_path):
    return OuroConfig(
        model_dir=tmp_path / "model",
        corpus_dir=tmp_path / "corpus",
        seed_docs_dir=tmp_path / "seeds",
        feedback_state_path=tmp_path / "state.json",
        kuzu_db_path=tmp_path / "kuzu",
    )


def test_is_ready_false_without_checkpoint(tmp_config):
    engine = OuroInferenceEngine(tmp_config)
    assert not engine.is_ready()


def test_is_ready_true_with_checkpoint(tmp_config):
    checkpoint = tmp_config.model_dir / "checkpoint"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("{}")
    engine = OuroInferenceEngine(tmp_config)
    assert engine.is_ready()


def test_load_raises_without_checkpoint(tmp_config):
    engine = OuroInferenceEngine(tmp_config)
    with pytest.raises(FileNotFoundError):
        engine.load()


def test_predict_falls_back_to_llm(tmp_config):
    checkpoint = tmp_config.model_dir / "checkpoint"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("{}")
    engine = OuroInferenceEngine(tmp_config)
    with patch.object(engine, "_llm_predict", return_value="mock response") as mock_llm:
        result = engine.predict("hello")
    assert result == "mock response"


def test_llm_predict_handles_exception(tmp_config):
    engine = OuroInferenceEngine(tmp_config)
    engine._is_ready = True
    with patch("openai.OpenAI", side_effect=Exception("no network")):
        result = engine._llm_predict("test", 64)
    assert "error" in result.lower() or "Inference" in result
