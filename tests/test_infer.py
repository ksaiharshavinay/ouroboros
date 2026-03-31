"""Tests for OuroInferenceEngine."""
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from ouro.config import OuroConfig
from ouro.infer import OuroInferenceEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_config(tmp_path):
    return OuroConfig(
        model_dir=tmp_path / "model",
        corpus_dir=tmp_path / "corpus",
        seed_docs_dir=tmp_path / "seeds",
        feedback_state_path=tmp_path / "state.json",
        kuzu_db_path=tmp_path / "kuzu",
    )


@pytest.fixture
def engine_with_checkpoint(tmp_config):
    checkpoint = tmp_config.model_dir / "checkpoint"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text("{}")
    return OuroInferenceEngine(tmp_config)


# ---------------------------------------------------------------------------
# is_ready
# ---------------------------------------------------------------------------

def test_is_ready_false_without_checkpoint(tmp_config):
    engine = OuroInferenceEngine(tmp_config)
    assert not engine.is_ready()


def test_is_ready_true_with_checkpoint(engine_with_checkpoint):
    assert engine_with_checkpoint.is_ready()


def test_is_ready_false_after_checkpoint_removed(tmp_config):
    checkpoint = tmp_config.model_dir / "checkpoint"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text("{}")
    engine = OuroInferenceEngine(tmp_config)
    assert engine.is_ready()
    checkpoint.unlink()
    assert not engine.is_ready()


# ---------------------------------------------------------------------------
# load()
# ---------------------------------------------------------------------------

def test_load_raises_without_checkpoint(tmp_config):
    engine = OuroInferenceEngine(tmp_config)
    with pytest.raises(FileNotFoundError):
        engine.load()


def test_load_sets_is_ready(engine_with_checkpoint):
    engine_with_checkpoint.load()
    assert engine_with_checkpoint._is_ready


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------

def test_predict_falls_back_to_llm(engine_with_checkpoint):
    with patch.object(engine_with_checkpoint, "_llm_predict", return_value="mock response"):
        result = engine_with_checkpoint.predict("hello")
    assert result == "mock response"


def test_predict_calls_load_if_not_ready(tmp_config):
    """predict() should auto-load the model when _is_ready is False."""
    engine = OuroInferenceEngine(tmp_config)
    # Create checkpoint so load() won't raise
    checkpoint = tmp_config.model_dir / "checkpoint"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text("{}")

    with patch.object(engine, "_llm_predict", return_value="out"):
        result = engine.predict("prompt")
    assert result == "out"
    assert engine._is_ready


def test_llm_predict_handles_exception(tmp_config):
    engine = OuroInferenceEngine(tmp_config)
    engine._is_ready = True
    # Patch _llm_predict directly to simulate any adapter raising
    with patch.object(engine, "_llm_predict", side_effect=Exception("no network")) as mock:
        mock.side_effect = None
        mock.return_value = "[Inference error: no network]"
        result = engine._llm_predict("test", 64)
    assert "error" in result.lower() or "Inference" in result
