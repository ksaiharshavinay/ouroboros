"""Tests for OuroPipeline."""
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from ouro.config import OuroConfig
from ouro.pipeline import OuroPipeline


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


# ---------------------------------------------------------------------------
# _load_seeds
# ---------------------------------------------------------------------------

def test_load_seeds_sorted(tmp_config):
    seeds_dir = tmp_config.seed_docs_dir
    seeds_dir.mkdir(parents=True)
    (seeds_dir / "b.txt").write_text("b")
    (seeds_dir / "a.txt").write_text("a")
    pipeline = OuroPipeline(tmp_config)
    seeds = pipeline._load_seeds()
    names = [s.name for s in seeds]
    assert names == sorted(names)


def test_load_seeds_empty(tmp_config):
    tmp_config.seed_docs_dir.mkdir(parents=True)
    pipeline = OuroPipeline(tmp_config)
    assert pipeline._load_seeds() == []


def test_load_seeds_includes_md_files(tmp_config):
    seeds_dir = tmp_config.seed_docs_dir
    seeds_dir.mkdir(parents=True)
    (seeds_dir / "note.md").write_text("# heading")
    (seeds_dir / "data.txt").write_text("data")
    pipeline = OuroPipeline(tmp_config)
    names = [s.name for s in pipeline._load_seeds()]
    assert "note.md" in names
    assert "data.txt" in names


def test_load_seeds_ignores_non_text_files(tmp_config):
    seeds_dir = tmp_config.seed_docs_dir
    seeds_dir.mkdir(parents=True)
    (seeds_dir / "image.png").write_bytes(b"\x89PNG")
    (seeds_dir / "real.txt").write_text("text")
    pipeline = OuroPipeline(tmp_config)
    names = [s.name for s in pipeline._load_seeds()]
    assert "image.png" not in names
    assert "real.txt" in names


# ---------------------------------------------------------------------------
# run_feedback_from_disk
# ---------------------------------------------------------------------------

def test_run_feedback_from_disk_no_state_raises(tmp_config):
    pipeline = OuroPipeline(tmp_config)
    with pytest.raises(FileNotFoundError):
        pipeline.run_feedback_from_disk()


def test_run_feedback_from_disk_reads_persisted_state(tmp_config):
    pipeline = OuroPipeline(tmp_config)
    state_path = tmp_config.feedback_state_path.parent / "last_cycle.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({"sim_result": {"num_turns": 5, "num_agents": 10}}))

    with patch.object(pipeline.infer, "predict", return_value="mock prediction"):
        seed_path = pipeline.run_feedback_from_disk(prompt="test")
    assert seed_path.exists()


def test_run_feedback_from_disk_uses_sim_result(tmp_config):
    """Feedback seed file must contain the prediction text."""
    pipeline = OuroPipeline(tmp_config)
    state_path = tmp_config.feedback_state_path.parent / "last_cycle.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({"sim_result": {"num_turns": 3}}))

    with patch.object(pipeline.infer, "predict", return_value="future_prediction"):
        seed_path = pipeline.run_feedback_from_disk()
    assert "future_prediction" in seed_path.read_text()


# ---------------------------------------------------------------------------
# infer.is_ready
# ---------------------------------------------------------------------------

def test_is_ready_false_when_no_checkpoint(tmp_config):
    pipeline = OuroPipeline(tmp_config)
    assert not pipeline.infer.is_ready()


def test_is_ready_true_after_checkpoint_created(tmp_config):
    checkpoint = tmp_config.model_dir / "checkpoint"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text("{}")
    pipeline = OuroPipeline(tmp_config)
    assert pipeline.infer.is_ready()


def test_run_full_cycle_skips_inference_when_not_ready(tmp_config):
    """run_full_cycle must return early without crashing if no model checkpoint exists."""
    pipeline = OuroPipeline(tmp_config)

    mock_sim = {"num_turns": 5, "num_agents": 10, "corpus_path": "x", "cycle": 0, "report_preview": ""}
    mock_research = {"stdout": "", "returncode": 0}

    with patch.object(pipeline, "run_swarm", return_value=mock_sim), \
         patch.object(pipeline, "run_autoresearch", return_value=mock_research):
        sim_result, research_result, seed_path = pipeline.run_full_cycle()

    assert sim_result == mock_sim
    assert research_result == mock_research
    assert seed_path == Path()
