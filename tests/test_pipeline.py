"""Tests for OuroPipeline."""
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from ouro.config import OuroConfig
from ouro.pipeline import OuroPipeline


@pytest.fixture
def tmp_config(tmp_path):
    return OuroConfig(
        seed_docs_dir=tmp_path / "seeds",
        corpus_dir=tmp_path / "corpus",
        model_dir=tmp_path / "model",
        feedback_state_path=tmp_path / "state.json",
        kuzu_db_path=tmp_path / "kuzu",
    )


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
    seeds = pipeline._load_seeds()
    assert seeds == []


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


def test_is_ready_false_when_no_checkpoint(tmp_config):
    pipeline = OuroPipeline(tmp_config)
    assert not pipeline.infer.is_ready()
