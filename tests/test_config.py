"""Tests for OuroConfig."""
import pytest
from pathlib import Path
from ouro.config import OuroConfig


def test_defaults():
    cfg = OuroConfig()
    assert cfg.num_agents == 500
    assert cfg.simulation_rounds == 30
    assert cfg.experiment_budget_minutes == 5


def test_str_paths_converted():
    cfg = OuroConfig(seed_docs_dir="my/seeds", corpus_dir="my/corpus", model_dir="my/model")
    assert isinstance(cfg.seed_docs_dir, Path)
    assert isinstance(cfg.corpus_dir, Path)
    assert isinstance(cfg.model_dir, Path)


def test_custom_model():
    cfg = OuroConfig(llm_model="gpt-4o")
    assert cfg.llm_model == "gpt-4o"


def test_custom_base_url():
    cfg = OuroConfig(llm_base_url="https://my.openai.proxy/v1")
    assert "proxy" in cfg.llm_base_url
