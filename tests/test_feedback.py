"""Tests for OuroFeedbackModule — including Algorithm 7: Confidence-Weighted Injection."""
import json
import math
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from ouro.config import OuroConfig
from ouro.feedback import OuroFeedbackModule, _HIGH_CONFIDENCE_THRESHOLD, _LOW_CONFIDENCE_THRESHOLD


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


# ===========================================================================
# Existing tests (run_feedback, cycle persistence, history)
# ===========================================================================

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
    fb2 = OuroFeedbackModule(tmp_config)
    assert fb2.cycle_count == 1


def test_corrupted_state_file(tmp_config):
    tmp_config.feedback_state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_config.feedback_state_path.write_text("INVALID JSON")
    fb = OuroFeedbackModule(tmp_config)
    assert fb.cycle_count == 0


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


# ===========================================================================
# Algorithm 7: _score_prediction
# ===========================================================================

def _make_logprob_client(logprob_value: float):
    """Build a mock LLM client that returns a single token with the given logprob."""
    client = MagicMock()
    token = MagicMock()
    token.logprob = logprob_value
    logprobs_obj = MagicMock()
    logprobs_obj.content = [token]
    choice = MagicMock()
    choice.logprobs = logprobs_obj
    client.chat.completions.create.return_value = MagicMock(choices=[choice])
    return client


def test_score_prediction_high_confidence(tmp_config):
    """log-prob close to 0 → exp(0) = 1 → high confidence score."""
    fb = OuroFeedbackModule(tmp_config)
    fb._llm_client = _make_logprob_client(-0.05)   # exp(-0.05) ≈ 0.95
    score = fb._score_prediction("prompt", "text")
    assert score >= _HIGH_CONFIDENCE_THRESHOLD


def test_score_prediction_low_confidence(tmp_config):
    """log-prob very negative → exp(-3) ≈ 0.05 → low confidence score."""
    fb = OuroFeedbackModule(tmp_config)
    fb._llm_client = _make_logprob_client(-3.0)    # exp(-3) ≈ 0.05
    score = fb._score_prediction("prompt", "text")
    assert score <= _LOW_CONFIDENCE_THRESHOLD


def test_score_prediction_neutral_on_api_failure(tmp_config):
    """Any exception during scoring must return neutral 0.5."""
    fb = OuroFeedbackModule(tmp_config)
    client = MagicMock()
    client.chat.completions.create.side_effect = Exception("API error")
    fb._llm_client = client
    score = fb._score_prediction("prompt", "text")
    assert score == 0.5


def test_score_prediction_neutral_when_no_client(tmp_config):
    """If LLM client cannot be initialised, score returns 0.5."""
    fb = OuroFeedbackModule(tmp_config)
    fb._llm_client = None
    with patch.object(fb, "_get_llm_client", return_value=None):
        score = fb._score_prediction("prompt", "text")
    assert score == 0.5


def test_score_prediction_neutral_when_logprobs_missing(tmp_config):
    """If logprobs object is absent from response, return neutral 0.5."""
    fb = OuroFeedbackModule(tmp_config)
    client = MagicMock()
    choice = MagicMock()
    choice.logprobs = None
    client.chat.completions.create.return_value = MagicMock(choices=[choice])
    fb._llm_client = client
    score = fb._score_prediction("prompt", "text")
    assert score == 0.5


def test_score_prediction_clamped_to_one(tmp_config):
    """Score must never exceed 1.0."""
    fb = OuroFeedbackModule(tmp_config)
    fb._llm_client = _make_logprob_client(0.0)   # exp(0) = 1 exactly
    score = fb._score_prediction("prompt", "text")
    assert score <= 1.0


def test_score_uses_mean_of_multiple_logprobs(tmp_config):
    """When multiple tokens are returned, the score must reflect their mean log-prob."""
    fb = OuroFeedbackModule(tmp_config)
    client = MagicMock()
    tokens = [MagicMock() for _ in range(3)]
    # log-probs: -1, -1, -1 → mean = -1 → exp(-1) ≈ 0.368
    for t in tokens:
        t.logprob = -1.0
    logprobs_obj = MagicMock()
    logprobs_obj.content = tokens
    choice = MagicMock()
    choice.logprobs = logprobs_obj
    client.chat.completions.create.return_value = MagicMock(choices=[choice])
    fb._llm_client = client
    score = fb._score_prediction("prompt", "text")
    assert abs(score - math.exp(-1.0)) < 0.01


# ===========================================================================
# Algorithm 7: _select_predictions
# ===========================================================================

def _make_scored(score: float, text: str = "pred") -> dict:
    return {"index": 0, "prompt": "p", "text": text, "confidence_score": score}


def test_select_keeps_high_confidence(tmp_config):
    fb = OuroFeedbackModule(tmp_config)
    items = [_make_scored(0.95)]
    assert len(fb._select_predictions(items)) == 1


def test_select_keeps_low_confidence(tmp_config):
    fb = OuroFeedbackModule(tmp_config)
    items = [_make_scored(0.10)]
    assert len(fb._select_predictions(items)) == 1


def test_select_discards_mid_confidence(tmp_config):
    fb = OuroFeedbackModule(tmp_config)
    items = [_make_scored(0.50)]
    assert len(fb._select_predictions(items)) == 0


def test_select_sorted_by_distance_from_midpoint(tmp_config):
    fb = OuroFeedbackModule(tmp_config)
    items = [_make_scored(0.72), _make_scored(0.98)]  # both high, 0.98 further from 0.5
    result = fb._select_predictions(items)
    assert result[0]["confidence_score"] == 0.98


def test_select_empty_input_returns_empty(tmp_config):
    fb = OuroFeedbackModule(tmp_config)
    assert fb._select_predictions([]) == []


# ===========================================================================
# Algorithm 7: run_feedback_scored (integration)
# ===========================================================================

def test_run_feedback_scored_writes_files(tmp_config):
    fb = OuroFeedbackModule(tmp_config)
    predictions = [{"prompt": "p1", "text": "prediction one"}]
    with patch.object(fb, "_score_prediction", return_value=0.95):
        paths = fb.run_feedback_scored(predictions, {"num_turns": 5, "num_agents": 10})
    assert len(paths) == 1
    assert paths[0].exists()


def test_run_feedback_scored_records_injection_mode(tmp_config):
    fb = OuroFeedbackModule(tmp_config)
    with patch.object(fb, "_score_prediction", return_value=0.90):
        fb.run_feedback_scored([{"prompt": "p", "text": "t"}], {})
    history = fb.get_history()
    assert history[0]["injection_mode"] == "confidence_weighted"


def test_run_feedback_scored_discards_mid_confidence(tmp_config):
    """Mid-confidence predictions must NOT produce seed files."""
    fb = OuroFeedbackModule(tmp_config)
    predictions = [
        {"prompt": "p1", "text": "high confidence"},
        {"prompt": "p2", "text": "mid confidence"},
        {"prompt": "p3", "text": "low confidence"},
    ]
    scores = [0.92, 0.50, 0.08]
    call_count = [0]

    def mock_score(prompt, text):
        s = scores[call_count[0]]
        call_count[0] += 1
        return s

    with patch.object(fb, "_score_prediction", side_effect=mock_score):
        paths = fb.run_feedback_scored(predictions, {})

    # Only high (0.92) and low (0.08) should be injected; mid (0.50) discarded
    assert len(paths) == 2


def test_run_feedback_scored_fallback_when_all_mid(tmp_config):
    """If all predictions are mid-confidence, fall back to injecting all."""
    fb = OuroFeedbackModule(tmp_config)
    predictions = [{"prompt": "p", "text": "t"} for _ in range(3)]
    with patch.object(fb, "_score_prediction", return_value=0.50):
        paths = fb.run_feedback_scored(predictions, {})
    assert len(paths) == 3   # all injected as fallback


def test_run_feedback_scored_seed_contains_confidence_score(tmp_config):
    """Seed file must include the confidence score for auditability."""
    fb = OuroFeedbackModule(tmp_config)
    with patch.object(fb, "_score_prediction", return_value=0.88):
        paths = fb.run_feedback_scored([{"prompt": "p", "text": "hello"}], {})
    content = paths[0].read_text()
    assert "0.88" in content


def test_run_feedback_scored_increments_cycle(tmp_config):
    fb = OuroFeedbackModule(tmp_config)
    with patch.object(fb, "_score_prediction", return_value=0.95):
        fb.run_feedback_scored([{"prompt": "p", "text": "t"}], {})
    assert fb.cycle_count == 1
