"""Tests for the Phase 2 multi-signal feedback gate (ouro/gate.py) and
its wiring through OuroFeedbackModule.run_feedback_gated.

Design intent under test (brainstorm §4.1 C3 / Finding 4):
- decisions carry per-signal values, with UNAVAILABLE where a signal
  cannot be measured — never an invented score,
- degenerate/repetitive candidates are vetoed,
- there is no inject-everything fallback,
- gate-off is a clean, recorded bypass (the ungated baseline condition).
"""
import json

import pytest
from pathlib import Path

from ouro.config import OuroConfig
from ouro.feedback import OuroFeedbackModule
from ouro.gate import (
    FeedbackGate,
    SIGNAL_CONSISTENCY,
    SIGNAL_DIVERSITY,
    SIGNAL_OUTCOME,
    SIGNAL_PROVENANCE,
)


@pytest.fixture
def gate_env(tmp_path):
    """Config + on-disk seeds/corpus for gate evaluation."""
    cfg = OuroConfig(
        seed_docs_dir=tmp_path / "seeds",
        corpus_dir=tmp_path / "corpus",
        model_dir=tmp_path / "model",
        feedback_state_path=tmp_path / "state.json",
        kuzu_db_path=tmp_path / "kuzu",
    )
    cfg.seed_docs_dir.mkdir(parents=True, exist_ok=True)
    (cfg.seed_docs_dir / "seed1.txt").write_text(
        "Baseline briefing: quarterly logistics costs rose across regional "
        "depots while diesel prices steadied and rail capacity tightened."
    )
    cfg.corpus_dir.mkdir(parents=True, exist_ok=True)
    corpus0 = cfg.corpus_dir / "corpus_cycle_0000.jsonl"
    corpus0.write_text(
        json.dumps({"round": 1, "agent_id": 0, "content": "agents discuss logistics",
                    "influence": 0.5, "action": "create_post"}) + "\n"
    )
    return cfg, corpus0


def _good_candidate(corpus0: Path, **over) -> dict:
    cand = {
        "prompt": "Summarize the key predictions from this cycle.",
        "text": (
            "Verification tooling for data quality pipelines will become the "
            "default gate in enterprise governance stacks, with provenance "
            "graphs tracking every transformation and auditors demanding "
            "reproducible lineage evidence before deployment sign-off."
        ),
        "backend": "local",
        "checkpoint_sha256": "ab12" * 16,
        "corpus_path": str(corpus0),
        "cycle": 0,
    }
    cand.update(over)
    return cand


# ---------------------------------------------------------------------------
# Known-good candidate is accepted, with a full decision record
# ---------------------------------------------------------------------------

def test_known_good_candidate_accepted(gate_env):
    cfg, corpus0 = gate_env
    gate = FeedbackGate(cfg)
    cands = [_good_candidate(corpus0) for _ in range(3)]  # resamples of one prompt
    (decision,) = [gate.evaluate(cands)[0]]
    assert decision["accepted"] is True
    assert decision["veto"] is None
    assert decision["weighted_score"] >= cfg.gate_accept_threshold
    sig = decision["signals"]
    assert sig[SIGNAL_DIVERSITY]["status"] == "ok"
    assert sig[SIGNAL_DIVERSITY]["value"] > 0.8, "novel text vs the seed set"
    assert sig[SIGNAL_CONSISTENCY]["status"] == "ok"
    assert sig[SIGNAL_CONSISTENCY]["value"] == 1.0, "identical resamples agree fully"
    assert sig[SIGNAL_PROVENANCE]["value"] == 1.0
    # No later-cycle corpus exists: outcome verification must be honest
    assert sig[SIGNAL_OUTCOME]["status"] == "unavailable"
    assert sig[SIGNAL_OUTCOME]["value"] is None


# ---------------------------------------------------------------------------
# Degenerate / repetitive candidate is rejected by veto
# ---------------------------------------------------------------------------

def test_repetitive_candidate_vetoed(gate_env):
    cfg, corpus0 = gate_env
    gate = FeedbackGate(cfg)
    spam = _good_candidate(corpus0, text="buy now buy now buy now buy now buy now")
    (decision,) = gate.evaluate([spam])
    assert decision["accepted"] is False
    assert decision["veto"] is not None
    assert "degenerate_repetition" in decision["veto"]


def test_inference_error_string_vetoed(gate_env):
    cfg, corpus0 = gate_env
    gate = FeedbackGate(cfg)
    err = _good_candidate(corpus0, text="[Inference error: connection refused]")
    (decision,) = gate.evaluate([err])
    assert decision["accepted"] is False
    assert "inference_error" in (decision["veto"] or "")


def test_duplicate_of_seed_scores_low_diversity(gate_env):
    cfg, corpus0 = gate_env
    gate = FeedbackGate(cfg)
    seed_text = (cfg.seed_docs_dir / "seed1.txt").read_text()
    dup = _good_candidate(corpus0, text=seed_text)
    (decision,) = gate.evaluate([dup])
    assert decision["signals"][SIGNAL_DIVERSITY]["value"] == 0.0
    assert decision["accepted"] is False, "re-injecting the seed verbatim must not pass"


# ---------------------------------------------------------------------------
# UNAVAILABLE-signal handling + delayed-outcome verification
# ---------------------------------------------------------------------------

def test_unavailable_signals_when_no_references(gate_env):
    cfg, corpus0 = gate_env
    # Remove the seed set: diversity becomes unmeasurable, not zero.
    (cfg.seed_docs_dir / "seed1.txt").unlink()
    gate = FeedbackGate(cfg)
    (decision,) = gate.evaluate([_good_candidate(corpus0)])
    assert decision["signals"][SIGNAL_DIVERSITY]["status"] == "unavailable"
    assert decision["signals"][SIGNAL_CONSISTENCY]["status"] == "unavailable"
    assert decision["signals"][SIGNAL_OUTCOME]["status"] == "unavailable"
    # Provenance alone (1.0 * its weight, renormalised) still decides
    assert decision["accepted"] is True
    assert decision["weighted_score"] == 1.0


def test_outcome_verification_uses_later_cycle_corpus(gate_env):
    cfg, corpus0 = gate_env
    # A later cycle's corpus records what actually happened next.
    later = cfg.corpus_dir / "corpus_cycle_0001.jsonl"
    later.write_text(
        json.dumps({
            "round": 1, "agent_id": 1,
            "content": "Everyone is talking about verification tooling and "
                       "provenance graphs in governance tooling today.",
            "influence": 0.5, "action": "create_post",
        }) + "\n"
    )
    gate = FeedbackGate(cfg)
    (decision,) = gate.evaluate([_good_candidate(corpus0)])
    sig = decision["signals"][SIGNAL_OUTCOME]
    assert sig["status"] == "ok"
    assert sig["value"] is not None and sig["value"] > 0.0, (
        "candidate terms that materialised later must score support"
    )


def test_outcome_verification_zero_when_terms_never_materialise(gate_env):
    cfg, corpus0 = gate_env
    later = cfg.corpus_dir / "corpus_cycle_0001.jsonl"
    later.write_text(
        json.dumps({
            "round": 1, "agent_id": 1,
            "content": "completely unrelated chatter about weather and lunch",
            "influence": 0.5, "action": "create_post",
        }) + "\n"
    )
    gate = FeedbackGate(cfg)
    (decision,) = gate.evaluate([_good_candidate(corpus0)])
    sig = decision["signals"][SIGNAL_OUTCOME]
    assert sig["status"] == "ok"
    assert sig["value"] == 0.0


# ---------------------------------------------------------------------------
# Gate-off bypass
# ---------------------------------------------------------------------------

def test_gate_off_accepts_everything_with_bypass_records(gate_env):
    cfg, corpus0 = gate_env
    cfg.feedback_gate_enabled = False
    gate = FeedbackGate(cfg)
    spam = _good_candidate(corpus0, text="buy now buy now buy now")
    decisions = gate.evaluate([_good_candidate(corpus0), spam])
    assert all(d["accepted"] for d in decisions)
    assert all(d["bypassed"] for d in decisions)
    assert all(d["signals"] == {} for d in decisions)


# ---------------------------------------------------------------------------
# Feedback-module integration: audit log, state, seed files
# ---------------------------------------------------------------------------

def test_run_feedback_gated_writes_seed_and_audit_trail(gate_env):
    cfg, corpus0 = gate_env
    fb = OuroFeedbackModule(cfg)
    good = _good_candidate(corpus0)
    spam = _good_candidate(corpus0, text="spam spam spam spam spam spam")
    paths = fb.run_feedback_gated([good, spam], {"num_turns": 5, "num_agents": 10})

    assert len(paths) == 1, "only the good candidate becomes a seed"
    seed_text = paths[0].read_text()
    assert "Gate-decision: ACCEPTED" in seed_text
    assert "outcome_verification=UNAVAILABLE" in seed_text

    history = fb.get_history()
    assert history[0]["injection_mode"] == "gated"
    assert history[0]["total_candidates"] == 2
    assert history[0]["selected_count"] == 1
    assert len(history[0]["gate_decisions"]) == 2

    log_path = cfg.feedback_state_path.parent / "gate_decisions.jsonl"
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 2, "one audit record per candidate"
    records = [json.loads(line) for line in lines]
    assert records[0]["accepted"] is True and records[1]["accepted"] is False
    assert all("signals" in r for r in records)


def test_run_feedback_gated_all_rejected_is_not_a_fallback(gate_env):
    """The Algorithm 7 inject-everything fallback is gone: an empty
    accept set stays empty."""
    cfg, corpus0 = gate_env
    fb = OuroFeedbackModule(cfg)
    spam = _good_candidate(corpus0, text="spam spam spam spam spam spam")
    paths = fb.run_feedback_gated([spam], {"num_turns": 1, "num_agents": 1})
    assert paths == []
    assert fb.get_history()[0]["selected_count"] == 0
    assert fb.cycle_count == 1, "the cycle is still recorded for audit"


def test_run_feedback_gated_gate_off_uses_uniform_mode_label(gate_env):
    cfg, corpus0 = gate_env
    cfg.feedback_gate_enabled = False
    fb = OuroFeedbackModule(cfg)
    paths = fb.run_feedback_gated([_good_candidate(corpus0)], {})
    assert len(paths) == 1
    assert fb.get_history()[0]["injection_mode"] == "uniform_gate_off"
