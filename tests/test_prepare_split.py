"""Tests for the Phase 2 leakage-resistant split in autoresearch/prepare.py.

The guarantee under test: examples are split by GROUP (source corpus file
+ target-turn round) — no group id ever appears in both train and eval —
and the split is deterministic for a fixed seed.
"""
import importlib.util
from pathlib import Path

import pytest

_PREPARE_PATH = Path(__file__).parent.parent / "autoresearch" / "prepare.py"
_spec = importlib.util.spec_from_file_location("ouro_prepare", _PREPARE_PATH)
prepare = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prepare)


def _turn(source: str, round_: int, content: str) -> dict:
    return {
        "round": round_,
        "agent_id": 0,
        "content": content,
        "influence": 0.5,
        "action": "create_post",
        "_source_file": source,
    }


def _two_file_turns() -> list[dict]:
    turns = []
    for f, rounds in (("corpus_cycle_0000.jsonl", (1, 1, 2, 2)),
                      ("corpus_cycle_0001.jsonl", (1, 1, 2, 2))):
        for i, r in enumerate(rounds):
            turns.append(_turn(f, r, f"{f} turn {i} round {r}"))
    return turns


# ---------------------------------------------------------------------------
# Pair formation
# ---------------------------------------------------------------------------

def test_pairs_never_cross_source_files():
    examples = prepare.turns_to_training_examples(_two_file_turns())
    # 3 pairs per file (4 turns each); the file boundary pair is dropped
    assert len(examples) == 6
    for ex in examples:
        assert ex["source_file"] in {
            "corpus_cycle_0000.jsonl", "corpus_cycle_0001.jsonl"
        }
        assert ex["group"].startswith(ex["source_file"] + "#round")


def test_example_group_uses_target_turn_round():
    turns = [
        _turn("corpus_cycle_0000.jsonl", 1, "first"),
        _turn("corpus_cycle_0000.jsonl", 2, "second"),
    ]
    (ex,) = prepare.turns_to_training_examples(turns)
    assert ex["group"] == "corpus_cycle_0000.jsonl#round2"


# ---------------------------------------------------------------------------
# Group-disjoint split
# ---------------------------------------------------------------------------

def test_split_is_group_disjoint():
    examples = prepare.turns_to_training_examples(_two_file_turns())
    train, eval_ = prepare.split_examples(examples, seed=42)
    prepare.assert_no_leakage(train, eval_)  # must not raise
    assert {e["group"] for e in train}.isdisjoint({e["group"] for e in eval_})
    assert len(train) + len(eval_) == len(examples)
    assert train and eval_, "both sides should be non-empty with 4 groups"


def test_split_is_deterministic_for_same_seed():
    examples = prepare.turns_to_training_examples(_two_file_turns())
    t1, e1 = prepare.split_examples(examples, seed=7)
    t2, e2 = prepare.split_examples(examples, seed=7)
    assert [x["group"] for x in t1] == [x["group"] for x in t2]
    assert [x["group"] for x in e1] == [x["group"] for x in e2]


def test_single_group_goes_all_to_train():
    turns = [_turn("corpus_cycle_0000.jsonl", 1, f"t{i}") for i in range(4)]
    examples = prepare.turns_to_training_examples(turns)
    train, eval_ = prepare.split_examples(examples, seed=42)
    assert len(train) == len(examples)
    assert eval_ == []


def test_assert_no_leakage_raises_on_overlap():
    ex = {"group": "f#round1", "prompt": "p", "completion": "c"}
    with pytest.raises(AssertionError, match="leakage"):
        prepare.assert_no_leakage([ex], [dict(ex)])


def test_assert_no_leakage_passes_when_disjoint():
    a = {"group": "f#round1", "prompt": "p", "completion": "c"}
    b = {"group": "f#round2", "prompt": "p", "completion": "c"}
    prepare.assert_no_leakage([a], [b])  # must not raise


# ---------------------------------------------------------------------------
# load_corpus tagging (against a patched corpus dir)
# ---------------------------------------------------------------------------

def test_load_corpus_tags_source_file(tmp_path, monkeypatch):
    import json as _json
    f = tmp_path / "corpus_cycle_0007.jsonl"
    f.write_text(_json.dumps({"round": 3, "content": "hello"}) + "\n")
    monkeypatch.setattr(prepare, "CORPUS_DIR", tmp_path)
    turns = prepare.load_corpus()
    assert len(turns) == 1
    assert turns[0]["_source_file"] == "corpus_cycle_0007.jsonl"
