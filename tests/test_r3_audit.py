# OUROBOROS — tests/test_r3_audit.py
# Copyright (c) 2026 Kudikala Sai Harsha Vinay. All rights reserved.
# PATENT PENDING
#
# R3 C4: the offline gate audit (spec §6) against a fabricated B5 run
# whose expected precision/recall, UNAVAILABLE rates and
# counterfactual flips are computed by hand below.

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import r3_audit  # noqa: E402


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _turn(content: str) -> str:
    return json.dumps({"round": 1, "agent_id": 0, "content": content,
                       "influence": 0.5, "action": "create_post"})


def _decision(accepted, score, veto, signals):
    return {"cycle": 0, "index": 0, "accepted": accepted,
            "bypassed": False, "weighted_score": score,
            "threshold": 0.5, "veto": veto, "signals": signals,
            "text_preview": "x"}


def _sig(value):
    return {"value": value,
            "status": "ok" if value is not None else "unavailable",
            "detail": ""}


def make_fake_run(tmp_path: Path) -> Path:
    out = tmp_path / "run_b5"
    data = out / "state" / "data"
    _write(out / "r1_params.json", json.dumps(
        {"arm": "B5", "rng_seed": 1, "train_model": "m"}))
    _write(data / "corpus" / "corpus_cycle_0000.jsonl",
           _turn("initial turn") + "\n")
    _write(data / "corpus" / "corpus_cycle_0001.jsonl",
           _turn("zephyr quasar lantern harvest meadow willow drift")
           + "\n")
    _write(data / "corpus" / "corpus_cycle_0002.jsonl",
           _turn("sowing planting watering pruning grafting season")
           + "\n")
    _write(data / "seeds" / "seed1.txt",
           "sowing planting watering pruning grafting")
    # Logged live decisions (hand-computed expectations):
    # D1 accepted (0.75); D2 rejected as-run (0.405) but accepted
    # under BOTH ablations; D3 vetoed — rejected under everything.
    d1 = _decision(True, 0.75, None, {
        "outcome_verification": _sig(None), "diversity": _sig(0.8),
        "consistency": _sig(0.5), "provenance": _sig(1.0)})
    d2 = _decision(False, 0.405, None, {
        "outcome_verification": _sig(0.0), "diversity": _sig(0.9),
        "consistency": _sig(0.9), "provenance": _sig(0.0)})
    d3 = _decision(False, 0.9, "too_short: 3 chars", {
        "outcome_verification": _sig(None), "diversity": _sig(0.9),
        "consistency": _sig(0.9), "provenance": _sig(0.9)})
    _write(data / "gate_decisions.jsonl",
           "\n".join(json.dumps(d) for d in (d1, d2, d3)) + "\n")
    return out


# Pool candidates (cycle 0), engineered against the fake corpora:
#   A/C "zephyr ..."  gate accepts; all salient terms recur later -> TP
#   B   "zephyr vortex ..." gate accepts; 1/7 terms recur -> FP
#   D   seed-like, no generator identity; gate rejects; 5/6 recur -> FN
#   E   degenerate repetition; vetoed; term never recurs -> TN
POOL = [
    {"cycle": 0, "backend": "local_lora",
     "text": "zephyr quasar lantern harvest meadow willow"},
    {"cycle": 0, "backend": "local_lora",
     "text": "zephyr vortex nebula cinder basalt granite obsidian"},
    {"cycle": 0, "backend": "local_lora",
     "text": "zephyr quasar lantern harvest meadow willow"},
    {"cycle": 0, "corpus_path": "/nonexistent/corpus.jsonl",
     "text": "sowing planting watering pruning grafting vortex"},
    {"cycle": 0, "backend": "local_lora",
     "text": "banana banana banana banana banana"},
]


def test_audit_report_hand_computed(tmp_path):
    out = make_fake_run(tmp_path)
    state = r3_audit.RunState(out)
    report = r3_audit.build_report(state, POOL, [])
    pool = report["pool"]
    assert (pool["tp"], pool["fp"], pool["fn"], pool["tn"]) == (2, 1, 1, 1)
    assert pool["precision"] == round(2 / 3, 4)
    assert pool["recall"] == round(2 / 3, 4)
    assert pool["accept_rate"] == 0.6
    # As-run conditions: no later outcomes exist inside the gate's
    # truncated view, so the outcome signal is UNAVAILABLE for every
    # pool candidate — exactly as it was live.
    assert pool["signal_unavailable_rates"]["outcome_verification"] == 1.0
    assert pool["signal_unavailable_rates"]["diversity"] == 0.0

    logged = report["logged"]
    assert logged["decisions"] == 3
    assert logged["accept_rate"] == round(1 / 3, 4)
    assert logged["signal_unavailable_rates"]["outcome_verification"] \
        == round(2 / 3, 4)
    cf = logged["counterfactual"]
    # Self-check: rescoring with the as-run weights must reproduce
    # the logged decisions exactly.
    assert cf["as_run_replay"]["accepts"] == 1
    assert cf["as_run_replay"]["flips_to_accept"] == 0
    assert cf["as_run_replay"]["flips_to_reject"] == 0
    assert cf["consistency_only"]["accepts"] == 2
    assert cf["consistency_only"]["flips_to_accept"] == 1
    assert cf["consistency_only"]["flips_to_reject"] == 0
    assert cf["no_outcome"]["accepts"] == 2
    assert cf["no_outcome"]["flips_to_accept"] == 1


def test_label_candidate_threshold():
    texts = {0: "a", 1: "zephyr quasar lantern harvest meadow"}
    lab = r3_audit.label_candidate(
        "zephyr quasar lantern harvest meadow willow", 0, texts)
    assert lab["label"] is True and lab["hit_fraction"] == round(5 / 6, 4)
    lab = r3_audit.label_candidate("zephyr vortex nebula", 0, texts)
    assert lab["label"] is True  # 1/3 = 0.333 >= the 0.20 threshold
    assert lab["hit_fraction"] == round(1 / 3, 4)
    lab = r3_audit.label_candidate("vortex nebula cinder basalt", 0, texts)
    assert lab["label"] is False and lab["hit_fraction"] == 0.0


def test_rescore_veto_stays_rejected():
    d = _decision(False, 0.99, "too_short", {
        "outcome_verification": _sig(1.0), "diversity": _sig(1.0),
        "consistency": _sig(1.0), "provenance": _sig(1.0)})
    for weights in r3_audit.ABLATIONS.values():
        assert r3_audit.rescore_decision(d, weights) is False


def test_sample_candidates_injected_generate_fn(tmp_path):
    out = make_fake_run(tmp_path)
    state = r3_audit.RunState(out)
    calls = []

    def gen(cycle, prompt):
        calls.append((cycle, prompt))
        return {"text": f"text for cycle {cycle}", "backend": "fake"}

    candidates, skipped = r3_audit.sample_candidates(
        state, cycles=(0, 1), per_cycle=3, generate_fn=gen)
    assert skipped == []
    assert len(candidates) == 6
    assert {c["cycle"] for c in candidates} == {0, 1}
    assert all(c["prompt"] == r3_audit.PROMPT for c in candidates)
    assert len(calls) == 6


def test_sample_candidates_skips_cycles_without_checkpoints(tmp_path):
    out = make_fake_run(tmp_path)
    state = r3_audit.RunState(out)
    candidates, skipped = r3_audit.sample_candidates(
        state, cycles=(0,), per_cycle=2)
    assert candidates == []
    assert len(skipped) == 1 and skipped[0]["cycle"] == 0


def test_markdown_report_renders(tmp_path):
    out = make_fake_run(tmp_path)
    report = r3_audit.build_report(r3_audit.RunState(out), POOL, [])
    md = r3_audit.render_markdown(report)
    assert "precision: 0.6667" in md
    assert "consistency_only" in md
