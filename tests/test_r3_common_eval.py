# OUROBOROS — tests/test_r3_common_eval.py
# Copyright (c) 2026 Kudikala Sai Harsha Vinay. All rights reserved.
# PATENT PENDING
#
# R3 C2: the common frozen eval set (spec §3) — shape, provenance,
# no-overlap with the loop's texts, and a perplexity computation that
# mirrors autoresearch/train.py's evaluate().

import json
from pathlib import Path

import pytest

from ouro import common_eval

REPO = Path(__file__).resolve().parent.parent
EVAL_PATH = REPO / "data" / "eval" / "common_eval.jsonl"
SOURCES_PATH = REPO / "data" / "eval" / "common_eval_sources.json"


def test_common_eval_file_shape():
    rows = common_eval.load_common_eval(EVAL_PATH)
    assert len(rows) == 40
    sources = {}
    for row in rows:
        assert row["prompt"].strip()
        # prepare.py convention: completions carry a leading space.
        assert row["completion"].startswith(" ")
        assert len(row["completion"]) > 50
        sources[row["source"]] = sources.get(row["source"], 0) + 1
    assert sources == {"gutenberg-11-alice": 20,
                       "gutenberg-1342-pride-and-prejudice": 20}


def test_common_eval_sources_recorded():
    meta = json.loads(SOURCES_PATH.read_text())
    ids = {s["project_gutenberg_ebook"] for s in meta["sources"]}
    assert ids == {11, 1342}
    for s in meta["sources"]:
        assert s["url"].startswith("http")
        assert "Public domain" in s["copyright"]


def test_common_eval_no_overlap_with_seeds():
    rows = common_eval.load_common_eval(EVAL_PATH)
    seed_text = "\n".join(
        p.read_text(encoding="utf-8")
        for p in sorted((REPO / "data" / "seeds").glob("*.txt")))
    for row in rows:
        assert row["prompt"] not in seed_text
        assert row["completion"].strip() not in seed_text


def test_common_eval_constants_mirror_train():
    from autoresearch import train
    assert common_eval.MAX_SEQ_LEN == train.MAX_SEQ_LEN
    assert common_eval.EVAL_LIMIT == train.EVAL_LIMIT


@pytest.mark.slow
def test_common_eval_perplexity_matches_train_method():
    # Same model, same rows: ouro.common_eval._eval_nll must equal
    # autoresearch.train._eval_nll exactly (it is a mirror), and the
    # public perplexity must be exp of it and finite.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from autoresearch.train import _eval_nll as train_eval_nll

    rows = common_eval.load_common_eval(EVAL_PATH)[:3]
    tok = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        "sshleifer/tiny-gpt2", dtype=torch.float32)
    from autoresearch.train import MAX_SEQ_LEN as TRAIN_MAX_SEQ_LEN
    assert common_eval._eval_nll(model, tok, rows) == \
        train_eval_nll(model, tok, rows, TRAIN_MAX_SEQ_LEN)
    ppl = common_eval.common_eval_perplexity(
        rows, "sshleifer/tiny-gpt2", None)
    assert ppl == ppl and 1.0 < ppl < float("inf")
