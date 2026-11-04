#!/usr/bin/env python3
# OUROBOROS — autoresearch/prepare.py
# Copyright (c) 2026 Kudikala Sai Harsha Vinay. All rights reserved.
# PATENT PENDING
#
# Prepares the OASIS corpus for model training.
#
# Phase 2 (S2 data protocol): the train/eval split is leakage-resistant.
# The old code shuffled all adjacent-turn pairs globally and split at
# random, so near-identical pairs drawn from the SAME simulation round
# (same conversation burst, same template family) landed on both sides —
# eval measured memorisation of the train rounds, not generalisation.
# Now every example carries a group id — its source corpus file plus the
# round its target turn was produced in, the finest grouping key the
# exported corpus schema actually carries (file = one simulation run =
# one source document; round = one temporal slice of that run) — and the
# split is by GROUP: no group id ever appears in both train and eval.
# An assertion in main() enforces that invariant on every run.

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CORPUS_DIR = Path("data/corpus")
TRAIN_FILE = CORPUS_DIR / "train.jsonl"
EVAL_FILE = CORPUS_DIR / "eval.jsonl"
EVAL_SPLIT = 0.1
# Split seed: fixed by default so splits are reproducible run-to-run;
# override with $OURO_SPLIT_SEED for split-seed sensitivity checks.
DEFAULT_SPLIT_SEED = 42


def load_corpus() -> list[dict]:
    """Load all corpus turns, tagging each with its source corpus file."""
    turns = []
    for jsonl_file in sorted(CORPUS_DIR.glob("corpus_cycle_*.jsonl")):
        with jsonl_file.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        turn = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    turn["_source_file"] = jsonl_file.name
                    turns.append(turn)
    return turns


def example_group(source_file: str, target_round: Any) -> str:
    """Group id for a training example: source corpus file + target round."""
    return f"{source_file}#round{target_round}"


def turns_to_training_examples(turns: list[dict]) -> list[dict]:
    """
    Convert raw simulation turns into (prompt, completion) training pairs.

    Pairs are formed only from consecutive turns of the SAME source file
    (a pair spanning two simulation runs is meaningless), and each example
    records the group it belongs to for the group-disjoint split.
    """
    examples = []
    for i in range(1, len(turns)):
        prev, curr = turns[i - 1], turns[i]
        prev_file = prev.get("_source_file", "unknown")
        curr_file = curr.get("_source_file", "unknown")
        if prev_file != curr_file:
            continue
        prev_text = prev.get("content", "")
        curr_text = curr.get("content", "")
        if prev_text and curr_text:
            examples.append({
                "prompt": f"Agent interaction context: {prev_text}\n\nNext agent response:",
                "completion": " " + curr_text,
                "influence": curr.get("influence", 0.5),
                "source_file": curr_file,
                "group": example_group(curr_file, curr.get("round", 0)),
            })
    return examples


def split_examples(
    examples: list[dict],
    eval_split: float = EVAL_SPLIT,
    seed: int = DEFAULT_SPLIT_SEED,
) -> tuple[list[dict], list[dict]]:
    """
    Split examples into (train, eval) by GROUP: every group id goes
    entirely to one side. Deterministic for a given seed.

    With >= 2 groups, at least one group lands in eval and at least one
    in train; eval grows group-by-group until it reaches the requested
    fraction of examples (approximately — groups are indivisible).
    With a single group no honest split exists: everything goes to
    train and eval is empty (logged by the caller).
    """
    by_group: dict[str, list[dict]] = {}
    for ex in examples:
        by_group.setdefault(ex["group"], []).append(ex)

    group_ids = sorted(by_group)
    if len(group_ids) <= 1:
        return list(examples), []

    rng = random.Random(seed)
    shuffled = group_ids[:]
    rng.shuffle(shuffled)

    target_eval = max(1, int(round(len(examples) * eval_split)))
    eval_ids: set[str] = set()
    eval_count = 0
    for gid in shuffled:
        if eval_count >= target_eval:
            break
        if len(eval_ids) >= len(group_ids) - 1:
            break  # always leave at least one group for train
        eval_ids.add(gid)
        eval_count += len(by_group[gid])

    train = [ex for ex in examples if ex["group"] not in eval_ids]
    eval_ = [ex for ex in examples if ex["group"] in eval_ids]
    rng.shuffle(train)
    rng.shuffle(eval_)
    return train, eval_


def assert_no_leakage(train: list[dict], eval_: list[dict]) -> None:
    """Fail loudly if any group id appears in both splits."""
    overlap = {ex["group"] for ex in train} & {ex["group"] for ex in eval_}
    if overlap:
        raise AssertionError(
            f"Train/eval leakage: group(s) present in both splits: {sorted(overlap)}"
        )


def write_jsonl(examples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    logger.info("Wrote %d examples to %s", len(examples), path)


def main() -> None:
    logger.info("Loading corpus from %s ...", CORPUS_DIR)
    turns = load_corpus()
    if not turns:
        logger.error("No corpus files found. Run the swarm simulation first.")
        return

    seed = int(os.environ.get("OURO_SPLIT_SEED", DEFAULT_SPLIT_SEED))
    examples = turns_to_training_examples(turns)
    train_examples, eval_examples = split_examples(examples, seed=seed)
    assert_no_leakage(train_examples, eval_examples)

    n_groups = len({ex["group"] for ex in examples})
    n_eval_groups = len({ex["group"] for ex in eval_examples})
    write_jsonl(train_examples, TRAIN_FILE)
    write_jsonl(eval_examples, EVAL_FILE)
    logger.info(
        "Corpus prepared: %d train / %d eval "
        "(%d groups: %d train / %d eval, split seed %d)",
        len(train_examples), len(eval_examples),
        n_groups, n_groups - n_eval_groups, n_eval_groups, seed,
    )
    if not eval_examples:
        logger.warning(
            "Eval split is EMPTY — the corpus has a single group, so no "
            "group-disjoint eval exists. Eval metrics this cycle are not "
            "meaningful."
        )


if __name__ == "__main__":
    main()
