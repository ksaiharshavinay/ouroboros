#!/usr/bin/env python3
# OUROBOROS — autoresearch/prepare.py
# Copyright (c) 2026 Sai Harsha Vinaykudikala. All rights reserved.
# PATENT PENDING
#
# Prepares the OASIS corpus for model training.

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CORPUS_DIR = Path("data/corpus")
TRAIN_FILE = CORPUS_DIR / "train.jsonl"
EVAL_FILE = CORPUS_DIR / "eval.jsonl"
EVAL_SPLIT = 0.1


def load_corpus() -> list[dict]:
    turns = []
    for jsonl_file in sorted(CORPUS_DIR.glob("corpus_cycle_*.jsonl")):
        with jsonl_file.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        turns.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return turns


def turns_to_training_examples(turns: list[dict]) -> list[dict]:
    """Convert raw simulation turns into (prompt, completion) training pairs."""
    examples = []
    for i in range(1, len(turns)):
        prev = turns[i - 1].get("content", "")
        curr = turns[i].get("content", "")
        if prev and curr:
            examples.append({
                "prompt": f"Agent interaction context: {prev}\n\nNext agent response:",
                "completion": " " + curr,
                "influence": turns[i].get("influence", 0.5),
            })
    return examples


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

    examples = turns_to_training_examples(turns)
    random.shuffle(examples)

    split = max(1, int(len(examples) * EVAL_SPLIT))
    eval_examples = examples[:split]
    train_examples = examples[split:]

    write_jsonl(train_examples, TRAIN_FILE)
    write_jsonl(eval_examples, EVAL_FILE)
    logger.info("Corpus prepared: %d train / %d eval", len(train_examples), len(eval_examples))


if __name__ == "__main__":
    main()
