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
#
# Run Spec R2 (anchoring sweep): with $OURO_REAL_FRACTION set to a float
# in (0, 1), the TRAIN split — and only the train split — is augmented
# with examples built from the human seed documents so that real
# examples account for that fraction of total training tokens (the
# B6 "anchored accumulation" arm). The split and the eval set are
# computed exactly as without the variable; real examples are
# train-only, never enter eval, and carry group ids prefixed
# "human-seed" so their origin is always accounted. With the variable
# unset or 0 the outputs are byte-identical to the unanchored path.

from __future__ import annotations

import json
import logging
import os
import random
import re
from pathlib import Path
from typing import Any, Callable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CORPUS_DIR = Path("data/corpus")
SEED_DIR = Path("data/seeds")
TRAIN_FILE = CORPUS_DIR / "train.jsonl"
EVAL_FILE = CORPUS_DIR / "eval.jsonl"
MIXTURE_FILE = CORPUS_DIR / "mixture.json"
EVAL_SPLIT = 0.1
# Split seed: fixed by default so splits are reproducible run-to-run;
# override with $OURO_SPLIT_SEED for split-seed sensitivity checks.
DEFAULT_SPLIT_SEED = 42
# Group-id prefix for examples built from the human seed documents.
REAL_GROUP_PREFIX = "human-seed"
# Mirror of autoresearch/train.py's MAX_SEQ_LEN: an example's effective
# training tokens are its prompt+completion tokens truncated to this.
MAX_SEQ_LEN = 512
# Mirror of train.py's base-model resolution (env OURO_TRAIN_BASE_MODEL,
# same default): token counting must use the tokenizer training uses.
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


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


def seed_documents_to_examples(seed_dir: Path | None = None) -> list[dict]:
    """
    Build training examples from the human seed documents (R2 anchor).

    Each document is one source unit — the analogue of one corpus file.
    A document is far longer than a simulation turn (580–720 tokenizer
    tokens vs the 512-token training window), and the corpus example
    format's length unit is the turn, so each document is chunked into
    its natural paragraph blocks and consecutive blocks are paired
    exactly as turns_to_training_examples pairs consecutive turns of
    one file: previous block as prompt context, block as completion.
    Every resulting example carries the document's group id,
    "human-seed:<file>", so origin accounting can always separate real
    examples from simulation ones.
    """
    seed_dir = SEED_DIR if seed_dir is None else seed_dir
    turns: list[dict] = []
    for path in sorted(seed_dir.glob("seed*.txt")):
        text = path.read_text(encoding="utf-8")
        blocks = [b.strip() for b in re.split(r"\n\s*\n", text)]
        for block in blocks:
            if block:
                turns.append({"content": block, "_source_file": path.name})
    examples = turns_to_training_examples(turns)
    for ex in examples:
        ex["group"] = f"{REAL_GROUP_PREFIX}:{ex['source_file']}"
    return examples


def load_base_tokenizer():
    """The base model's tokenizer — the one train.py tokenises with."""
    from transformers import AutoTokenizer

    model = os.environ.get("OURO_TRAIN_BASE_MODEL", DEFAULT_BASE_MODEL)
    return AutoTokenizer.from_pretrained(model)


def example_token_count(example: dict, tokenizer) -> int:
    """Effective training tokens of one example, counted as train.py's
    dataset builder counts them: prompt tokens (special tokens added)
    plus completion tokens (none added), the concatenation truncated
    to MAX_SEQ_LEN."""
    prompt_ids = tokenizer(
        example["prompt"], add_special_tokens=True)["input_ids"]
    completion_ids = tokenizer(
        example["completion"], add_special_tokens=False)["input_ids"]
    return min(len(prompt_ids) + len(completion_ids), MAX_SEQ_LEN)


def augment_train_with_real(
    train: list[dict],
    real_examples: list[dict],
    target_fraction: float,
    count_tokens: Callable[[dict], int],
    seed: int,
) -> tuple[list[dict], dict]:
    """
    Repeat/sample real examples into the train split until they account
    for `target_fraction` of total training tokens.

    Deterministic for a given seed: real examples are drawn in repeated
    passes over a pool shuffled by an RNG seeded from the split seed,
    and the augmented list is shuffled with the same RNG. Examples are
    indivisible, so the achieved fraction is approximate — after a
    greedy fill to the target token count, a bounded refinement pass
    swaps or drops chosen examples whenever doing so lands the
    achieved fraction closer to the target (this is what keeps the
    error small when the corpus is small and single examples are a
    large share of the total).

    Returns (augmented_train, stats). With a non-positive target, an
    empty train split, or an empty real pool, the train split is
    returned unchanged and the stats report an achieved fraction of 0.
    """
    stats: dict[str, Any] = {
        "target_fraction": target_fraction,
        "achieved_fraction": 0.0,
        "real_examples_added": 0,
        "real_tokens": 0,
        "corpus_train_tokens": 0,
        "total_train_tokens": 0,
    }
    if target_fraction <= 0 or not train or not real_examples:
        return train, stats
    corpus_tokens = sum(count_tokens(ex) for ex in train)
    stats["corpus_train_tokens"] = corpus_tokens
    stats["total_train_tokens"] = corpus_tokens
    if corpus_tokens <= 0:
        return train, stats

    needed = target_fraction / (1.0 - target_fraction) * corpus_tokens
    pool_counts = [count_tokens(ex) for ex in real_examples]
    rng = random.Random(f"ouro-real-mixture:{seed}")
    chosen_idx: list[int] = []
    real_tokens = 0
    while real_tokens < needed:
        order = list(range(len(real_examples)))
        rng.shuffle(order)
        pass_tokens = 0
        for j in order:
            if real_tokens >= needed:
                break
            chosen_idx.append(j)
            real_tokens += pool_counts[j]
            pass_tokens += pool_counts[j]
        if pass_tokens <= 0:
            break  # pool contributes no tokens; cannot approach target

    def _err(rt: int) -> float:
        return abs(rt / (corpus_tokens + rt) - target_fraction)

    for _ in range(8):  # bounded refinement passes
        cur_err = _err(real_tokens)
        improved = False
        for i in range(len(chosen_idx)):
            for j in range(len(real_examples)):
                if j == chosen_idx[i]:
                    continue
                new_rt = real_tokens - pool_counts[chosen_idx[i]] \
                    + pool_counts[j]
                if new_rt > 0 and _err(new_rt) < cur_err - 1e-12:
                    chosen_idx[i] = j
                    real_tokens = new_rt
                    improved = True
                    break
            if improved:
                break
        if improved:
            continue
        for i in range(len(chosen_idx)):
            new_rt = real_tokens - pool_counts[chosen_idx[i]]
            if new_rt > 0 and _err(new_rt) < cur_err - 1e-12:
                chosen_idx.pop(i)
                real_tokens = new_rt
                improved = True
                break
        if not improved:
            break

    chosen = [real_examples[j] for j in chosen_idx]
    combined = list(train) + chosen
    rng.shuffle(combined)
    stats.update({
        "achieved_fraction": real_tokens / (corpus_tokens + real_tokens),
        "real_examples_added": len(chosen),
        "real_tokens": real_tokens,
        "total_train_tokens": corpus_tokens + real_tokens,
    })
    return combined, stats


def real_fraction_from_env() -> float:
    """$OURO_REAL_FRACTION as a float in [0, 1); unset means 0 (off)."""
    raw = os.environ.get("OURO_REAL_FRACTION", "")
    if not raw:
        return 0.0
    try:
        value = float(raw)
    except ValueError:
        raise SystemExit(
            f"OURO_REAL_FRACTION must be a float in [0, 1), got {raw!r}")
    if not 0.0 <= value < 1.0:
        raise SystemExit(
            f"OURO_REAL_FRACTION must be a float in [0, 1), got {raw!r}")
    return value


def split_examples(
    examples: list[dict],
    eval_split: float = EVAL_SPLIT,
    seed: int = DEFAULT_SPLIT_SEED,
    frozen_eval_ids: set[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Split examples into (train, eval) by GROUP: every group id goes
    entirely to one side. Deterministic for a given seed.

    With frozen_eval_ids given, those groups — and only those groups —
    form eval, whatever new groups have appeared since cycle 0. That is
    the frozen test set the paper's Table III specifies: the same items
    are evaluated at every cycle, so cross-cycle perplexity is comparable.

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

    if frozen_eval_ids is not None:
        eval_ids = {g for g in frozen_eval_ids if g in by_group}
        if not eval_ids:
            raise ValueError(
                "Frozen eval groups not found in corpus: "
                f"{sorted(frozen_eval_ids)}"
            )
        train = [ex for ex in examples if ex["group"] not in eval_ids]
        eval_ = [ex for ex in examples if ex["group"] in eval_ids]
        rng.shuffle(train)
        rng.shuffle(eval_)
        return train, eval_

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
    target_fraction = real_fraction_from_env()
    examples = turns_to_training_examples(turns)

    # Frozen test set: cycle 0's eval groups are written to
    # data/corpus/frozen_eval_groups.json on first use and reused verbatim
    # on every later cycle. The file lives under data/corpus/ so
    # r1_driver init() (which clears data/corpus/) resets it for a fresh
    # trajectory. Only write it when eval is non-empty: with a single
    # group there is no honest split, and an empty frozen set must not
    # lock later cycles into an empty eval.
    frozen_path = CORPUS_DIR / "frozen_eval_groups.json"
    if frozen_path.exists():
        frozen_ids = set(json.loads(frozen_path.read_text()))
        if frozen_ids:
            train_examples, eval_examples = split_examples(
                examples, seed=seed, frozen_eval_ids=frozen_ids)
            logger.info("Frozen eval groups reused: %s (%d examples)",
                        sorted(frozen_ids), len(eval_examples))
        else:
            train_examples, eval_examples = split_examples(examples, seed=seed)
    else:
        train_examples, eval_examples = split_examples(examples, seed=seed)
        if eval_examples:
            frozen_ids = sorted({ex["group"] for ex in eval_examples})
            frozen_path.write_text(json.dumps(frozen_ids, indent=2))
            logger.info("Frozen eval groups written: %s (%d examples)",
                        frozen_ids, len(eval_examples))

    assert_no_leakage(train_examples, eval_examples)

    # R2 anchoring: augment the TRAIN split only, after the split and
    # the leakage assertion, so the eval set and the frozen eval groups
    # are exactly what the unanchored path computed. The achieved
    # mixture is written to a sidecar the R1 driver records per cycle;
    # with the anchor off, any stale sidecar from an earlier anchored
    # run in this corpus dir is removed so it can never be misread.
    if target_fraction > 0:
        tokenizer = load_base_tokenizer()
        real_examples = seed_documents_to_examples()
        train_examples, mix = augment_train_with_real(
            train_examples, real_examples, target_fraction,
            lambda ex: example_token_count(ex, tokenizer), seed)
        mix["split_seed"] = seed
        mix["real_groups"] = sorted({ex["group"] for ex in real_examples})
        MIXTURE_FILE.write_text(json.dumps(mix, indent=2))
        logger.info(
            "Real-data anchor: target %.3f, achieved %.3f "
            "(%d real examples, %d real tokens of %d train tokens)",
            mix["target_fraction"], mix["achieved_fraction"],
            mix["real_examples_added"], mix["real_tokens"],
            mix["total_train_tokens"])
    elif MIXTURE_FILE.exists():
        MIXTURE_FILE.unlink()

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
