# OUROBOROS — ouro/common_eval.py
# Copyright (c) 2026 Kudikala Sai Harsha Vinay. All rights reserved.
# PATENT PENDING
#
# Run Spec R3 §3: ONE common frozen eval set for every arm and seed.
#
# R1/R2 froze each trajectory's eval groups from its own cycle-0
# corpus, so every run had a different test set and a different base
# perplexity (12–61) and only the percent drop from base was honestly
# comparable across runs. The common eval is a fixed file of
# (prompt, completion) pairs built from public-domain texts that
# appear nowhere in the loop (see data/eval/common_eval_sources.json);
# it never changes within or across runs, so absolute trajectories
# are comparable, not just drops.
#
# The perplexity computation MIRRORS autoresearch/train.py's
# evaluate(): mean token-level NLL over completion tokens only
# (prompt tokens masked), prompt tokenised with special tokens,
# completion without, the pair truncated to train.py's MAX_SEQ_LEN,
# at most EVAL_LIMIT rows, perplexity = exp(NLL). It is implemented
# here by loading base (+ optional LoRA adapter) exactly as
# train.evaluate does, so the numbers are comparable in method to the
# per-run corpus eval the driver already records.

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_COMMON_EVAL_PATH = Path("data/eval/common_eval.jsonl")
# Mirrors of autoresearch/train.py constants (kept in step by test).
MAX_SEQ_LEN = 512
EVAL_LIMIT = 50


def load_common_eval(path: Path | None = None) -> list[dict]:
    """Load the common eval rows: [{"prompt":..., "completion":...}].

    Extra per-row provenance keys (source, paragraph_index) are kept
    on the returned dicts; the scorer reads only prompt/completion.
    """
    p = Path(path) if path is not None else DEFAULT_COMMON_EVAL_PATH
    rows: list[dict] = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _eval_nll(model, tokenizer, eval_data: list[dict],
              max_len: int = MAX_SEQ_LEN, limit: int = EVAL_LIMIT) -> float:
    """Mean token-level NLL over completion tokens.

    A line-for-line mirror of autoresearch/train.py's _eval_nll, so a
    common-eval perplexity and a corpus-eval perplexity are the same
    statistic computed the same way.
    """
    import torch as _torch

    model.eval()
    _torch.set_grad_enabled(False)
    device = next(model.parameters()).device
    total_nll, total_tokens = 0.0, 0
    for ex in eval_data[:limit]:
        prompt_ids = tokenizer(ex["prompt"], add_special_tokens=True)["input_ids"]
        completion_ids = tokenizer(ex["completion"], add_special_tokens=False)["input_ids"]
        input_ids = (prompt_ids + completion_ids)[:max_len]
        labels = ([-100] * len(prompt_ids) + completion_ids)[:max_len]
        if all(t == -100 for t in labels):
            continue
        ids = _torch.tensor([input_ids], dtype=_torch.long, device=device)
        lab = _torch.tensor([labels], dtype=_torch.long, device=device)
        out = model(input_ids=ids, labels=lab)
        n_tokens = int((lab != -100).sum().item())
        total_nll += float(out.loss.item()) * n_tokens
        total_tokens += n_tokens
    if total_tokens == 0:
        return float("nan")
    return total_nll / total_tokens


def common_eval_perplexity(
    eval_data: list[dict],
    base_model: str,
    adapter_dir: str | Path | None = None,
) -> float:
    """Common-eval perplexity of base_model (+ optional LoRA adapter).

    Loading mirrors autoresearch/train.py's evaluate(): tokenizer
    from the adapter dir when given, fp32 weights, GPU when available.
    Returns exp(mean completion-token NLL); NaN on empty data.
    """
    if not eval_data:
        return float("nan")
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Common eval requires torch, transformers and peft. "
            "Use the Python 3.11 environment (see "
            "~/workspace/eb1a/ouro-venv311).") from exc
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir or base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.float32)
    if adapter_dir:
        model = PeftModel.from_pretrained(model, str(adapter_dir))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    try:
        loss = _eval_nll(model, tokenizer, eval_data, MAX_SEQ_LEN)
    finally:
        # The driver scores the base model immediately before the
        # training subprocess starts; without this, the eval model's
        # GPU allocation stays reserved in the driver process for the
        # whole training run.
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return math.exp(loss) if loss == loss and loss < 50 else float("inf")


def evaluate_common_eval(
    base_model: str,
    adapter_dir: str | Path | None = None,
    path: Path | None = None,
) -> dict[str, Any]:
    """Load the frozen file and score it; a small dict for the summary."""
    rows = load_common_eval(path)
    ppl = common_eval_perplexity(rows, base_model, adapter_dir)
    return {
        "common_eval_perplexity": round(ppl, 2) if ppl == ppl else None,
        "common_eval_examples": min(len(rows), EVAL_LIMIT),
        "common_eval_base_model": base_model,
        "common_eval_adapter": str(adapter_dir) if adapter_dir else None,
    }
