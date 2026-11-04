#!/usr/bin/env python3
# OUROBOROS — autoresearch/train.py
# Copyright (c) 2026 Kudikala Sai Harsha Vinay. All rights reserved.
#
# Real LoRA SFT training loop (Phase 1 of the paper plan, §4.2 S1.2).
#
# This replaces the original stub, in which fine_tune() counted steps without
# ever running a forward/backward pass, evaluate() scored empty predictions
# with a character-overlap metric, and save_checkpoint() wrote only a JSON
# metadata file. Everything here is real:
#   * fine_tune() runs HuggingFace Trainer with a PEFT LoRA adapter —
#     real forward/backward/optimizer steps, per-step loss logged.
#   * evaluate() computes mean token-level NLL (and perplexity) of the
#     completions in eval.jsonl under the specified model/adapter.
#   * save_checkpoint() writes loadable adapter weights
#     (adapter_model.safetensors + adapter_config.json + tokenizer files)
#     plus a metadata JSON recording the adapter file's SHA-256, so the
#     lineage of every inference run is provable, not asserted.
#
# Configuration (environment variables):
#   OURO_TRAIN_BASE_MODEL  base model id/path (default below)
#   OURO_TRAIN_MAX_STEPS   cap optimizer steps (default: NUM_EPOCHS epochs)
#   OURO_TRAIN_OUTPUT_DIR  checkpoint output dir (default: data/model)

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import shutil
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# The hf-xet download path fails through some corporate/sandbox proxies
# (RemoteDisconnected on xet-read-token). Disable it unless the operator
# explicitly configured it; plain HTTPS downloads work everywhere.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

TRAIN_FILE = Path("data/corpus/train.jsonl")
EVAL_FILE = Path("data/corpus/eval.jsonl")
MODEL_DIR = Path(os.environ.get("OURO_TRAIN_OUTPUT_DIR", "data/model"))
LOG_FILE = Path("data/research_log.jsonl")
BUDGET_MINUTES = 5

# --- Hyperparameters ---
LEARNING_RATE = 2e-4
BATCH_SIZE = 8
MAX_SEQ_LEN = 512
NUM_EPOCHS = 3

DEFAULT_BASE_MODEL = os.environ.get(
    "OURO_TRAIN_BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"
)

# LoRA configuration. "all-linear" keeps this architecture-agnostic
# (Qwen2, Llama, GPT-2, ...) which matters for tiny smoke-test models.
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.0
LORA_TARGET_MODULES = "all-linear"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _import_stack():
    """Import the ML stack lazily, with an actionable error if absent."""
    try:
        import torch
        import transformers
        import peft
    except ImportError as exc:
        raise RuntimeError(
            "Real training requires torch, transformers and peft. "
            "Use the Python 3.11 environment (see "
            "~/workspace/eb1a/ouro-venv311) — the system Python 3.12 env "
            "does not carry the ML stack."
        ) from exc
    return torch, transformers, peft


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_dataset(torch, tokenizer, data: list[dict], max_len: int):
    """Tokenise (prompt, completion) pairs; loss is on completion tokens only."""
    from torch.utils.data import Dataset

    class SFTDataset(Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, idx):
            ex = self.rows[idx]
            prompt_ids = tokenizer(ex["prompt"], add_special_tokens=True)["input_ids"]
            completion_ids = tokenizer(ex["completion"], add_special_tokens=False)["input_ids"]
            input_ids = (prompt_ids + completion_ids)[:max_len]
            labels = ([-100] * len(prompt_ids) + completion_ids)[:max_len]
            return {"input_ids": input_ids, "labels": labels}

    return SFTDataset(data)


def _collate(batch, pad_token_id: int):
    import torch

    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        n = maxlen - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_token_id] * n)
        labels.append(b["labels"] + [-100] * n)
        attn.append([1] * len(b["input_ids"]) + [0] * n)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
    }


def fine_tune(
    train_data: list[dict],
    budget_seconds: float,
    base_model: str = DEFAULT_BASE_MODEL,
    max_steps: int | None = None,
    output_dir: Path | None = None,
    learning_rate: float = LEARNING_RATE,
    num_epochs: float = NUM_EPOCHS,
    seed: int = 42,
) -> dict:
    """
    Really fine-tune `base_model` with LoRA on train_data.

    Returns a result dict including the per-step loss history and the path
    to the saved adapter (adapter staging dir under output_dir).
    """
    torch, transformers, peft = _import_stack()
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model

    output_dir = Path(output_dir) if output_dir else MODEL_DIR
    staging = output_dir / "adapter_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.float32)
    model.config.use_cache = False
    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "[train] Base %s | trainable LoRA params: %d | examples: %d",
        base_model, trainable, len(train_data),
    )

    dataset = _build_dataset(torch, tokenizer, train_data, MAX_SEQ_LEN)
    loss_history: list[float] = []
    start = time.time()

    from transformers import TrainerCallback

    class _Probe(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            if logs and "loss" in logs:
                loss_history.append(float(logs["loss"]))

        def on_step_end(self, args, state, control, **kw):
            if time.time() - start > budget_seconds:
                logger.info("[train] Budget exhausted at step %d", state.global_step)
                control.should_training_stop = True
            return control

    args = TrainingArguments(
        output_dir=str(staging / "trainer"),
        per_device_train_batch_size=min(BATCH_SIZE, max(1, len(train_data))),
        gradient_accumulation_steps=1,
        num_train_epochs=num_epochs,
        max_steps=max_steps if max_steps else -1,
        learning_rate=learning_rate,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        seed=seed,
        data_seed=seed,
        optim="adamw_torch",
        lr_scheduler_type="constant",
        warmup_steps=0,
        disable_tqdm=True,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=lambda batch: _collate(batch, tokenizer.pad_token_id),
        callbacks=[_Probe()],
    )
    train_output = trainer.train()

    # Save the adapter (weights + config) and tokenizer — loadable artifacts.
    model.save_pretrained(staging)
    tokenizer.save_pretrained(staging)
    adapter_file = staging / "adapter_model.safetensors"
    if not adapter_file.exists():
        raise RuntimeError(f"Adapter weights were not written to {adapter_file}")
    adapter_hash = sha256_file(adapter_file)

    steps = int(train_output.global_step)
    result = {
        "steps": steps,
        "loss_history": loss_history,
        "initial_loss": loss_history[0] if loss_history else None,
        "final_loss": loss_history[-1] if loss_history else None,
        "train_runtime_seconds": round(time.time() - start, 2),
        "base_model": base_model,
        "adapter_dir": str(staging),
        "adapter_file": str(adapter_file),
        "adapter_sha256": adapter_hash,
        "trainable_params": trainable,
        "num_train_examples": len(train_data),
        "trained": True,
    }
    logger.info(
        "[train] Done: %d steps, loss %s -> %s, adapter sha256 %s",
        steps,
        result["initial_loss"],
        result["final_loss"],
        adapter_hash[:16],
    )
    return result


def _eval_nll(model, tokenizer, eval_data: list[dict], max_len: int, limit: int = 50) -> float:
    """Mean token-level NLL over completion tokens (lower = better)."""
    import torch as _torch

    model.eval()
    _torch.set_grad_enabled(False)
    total_nll, total_tokens = 0.0, 0
    for ex in eval_data[:limit]:
        prompt_ids = tokenizer(ex["prompt"], add_special_tokens=True)["input_ids"]
        completion_ids = tokenizer(ex["completion"], add_special_tokens=False)["input_ids"]
        input_ids = (prompt_ids + completion_ids)[:max_len]
        labels = ([-100] * len(prompt_ids) + completion_ids)[:max_len]
        if all(t == -100 for t in labels):
            continue
        ids = _torch.tensor([input_ids], dtype=_torch.long)
        lab = _torch.tensor([labels], dtype=_torch.long)
        out = model(input_ids=ids, labels=lab)
        n_tokens = int((lab != -100).sum().item())
        total_nll += float(out.loss.item()) * n_tokens
        total_tokens += n_tokens
    if total_tokens == 0:
        return float("nan")
    return total_nll / total_tokens


def evaluate(
    eval_data: list[dict],
    base_model: str = DEFAULT_BASE_MODEL,
    adapter_dir: str | Path | None = None,
) -> float:
    """
    Evaluate on eval_data. Returns mean completion-token NLL (lower = better).

    Loads `base_model` and, when adapter_dir is given, the LoRA adapter from
    that directory — i.e. this measures the actually-trained model, never an
    empty-string prediction.
    """
    if not eval_data:
        return float("nan")
    torch, transformers, peft = _import_stack()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir or base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.float32)
    if adapter_dir:
        model = PeftModel.from_pretrained(model, str(adapter_dir))
    return _eval_nll(model, tokenizer, eval_data, MAX_SEQ_LEN)


def evaluate_detailed(
    eval_data: list[dict],
    base_model: str = DEFAULT_BASE_MODEL,
    adapter_dir: str | Path | None = None,
) -> dict:
    loss = evaluate(eval_data, base_model=base_model, adapter_dir=adapter_dir)
    return {
        "loss": loss,
        "perplexity": math.exp(loss) if loss == loss and loss < 50 else float("inf"),
        "num_eval_examples": min(len(eval_data), 50),
        "base_model": base_model,
        "adapter_dir": str(adapter_dir) if adapter_dir else None,
    }


def log_experiment(result: dict) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a") as f:
        f.write(json.dumps({**result, "timestamp": time.time()}) + "\n")


def save_checkpoint(result: dict) -> Path:
    """
    Persist a trained adapter as THE checkpoint.

    Writes MODEL_DIR/checkpoint/ containing the adapter weights, adapter
    config and tokenizer files, plus checkpoint_metadata.json (the training
    result + adapter SHA-256). Replaces the pre-Phase-1 format, which was a
    single JSON file named "checkpoint" containing no weights at all.
    """
    staging = Path(result.get("adapter_dir", ""))
    if not staging.exists() or not (staging / "adapter_model.safetensors").exists():
        raise RuntimeError(
            "save_checkpoint requires a fine_tune() result with a real adapter "
            f"directory; got adapter_dir={staging!r}"
        )
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = MODEL_DIR / "checkpoint"
    legacy_file = MODEL_DIR / "checkpoint"
    if legacy_file.is_file():
        legacy_file.unlink()
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    shutil.copytree(staging, checkpoint_dir)

    metadata = {
        **{k: v for k, v in result.items() if k != "adapter_dir"},
        # Hoist base_model to the top level: main() nests it inside
        # train_result, but inference reads it from the metadata root.
        "base_model": result.get("base_model")
        or (result.get("train_result") or {}).get("base_model"),
        "adapter_dir": str(checkpoint_dir),
        "adapter_file": str(checkpoint_dir / "adapter_model.safetensors"),
        "adapter_sha256": sha256_file(checkpoint_dir / "adapter_model.safetensors"),
        "saved_at": time.time(),
    }
    (checkpoint_dir / "checkpoint_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )
    logger.info(
        "Checkpoint saved to %s (adapter sha256 %s)",
        checkpoint_dir, metadata["adapter_sha256"][:16],
    )
    return checkpoint_dir


def main() -> None:
    logger.info("Loading training data...")
    train_data = load_jsonl(TRAIN_FILE)
    eval_data = load_jsonl(EVAL_FILE)

    if not train_data:
        logger.error("No training data found. Run autoresearch/prepare.py first.")
        return

    base_model = DEFAULT_BASE_MODEL
    max_steps_env = os.environ.get("OURO_TRAIN_MAX_STEPS")
    max_steps = int(max_steps_env) if max_steps_env else None

    logger.info(
        "Training %s on %d examples (budget: %d min)...",
        base_model, len(train_data), BUDGET_MINUTES,
    )
    train_result = fine_tune(
        train_data,
        budget_seconds=BUDGET_MINUTES * 60,
        base_model=base_model,
        max_steps=max_steps,
    )

    eval_before = evaluate_detailed(eval_data, base_model=base_model)
    eval_after = evaluate_detailed(
        eval_data, base_model=base_model, adapter_dir=train_result["adapter_dir"]
    )
    logger.info("Eval loss (base):    %.4f (ppl %.2f)", eval_before["loss"], eval_before["perplexity"])
    logger.info("Eval loss (trained): %.4f (ppl %.2f)", eval_after["loss"], eval_after["perplexity"])

    result = {
        "train_result": train_result,
        "eval_loss": eval_after["loss"],
        "eval_perplexity": eval_after["perplexity"],
        "eval_loss_base": eval_before["loss"],
        "lr": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "max_seq_len": MAX_SEQ_LEN,
        "adapter_dir": train_result["adapter_dir"],
    }
    log_experiment(result)
    checkpoint_dir = save_checkpoint(result)
    result["checkpoint_dir"] = str(checkpoint_dir)
    logger.info("Training complete. Checkpoint: %s", checkpoint_dir)


if __name__ == "__main__":
    main()
