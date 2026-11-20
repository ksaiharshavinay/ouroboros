# OUROBOROS — ouro/direct.py
# Copyright (c) 2026 Kudikala Sai Harsha Vinay. All rights reserved.
# PATENT PENDING
#
# Run Spec R3 §4, arm B3: direct self-training, no simulation.
#
# B3 removes the swarm from the loop entirely. Each cycle the CURRENT
# checkpoint (the base model at cycle 0, the latest LoRA adapter
# after) generates the next corpus directly: TURNS_PER_CYCLE turns,
# TURNS_PER_SEED per human seed document, each conditioned on that
# seed document's opening. The turns are written as a
# corpus_cycle_NNNN.jsonl in the IDENTICAL format to a simulation
# corpus (same five fields per line), so prepare/training/metrics
# code cannot tell the difference — the only thing that changed is
# where the text came from.
#
# Generation uses the pipeline's inference settings unchanged
# (ouro/infer.py: sampling, temperature 0.7, 512 max new tokens) via
# DirectCheckpointGenerator; there is no Ollama and no swarm anywhere
# in this path.
#
# Turn layout: the 12 turns of one seed document are spread over
# ROUNDS_PER_SEED distinct `round` values. prepare.py groups training
# examples by (file, round) for the group-disjoint train/eval split,
# so rounds must vary within a file — a single round for the whole
# file would collapse the split to one group.

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

TURNS_PER_CYCLE = 60
TURNS_PER_SEED = 12
ROUNDS_PER_SEED = 10
SEED_OPENING_CHARS = 1000
# A recorded turn is capped at this many characters (word boundary).
# Generation itself samples up to the inference path's 512 tokens,
# but an uncapped turn would break the training pairs prepare.py
# builds: the previous turn becomes the prompt and train.py truncates
# each pair at 512 tokens prompt-first, so a near-512-token prompt
# leaves a completion with every label masked — nothing to learn
# from. Simulation turns are short posts; the cap keeps B3 pairs in
# the same length regime.
MAX_TURN_CHARS = 1000


def seed_opening(seed_text: str, max_chars: int = SEED_OPENING_CHARS) -> str:
    """The conditioning prompt for one seed document: its opening.

    The first two paragraph blocks of the document (a document title
    block followed by its first prose block), hard-capped at
    max_chars on a sentence boundary when the blocks run long.
    """
    blocks = [b.strip() for b in seed_text.split("\n\n") if b.strip()]
    opening = "\n\n".join(blocks[:2]) if blocks else seed_text.strip()
    if len(opening) <= max_chars:
        return opening
    cut = opening[:max_chars]
    for sep in (". ", "! ", "? "):
        idx = cut.rfind(sep)
        if idx > max_chars // 2:
            return cut[: idx + 1].strip()
    return cut.strip()


def build_direct_turns(
    seed_texts: list[str],
    generate_fn: Callable[[str], str],
    cycle: int,
    per_seed: int = TURNS_PER_SEED,
    rounds: int = ROUNDS_PER_SEED,
) -> list[dict]:
    """Generate one B3 cycle's corpus turns.

    generate_fn(prompt) -> text is called once per turn with the seed
    document's opening as the prompt. Exactly per_seed turns per seed
    document, in simulation-corpus field format. An empty or errored
    generation raises: a silently short or placeholder corpus would
    corrupt the trajectory, and the driver fails the cycle loudly
    instead (same discipline as the swarm path).
    """
    turns: list[dict] = []
    for agent_id, seed_text in enumerate(seed_texts):
        prompt = seed_opening(seed_text)
        for k in range(per_seed):
            content = (generate_fn(prompt) or "").strip()
            if not content or content.startswith("[Inference error"):
                raise RuntimeError(
                    f"B3 direct generation failed at cycle {cycle}, "
                    f"seed {agent_id}, turn {k}: {content[:80]!r}")
            if len(content) > MAX_TURN_CHARS:
                cut = content[:MAX_TURN_CHARS]
                idx = cut.rfind(" ")
                content = (cut[:idx] if idx > 0 else cut).strip()
            turns.append({
                "round": (k % rounds) + 1,
                "agent_id": agent_id,
                "content": content,
                "influence": 0.5,
                "action": "create_post",
            })
    # Round-major order, agents interleaved — the layout a simulation
    # corpus has (turns exported in interaction order), so prepare.py's
    # consecutive-turn pairs mix seed documents the same way.
    turns.sort(key=lambda t: (t["round"], t["agent_id"]))
    return turns


def write_direct_corpus(turns: list[dict], corpus_dir: Path,
                        cycle: int) -> Path:
    """Write turns as corpus_cycle_NNNN.jsonl; return the path."""
    corpus_dir = Path(corpus_dir)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    path = corpus_dir / f"corpus_cycle_{cycle:04d}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for turn in turns:
            f.write(json.dumps(turn, ensure_ascii=False) + "\n")
    return path


class DirectCheckpointGenerator:
    """generate_fn backed by the current checkpoint, locally.

    Uses OuroInferenceEngine (the pipeline's inference path and its
    sampling settings) when an adapter checkpoint exists. At cycle 0
    there is no adapter yet and OuroInferenceEngine would fall back
    to the LLM API — B3 must generate from the BASE MODEL itself, so
    the base model is loaded locally through HuggingFace with the
    same sampling settings instead. No Ollama, no API.
    """

    def __init__(self, config, pipeline=None):
        self.config = config
        self._pipeline = pipeline
        self._engine = None
        self._base_model = None
        self._base_tokenizer = None

    def _ensure_engine(self):
        if self._engine is None:
            if self._pipeline is not None:
                # The pipeline's own engine: its checkpoint discovery
                # (config.model_dir/checkpoint) is the run's.
                self._engine = self._pipeline.infer
            else:
                from ouro.infer import OuroInferenceEngine
                self._engine = OuroInferenceEngine(self.config)
        return self._engine

    def _ensure_base(self):
        if self._base_model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            name = getattr(self.config, "train_base_model", None) \
                or "Qwen/Qwen2.5-0.5B-Instruct"
            self._base_tokenizer = AutoTokenizer.from_pretrained(name)
            if self._base_tokenizer.pad_token is None:
                self._base_tokenizer.pad_token = \
                    self._base_tokenizer.eos_token
            self._base_model = AutoModelForCausalLM.from_pretrained(
                name, dtype=torch.float32)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._base_model = self._base_model.to(device)
            self._base_model.eval()
        return self._base_model, self._base_tokenizer

    def __call__(self, prompt: str) -> str:
        if self._ensure_engine().has_adapter_checkpoint():
            return self._ensure_engine().predict(prompt)
        import torch
        model, tokenizer = self._ensure_base()
        device = next(model.parameters()).device
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True)
        return text.strip()
