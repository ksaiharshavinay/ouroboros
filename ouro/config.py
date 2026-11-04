# OUROBOROS — ouro/config.py
# Copyright (c) 2026 Kudikala Sai Harsha Vinay. All rights reserved.
# PATENT PENDING

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class OuroConfig:
    """Central configuration for the OUROBOROS pipeline."""

    # --- LLM / API ---
    # Provider: "claude" (default) or "ollama" (local)
    llm_provider: str = "ollama"
    # Local Ollama model. qwen2.5:1.5b, not llama3.2:1b: inside OASIS
    # agent turns the 1B llama echoes the tool JSON *schema* back as
    # the tool-call arguments (every create_post failed on a stray
    # 'type' kwarg); qwen2.5:1.5b emits well-formed tool calls.
    # Set to "claude-haiku-4-5" when using claude.
    llm_model: str = "qwen2.5:1.5b"
    llm_base_url: str = "http://localhost:11434/v1"   # Ollama endpoint; change for Claude
    llm_api_key_env: str = "ANTHROPIC_API_KEY"   # only used when llm_provider="claude"
    # Per-call timeout (seconds) for the swarm's camel model backend.
    # camel-ai defaults to 180s, which a CPU Ollama backend cannot meet
    # when a whole OASIS step of agents queues behind one model — calls
    # time out and retry in lockstep. 1800s is sized for local CPU runs.
    llm_timeout: float = 1800.0
    # Max tokens per swarm-agent response. Uncapped, a 1B local model
    # rambles past 1,200 tokens in a single agent turn without emitting
    # a tool call or stop (observed live on CPU at ~2 tok/s = 10+ min
    # per runaway response). A tool call plus short post needs far less.
    llm_max_tokens: int = 512

    # --- Swarm simulation ---
    num_agents: int = 500
    simulation_rounds: int = 30
    seed_docs_dir: Path = Path("data/seeds")
    # OASIS platform: "twitter" or "reddit"
    oasis_platform: str = "twitter"
    # Stub simulation is EXPLICIT opt-in. When False (default), a missing or
    # failing OASIS install raises instead of silently substituting templated
    # stub turns — paper runs must never degrade to the stub unnoticed.
    allow_stub_simulation: bool = False

    # --- AutoResearch ---
    autoresearch_program: Path = Path("autoresearch/program.md")
    experiment_budget_minutes: int = 5
    max_experiments: int = 100
    # Base model for the real LoRA SFT loop (autoresearch/train.py).
    # Small enough for CPU smoke runs; override for real experiments.
    train_base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    # Interpreter used by OuroPipeline to launch the autoresearch scripts
    # (prepare.py / train.py). Those need the pinned Python 3.11 ML env,
    # not whatever "python3" is on PATH. None -> resolve at runtime:
    # $OURO_TRAIN_PYTHON, then a sibling ``ouro-venv311`` venv next to the
    # repo checkout, then sys.executable.
    train_python: str | None = None

    # --- Corpus / training ---
    corpus_dir: Path = Path("data/corpus")
    model_dir: Path = Path("data/model")

    # --- Feedback loop ---
    feedback_state_path: Path = Path("data/ouro_state.json")
    max_feedback_cycles: int = 10

    # --- Feedback gate (Phase 2, Finding 4) ---
    # The multi-signal gate (ouro/gate.py) decides which inference
    # predictions re-enter as next-cycle seeds. It replaces the old
    # Algorithm 7 design, whose logprob score could not be computed by
    # the shipped adapters and which was never called by the pipeline.
    # Enabled by default; disable for the paper's ungated baselines
    # (B3/B4 conditions), which must use the historical uniform path.
    feedback_gate_enabled: bool = True
    # Candidates sampled per feedback stage. >= 2 makes the
    # self-consistency signal measurable; 1 leaves it UNAVAILABLE.
    feedback_num_candidates: int = 3
    # Weighted-mean acceptance threshold over the AVAILABLE signals
    # (weights renormalise over whichever signals are available).
    gate_accept_threshold: float = 0.50
    gate_weight_outcome: float = 0.40      # delayed-outcome verification
    gate_weight_diversity: float = 0.25    # novelty vs the existing seed set
    gate_weight_consistency: float = 0.20  # self-consistency across samples
    gate_weight_provenance: float = 0.15   # provenance completeness
    # Hard vetoes (applied before scoring): a candidate whose distinct
    # token ratio falls below this is degenerate/repetitive; one shorter
    # than gate_min_chars carries no seedable content.
    gate_min_distinct_ratio: float = 0.35
    gate_min_chars: int = 10
    # Per-candidate decision audit log (JSONL). None -> written next to
    # feedback_state_path as gate_decisions.jsonl.
    gate_log_path: Path | None = None

    # --- KuzuDB ---
    kuzu_db_path: Path = Path("data/kuzu_graph")

    def __post_init__(self):
        for attr in ("seed_docs_dir", "corpus_dir", "model_dir", "kuzu_db_path"):
            p = getattr(self, attr)
            if isinstance(p, str):
                setattr(self, attr, Path(p))
        if isinstance(self.gate_log_path, str):
            self.gate_log_path = Path(self.gate_log_path)
        # Ensure output dirs exist
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.feedback_state_path.parent.mkdir(parents=True, exist_ok=True)
