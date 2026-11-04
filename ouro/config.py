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
    llm_model: str = "llama3.2:1b"       # local Ollama model; set to "claude-haiku-4-5" when using claude
    llm_base_url: str = "http://localhost:11434/v1"   # Ollama endpoint; change for Claude
    llm_api_key_env: str = "ANTHROPIC_API_KEY"   # only used when llm_provider="claude"
    # Per-call timeout (seconds) for the swarm's camel model backend.
    # camel-ai defaults to 180s, which a CPU Ollama backend cannot meet
    # when a whole OASIS step of agents queues behind one model — calls
    # time out and retry in lockstep. 1800s is sized for local CPU runs.
    llm_timeout: float = 1800.0

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

    # --- KuzuDB ---
    kuzu_db_path: Path = Path("data/kuzu_graph")

    def __post_init__(self):
        for attr in ("seed_docs_dir", "corpus_dir", "model_dir", "kuzu_db_path"):
            p = getattr(self, attr)
            if isinstance(p, str):
                setattr(self, attr, Path(p))
        # Ensure output dirs exist
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.feedback_state_path.parent.mkdir(parents=True, exist_ok=True)
