# OUROBOROS — ouro/config.py
# Copyright (c) 2026 Sai Harsha Vinaykudikala. All rights reserved.
# PATENT PENDING

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class OuroConfig:
    """Central configuration for the OUROBOROS pipeline."""

    # --- LLM / API ---
    llm_model: str = "claude-haiku-4-5"
    llm_base_url: str = "https://api.anthropic.com/v1"
    llm_api_key_env: str = "ANTHROPIC_API_KEY"   # name of the env-var, not the key itself

    # --- Swarm simulation ---
    num_agents: int = 500
    simulation_rounds: int = 30
    seed_docs_dir: Path = Path("data/seeds")

    # --- AutoResearch ---
    autoresearch_program: Path = Path("autoresearch/program.md")
    experiment_budget_minutes: int = 5
    max_experiments: int = 100

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
