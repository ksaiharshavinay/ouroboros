# OUROBOROS — swarm/base.py
# Copyright (c) 2026 Kudikala Sai Harsha Vinay. All rights reserved.
# PATENT PENDING

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseSwarmAdapter(ABC):
    """
    Abstract interface for all swarm simulation backends.

    Implementing this interface decouples the pipeline from any specific
    simulation library, making it easy to swap backends for research or
    commercial deployment.
    """

    @abstractmethod
    def run(self, seed_docs: list[Path], cycle: int = 0) -> dict[str, Any]:
        """
        Run a swarm simulation from seed_docs.

        Args:
            seed_docs: Paths to seed documents that initialise agent personas.
            cycle: Current feedback cycle number (used to namespace graph IDs).

        Returns:
            A dict with at least:
                num_agents (int)
                num_turns  (int)
                corpus_path (str) — path to the exported JSONL corpus
        """
        ...

    @abstractmethod
    def get_top_entities(self, n: int = 10) -> list[dict[str, Any]]:
        """Return the top-n most influential entities from the last simulation."""
        ...
