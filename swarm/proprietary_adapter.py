# OUROBOROS — swarm/proprietary_adapter.py
# Copyright (c) 2026 Sai Harsha Vinaykudikala. All rights reserved.
# PATENT PENDING

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import BaseSwarmAdapter
from .generator import OuroborosSwarmGenerator


class ProprietarySwarmAdapter(BaseSwarmAdapter):
    """
    The commercial swarm adapter — delegates to OuroborosSwarmGenerator
    which is built entirely on permissive-licensed libraries (OASIS/Apache 2.0,
    KuzuDB/MIT). No AGPL dependencies.

    Swap out OuroborosSwarmGenerator here when a fully proprietary swarm
    engine is ready.
    """

    def __init__(self, config: Any):
        self._generator = OuroborosSwarmGenerator(config)

    def run(self, seed_docs: list[Path], cycle: int = 0) -> dict[str, Any]:
        return self._generator.run(seed_docs, cycle=cycle)

    def get_top_entities(self, n: int = 10) -> list[dict[str, Any]]:
        return self._generator.get_top_entities(n=n)

    def get_report(self) -> str:
        return self._generator.get_report()
