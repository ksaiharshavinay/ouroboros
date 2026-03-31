# OUROBOROS — ouro/feedback.py
# Copyright (c) 2026 Sai Harsha Vinaykudikala. All rights reserved.
# PATENT PENDING
#
# The recursive feedback loop: model prediction outputs are fed back as new
# simulation seeds, driving the next swarm generation cycle. This closed loop
# is the core novel mechanism of OUROBOROS.

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .config import OuroConfig

logger = logging.getLogger(__name__)


class OuroFeedbackModule:
    """
    Manages the recursive feedback loop between inference outputs and the
    swarm simulation. Persists cycle metadata to disk so the loop survives
    restarts.
    """

    def __init__(self, config: OuroConfig):
        self.config = config
        self._state_path = config.feedback_state_path
        self._state = self._load_state()

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _load_state(self) -> dict[str, Any]:
        if self._state_path.exists():
            try:
                with self._state_path.open() as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not read state file (%s). Starting fresh.", exc)
        return {"cycle_count": 0, "cycles": []}

    def _save_state(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        with self._state_path.open("w") as f:
            json.dump(self._state, f, indent=2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def cycle_count(self) -> int:
        return self._state["cycle_count"]

    def run_feedback(
        self,
        prediction_report: str,
        sim_result: dict[str, Any],
        seeds_dir: Path | None = None,
    ) -> Path:
        """
        Write model prediction outputs back as new seed documents for the
        next swarm simulation cycle.

        Args:
            prediction_report: Text output from OuroInferenceEngine.predict().
            sim_result: Metadata dict from SwarmGenerator.run().
            seeds_dir: Directory to write new seeds into. Defaults to config.seed_docs_dir.

        Returns:
            Path of the newly written seed file.
        """
        seeds_dir = seeds_dir or self.config.seed_docs_dir
        seeds_dir.mkdir(parents=True, exist_ok=True)

        cycle = self.cycle_count
        seed_path = seeds_dir / f"feedback_cycle_{cycle:04d}.txt"
        seed_path.write_text(prediction_report, encoding="utf-8")

        cycle_meta = {
            "cycle": cycle,
            "seed_file": str(seed_path),
            "num_turns": sim_result.get("num_turns", 0),
            "num_agents": sim_result.get("num_agents", 0),
        }
        self._state["cycles"].append(cycle_meta)
        self._state["cycle_count"] = cycle + 1
        self._save_state()

        logger.info("[ouro] Feedback cycle %d complete — seed written to %s", cycle, seed_path)
        return seed_path

    def get_history(self) -> list[dict[str, Any]]:
        """Return metadata for all completed feedback cycles."""
        return list(self._state.get("cycles", []))
