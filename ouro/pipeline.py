# OUROBOROS — ouro/pipeline.py
# Copyright (c) 2026 Sai Harsha Vinaykudikala. All rights reserved.
# PATENT PENDING

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

from .config import OuroConfig
from .feedback import OuroFeedbackModule
from .infer import OuroInferenceEngine

logger = logging.getLogger(__name__)


class OuroPipeline:
    """
    Orchestrates the full OUROBOROS loop:
      1. run_swarm()        — spawn agents, simulate social interaction, export corpus
      2. run_autoresearch() — autonomously train/improve model on corpus
      3. run_feedback()     — feed model predictions back as new simulation seeds
    """

    def __init__(self, config: OuroConfig | None = None):
        self.config = config or OuroConfig()
        self.feedback = OuroFeedbackModule(self.config)
        self.infer = OuroInferenceEngine(self.config)

    # ------------------------------------------------------------------
    # Stage 1 — Swarm simulation
    # ------------------------------------------------------------------

    def run_swarm(self, seed_docs: list[Path] | None = None) -> dict[str, Any]:
        """
        Run the OASIS-powered swarm simulation on seed_docs and export the
        interaction corpus to config.corpus_dir.
        """
        from swarm.generator import OuroborosSwarmGenerator

        seeds = seed_docs or self._load_seeds()
        logger.info("[ouro] Running swarm with %d seed documents", len(seeds))

        generator = OuroborosSwarmGenerator(self.config)
        sim_result = generator.run(seeds, cycle=self.feedback.cycle_count)

        logger.info("[ouro] Swarm complete — %d turns exported", sim_result.get("num_turns", 0))
        return sim_result

    def _load_seeds(self) -> list[Path]:
        """Load seed documents from config.seed_docs_dir (sorted for determinism)."""
        seed_dir = self.config.seed_docs_dir
        seeds = sorted(seed_dir.glob("*.txt")) + sorted(seed_dir.glob("*.md"))
        if not seeds:
            logger.warning("[ouro] No seed documents found in %s", seed_dir)
        return seeds

    # ------------------------------------------------------------------
    # Stage 2 — AutoResearch training loop
    # ------------------------------------------------------------------

    def run_autoresearch(self) -> dict[str, Any]:
        """
        Launch the AutoResearch agent loop which modifies training code, runs
        5-minute experiments, and keeps improvements.
        """
        prepare_script = Path(__file__).parent.parent / "autoresearch" / "prepare.py"
        train_script = Path(__file__).parent.parent / "autoresearch" / "train.py"

        logger.info("[ouro] Preparing corpus for AutoResearch...")
        subprocess.run(["python3", str(prepare_script)], check=True)

        logger.info("[ouro] Starting AutoResearch training loop...")
        result = subprocess.run(
            ["python3", str(train_script)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.error("[ouro] AutoResearch training failed:\n%s", result.stderr)
            raise RuntimeError("AutoResearch training loop failed.")

        logger.info("[ouro] AutoResearch complete.")
        return {"stdout": result.stdout, "returncode": result.returncode}

    # ------------------------------------------------------------------
    # Stage 3 — Full cycle with feedback
    # ------------------------------------------------------------------

    def run_full_cycle(self, prompt: str = "Summarize the key predictions from this cycle.") -> tuple[dict, dict, Path]:
        """
        Run one complete OUROBOROS cycle:
          swarm → autoresearch → infer → feedback

        Returns:
            (sim_result, research_result, feedback_seed_path)
        """
        sim_result = self.run_swarm()
        research_result = self.run_autoresearch()

        if not self.infer.is_ready():
            logger.warning("[ouro] No trained model yet; skipping inference & feedback.")
            return sim_result, research_result, Path()

        prediction = self.infer.predict(prompt)
        seed_path = self.feedback.run_feedback(prediction, sim_result)

        # Persist full cycle metadata to disk
        state_path = self.config.feedback_state_path.parent / "last_cycle.json"
        state_path.write_text(
            json.dumps({"sim_result": sim_result, "research_result": research_result}, indent=2)
        )

        logger.info("[ouro] Full cycle %d complete.", self.feedback.cycle_count - 1)
        return sim_result, research_result, seed_path

    def run_feedback_from_disk(self, prompt: str = "Summarize the predictions.") -> Path:
        """
        Run only the inference + feedback stage, loading sim_result from the
        last persisted cycle state on disk.
        """
        state_path = self.config.feedback_state_path.parent / "last_cycle.json"
        if not state_path.exists():
            raise FileNotFoundError(
                "No persisted cycle state found. Run run_full_cycle() first."
            )
        with state_path.open() as f:
            state = json.load(f)

        sim_result = state.get("sim_result", {})
        prediction = self.infer.predict(prompt)
        return self.feedback.run_feedback(prediction, sim_result)
