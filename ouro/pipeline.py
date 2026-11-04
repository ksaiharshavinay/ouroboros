# OUROBOROS — ouro/pipeline.py
# Copyright (c) 2026 Kudikala Sai Harsha Vinay. All rights reserved.
# PATENT PENDING

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
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

    def _train_python(self) -> str:
        """Interpreter for the autoresearch subprocesses.

        train.py needs the pinned Python 3.11 environment (torch/peft);
        the system python3 (3.12) has no ML stack. Resolution order:
        config.train_python -> $OURO_TRAIN_PYTHON -> sibling
        ``ouro-venv311`` venv -> sys.executable.
        """
        configured = getattr(self.config, "train_python", None)
        if configured:
            return configured
        env = os.environ.get("OURO_TRAIN_PYTHON")
        if env:
            return env
        venv_python = (
            Path(__file__).parent.parent.parent / "ouro-venv311" / "bin" / "python"
        )
        if venv_python.exists():
            return str(venv_python)
        return sys.executable

    def run_autoresearch(self) -> dict[str, Any]:
        """
        Launch the AutoResearch agent loop which modifies training code, runs
        5-minute experiments, and keeps improvements.
        """
        prepare_script = Path(__file__).parent.parent / "autoresearch" / "prepare.py"
        train_script = Path(__file__).parent.parent / "autoresearch" / "train.py"
        python = self._train_python()

        logger.info("[ouro] Preparing corpus for AutoResearch (python: %s)...", python)
        subprocess.run([python, str(prepare_script)], check=True)

        logger.info("[ouro] Starting AutoResearch training loop...")
        result = subprocess.run(
            [python, str(train_script)],
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

        context = self._load_corpus_context(sim_result)
        seed_path = self._infer_and_feed_back(prompt, context, sim_result)

        # Persist full cycle metadata to disk
        state_path = self.config.feedback_state_path.parent / "last_cycle.json"
        state_path.write_text(
            json.dumps({"sim_result": sim_result, "research_result": research_result}, indent=2)
        )

        logger.info("[ouro] Full cycle %d complete.", self.feedback.cycle_count - 1)
        return sim_result, research_result, seed_path

    def _load_corpus_context(self, sim_result: dict[str, Any]) -> str | None:
        """
        Build a system context string for inference grounded in simulation output.
        Prefers the LLM-generated prediction report (richest signal); falls back
        to raw corpus turns if the report is unavailable.
        """
        # Prefer the full LLM-generated report — much richer than raw stub turns
        report = sim_result.get("report", "").strip()
        if report:
            logger.info("[ouro] Injecting prediction report as inference context.")
            return (
                "You are synthesising results from an OUROBOROS swarm simulation.\n"
                "Below is the prediction report generated from the latest simulation cycle.\n"
                "Use it to answer the user's question with specific observations, "
                "trends, and forward-looking predictions grounded in the simulation.\n\n"
                "--- SIMULATION REPORT ---\n"
                + report
                + "\n--- END REPORT ---"
            )

        # Fallback: load raw corpus turns
        corpus_path = sim_result.get("corpus_path")
        if not corpus_path or not Path(corpus_path).exists():
            return None
        try:
            lines = Path(corpus_path).read_text(encoding="utf-8").splitlines()
            turns: list[str] = []
            for line in lines[:60]:
                try:
                    obj = json.loads(line)
                    content = obj.get("content") or obj.get("message") or ""
                    if content:
                        turns.append(content)
                except (json.JSONDecodeError, AttributeError):
                    continue
            if not turns:
                return None
            logger.info("[ouro] Injecting %d corpus turns as inference context.", len(turns))
            return (
                "You are synthesising results from an OUROBOROS swarm simulation. "
                "Below are agent interaction turns from the simulation corpus. "
                "Use this data to answer the user's question with specific observations, "
                "trends, and predictions grounded in the simulation output.\n\n"
                "--- SIMULATION CORPUS (excerpt) ---\n"
                + "\n".join(f"[Turn {i+1}] {t}" for i, t in enumerate(turns))
                + "\n--- END CORPUS ---"
            )
        except Exception as exc:
            logger.warning("[ouro] Could not load corpus for context: %s", exc)
            return None

    def _infer_and_feed_back(
        self, prompt: str, context: str | None, sim_result: dict[str, Any]
    ) -> Path:
        """
        The pipeline's feedback stage (Phase 2): inference predictions
        become next-cycle seeds THROUGH the multi-signal gate
        (ouro/gate.py) whenever it is enabled — the default. Candidates
        are sampled with their inference provenance attached so the gate
        can score them. With the gate disabled (the paper's ungated
        baseline conditions), the historical uniform path runs instead:
        one prediction, injected unconditionally.

        Returns the first accepted seed path, or Path() when the gate
        accepted nothing (a legitimate, logged result).
        """
        if not getattr(self.config, "feedback_gate_enabled", True):
            prediction = self.infer.predict(prompt, context=context)
            return self.feedback.run_feedback(prediction, sim_result)

        n = max(1, int(getattr(self.config, "feedback_num_candidates", 3)))
        candidates: list[dict[str, Any]] = []
        for _ in range(n):
            meta = self.infer.predict_with_metadata(prompt, context=context)
            candidates.append({
                "prompt": prompt,
                "text": meta["text"],
                "backend": meta["backend"],
                "checkpoint_sha256": meta["checkpoint_sha256"],
                "corpus_path": sim_result.get("corpus_path"),
                "cycle": self.feedback.cycle_count,
            })
        paths = self.feedback.run_feedback_gated(candidates, sim_result)
        if not paths:
            logger.warning(
                "[ouro] Gate accepted no candidates this cycle; no feedback seed written."
            )
            return Path()
        return paths[0]

    def run_feedback_from_disk(self, prompt: str = "Summarize the predictions.") -> Path:
        """
        Run only the inference + feedback stage, loading sim_result from the
        last persisted cycle state on disk. The corpus from the last cycle is
        injected as system context so the model synthesises real simulation data.
        """
        state_path = self.config.feedback_state_path.parent / "last_cycle.json"
        if not state_path.exists():
            raise FileNotFoundError(
                "No persisted cycle state found. Run run_full_cycle() first."
            )
        with state_path.open() as f:
            state = json.load(f)

        sim_result = state.get("sim_result", {})
        context = self._load_corpus_context(sim_result)
        return self._infer_and_feed_back(prompt, context, sim_result)
