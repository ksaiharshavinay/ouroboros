# OUROBOROS — swarm/generator.py
# Copyright (c) 2026 Sai Harsha Vinaykudikala. All rights reserved.
# PATENT PENDING
#
# Swarm generator built on:
#   OASIS / camel-ai  — Apache 2.0
#   KuzuDB            — MIT
#   openai SDK        — MIT

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

from .base import BaseSwarmAdapter
from .graph import GraphBuilder
from .personas import PersonaGenerator
from .report import ReportGenerator

logger = logging.getLogger(__name__)


class OuroborosSwarmGenerator(BaseSwarmAdapter):
    """
    Full swarm simulation pipeline:
      1. Build knowledge graph from seed documents (KuzuDB)
      2. Generate N agent personas informed by top graph entities (LLM)
      3. Run OASIS social simulation
      4. Export interaction corpus to JSONL
      5. Generate prediction report (feeds back into next cycle)
    """

    def __init__(self, config: Any):
        self.config = config
        self._llm_client = self._make_llm_client()
        self._graph_builder = GraphBuilder(
            db_path=config.kuzu_db_path,
            llm_client=self._llm_client,
            model=config.llm_model,
        )
        self._persona_gen = PersonaGenerator(self._llm_client, config.llm_model)
        self._report_gen = ReportGenerator(self._llm_client, config.llm_model)
        self._last_report: str = ""
        self._last_turns: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "OuroborosSwarmGenerator v0.1.0"

    # ------------------------------------------------------------------
    # BaseSwarmAdapter interface
    # ------------------------------------------------------------------

    def run(self, seed_docs: list[Path], cycle: int = 0) -> dict[str, Any]:
        """Run the full swarm pipeline and return result metadata."""
        logger.info("[swarm] Starting OuroborosSwarmGenerator — cycle %d", cycle)

        self._graph_builder.open(cycle=cycle)
        try:
            # Stage 1: Build knowledge graph
            self._graph_builder.build_from_docs(seed_docs)
            top_entities = self._graph_builder.get_top_entities(n=20)

            # Stage 2: Generate personas
            personas = self._persona_gen.generate(
                n=self.config.num_agents, top_entities=top_entities
            )
            oasis_personas = PersonaGenerator.to_oasis_format(personas)

            # Stage 3: Run OASIS simulation
            turns = self._run_async(
                self._run_oasis(oasis_personas, rounds=self.config.simulation_rounds)
            )
            self._last_turns = turns

            # Stage 4: Export corpus
            corpus_path = self._export_corpus(turns, cycle=cycle)

            # Stage 5: Generate prediction report
            self._last_report = self._report_gen.generate(turns, top_entities)

        finally:
            self._graph_builder.close()

        result = {
            "num_agents": len(personas),
            "num_turns": len(turns),
            "corpus_path": str(corpus_path),
            "cycle": cycle,
            "report_preview": self._last_report[:300],
        }
        logger.info("[swarm] Cycle %d complete — %d turns", cycle, len(turns))
        return result

    def get_top_entities(self, n: int = 10) -> list[dict[str, Any]]:
        """Return top entities from the last completed simulation."""
        return self._graph_builder.get_top_entities(n=n)

    def get_report(self) -> str:
        """Return the prediction report from the last simulation."""
        return self._last_report

    # ------------------------------------------------------------------
    # OASIS simulation
    # ------------------------------------------------------------------

    async def _run_oasis(
        self, oasis_personas: list[dict[str, Any]], rounds: int
    ) -> list[dict[str, Any]]:
        """Run an OASIS Twitter-style social simulation."""
        try:
            import oasis

            env = oasis.make(
                "twitter",
                agent_profiles=oasis_personas,
                num_steps=rounds,
            )
            all_turns: list[dict[str, Any]] = []
            obs = await env.reset()
            for _ in range(rounds):
                actions = [agent.act(obs) for agent in env.agents]
                obs, rewards, done, info = await env.step(actions)
                if hasattr(obs, "__iter__"):
                    for o in obs:
                        if isinstance(o, dict):
                            all_turns.append(o)
                if done:
                    break
            return all_turns
        except ImportError:
            logger.warning("[swarm] camel-oasis not installed; using stub simulation.")
            return self._stub_simulation(len(oasis_personas), rounds)

    def _stub_simulation(self, n_agents: int, rounds: int) -> list[dict[str, Any]]:
        """Minimal fallback simulation when OASIS is not installed."""
        turns = []
        for r in range(rounds):
            for a in range(min(n_agents, 10)):
                turns.append({
                    "round": r,
                    "agent_id": a,
                    "content": f"Agent {a} observes trends in round {r}.",
                    "influence": (a % 5) * 0.1,
                })
        return turns

    # ------------------------------------------------------------------
    # Corpus export
    # ------------------------------------------------------------------

    def _export_corpus(self, turns: list[dict[str, Any]], cycle: int) -> Path:
        """Export simulation turns to a JSONL corpus file."""
        corpus_dir = self.config.corpus_dir
        corpus_dir.mkdir(parents=True, exist_ok=True)
        corpus_path = corpus_dir / f"corpus_cycle_{cycle:04d}.jsonl"
        with corpus_path.open("w", encoding="utf-8") as f:
            for turn in turns:
                f.write(json.dumps(turn) + "\n")
        logger.info("[swarm] Corpus exported to %s (%d turns)", corpus_path, len(turns))
        return corpus_path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_llm_client(self) -> Any:
        """Create an LLM client from config (Claude adapter)."""
        from ouro.claude_adapter import ClaudeAdapter

        api_key = os.environ.get(self.config.llm_api_key_env, "")
        return ClaudeAdapter(api_key=api_key, model=self.config.llm_model)

    @staticmethod
    def _run_async(coro: Any) -> Any:
        """Run a coroutine safely whether or not an event loop is running."""
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        except RuntimeError:
            return asyncio.run(coro)
