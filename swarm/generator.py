# OUROBOROS — swarm/generator.py
# Copyright (c) 2026 Kudikala Sai Harsha Vinay. All rights reserved.
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


class OasisUnavailableError(RuntimeError):
    """Raised when OASIS cannot run and the stub fallback was not opted into."""


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
                self._run_oasis(
                    oasis_personas,
                    rounds=self.config.simulation_rounds,
                    personas=personas,
                    cycle=cycle,
                )
            )
            self._last_turns = turns

            # Stage 3b (Phase 2): provenance enrichment — record which
            # turn produced which entity in the knowledge graph. Guarded:
            # enrichment failure must never fail the simulation.
            try:
                self._graph_builder.build_from_turns(
                    turns, source_ref=f"corpus_cycle_{cycle:04d}"
                )
            except Exception as exc:
                logger.warning("[swarm] Turn graph enrichment failed: %s", exc)

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
            "report": self._last_report,
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
        self, oasis_personas: list[dict[str, Any]], rounds: int,
        personas: list[dict[str, Any]] | None = None,
        cycle: int = 0,
    ) -> list[dict[str, Any]]:
        """
        Run an OASIS social simulation via the documented camel-oasis API.

        Failure policy (Phase 1): the stub simulation is an EXPLICIT opt-in
        (config.allow_stub_simulation). When OASIS is missing or errors and
        the flag is off, OasisUnavailableError is raised — a paper run must
        never silently degrade to templated stub turns.
        """
        allow_stub = bool(getattr(self.config, "allow_stub_simulation", False))
        try:
            import oasis
        except ImportError as exc:
            if allow_stub:
                logger.warning(
                    "[swarm] camel-oasis not installed and allow_stub_simulation "
                    "is set; using EXPLICIT stub simulation."
                )
                return self._stub_simulation(
                    len(oasis_personas), rounds, personas=personas or []
                )
            raise OasisUnavailableError(
                "camel-oasis is not installed and allow_stub_simulation is "
                "False. Install camel-oasis in the Python 3.11 environment "
                "or set OuroConfig.allow_stub_simulation=True to explicitly "
                "opt into the templated stub simulation."
            ) from exc

        try:
            return await self._run_oasis_env(
                oasis, oasis_personas, rounds, cycle, personas=personas
            )
        except Exception:
            if allow_stub:
                logger.warning(
                    "[swarm] OASIS run failed and allow_stub_simulation is "
                    "set; using EXPLICIT stub simulation.", exc_info=True,
                )
                return self._stub_simulation(
                    len(oasis_personas), rounds, personas=personas or []
                )
            raise

    def _make_oasis_model(self) -> Any:
        """Build the camel-ai model handle OASIS agents will act with.

        Points camel's OpenAI-compatible factory at the configured backend
        (Ollama by default), or Anthropic when llm_provider="claude".
        """
        from camel.models import ModelFactory
        from camel.types import ModelPlatformType

        provider = getattr(self.config, "llm_provider", "ollama").lower()
        timeout = getattr(self.config, "llm_timeout", None)
        if provider == "claude":
            return ModelFactory.create(
                model_platform=ModelPlatformType.ANTHROPIC,
                model_type=self.config.llm_model,
                api_key=os.environ.get(self.config.llm_api_key_env, "") or None,
                timeout=timeout,
            )
        return ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=self.config.llm_model,
            model_config_dict={
                "max_tokens": getattr(self.config, "llm_max_tokens", 512),
            },
            url=self.config.llm_base_url,
            api_key=os.environ.get("OLLAMA_API_KEY", "ollama"),
            timeout=timeout,
        )

    def _write_oasis_profile_file(
        self, platform: str, oasis_personas: list[dict[str, Any]],
        personas: list[dict[str, Any]] | None, cycle: int,
    ) -> Path:
        """
        Materialise agent profiles in the exact format camel-oasis 0.2.6
        expects for its generate_*_agent_graph(profile_path=...) calls:
          * twitter: CSV with columns username, description, user_char
          * reddit:  JSON list of {username, bio, persona, mbti, gender,
                     age, country}
        """
        import csv
        import re

        rows = []
        for i, op in enumerate(oasis_personas):
            raw = personas[i] if personas and i < len(personas) else {}
            name = raw.get("name") or f"agent_{i}"
            background = raw.get("background", "")
            beliefs = raw.get("beliefs", [])
            user_char = op.get("user_profile") or (
                f"{name}: {background} Beliefs: {'; '.join(beliefs)}"
            )
            username = re.sub(r"[^A-Za-z0-9_]", "_", str(name))[:25] or f"agent_{i}"
            rows.append({
                "username": f"{username}_{i}" if platform == "twitter" else username,
                "description": background or user_char[:200],
                "user_char": user_char,
            })

        corpus_dir = self.config.corpus_dir
        corpus_dir.mkdir(parents=True, exist_ok=True)
        if platform == "reddit":
            profile_path = corpus_dir / f"oasis_profiles_cycle_{cycle:04d}.json"
            payload = [{
                "username": r["username"],
                "bio": r["description"],
                "persona": r["user_char"],
                "mbti": "", "gender": "", "age": "", "country": "",
            } for r in rows]
            profile_path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        else:
            profile_path = corpus_dir / f"oasis_profiles_cycle_{cycle:04d}.csv"
            with profile_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["username", "description", "user_char"]
                )
                writer.writeheader()
                writer.writerows(rows)
        return profile_path

    async def _run_oasis_env(
        self, oasis: Any, oasis_personas: list[dict[str, Any]], rounds: int,
        cycle: int, personas: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Documented-API OASIS run (camel-oasis 0.2.6):
          1. generate_<platform>_agent_graph(profile_path=..., model=...,
             available_actions=[...]) -> AgentGraph   (async)
          2. oasis.make(agent_graph=..., platform=..., database_path=...)
          3. await env.step({agent: LLMAction(), ...}) for `rounds` steps
             (AgentGraph.get_agents() yields (agent_id, agent) tuples)
          4. Export turns from the simulation's own trace database.
        """
        import inspect

        platform = getattr(self.config, "oasis_platform", "twitter").lower()
        generate = (
            oasis.generate_reddit_agent_graph
            if platform == "reddit"
            else oasis.generate_twitter_agent_graph
        )
        platform_type = oasis.DefaultPlatformType.REDDIT if platform == "reddit" \
            else oasis.DefaultPlatformType.TWITTER

        action_type = oasis.ActionType
        available_actions = [
            action_type.CREATE_POST,
            action_type.CREATE_COMMENT,
            action_type.LIKE_POST,
            action_type.REPOST,
            action_type.FOLLOW,
            action_type.DO_NOTHING,
        ]
        model = self._make_oasis_model()
        profile_path = self._write_oasis_profile_file(
            platform, oasis_personas, personas, cycle
        )
        agent_graph = generate(
            profile_path=str(profile_path),
            model=model,
            available_actions=available_actions,
        )
        if inspect.isawaitable(agent_graph):
            agent_graph = await agent_graph

        db_path = self.config.corpus_dir / f"oasis_trace_cycle_{cycle:04d}.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        if db_path.exists():
            db_path.unlink()

        env = oasis.make(
            agent_graph=agent_graph,
            platform=platform_type,
            database_path=str(db_path),
        )
        try:
            await env.reset()
            agents = [
                item[1] if isinstance(item, tuple) else item
                for item in agent_graph.get_agents()
            ]
            for _ in range(rounds):
                actions = {agent: oasis.LLMAction() for agent in agents}
                await env.step(actions)
        finally:
            close = getattr(env, "close", None)
            if close is not None:
                result = close()
                if inspect.isawaitable(result):
                    await result

        turns = self._export_trace_turns(db_path, n_agents=len(oasis_personas))
        logger.info(
            "[swarm] OASIS run complete: %d agents, %d rounds, %d exported turns",
            len(oasis_personas), rounds, len(turns),
        )
        return turns

    @staticmethod
    def _export_trace_turns(db_path: Path, n_agents: int) -> list[dict[str, Any]]:
        """
        Export corpus turns from an OASIS trace database into the corpus
        schema used by _export_corpus / autoresearch.prepare:
        {"round", "agent_id", "content", "influence", "action"}.

        Prefers the `trace` table (every agent action, in order); falls
        back to the `post`/`comment` tables if no trace content exists.
        """
        import sqlite3

        if not Path(db_path).exists():
            return []
        conn = sqlite3.connect(str(db_path))
        try:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            turns: list[dict[str, Any]] = []
            if "trace" in tables:
                rows = conn.execute(
                    "SELECT user_id, action, info FROM trace ORDER BY rowid"
                ).fetchall()
                for i, (user_id, action, info) in enumerate(rows):
                    try:
                        info_dict = json.loads(info) if info else {}
                    except (json.JSONDecodeError, TypeError):
                        info_dict = {}
                    content = (
                        info_dict.get("content")
                        or info_dict.get("post_content")
                        or info_dict.get("comment_content")
                        or ""
                    )
                    if not content:
                        continue
                    turns.append({
                        "round": i // max(1, n_agents),
                        "agent_id": int(user_id) if user_id is not None else 0,
                        "content": str(content),
                        "influence": 0.5,
                        "action": str(action),
                    })
            if not turns and "post" in tables:
                rows = conn.execute(
                    "SELECT user_id, content FROM post ORDER BY rowid"
                ).fetchall()
                for i, (user_id, content) in enumerate(rows):
                    if not content:
                        continue
                    turns.append({
                        "round": i // max(1, n_agents),
                        "agent_id": int(user_id) if user_id is not None else 0,
                        "content": str(content),
                        "influence": 0.5,
                        "action": "create_post",
                    })
            return turns
        finally:
            conn.close()

    def _stub_simulation(
        self, n_agents: int, rounds: int, personas: list[dict[str, Any]] | None = None
    ) -> list[dict[str, Any]]:
        """
        Fallback simulation when OASIS is not installed.
        Uses persona backgrounds and beliefs so corpus content is domain-specific,
        not generic placeholder text.
        """
        import random
        rng = random.Random(42)  # deterministic
        active = min(n_agents, 10)
        turns = []
        for r in range(rounds):
            for a in range(active):
                if personas and a < len(personas):
                    p = personas[a]
                    name = p.get("name", f"Agent_{a}")
                    beliefs = p.get("beliefs", [])
                    background = p.get("background", "")
                    belief = rng.choice(beliefs) if beliefs else background
                    stances = ["supports", "questions", "analyses", "challenges", "advocates for"]
                    stance = rng.choice(stances)
                    content = f"{name} {stance}: \"{belief}\""
                else:
                    content = f"Agent {a} observes trends in round {r}."
                turns.append({
                    "round": r,
                    "agent_id": a,
                    "content": content,
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
        return ClaudeAdapter(
            api_key=api_key,
            model=self.config.llm_model,
            provider=self.config.llm_provider,
            base_url=self.config.llm_base_url,
        )

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
