"""Phase 1 proof tests — the "loop is real" contract (§4.2 S1).

Every test in this file FAILS against the pre-Phase-1 code:

* Training was a stub (no weights, char-overlap metric, JSON-only
  checkpoint) — there was no fine_tune() result with a loss history,
  no adapter file, and no adapter hash to assert on.
* The OASIS adapter called a non-existent API and silently substituted
  a templated stub simulation on ImportError — there was no
  OasisUnavailableError and no allow_stub_simulation flag.
* Inference checked that a metadata file existed and then called the
  base LLM API — there was no local adapter loading path at all.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ouro.config import OuroConfig
from ouro.infer import OuroInferenceEngine
from swarm.generator import OuroborosSwarmGenerator

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "autoresearch"))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_config(tmp_path):
    return OuroConfig(
        seed_docs_dir=tmp_path / "seeds",
        corpus_dir=tmp_path / "corpus",
        model_dir=tmp_path / "model",
        feedback_state_path=tmp_path / "state.json",
        kuzu_db_path=tmp_path / "kuzu",
    )


@pytest.fixture
def generator(tmp_config):
    with patch.object(OuroborosSwarmGenerator, "_make_llm_client", return_value=MagicMock()):
        gen = OuroborosSwarmGenerator(tmp_config)
    return gen


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# B. OASIS adapter — stub is explicit, never silent
# ---------------------------------------------------------------------------

def test_oasis_missing_raises_when_stub_not_allowed(generator, monkeypatch):
    """ImportError + allow_stub_simulation=False must raise, not stub."""
    from swarm.generator import OasisUnavailableError

    monkeypatch.setitem(sys.modules, "oasis", None)  # import oasis -> ImportError
    assert generator.config.allow_stub_simulation is False
    with pytest.raises(OasisUnavailableError):
        asyncio.run(generator._run_oasis([], rounds=1))


def test_oasis_missing_stubs_only_when_explicitly_allowed(tmp_config, monkeypatch):
    """The stub path still exists — but only as an explicit opt-in."""
    cfg = OuroConfig(
        seed_docs_dir=tmp_config.seed_docs_dir,
        corpus_dir=tmp_config.corpus_dir,
        model_dir=tmp_config.model_dir,
        feedback_state_path=tmp_config.feedback_state_path,
        kuzu_db_path=tmp_config.kuzu_db_path,
        allow_stub_simulation=True,
    )
    with patch.object(OuroborosSwarmGenerator, "_make_llm_client", return_value=MagicMock()):
        gen = OuroborosSwarmGenerator(cfg)
    monkeypatch.setitem(sys.modules, "oasis", None)
    fake_personas = [{"user_profile": f"agent {i}", "activity_level": 0.5} for i in range(3)]
    turns = asyncio.run(gen._run_oasis(fake_personas, rounds=2))
    assert isinstance(turns, list) and len(turns) > 0


def _install_fake_oasis(monkeypatch, db_posts: list[str]):
    """Install a fake `oasis` (+ `camel`) module tree recording every call."""
    calls: dict = {"make": {}, "gen": {}, "steps": []}

    agents = [MagicMock(name=f"agent{i}") for i in range(2)]

    class FakeAgentGraph:
        def get_agents(self, agent_ids=None):
            # camel-oasis 0.2.6: list of (agent_id, agent) tuples
            return list(enumerate(agents))

    async def fake_generate(profile_path=None, model=None, available_actions=None, **kw):
        calls["gen"] = {"profile_path": profile_path, "available_actions": available_actions}
        return FakeAgentGraph()

    class FakeEnv:
        async def reset(self):
            return None

        async def step(self, actions):
            calls["steps"].append(actions)
            return None

        async def close(self):
            return None

    def fake_make(agent_graph=None, platform=None, database_path=None):
        calls["make"] = {
            "agent_graph": agent_graph,
            "platform": platform,
            "database_path": database_path,
        }
        # Simulate OASIS writing its trace database.
        conn = sqlite3.connect(database_path)
        conn.execute(
            "CREATE TABLE trace (user_id INTEGER, created_at TEXT, action TEXT, info TEXT)"
        )
        for i, content in enumerate(db_posts):
            conn.execute(
                "INSERT INTO trace VALUES (?, ?, ?, ?)",
                (i % 2, "2026-01-01 00:00:00", "create_post",
                 json.dumps({"content": content})),
            )
        conn.commit()
        conn.close()
        return FakeEnv()

    oasis = types.ModuleType("oasis")
    oasis.generate_twitter_agent_graph = fake_generate
    oasis.generate_reddit_agent_graph = fake_generate
    oasis.make = fake_make
    oasis.LLMAction = type("LLMAction", (), {"__init__": lambda self: None})
    oasis.ManualAction = type("ManualAction", (), {"__init__": lambda self, **kw: None})
    oasis.ActionType = types.SimpleNamespace(
        CREATE_POST="create_post", CREATE_COMMENT="create_comment",
        LIKE_POST="like_post", REPOST="repost", FOLLOW="follow",
        DO_NOTHING="do_nothing", QUOTE_POST="quote_post",
    )
    oasis.DefaultPlatformType = types.SimpleNamespace(TWITTER="twitter", REDDIT="reddit")

    camel = types.ModuleType("camel")
    camel_models = types.ModuleType("camel.models")
    camel_models.ModelFactory = MagicMock()
    camel_models.ModelFactory.create = MagicMock(return_value=MagicMock())
    camel_types = types.ModuleType("camel.types")
    camel_types.ModelPlatformType = types.SimpleNamespace(OPENAI="openai", ANTHROPIC="anthropic")

    monkeypatch.setitem(sys.modules, "oasis", oasis)
    monkeypatch.setitem(sys.modules, "camel", camel)
    monkeypatch.setitem(sys.modules, "camel.models", camel_models)
    monkeypatch.setitem(sys.modules, "camel.types", camel_types)
    return calls


def test_oasis_adapter_uses_documented_api_and_exports_trace(generator, monkeypatch):
    """The adapter must build an AgentGraph, call oasis.make(agent_graph=...,
    platform=..., database_path=...), step with LLMAction dicts, and export
    the corpus from the simulation's own trace database."""
    calls = _install_fake_oasis(monkeypatch, ["first post from trace", "second post"])
    personas = [
        {"user_profile": "Alice: researcher", "activity_level": 0.7},
        {"user_profile": "Bob: sceptic", "activity_level": 0.4},
    ]
    turns = asyncio.run(generator._run_oasis(personas, rounds=2, cycle=0))

    # Documented-API construction (camel-oasis 0.2.6: profile_path file)
    assert calls["gen"]["profile_path"] is not None
    assert Path(calls["gen"]["profile_path"]).exists()
    assert isinstance(calls["make"]["agent_graph"], object)
    assert calls["make"]["database_path"] is not None
    assert len(calls["steps"]) == 2
    for step_actions in calls["steps"]:
        assert isinstance(step_actions, dict)
        assert all(type(a).__name__ == "LLMAction" for a in step_actions.values())

    # Corpus exported from the trace database, in the existing turn schema
    contents = [t["content"] for t in turns]
    assert "first post from trace" in contents
    assert "second post" in contents
    for t in turns:
        assert "round" in t and "agent_id" in t and "content" in t


# ---------------------------------------------------------------------------
# C. Real training — weights move, loss is real, checkpoint is loadable
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_training_actually_trains(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    import train as T

    monkeypatch.setattr(T, "MODEL_DIR", tmp_path / "model")

    examples = [
        {
            "prompt": f"Agent interaction context: Agents discuss topic {i}.\n\nNext agent response:",
            "completion": " The swarm converges on verified, well-sourced data.",
        }
        for i in range(8)
    ]
    result = T.fine_tune(
        examples,
        budget_seconds=280,
        base_model="sshleifer/tiny-gpt2",
        max_steps=12,
        output_dir=tmp_path / "model",
        learning_rate=5e-3,
        num_epochs=1,
    )

    # (a) Real optimisation happened: multiple steps, loss values change.
    assert result["steps"] >= 2
    history = result["loss_history"]
    assert len(history) >= 2, f"expected per-step losses, got {history}"
    assert len(set(history)) > 1, f"loss never changed: {history}"

    # (b) Adapter weights exist, are non-trivial, and the hash is recorded.
    adapter_file = Path(result["adapter_dir"]) / "adapter_model.safetensors"
    assert adapter_file.exists()
    assert result["adapter_sha256"] == _sha256(adapter_file)
    from safetensors.torch import load_file

    state = load_file(str(adapter_file))
    assert state, "adapter file contains no tensors"
    # LoRA B matrices are zero-initialised; after real training at least
    # some adapter weights must be non-zero (i.e. differ from init).
    assert any(float(t.abs().sum()) > 0 for t in state.values())

    # save_checkpoint produces a loadable checkpoint dir + metadata hash.
    ckpt = T.save_checkpoint(result)
    assert (ckpt / "adapter_model.safetensors").exists()
    assert (ckpt / "adapter_config.json").exists()
    metadata = json.loads((ckpt / "checkpoint_metadata.json").read_text())
    assert metadata["adapter_sha256"] == _sha256(ckpt / "adapter_model.safetensors")

    # evaluate() returns a real, finite NLL for the trained adapter.
    loss = T.evaluate(examples[:2], base_model="sshleifer/tiny-gpt2", adapter_dir=str(ckpt))
    assert loss == loss and loss > 0  # not NaN, not the old char-overlap score


# ---------------------------------------------------------------------------
# D. Inference — a real checkpoint is loaded and used
# ---------------------------------------------------------------------------

def _write_fake_checkpoint(config: OuroConfig) -> Path:
    ckpt = config.model_dir / "checkpoint"
    ckpt.mkdir(parents=True, exist_ok=True)
    weights = ckpt / "adapter_model.safetensors"
    weights.write_bytes(b"fake-adapter-weights")
    (ckpt / "adapter_config.json").write_text("{}")
    (ckpt / "checkpoint_metadata.json").write_text(json.dumps({
        "base_model": "some/base-model",
        "adapter_sha256": _sha256(weights),
    }))
    return ckpt


def test_infer_loads_adapter_and_records_hash(tmp_config, monkeypatch):
    """_load_local must run the HF/PEFT load path and record the hash of
    the adapter actually loaded. Fails on the old code, which had no
    _load_local at all and called the base LLM API instead."""
    ckpt = _write_fake_checkpoint(tmp_config)
    engine = OuroInferenceEngine(tmp_config)
    assert engine.has_adapter_checkpoint()

    fake_torch = types.ModuleType("torch")
    fake_torch.float32 = "float32"
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = MagicMock()
    fake_transformers.AutoTokenizer.from_pretrained = MagicMock(return_value=MagicMock())
    fake_transformers.AutoModelForCausalLM = MagicMock()
    fake_transformers.AutoModelForCausalLM.from_pretrained = MagicMock(return_value=MagicMock())
    fake_peft = types.ModuleType("peft")
    fake_peft.PeftModel = MagicMock()
    fake_peft.PeftModel.from_pretrained = MagicMock(return_value=MagicMock())

    with patch.dict(sys.modules, {
        "torch": fake_torch, "transformers": fake_transformers, "peft": fake_peft,
    }):
        engine.load()

    assert engine._backend == "local"
    assert engine.checkpoint_sha256 == _sha256(ckpt / "adapter_model.safetensors")
    fake_peft.PeftModel.from_pretrained.assert_called_once()
    # The adapter dir (not some other path) is what got loaded.
    assert fake_peft.PeftModel.from_pretrained.call_args[0][1] == str(ckpt)


def test_infer_predict_prefers_local_backend(tmp_config):
    """With an adapter checkpoint present, predict() must use the local
    model path, not the LLM API."""
    _write_fake_checkpoint(tmp_config)
    engine = OuroInferenceEngine(tmp_config)

    def fake_load_local(self):
        self._backend = "local"
        self._model = object()
        self.checkpoint_sha256 = "deadbeef"

    with patch.object(OuroInferenceEngine, "_load_local", fake_load_local), \
         patch.object(engine, "_local_predict", return_value="local output") as local_mock, \
         patch.object(engine, "_llm_predict", return_value="api output") as api_mock:
        result = engine.predict("hello")

    assert result == "local output"
    assert local_mock.called and not api_mock.called


def test_infer_rejects_checkpoint_with_mismatched_hash(tmp_config, monkeypatch):
    """A checkpoint whose weights don't match its recorded hash must be
    refused — lineage is provable or it doesn't run."""
    ckpt = _write_fake_checkpoint(tmp_config)
    meta = json.loads((ckpt / "checkpoint_metadata.json").read_text())
    meta["adapter_sha256"] = "0" * 64
    (ckpt / "checkpoint_metadata.json").write_text(json.dumps(meta))

    engine = OuroInferenceEngine(tmp_config)
    fake_torch = types.ModuleType("torch")
    fake_torch.float32 = "float32"
    with patch.dict(sys.modules, {"torch": fake_torch}):
        with pytest.raises(RuntimeError, match="hash mismatch"):
            engine._load_local()
