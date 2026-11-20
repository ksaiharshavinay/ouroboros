"""Tests for Run Spec R3 C3 — gate configuration passthrough.

Tier 2 gate ablations (diversity-only, no-vetoes) must be run
parameters, not code forks: r1_params.json gains an optional "gate"
object, the driver threads it into OuroConfig, and the gate honours a
veto on/off toggle. With no overrides anywhere, behaviour is exactly
the historical gate — that default path is what most tests here pin.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from ouro.config import OuroConfig
from ouro.gate import FeedbackGate

REPO = Path(__file__).parent.parent
RUNNER = REPO / "scripts" / "r1_run.py"


@pytest.fixture
def gate_env(tmp_path):
    cfg = OuroConfig(
        seed_docs_dir=tmp_path / "seeds",
        corpus_dir=tmp_path / "corpus",
        model_dir=tmp_path / "model",
        feedback_state_path=tmp_path / "state.json",
        kuzu_db_path=tmp_path / "kuzu",
    )
    cfg.seed_docs_dir.mkdir(parents=True, exist_ok=True)
    (cfg.seed_docs_dir / "seed1.txt").write_text(
        "Baseline briefing: quarterly logistics costs rose across "
        "regional depots while diesel prices steadied.")
    cfg.corpus_dir.mkdir(parents=True, exist_ok=True)
    (cfg.corpus_dir / "corpus_cycle_0000.jsonl").write_text(
        json.dumps({"round": 1, "agent_id": 0,
                    "content": "agents discuss logistics",
                    "influence": 0.5, "action": "create_post"}) + "\n")
    return cfg


# A candidate whose distinct-token ratio is far below the veto floor:
# vetoed with vetoes on, scored (and judged on its signals) with them off.
DEGENERATE = {"prompt": "p", "text": "same same same same same same",
              "cycle": 0}
HEALTHY = {"prompt": "p",
           "text": "Rail capacity tightened across northern depots "
                   "while diesel held steady and costs kept rising "
                   "through the quarter for regional carriers.",
           "cycle": 0}


def test_vetoes_on_by_default(gate_env):
    assert gate_env.gate_vetoes_enabled is True
    [decision] = FeedbackGate(gate_env).evaluate([DEGENERATE])
    assert decision["accepted"] is False
    assert decision["veto"] is not None
    assert "degenerate_repetition" in decision["veto"]


def test_vetoes_off_scores_instead_of_vetoing(gate_env):
    gate_env.gate_vetoes_enabled = False
    [decision] = FeedbackGate(gate_env).evaluate([DEGENERATE])
    assert decision["veto"] is None
    assert not decision["reason"].startswith("vetoed")


def test_vetoes_off_does_not_change_healthy_candidates(gate_env):
    [on] = FeedbackGate(gate_env).evaluate([HEALTHY])
    gate_env.gate_vetoes_enabled = False
    [off] = FeedbackGate(gate_env).evaluate([HEALTHY])
    assert on["veto"] is None and off["veto"] is None
    assert on["weighted_score"] == off["weighted_score"]
    assert on["accepted"] == off["accepted"]


def test_weight_override_changes_decision(gate_env):
    # Provenance is this candidate's weakest signal (no corpus on
    # disk, no generator identity): zeroing the other weights leaves
    # the score at the provenance value alone.
    [base] = FeedbackGate(gate_env).evaluate([HEALTHY])
    gate_env.gate_weight_outcome = 0.0
    gate_env.gate_weight_diversity = 0.0
    gate_env.gate_weight_consistency = 0.0
    gate_env.gate_weight_provenance = 1.0
    [prov_only] = FeedbackGate(gate_env).evaluate([HEALTHY])
    assert prov_only["weighted_score"] == \
        prov_only["signals"]["provenance"]["value"]
    assert prov_only["weighted_score"] != base["weighted_score"]


# ---------------------------------------------------------------------------
# Runner CLI: overrides land in r1_params.json; misuse is refused
# ---------------------------------------------------------------------------

def _run_runner(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(RUNNER), *args],
        capture_output=True, text=True, timeout=600)


def _write_params(tmp_path, extra: list[str]) -> dict:
    """Invoke the runner far enough to write params (it then stops at
    the nonexistent Ollama binary — after the params write)."""
    out = tmp_path / "out"
    proc = _run_runner(["--arm", "B5", "--seed", "1", "--out", str(out),
                        "--venv-python", sys.executable,
                        "--ollama-bin", "/nonexistent/ollama-bin",
                        *extra])
    assert proc.returncode != 0
    return json.loads((out / "r1_params.json").read_text())


def test_runner_records_gate_overrides(tmp_path):
    params = _write_params(tmp_path, [
        "--gate-weight-diversity", "1.0", "--gate-weight-outcome", "0",
        "--no-gate-vetoes"])
    assert params["gate"] == {
        "weights": {"diversity": 1.0, "outcome": 0.0}, "vetoes": False}


def test_runner_no_overrides_no_gate_key(tmp_path):
    params = _write_params(tmp_path, [])
    assert "gate" not in params


def test_runner_gate_overrides_refused_for_b4(tmp_path):
    proc = _run_runner(["--arm", "B4", "--seed", "1",
                        "--out", str(tmp_path / "out"),
                        "--gate-weight-diversity", "1.0"])
    assert proc.returncode != 0
    assert "meaningless for --arm B4" in proc.stderr + proc.stdout


def test_runner_negative_gate_weight_refused(tmp_path):
    proc = _run_runner(["--arm", "B5", "--seed", "1",
                        "--out", str(tmp_path / "out"),
                        "--gate-weight-diversity", "-0.5"])
    assert proc.returncode != 0
    assert "must be >= 0" in proc.stderr + proc.stdout


def test_runner_gate_config_is_trajectory_identity(tmp_path):
    out = tmp_path / "out"
    proc = _run_runner(["--arm", "B5", "--seed", "1", "--out", str(out),
                        "--venv-python", sys.executable,
                        "--ollama-bin", "/nonexistent/ollama-bin",
                        "--gate-weight-diversity", "1.0"])
    assert proc.returncode != 0  # stopped at Ollama, params written
    # Same gate config: dry-run reuses the params fine.
    proc = _run_runner(["--arm", "B5", "--seed", "1", "--out", str(out),
                        "--venv-python", sys.executable, "--dry-run",
                        "--gate-weight-diversity", "1.0"])
    assert proc.returncode == 0, proc.stderr
    # A different gate config against the same --out is mixing.
    proc = _run_runner(["--arm", "B5", "--seed", "1", "--out", str(out),
                        "--venv-python", sys.executable])
    assert proc.returncode != 0
    assert "does not match" in proc.stderr + proc.stdout
