# OUROBOROS — tests/test_r3_direct_b3.py
# Copyright (c) 2026 Kudikala Sai Harsha Vinay. All rights reserved.
# PATENT PENDING
#
# R3 C1: the B3 direct self-training arm (spec §4) — turn building in
# simulation-corpus format, no-feedback cycle bookkeeping, runner
# refusals, and the driver's snapshot validation treating B3's
# missing feedback seeds as by-construction, not corruption.

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ouro.direct import (build_direct_turns, seed_opening,
                         write_direct_corpus)

REPO = Path(__file__).resolve().parent.parent
SEEDS = [p.read_text(encoding="utf-8")
         for p in sorted((REPO / "data" / "seeds").glob("seed*.txt"))]
assert len(SEEDS) == 5


def test_seed_opening_takes_title_and_first_block():
    text = "Title Block\n\nFirst prose block here.\n\nSecond block."
    assert seed_opening(text) == "Title Block\n\nFirst prose block here."


def test_seed_opening_caps_long_documents():
    text = "Head. " + ("Sentence number two is here. " * 100)
    opening = seed_opening(text)
    assert len(opening) <= 1000
    assert opening.endswith(".")


def test_build_direct_turns_shape_and_counts():
    prompts = []

    def fake(prompt: str) -> str:
        prompts.append(prompt)
        return f"generated text {len(prompts)}"

    turns = build_direct_turns(SEEDS, fake, cycle=0)
    assert len(turns) == 60
    for turn in turns:
        assert set(turn) == {"round", "agent_id", "content",
                             "influence", "action"}
        assert turn["action"] == "create_post"
        assert turn["influence"] == 0.5
    for agent_id in range(5):
        mine = [t for t in turns if t["agent_id"] == agent_id]
        assert len(mine) == 12
        # prepare.py groups by (file, round): rounds must vary within
        # the file or the group-disjoint split collapses.
        assert len({t["round"] for t in mine}) == 10
    # Every generation was conditioned on its seed document's opening.
    assert prompts[0] == seed_opening(SEEDS[0])
    assert prompts[12] == seed_opening(SEEDS[1])


def test_build_direct_turns_refuses_empty_generation():
    with pytest.raises(RuntimeError, match="direct generation failed"):
        build_direct_turns(SEEDS, lambda prompt: "  ", cycle=0)
    with pytest.raises(RuntimeError, match="direct generation failed"):
        build_direct_turns(SEEDS, lambda prompt: "[Inference error: x]",
                           cycle=0)


def test_write_direct_corpus_format(tmp_path):
    turns = build_direct_turns(SEEDS, lambda p: "some generated text",
                               cycle=2)
    path = write_direct_corpus(turns, tmp_path, 2)
    assert path.name == "corpus_cycle_0002.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 60
    for line in lines:
        obj = json.loads(line)
        assert isinstance(obj["content"], str) and obj["content"]


def test_record_cycle_without_feedback(tmp_path):
    from ouro.config import OuroConfig
    from ouro.feedback import OuroFeedbackModule

    cfg = OuroConfig()
    cfg.feedback_state_path = tmp_path / "ouro_state.json"
    cfg.seed_docs_dir = tmp_path / "seeds"
    fb = OuroFeedbackModule(cfg)
    candidates = [{"text": f"cand {i}", "prompt": "p",
                   "backend": "local_lora", "checkpoint_sha256": "ab",
                   "corpus_path": "x", "cycle": 0} for i in range(3)]
    out = fb.record_cycle_without_feedback(
        {"num_turns": 60, "num_agents": 5}, candidates)
    assert out == []
    assert fb.cycle_count == 1
    seeds_dir = tmp_path / "seeds"
    assert not seeds_dir.exists() or \
        not list(seeds_dir.glob("feedback_cycle_*.txt"))
    meta = fb.get_history()[-1]
    assert meta["injection_mode"] == "direct_no_feedback"
    assert meta["selected_count"] == 0
    assert meta["seed_files"] == []
    assert len(meta["candidates"]) == 3
    assert meta["num_turns"] == 60


def _run_runner(*argv: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPO / "scripts" / "r1_run.py"), *argv],
        capture_output=True, text=True, cwd=REPO)


def test_runner_b3_refuses_anchor_and_gate_overrides(tmp_path):
    out = tmp_path / "out"
    p = _run_runner("--arm", "B3", "--seed", "1", "--out", str(out),
                    "--real-fraction", "0.1")
    assert p.returncode != 0
    assert "no anchor variant" in (p.stderr + p.stdout)
    p = _run_runner("--arm", "B3", "--seed", "1", "--out", str(out),
                    "--gate-weight-diversity", "1.0")
    assert p.returncode != 0
    assert "gate overrides are meaningless" in (p.stderr + p.stdout)


def _driver_state_problems(arm: str) -> list[str]:
    params = {"arm": arm, "rng_seed": 1, "num_agents": 10, "rounds": 15,
              "train_steps": 30, "train_model": "m", "swarm_model": "s",
              "venv_python": "python3", "real_fraction": 0.0}
    code = (
        "import json, sys\n"
        "sys.path.insert(0, 'scripts')\n"
        "import r1_driver as d\n"
        "state = {\n"
        " 'data/ouro_state.json': json.dumps(\n"
        "     {'cycle_count': 1, 'cycles': [\n"
        "         {'cycle': 0, 'seed_files': []}]}).encode(),\n"
        " 'data/corpus/frozen_eval_groups.json': b'{}',\n"
        " 'data/corpus/corpus_cycle_0000.jsonl': b'',\n"
        "}\n"
        "last, problems = d._state_problems(state, 0)\n"
        "print(json.dumps(problems))\n"
    )
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "r1_params.json").write_text(json.dumps(params))
        env = dict(os.environ, OURO_R1_OUT=td)
        p = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, cwd=REPO, env=env)
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout.strip().splitlines()[-1])


def test_driver_state_problems_b3_waives_feedback_seeds():
    assert _driver_state_problems("B3") == []
    problems = _driver_state_problems("B5")
    assert any("feedback seed" in p for p in problems)
