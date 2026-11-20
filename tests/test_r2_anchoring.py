"""Tests for Run Spec R2 (anchoring sweep) support.

Three layers:
  * unit tests for the real-example builder and the mixture builder in
    autoresearch/prepare.py (stub token counter — hermetic);
  * end-to-end prepare.py runs on a fixture corpus dir with the real
    seed documents and the real base-model tokenizer, for
    OURO_REAL_FRACTION in {unset, 0, 0.1, 0.25, 0.5};
  * scripts/r1_run.py CLI behaviour: B6 refuses a 0 fraction, params
    carry arm + real_fraction, dry-run reuses them, and a different
    fraction against the same --out is refused as trajectory mixing.

Guarantees under test: with the fraction unset/0, prepare.py's train
and eval files are byte-identical to the unanchored path; the eval
file is identical at every fraction; the achieved fraction lands
within +/-2 percentage points of the target; everything is
deterministic.
"""
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
_PREPARE_PATH = REPO / "autoresearch" / "prepare.py"
_spec = importlib.util.spec_from_file_location("ouro_prepare_r2", _PREPARE_PATH)
prepare = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prepare)

RUNNER = REPO / "scripts" / "r1_run.py"
REAL_SEEDS = REPO / "data" / "seeds"


# ---------------------------------------------------------------------------
# Real-example construction
# ---------------------------------------------------------------------------

def test_seed_examples_schema_and_groups():
    examples = prepare.seed_documents_to_examples(REAL_SEEDS)
    assert len(examples) >= 40  # 5 docs, paragraph-chained
    for ex in examples:
        assert set(ex) == {"prompt", "completion", "influence",
                           "source_file", "group"}
        assert ex["group"] == f"human-seed:{ex['source_file']}"
        assert ex["source_file"] in {f"seed{i}.txt" for i in range(1, 6)}
        assert ex["prompt"].startswith("Agent interaction context: ")
        assert ex["prompt"].endswith("\n\nNext agent response:")
        assert ex["completion"].startswith(" ")


def test_seed_examples_pair_within_document_only(tmp_path):
    (tmp_path / "seed1.txt").write_text("Alpha one.\n\nAlpha two.\n\nAlpha three.")
    (tmp_path / "seed2.txt").write_text("Beta one.\n\nBeta two.")
    examples = prepare.seed_documents_to_examples(tmp_path)
    # 2 pairs from seed1 (3 blocks), 1 pair from seed2 (2 blocks);
    # no pair crosses documents.
    assert len(examples) == 3
    by_source = {}
    for ex in examples:
        by_source.setdefault(ex["source_file"], []).append(ex)
    assert set(by_source) == {"seed1.txt", "seed2.txt"}
    assert by_source["seed1.txt"][0]["completion"] == " Alpha two."
    assert by_source["seed1.txt"][1]["completion"] == " Alpha three."
    assert by_source["seed2.txt"][0]["completion"] == " Beta two."
    assert all("Beta" not in ex["prompt"] or ex["source_file"] == "seed2.txt"
               for ex in examples)


# ---------------------------------------------------------------------------
# Mixture builder (stub counter)
# ---------------------------------------------------------------------------

def _fake_train(n: int, tokens: int) -> list[dict]:
    return [{"prompt": f"p{i}", "completion": f"c{i}", "group": "g#round1",
             "source_file": "f", "influence": 0.5, "_tokens": tokens}
            for i in range(n)]


def _fake_real(n: int, tokens: int) -> list[dict]:
    return [{"prompt": f"rp{i}", "completion": f"rc{i}",
             "group": "human-seed:seed1.txt", "source_file": "seed1.txt",
             "influence": 0.5, "_tokens": tokens} for i in range(n)]


def _count(ex: dict) -> int:
    return ex["_tokens"]


@pytest.mark.parametrize("target", [0.1, 0.25, 0.5])
def test_augment_hits_target_within_2pp(target):
    train = _fake_train(40, 100)          # 4000 corpus tokens
    real = _fake_real(10, 100)
    augmented, stats = prepare.augment_train_with_real(
        train, real, target, _count, seed=42)
    assert abs(stats["achieved_fraction"] - target) <= 0.02
    assert stats["real_examples_added"] == (
        len(augmented) - len(train))
    n_real = sum(1 for ex in augmented
                 if ex["group"].startswith("human-seed"))
    assert n_real == stats["real_examples_added"]


def test_augment_is_deterministic():
    train = _fake_train(40, 100)
    real = _fake_real(10, 100)
    a1, s1 = prepare.augment_train_with_real(train, real, 0.25, _count, 42)
    a2, s2 = prepare.augment_train_with_real(train, real, 0.25, _count, 42)
    assert [json.dumps(e, sort_keys=True) for e in a1] == \
           [json.dumps(e, sort_keys=True) for e in a2]
    assert s1 == s2


def test_augment_fraction_zero_returns_train_unchanged():
    train = _fake_train(5, 100)
    augmented, stats = prepare.augment_train_with_real(
        train, _fake_real(3, 100), 0.0, _count, seed=42)
    assert augmented == train
    assert stats["achieved_fraction"] == 0.0
    assert stats["real_examples_added"] == 0


def test_augment_empty_pool_is_noop():
    train = _fake_train(5, 100)
    augmented, stats = prepare.augment_train_with_real(
        train, [], 0.5, _count, seed=42)
    assert augmented == train
    assert stats["achieved_fraction"] == 0.0


def test_real_fraction_from_env(monkeypatch):
    monkeypatch.delenv("OURO_REAL_FRACTION", raising=False)
    assert prepare.real_fraction_from_env() == 0.0
    monkeypatch.setenv("OURO_REAL_FRACTION", "0.25")
    assert prepare.real_fraction_from_env() == 0.25
    monkeypatch.setenv("OURO_REAL_FRACTION", "1")
    with pytest.raises(SystemExit):
        prepare.real_fraction_from_env()
    monkeypatch.setenv("OURO_REAL_FRACTION", "abc")
    with pytest.raises(SystemExit):
        prepare.real_fraction_from_env()


# ---------------------------------------------------------------------------
# prepare.py end-to-end on a fixture corpus
# ---------------------------------------------------------------------------

_SENTENCE = ("The committee reviewed the quarterly figures and noted "
             "that adoption across regional offices continued to grow "
             "steadily through the period under discussion today.")


def _fixture_corpus(root: Path) -> None:
    corpus = root / "data" / "corpus"
    corpus.mkdir(parents=True)
    seeds = root / "data" / "seeds"
    seeds.mkdir(parents=True)
    for i in range(1, 6):
        shutil.copy(REAL_SEEDS / f"seed{i}.txt", seeds / f"seed{i}.txt")
    for fname, tag in (("corpus_cycle_0000.jsonl", "alpha"),
                       ("corpus_cycle_0001.jsonl", "beta")):
        lines = []
        for i, rnd in enumerate((1, 1, 2, 2, 3, 3)):
            lines.append(json.dumps({
                "round": rnd, "agent_id": i % 3,
                "content": f"{tag} turn {i}: {_SENTENCE}",
                "influence": 0.5, "action": "create_post",
            }))
        (corpus / fname).write_text("\n".join(lines) + "\n")


def _run_prepare(root: Path, fraction: str | None) -> subprocess.CompletedProcess:
    env = dict(**{k: v for k, v in __import__("os").environ.items()})
    env["HF_HUB_OFFLINE"] = "1"  # tokenizer comes from the local cache
    if fraction is None:
        env.pop("OURO_REAL_FRACTION", None)
    else:
        env["OURO_REAL_FRACTION"] = fraction
    return subprocess.run(
        [sys.executable, str(_PREPARE_PATH)], cwd=root, env=env,
        capture_output=True, text=True, timeout=600)


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    """Run prepare.py at each fraction on a fresh copy of the fixture."""
    results = {}
    for label, fraction in (("unset", None), ("zero", "0"), ("f10", "0.1"),
                            ("f25", "0.25"), ("f50", "0.5"), ("f25b", "0.25")):
        root = tmp_path_factory.mktemp(f"prep_{label}")
        _fixture_corpus(root)
        proc = _run_prepare(root, fraction)
        assert proc.returncode == 0, proc.stderr
        results[label] = root
    return results


def _read(root: Path, name: str) -> bytes:
    return (root / "data" / "corpus" / name).read_bytes()


def test_e2e_fraction_zero_byte_identical(prepared):
    assert _read(prepared["zero"], "train.jsonl") == \
        _read(prepared["unset"], "train.jsonl")
    assert _read(prepared["zero"], "eval.jsonl") == \
        _read(prepared["unset"], "eval.jsonl")
    assert not (prepared["zero"] / "data/corpus/mixture.json").exists()
    assert not (prepared["unset"] / "data/corpus/mixture.json").exists()


def test_e2e_eval_identical_at_all_fractions(prepared):
    base = _read(prepared["unset"], "eval.jsonl")
    assert base  # the fixture must produce a non-empty eval
    for label in ("zero", "f10", "f25", "f50"):
        assert _read(prepared[label], "eval.jsonl") == base
        eval_rows = [json.loads(l) for l in
                     _read(prepared[label], "eval.jsonl").splitlines()]
        assert not any(r["group"].startswith("human-seed")
                       for r in eval_rows)


def test_e2e_deterministic_train_bytes(prepared):
    assert _read(prepared["f25"], "train.jsonl") == \
        _read(prepared["f25b"], "train.jsonl")


@pytest.mark.parametrize("label,target", [("f10", 0.1), ("f25", 0.25),
                                          ("f50", 0.5)])
def test_e2e_achieved_fraction_within_2pp(prepared, label, target):
    root = prepared[label]
    mix = json.loads(_read(root, "mixture.json"))
    assert mix["target_fraction"] == target
    assert abs(mix["achieved_fraction"] - target) <= 0.02
    # Independent recomputation from the written train file, counted
    # with the real tokenizer exactly as prepare counts.
    tokenizer = prepare.load_base_tokenizer()
    rows = [json.loads(l) for l in
            _read(root, "train.jsonl").decode().splitlines()]
    real_tokens = sum(prepare.example_token_count(r, tokenizer)
                      for r in rows if r["group"].startswith("human-seed"))
    total_tokens = sum(prepare.example_token_count(r, tokenizer)
                       for r in rows)
    assert real_tokens > 0
    assert abs(real_tokens / total_tokens - target) <= 0.02
    assert mix["achieved_fraction"] == pytest.approx(
        real_tokens / total_tokens, abs=1e-6)


def test_e2e_stale_sidecar_removed_when_fraction_zero(tmp_path):
    _fixture_corpus(tmp_path)
    stale = tmp_path / "data" / "corpus" / "mixture.json"
    stale.write_text("{}")
    proc = _run_prepare(tmp_path, "0")
    assert proc.returncode == 0, proc.stderr
    assert not stale.exists()


# ---------------------------------------------------------------------------
# r1_run.py CLI
# ---------------------------------------------------------------------------

def _run_runner(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(RUNNER), *args],
        capture_output=True, text=True, timeout=600)


def test_runner_b6_with_zero_fraction_refused(tmp_path):
    proc = _run_runner(["--arm", "B6", "--seed", "2",
                        "--out", str(tmp_path / "out")])
    assert proc.returncode != 0
    assert "requires --real-fraction" in proc.stderr + proc.stdout


def test_runner_fraction_one_refused(tmp_path):
    proc = _run_runner(["--arm", "B5", "--seed", "1", "--real-fraction", "1",
                        "--out", str(tmp_path / "out")])
    assert proc.returncode != 0
    assert "--real-fraction must be in [0, 1)" in proc.stderr + proc.stdout


def test_runner_b6_params_dry_run_and_mixing(tmp_path):
    out = tmp_path / "r1_b6_rep2"
    # Real invocation: params are written and hardware probed, then the
    # run stops at the (nonexistent) Ollama binary — after the params
    # write, which is what this test exercises.
    proc = _run_runner(["--arm", "B6", "--seed", "2",
                        "--real-fraction", "0.25", "--out", str(out),
                        "--venv-python", sys.executable,
                        "--ollama-bin", "/nonexistent/ollama-bin"])
    assert proc.returncode != 0
    params = json.loads((out / "r1_params.json").read_text())
    assert params["arm"] == "B6"
    assert params["real_fraction"] == 0.25

    # Dry-run against the same --out reuses the params and exits 0.
    proc = _run_runner(["--arm", "B6", "--seed", "2",
                        "--real-fraction", "0.25", "--out", str(out),
                        "--venv-python", sys.executable, "--dry-run"])
    assert proc.returncode == 0, proc.stderr

    # A different fraction (or arm) against the same --out is refused.
    for bad in (["--arm", "B6", "--seed", "2", "--real-fraction", "0.1"],
                ["--arm", "B5", "--seed", "2", "--real-fraction", "0.25"]):
        proc = _run_runner([*bad, "--out", str(out),
                            "--venv-python", sys.executable])
        assert proc.returncode != 0
        assert "does not match" in proc.stderr + proc.stdout
