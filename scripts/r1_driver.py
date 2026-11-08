"""OUROBOROS — R1 driver (Colab).

Runs ONE arm trajectory of Run Spec R1 (paper Section 8) cycle by cycle,
one stage per invocation, mirroring the proven pilot driver:

  r1_driver.py init    — fresh-trajectory setup inside the disposable clone
  r1_driver.py stage1  — OASIS swarm simulation -> corpus for this cycle
  r1_driver.py stage2  — prepare + LoRA training (caller unloads Ollama first)
  r1_driver.py stage3  — inference + gated feedback (arm B5) and cycle metrics

Arm configuration:
  B5 (default): feedback_gate_enabled=True  — full multi-signal gate.
  B4:           feedback_gate_enabled=False — uniform feedback ablation.

State that must survive a dead runtime lives in OUT (OURO_R1_OUT,
default /content/r1_run): r1_params.json (written by the notebook),
r1_summary.json (per-cycle records), per-stage logs are the notebook's job.

Trajectory isolation: `init` resets the CLONE's data/ state — fresh
ouro_state.json (cycle_count 0), empty corpus/model/graph — and removes
the historical feedback_cycle_*.txt files from the clone's data/seeds so
cycle 0's seed set is exactly the human seed documents (seed1..seed5).
The permanent human seeds stay in data/seeds for every cycle (the anchor);
gated feedback files accumulate alongside them, matching the paper's
accumulate-subsample regime (prepare.py trains on all corpus files).
Nothing here writes outside the clone and OUT.
"""
from __future__ import annotations

import json
import logging
import math
import os
import random
import sqlite3
import sys
import time
from pathlib import Path

REPO = Path.cwd()
OUT = Path(os.environ.get("OURO_R1_OUT", "/content/r1_run"))
OUT.mkdir(parents=True, exist_ok=True)
os.chdir(REPO)
sys.path.insert(0, str(REPO))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

PARAMS = json.loads((OUT / "r1_params.json").read_text())
ARM = PARAMS.get("arm", "B5")
RNG_SEED = int(PARAMS.get("rng_seed", 1))
NUM_AGENTS = int(PARAMS["num_agents"])
ROUNDS = int(PARAMS["rounds"])
TRAIN_STEPS = int(PARAMS["train_steps"])
TRAIN_MODEL = PARAMS["train_model"]
VENV_PY = PARAMS["venv_python"]
GATE_ENABLED = ARM != "B4"

random.seed(RNG_SEED)
os.environ.setdefault("PYTHONHASHSEED", str(RNG_SEED))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("r1-driver")

from ouro.config import OuroConfig  # noqa: E402
from ouro.pipeline import OuroPipeline  # noqa: E402

PROMPT = "Summarize the key predictions from this cycle."


def make_pipeline() -> OuroPipeline:
    config = OuroConfig(
        num_agents=NUM_AGENTS,
        simulation_rounds=ROUNDS,
        train_python=VENV_PY,
        feedback_gate_enabled=GATE_ENABLED,
    )
    assert config.allow_stub_simulation is False, "stub must stay OFF"
    assert config.llm_model == "qwen2.5:1.5b", config.llm_model
    return OuroPipeline(config)


def load_summary() -> dict:
    p = OUT / "r1_summary.json"
    if p.exists():
        return json.loads(p.read_text())
    return {
        "run_spec": "R1",
        "arm": ARM,
        "gate_enabled": GATE_ENABLED,
        "rng_seed": RNG_SEED,
        "seed_domains": ["seed1", "seed2", "seed3", "seed4", "seed5"],
        "agents": NUM_AGENTS,
        "rounds": ROUNDS,
        "train_steps": TRAIN_STEPS,
        "train_model": TRAIN_MODEL,
        "swarm_model": "qwen2.5:1.5b",
        "cycles": [],
    }


def save_summary(summary: dict) -> None:
    (OUT / "r1_summary.json").write_text(json.dumps(summary, indent=2, default=str))


def current_cycle_record(summary: dict, cycle: int) -> dict:
    for rec in summary["cycles"]:
        if rec["cycle"] == cycle:
            return rec
    rec = {"cycle": cycle}
    summary["cycles"].append(rec)
    return rec


# ---------------------------------------------------------------- init
def init() -> None:
    state_path = OUT / "run_state.json"
    if state_path.exists():
        log.info("init already done — %s exists, skipping reset", state_path)
        return
    data = REPO / "data"
    seeds = data / "seeds"
    human = sorted(seeds.glob("seed*.txt"))
    assert [p.name for p in human] == [
        "seed1.txt", "seed2.txt", "seed3.txt", "seed4.txt", "seed5.txt",
    ], f"expected exactly seed1..seed5, found {[p.name for p in human]}"
    for p in seeds.glob("feedback_cycle_*.txt"):
        p.unlink()
    for sub in ("corpus", "model"):
        d = data / sub
        if d.exists():
            for p in d.rglob("*"):
                if p.is_file():
                    p.unlink()
    for extra in ("kuzu_graph",):
        d = data / extra
        if d.exists():
            import shutil
            shutil.rmtree(d)
    for f in ("gate_decisions.jsonl", "research_log.jsonl", "last_cycle.json"):
        p = data / f
        if p.exists():
            p.unlink()
    (data / "ouro_state.json").write_text(
        json.dumps({"cycle_count": 0, "cycles": []}, indent=2))
    state_path.write_text(json.dumps({
        "initialized": True, "arm": ARM, "rng_seed": RNG_SEED,
        "seed_files": [p.name for p in human],
    }, indent=2))
    log.info("INIT DONE — fresh %s trajectory, seeds=%s",
             ARM, [p.name for p in human])


# ---------------------------------------------------------------- stage 1
def stage1() -> None:
    pipeline = make_pipeline()
    summary = load_summary()
    cycle = pipeline.feedback.cycle_count
    rec = current_cycle_record(summary, cycle)
    rec.update({
        "seed_files_present": sorted(
            p.name for p in (REPO / "data/seeds").glob("*.txt")),
    })
    t0 = time.time()
    sim_result = pipeline.run_swarm()
    rec.update({
        "stage1_seconds": round(time.time() - t0, 1),
        "num_turns": sim_result.get("num_turns"),
        "corpus_path": sim_result.get("corpus_path"),
    })
    db_path = (REPO / pipeline.config.corpus_dir
               / f"oasis_trace_cycle_{cycle:04d}.db")
    counts: dict = {}
    if db_path.exists():
        con = sqlite3.connect(db_path)
        try:
            tables = {r[0] for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            if "trace" in tables:
                counts = dict(con.execute(
                    "SELECT action, COUNT(*) FROM trace GROUP BY action"))
        finally:
            con.close()
    rec["trace_actions"] = counts
    (OUT / f"sim_result_cycle_{cycle:02d}.json").write_text(
        json.dumps(sim_result, indent=2, default=str))
    save_summary(summary)
    log.info("STAGE1 DONE cycle=%d in %.1fs — turns=%s actions=%s",
             cycle, rec["stage1_seconds"], sim_result.get("num_turns"), counts)
    if not sim_result.get("num_turns"):
        log.warning("0 corpus turns — check Ollama/swarm tool-calling.")


# ---------------------------------------------------------------- stage 2
def stage2() -> None:
    os.environ["OURO_TRAIN_MAX_STEPS"] = str(TRAIN_STEPS)
    os.environ["OURO_TRAIN_BASE_MODEL"] = TRAIN_MODEL
    pipeline = make_pipeline()
    summary = load_summary()
    cycle = pipeline.feedback.cycle_count
    rec = current_cycle_record(summary, cycle)
    t0 = time.time()
    pipeline.run_autoresearch()
    rec["stage2_seconds"] = round(time.time() - t0, 1)
    meta = json.loads(
        (REPO / "data/model/checkpoint/checkpoint_metadata.json").read_text())
    tr = meta["train_result"]
    losses = tr["loss_history"]
    corpus_dir = REPO / pipeline.config.corpus_dir

    def _lines(p: Path) -> int:
        return sum(1 for _ in p.open()) if p.exists() else 0

    rec.update({
        "train_steps": tr["steps"],
        "train_examples": tr["num_train_examples"],
        "train_file_examples": _lines(corpus_dir / "train.jsonl"),
        "eval_file_examples": _lines(corpus_dir / "eval.jsonl"),
        "train_loss_first": losses[0] if losses else None,
        "train_loss_last": losses[-1] if losses else None,
        "eval_loss_base": meta["eval_loss_base"],
        "eval_perplexity_base": round(math.exp(meta["eval_loss_base"]), 2),
        "eval_loss_trained": meta["eval_loss"],
        "eval_perplexity": meta["eval_perplexity"],
        "adapter_sha256": meta["adapter_sha256"],
    })
    save_summary(summary)
    log.info("STAGE2 DONE cycle=%d in %.1fs — loss %s -> %s, ppl %s -> %s",
             cycle, rec["stage2_seconds"], rec["train_loss_first"],
             rec["train_loss_last"], rec["eval_perplexity_base"],
             rec["eval_perplexity"])


# ---------------------------------------------------------------- metrics
def corpus_metrics(corpus_path: str | None) -> dict:
    """Lexical diversity + duplication for one cycle's corpus file."""
    if not corpus_path or not Path(corpus_path).exists():
        return {}
    texts: list[str] = []
    for line in Path(corpus_path).read_text(encoding="utf-8").splitlines():
        try:
            obj = json.loads(line)
            content = obj.get("content") or obj.get("message") or ""
        except (json.JSONDecodeError, AttributeError):
            continue
        if content:
            texts.append(content)
    if not texts:
        return {"corpus_texts": 0}
    toks = [t.lower().split() for t in texts]
    unigrams = [w for t in toks for w in t]
    bigrams = [tuple(t[i:i + 2]) for t in toks for i in range(len(t) - 1)]
    dup = 1.0 - (len(set(texts)) / len(texts))
    return {
        "corpus_texts": len(texts),
        "distinct_1": round(len(set(unigrams)) / max(1, len(unigrams)), 4),
        "distinct_2": round(len(set(bigrams)) / max(1, len(bigrams)), 4),
        "near_duplicate_rate_exact": round(dup, 4),
        # Tail-entity retention and ECE need the frozen harness / graded
        # outcomes (R4); not computed by this driver. Recorded as pending.
        "tail_entity_retention": None,
        "ece": None,
    }


# ---------------------------------------------------------------- stage 3
def stage3() -> None:
    pipeline = make_pipeline()
    summary = load_summary()
    cycle = pipeline.feedback.cycle_count
    rec = current_cycle_record(summary, cycle)
    sim_result = json.loads((OUT / f"sim_result_cycle_{cycle:02d}.json").read_text())
    t0 = time.time()
    context = pipeline._load_corpus_context(sim_result)
    # Peek at the first candidate's backend/sha via a metadata prediction,
    # then run the pipeline's own gated feedback path (B5) which samples
    # its own candidates. The peek is provenance reporting only.
    seed_path = pipeline._infer_and_feed_back(PROMPT, context, sim_result)
    rec["stage3_seconds"] = round(time.time() - t0, 1)
    state = json.loads((REPO / "data/ouro_state.json").read_text())
    cycle_meta = state["cycles"][-1] if state.get("cycles") else {}
    rec.update({
        "injection_mode": cycle_meta.get("injection_mode"),
        "gate_total_candidates": cycle_meta.get("total_candidates"),
        "gate_selected_count": cycle_meta.get("selected_count"),
        "feedback_seed_files": cycle_meta.get("seed_files"),
        "first_feedback_seed": str(seed_path),
    })
    rec.update(corpus_metrics(rec.get("corpus_path")))
    parts = [rec.get(k) for k in
             ("stage1_seconds", "stage2_seconds", "stage3_seconds")]
    if all(isinstance(x, (int, float)) for x in parts):
        rec["total_seconds"] = round(sum(parts), 1)
    rec["complete"] = True
    save_summary(summary)
    log.info("STAGE3 DONE cycle=%d in %.1fs — mode=%s accepted=%s/%s",
             cycle, rec["stage3_seconds"], rec["injection_mode"],
             rec["gate_selected_count"], rec["gate_total_candidates"])
    log.info("R1 CYCLE %d COMPLETE — summary at %s", cycle,
             OUT / "r1_summary.json")


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"init": init, "stage1": stage1, "stage2": stage2,
     "stage3": stage3}[cmd]()
