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
  B6:           gate on + a guaranteed real-data anchor (Run Spec R2):
                prepare.py mixes human-seed examples into the train
                split at the params' real_fraction of training tokens.
  Gate config:  optional params["gate"] (Run Spec R3 Tier 2) overrides
                individual signal weights and/or disables the hard
                vetoes ("vetoes": false). Absent = the historical gate.

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

import hashlib
import json
import logging
import math
import os
import random
import shutil
import sqlite3
import sys
import time
import zipfile
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
REAL_FRACTION = float(PARAMS.get("real_fraction", 0))
GATE_ENABLED = ARM not in ("B4", "B3")  # B3: gate off, like B4 (R3 §4)
# R3 Tier 2 gate configuration (optional; absent = the historical
# gate, exactly). r1_run.py records only the overrides the operator
# actually passed under params["gate"]: {"weights": {signal: w},
# "vetoes": false}.
_GATE_CFG = PARAMS.get("gate") or {}
GATE_WEIGHT_OVERRIDES = dict(_GATE_CFG.get("weights") or {})
GATE_VETOES = bool(_GATE_CFG.get("vetoes", True))
_GATE_WEIGHT_ATTRS = {
    "outcome": "gate_weight_outcome",
    "diversity": "gate_weight_diversity",
    "consistency": "gate_weight_consistency",
    "provenance": "gate_weight_provenance",
}

random.seed(RNG_SEED)
os.environ.setdefault("PYTHONHASHSEED", str(RNG_SEED))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("r1-driver")

# Stage logs ran to 24–29 MB per cycle, almost all of it library
# internals: the installed camel-oasis sets its social.* loggers to
# DEBUG (each agent's full observation prompt, every step) and httpx
# logs every Ollama request at INFO, and all of it propagates to the
# root handler. Floor the root handler at INFO (drops the DEBUG spam)
# and raise the noisy libraries to WARNING. social.agent and
# social.twitter carry their OWN DEBUG level — verified against the
# camel-oasis installed in the repo venv — which a level set on the
# parent "social" logger cannot override, so they are named
# explicitly. The driver's own lines and the ouro/swarm/autoresearch
# loggers are untouched.
if logging.getLogger().handlers:
    logging.getLogger().handlers[0].setLevel(logging.INFO)
for _lib_logger in ("social", "social.agent", "social.twitter",
                    "social.rec", "oasis", "camel", "httpx", "openai",
                    "urllib3"):
    logging.getLogger(_lib_logger).setLevel(logging.WARNING)

from ouro.config import OuroConfig  # noqa: E402
from ouro.pipeline import OuroPipeline  # noqa: E402

PROMPT = "Summarize the key predictions from this cycle."


def make_pipeline() -> OuroPipeline:
    gate_kwargs = {}
    for key, value in GATE_WEIGHT_OVERRIDES.items():
        attr = _GATE_WEIGHT_ATTRS.get(key)
        if attr is None:
            raise SystemExit(
                f"r1_params.json gate.weights has unknown signal "
                f"{key!r} — expected one of {sorted(_GATE_WEIGHT_ATTRS)}")
        gate_kwargs[attr] = float(value)
    if not GATE_VETOES:
        gate_kwargs["gate_vetoes_enabled"] = False
    config = OuroConfig(
        num_agents=NUM_AGENTS,
        simulation_rounds=ROUNDS,
        train_python=VENV_PY,
        train_base_model=TRAIN_MODEL,
        feedback_gate_enabled=GATE_ENABLED,
        **gate_kwargs,
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
        "real_fraction": REAL_FRACTION,
        "swarm_model": "qwen2.5:1.5b",
        "cycles": [],
        **({"gate_overrides": _GATE_CFG} if _GATE_CFG else {}),
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
    # The guard is keyed on the SUMMARY's completed cycles, not on
    # run_state.json existing: a partial sync/copy of OUT can lose
    # run_state.json while the summary survives, and taking the reset
    # path then would wipe the clone's corpus/feedback state for a
    # trajectory that is still resumable from its snapshot. So: whenever
    # the summary holds completed cycles, init NEVER resets — it either
    # confirms the clone already holds that state or resumes it.
    completed = _completed_cycles(load_summary())
    if completed:
        last = completed[-1]
        if not _clone_state_complete(last):
            log.warning(
                "init: OUT holds %d completed cycle(s) but the clone "
                "does not hold that state — resuming instead of "
                "resetting", len(completed))
            resume()
        else:
            if not state_path.exists():
                human = sorted((REPO / "data/seeds").glob("seed*.txt"))
                state_path.write_text(json.dumps({
                    "initialized": True, "arm": ARM, "rng_seed": RNG_SEED,
                    "seed_files": [p.name for p in human],
                }, indent=2))
            log.info("init: clone already holds state through cycle %d "
                     "— skipping reset", last)
        return
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
    _write_owner()
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
    if ARM == "B3":
        # R3 §4: no simulation — the current checkpoint generates the
        # corpus directly from the human seed documents. No Ollama.
        sim_result = pipeline.run_direct_corpus()
    else:
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
    # R2 anchor: prepare.py (a pipeline subprocess, so it inherits this
    # environment) reads OURO_REAL_FRACTION and mixes human-seed
    # examples into the train split. Set unconditionally — prepare
    # treats "0" exactly as unset.
    os.environ["OURO_REAL_FRACTION"] = str(REAL_FRACTION)
    pipeline = make_pipeline()
    summary = load_summary()
    cycle = pipeline.feedback.cycle_count
    rec = current_cycle_record(summary, cycle)
    if cycle == 0 and "common_eval_perplexity_base" not in summary:
        # Base model on the common eval, once, before any training
        # (R3 §3) — the anchor every trajectory in the grid shares.
        _score_common_eval(rec, None, base=True)
        summary["common_eval_perplexity_base"] = \
            rec["common_eval_perplexity"]
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
    # R2 anchor accounting, from prepare.py's mixture sidecar. With no
    # anchor requested the achieved fraction is 0 by construction (real
    # examples are only ever added by the anchored path).
    rec["real_fraction_target"] = REAL_FRACTION
    if REAL_FRACTION > 0:
        mix_path = REPO / "data/corpus/mixture.json"
        if mix_path.exists():
            mix = json.loads(mix_path.read_text())
            rec["real_fraction_achieved"] = mix.get("achieved_fraction")
            rec["real_examples_added"] = mix.get("real_examples_added")
        else:
            rec["real_fraction_achieved"] = None
            log.warning(
                "stage2: real_fraction=%s but %s is missing — prepare "
                "did not report an achieved fraction", REAL_FRACTION,
                mix_path)
    else:
        rec["real_fraction_achieved"] = 0.0
    frozen_path = REPO / "data/corpus/frozen_eval_groups.json"
    if frozen_path.exists():
        rec["frozen_eval_groups"] = json.loads(frozen_path.read_text())
    _score_common_eval(rec, str(REPO / "data/model/checkpoint"))
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


# ------------------------------------------------------- common eval (R3 §3)
def _score_common_eval(rec: dict, adapter_dir: str | None,
                       base: bool = False) -> None:
    """Score the frozen common eval for the current checkpoint.

    Recorded per cycle as common_eval_perplexity; the base model's
    value (cycle 0, before any training) is also lifted to the summary
    top level as common_eval_perplexity_base. If the eval file is
    absent the field is None with a warning (pre-R3 runs); if it is
    present, an evaluation failure is fatal — the frozen eval is the
    metric of record for R3 and a silent None would corrupt it.
    """
    from ouro.common_eval import (DEFAULT_COMMON_EVAL_PATH,
                                  evaluate_common_eval, load_common_eval)
    if not (REPO / DEFAULT_COMMON_EVAL_PATH).exists():
        log.warning("stage2: %s not found — common_eval_perplexity "
                    "recorded as null", DEFAULT_COMMON_EVAL_PATH)
        rec["common_eval_perplexity"] = None
        return
    rows = load_common_eval(REPO / DEFAULT_COMMON_EVAL_PATH)
    result = evaluate_common_eval(TRAIN_MODEL, adapter_dir,
                                  REPO / DEFAULT_COMMON_EVAL_PATH)
    rec["common_eval_perplexity"] = result["common_eval_perplexity"]
    rec["common_eval_examples"] = result["common_eval_examples"]
    log.info("stage2: common-eval perplexity %s = %s over %d examples",
             "base" if base else "trained",
             result["common_eval_perplexity"], len(rows))


# ---------------------------------------------------------------- stage 3
def stage3() -> None:
    pipeline = make_pipeline()
    summary = load_summary()
    cycle = pipeline.feedback.cycle_count
    rec = current_cycle_record(summary, cycle)
    sim_result = json.loads((OUT / f"sim_result_cycle_{cycle:02d}.json").read_text())
    t0 = time.time()
    context = pipeline._load_corpus_context(sim_result)
    if ARM == "B3":
        # R3 §4: sample and log the cycle's candidates (audit parity
        # with the gated arms) but write NO feedback seeds — in B3
        # the next cycle's direct corpus generation is the feedback.
        candidates = pipeline.sample_prediction_candidates(
            PROMPT, context, sim_result)
        (OUT / f"b3_candidates_cycle_{cycle:02d}.json").write_text(
            json.dumps(candidates, indent=2, default=str))
        pipeline.feedback.record_cycle_without_feedback(
            sim_result, candidates)
        seed_path = Path()
    else:
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


# ---------------------------------------------------------------- resume
# Trajectory state that must survive a dead runtime. Everything the next
# cycle reads lives in the clone's data/ tree:
#   data/ouro_state.json               cycle_count + per-cycle feedback meta
#   data/seeds/feedback_cycle_*.txt    accumulated feedback seeds — the
#                                      recursion channel itself
#   data/corpus/corpus_cycle_*.jsonl   accumulated corpus; prepare.py
#                                      rebuilds train/eval.jsonl from these
#   data/corpus/frozen_eval_groups.json  eval groups frozen at cycle 0;
#                                      without it prepare.py would freeze a
#                                      different split from the full corpus
#   data/gate_decisions.jsonl          B5 gate audit log (continuity only)
#   data/model/checkpoint/checkpoint_metadata.json  informational — stage 2
#                                      overwrites it; training always starts
#                                      from the base model, so the LoRA
#                                      adapter weights are per-cycle outputs,
#                                      NOT trajectory state, and are excluded.
#                                      EXCEPTION (B3): the adapter generates
#                                      the next cycle's corpus, so for B3
#                                      the whole checkpoint dir IS state
#                                      and is snapshotted/restored.
# Also excluded: data/kuzu_graph (a derived cache, rebuilt from the seed
# documents on every stage 1) and train/eval.jsonl (regenerated
# deterministically by prepare.py: fixed split seed + the frozen groups).
# No RNG state needs restoring: every stage is a fresh process that
# re-seeds random with RNG_SEED, exactly as in the notebook flow.
#
# A state source is one of:
#   OUT/state/            written by `r1_driver.py snapshot` after a cycle
#   OUT/*checkpoint*.zip  the notebook checkpoint layout (run/ + repo/)
# The loose per-cycle OUT files alone (summary/params/logs) are NOT enough
# to resume — they never contained the clone's data/ state.

def _completed_cycles(summary: dict) -> list[int]:
    return sorted(int(r["cycle"]) for r in summary.get("cycles", [])
                  if r.get("complete"))


def _clone_cycle_count() -> int:
    p = REPO / "data/ouro_state.json"
    if p.exists():
        try:
            return int(json.loads(p.read_text()).get("cycle_count", 0))
        except (json.JSONDecodeError, ValueError, TypeError):
            return 0
    return 0


FIXED_STATE_FILES = [
    "data/ouro_state.json",
    "data/corpus/frozen_eval_groups.json",
    "data/gate_decisions.jsonl",
    "data/model/checkpoint/checkpoint_metadata.json",
]
# Per-cycle state files. Feedback naming depends on the arm's injection
# mode (feedback.py): feedback_cycle_{i:04d}.txt (uniform),
# ..._pred_{NN}.txt, ..._cand_{NN}.txt (gated) — hence prefix matching.
_CYCLE_PATTERNS = (("data/seeds", "feedback_cycle_"),
                   ("data/corpus", "corpus_cycle_"))


def _cycle_index(rel: str) -> int | None:
    """Cycle index of a per-cycle state file, or None for fixed files."""
    for sub, prefix in _CYCLE_PATTERNS:
        if rel.startswith(f"{sub}/{prefix}"):
            token = rel[len(sub) + len(prefix) + 1:].split("_")[0].split(".")[0]
            try:
                return int(token)
            except ValueError:
                return None
    return None


def _expected_state_files(last: int) -> list[str]:
    rels = list(FIXED_STATE_FILES)
    for i in range(last + 1):
        rels.append(f"data/corpus/corpus_cycle_{i:04d}.jsonl")
    return rels


def _required_state_files(last: int) -> list[str]:
    """Files without which a resumed trajectory would diverge or restart."""
    rels = ["data/ouro_state.json", "data/corpus/frozen_eval_groups.json"]
    for i in range(last + 1):
        rels.append(f"data/corpus/corpus_cycle_{i:04d}.jsonl")
    return rels


def _read_state_dir(root: Path) -> dict[str, bytes]:
    state: dict[str, bytes] = {}
    for rel in FIXED_STATE_FILES:
        p = root / rel
        if p.exists():
            state[rel] = p.read_bytes()
    for sub, pat in (("data/seeds", "feedback_cycle_*.txt"),
                     ("data/corpus", "corpus_cycle_*.jsonl")):
        d = root / sub
        if d.exists():
            for p in d.glob(pat):
                state[str(p.relative_to(root))] = p.read_bytes()
    if ARM == "B3":
        # B3 only: the adapter weights are trajectory state (they
        # generate the next cycle's corpus), so they are read here
        # too. For every other arm training restarts from the base
        # model each cycle and the adapter is a per-cycle output.
        d = root / "data/model/checkpoint"
        if d.exists():
            for p in sorted(d.iterdir()):
                if p.is_file():
                    state[str(p.relative_to(root))] = p.read_bytes()
    return state


def _is_state_rel(rel: str) -> bool:
    """Allowlist: fixed state files + per-cycle feedback/corpus files.

    Anything else in a checkpoint zip (LoRA adapter weights, the kuzu
    graph, train/eval.jsonl, ...) is a per-cycle output or a derived
    cache, NOT trajectory state, and is never restored. Exception:
    for B3 the adapter weights ARE trajectory state — they generate
    the next cycle's corpus — so the checkpoint dir is allowlisted
    when ARM is B3.
    """
    if rel in FIXED_STATE_FILES:
        return True
    if ARM == "B3" and rel.startswith("data/model/checkpoint/"):
        return True
    return any(rel.startswith(f"{sub}/{prefix}")
               for sub, prefix in _CYCLE_PATTERNS)


def _read_state_zip(zpath: Path) -> dict[str, bytes]:
    state: dict[str, bytes] = {}
    with zipfile.ZipFile(zpath) as z:
        for name in z.namelist():
            if "/repo/" in name:
                rel = name.split("/repo/", 1)[1]
            elif name.startswith("repo/"):
                rel = name[len("repo/"):]
            else:
                continue
            # Zip-slip guard: never let an entry escape the repo tree,
            # and restore only allowlisted state files.
            if rel.endswith("/") or ".." in Path(rel).parts or \
                    Path(rel).is_absolute() or not _is_state_rel(rel):
                continue
            state[rel] = z.read(name)
    return state


def _sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _write_owner() -> None:
    """Mark the clone's data/ as belonging to this trajectory.

    Written by init (fresh reset), snapshot and resume. A clone whose
    owner marker names a different arm/seed/OUT is another trajectory's
    checkout and must never be mistaken for this one.
    """
    data = REPO / "data"
    data.mkdir(parents=True, exist_ok=True)
    (data / ".r1_owner.json").write_text(json.dumps({
        "arm": ARM, "rng_seed": RNG_SEED, "out": str(OUT),
    }, indent=2))


def _clone_owner_ok() -> bool:
    p = REPO / "data/.r1_owner.json"
    if not p.exists():
        return False
    try:
        owner = json.loads(p.read_text())
        return owner.get("arm") == ARM and \
            int(owner.get("rng_seed", -1)) == RNG_SEED and \
            owner.get("out") == str(OUT)
    except (json.JSONDecodeError, ValueError, TypeError):
        return False


def _state_problems(state: dict[str, bytes],
                    summary_last: int) -> tuple[int | None, list[str]]:
    """Validate a candidate state snapshot against the summary.

    Returns (snap_last, problems): snap_last is the snapshot's own last
    completed cycle (its ouro_state cycle_count - 1). A snapshot ahead
    of the summary is rejected; one behind it is fine — resume rolls
    the summary back to it (see resume()).
    """
    problems: list[str] = []
    raw = state.get("data/ouro_state.json")
    if raw is None:
        return None, ["data/ouro_state.json not in snapshot"]
    try:
        snap_state = json.loads(raw)
        snap_cc = int(snap_state.get("cycle_count", -1))
    except (json.JSONDecodeError, ValueError, TypeError):
        return None, ["snapshot ouro_state.json unreadable or "
                      "cycle_count not an int"]
    snap_last = snap_cc - 1
    if snap_last < 0:
        problems.append(f"snapshot cycle_count={snap_cc} — no completed "
                        "cycle to resume from")
        return None, problems
    if snap_last > summary_last:
        problems.append(
            f"snapshot covers cycle {snap_last} but the summary's last "
            f"completed cycle is {summary_last} — snapshot is ahead of "
            "the summary, refusing to trust either")
    for rel in _required_state_files(snap_last):
        if rel not in state:
            problems.append(f"missing {rel}")
    # B3 writes no feedback seeds by construction (R3 §4) — its
    # recursion channel is the next cycle's directly generated corpus.
    if ARM != "B3":
        for i in range(snap_last + 1):
            if not any(rel.startswith(f"data/seeds/feedback_cycle_{i:04d}")
                       for rel in state):
                problems.append(
                    f"missing feedback seed file(s) for cycle {i}")
    # The snapshot's own ouro_state names the exact seed file(s) each
    # cycle produced; every one of them must be in the snapshot. Paths
    # are normalised to data/seeds/<basename>: a snapshot produced
    # under a different seed_docs_dir (e.g. an absolute /content/...
    # path from a Colab run) names the same files.
    for cyc in snap_state.get("cycles", []):
        try:
            if int(cyc.get("cycle", 0)) > snap_last:
                continue
        except (ValueError, TypeError):
            continue
        named = cyc.get("seed_files") or ([cyc["seed_file"]]
                                          if cyc.get("seed_file") else [])
        for rel in named:
            norm = f"data/seeds/{Path(rel).name}"
            if rel not in state and norm not in state:
                problems.append(f"missing {rel} (named in ouro_state)")
    return snap_last, problems


def _seed_matches(value) -> bool:
    try:
        return value is not None and int(value) == RNG_SEED
    except (ValueError, TypeError):
        return False


def _with_loose_frozen(state: dict[str, bytes]) -> dict[str, bytes]:
    """The watch's per-cycle Drive sync also drops a loose copy of the
    frozen groups directly in OUT; accept it as a fallback for that
    one file if the snapshot itself lacks it."""
    frozen_rel = "data/corpus/frozen_eval_groups.json"
    if frozen_rel not in state and (OUT / "frozen_eval_groups.json").exists():
        state = dict(state)
        state[frozen_rel] = (OUT / "frozen_eval_groups.json").read_bytes()
    return state


def _validate_dir_source(path: Path,
                         summary_last: int) -> tuple[dict, int | None,
                                                      list[str]]:
    meta_path = path / "snapshot_meta.json"
    if not meta_path.exists():
        return {}, None, [f"{path}: no snapshot_meta.json"]
    try:
        meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return {}, None, [f"{path}: snapshot_meta.json unreadable"]
    problems: list[str] = []
    if meta.get("arm") != ARM or not _seed_matches(meta.get("rng_seed")):
        problems.append(
            f"{path}: snapshot_meta is arm={meta.get('arm')} "
            f"seed={meta.get('rng_seed')}, expected arm={ARM} "
            f"seed={RNG_SEED}")
    state = _with_loose_frozen(_read_state_dir(path))
    snap_last, probs = _state_problems(state, summary_last)
    problems.extend(f"{path}: {p}" for p in probs)
    if snap_last is not None and \
            meta.get("last_completed_cycle") != snap_last:
        problems.append(
            f"{path}: snapshot_meta last_completed_cycle="
            f"{meta.get('last_completed_cycle')} but ouro_state says "
            f"{snap_last}")
    hashes = meta.get("sha256")
    if hashes:
        for rel, want in hashes.items():
            if rel not in state:
                problems.append(f"{path}: {rel} listed in snapshot_meta "
                                "hashes but absent")
            elif _sha256(state[rel]) != want:
                problems.append(f"{path}: sha256 mismatch for {rel} — "
                                "file is truncated or corrupt")
    else:
        # Legacy metas (pre-hash snapshots) list files without hashes;
        # existence is all that can be checked for them.
        for rel in meta.get("files", []):
            if rel not in state:
                problems.append(f"{path}: {rel} listed in snapshot_meta "
                                "but absent")
    return state, snap_last, problems


def _zip_run_params(z: zipfile.ZipFile) -> dict | None:
    for name in z.namelist():
        if name == "run/r1_params.json" or \
                name.endswith("/run/r1_params.json"):
            try:
                return json.loads(z.read(name))
            except (json.JSONDecodeError, KeyError):
                return None
    return None


def _validate_zip_source(path: Path,
                         summary_last: int) -> tuple[dict, int | None,
                                                      list[str]]:
    try:
        with zipfile.ZipFile(path) as z:
            params = _zip_run_params(z)
    except zipfile.BadZipFile:
        return {}, None, [f"{path}: not a readable zip (truncated?)"]
    problems: list[str] = []
    if params is None:
        problems.append(f"{path}: no readable run/r1_params.json inside "
                        "— cannot verify which trajectory this is")
    elif params.get("arm") != ARM or \
            not _seed_matches(params.get("rng_seed")):
        problems.append(
            f"{path}: zip params are arm={params.get('arm')} "
            f"seed={params.get('rng_seed')}, expected arm={ARM} "
            f"seed={RNG_SEED}")
    state = _with_loose_frozen(_read_state_zip(path))
    snap_last, probs = _state_problems(state, summary_last)
    problems.extend(f"{path}: {p}" for p in probs)
    return state, snap_last, problems


def _find_valid_source(summary_last: int) -> tuple[dict, int, Path,
                                                   list[str]] | None:
    """Pick the best VALIDATED state source in OUT.

    Candidates: OUT/state/, OUT/state.prev/ (the previous snapshot,
    kept by the atomic swap in snapshot()), then every checkpoint zip.
    A candidate must prove it belongs to this arm+seed and be
    internally consistent; the winner is the one covering the most
    cycles (ties prefer the live state dir). Returns (state, snap_last,
    source_path, rejections) or None plus all rejection reasons.
    """
    candidates: list[tuple[str, Path]] = []
    for name in ("state", "state.prev"):
        d = OUT / name
        if (d / "data/ouro_state.json").exists() or \
                (d / "snapshot_meta.json").exists():
            candidates.append(("dir", d))
    for zp in sorted(OUT.glob("*checkpoint*.zip")):
        candidates.append(("zip", zp))
    valid: list[tuple[int, int, dict, Path]] = []
    rejections: list[str] = []
    for order, (kind, path) in enumerate(candidates):
        if kind == "dir":
            state, snap_last, problems = _validate_dir_source(
                path, summary_last)
        else:
            state, snap_last, problems = _validate_zip_source(
                path, summary_last)
        if problems:
            rejections.extend(problems)
        elif snap_last is not None:
            valid.append((snap_last, -order, state, path))
    if not valid:
        return None
    snap_last, _neg_order, state, path = max(valid, key=lambda v: (v[0], v[1]))
    # Drop per-cycle files from cycles beyond the snapshot's own last
    # cycle — they belong to an interrupted later cycle.
    state = {rel: blob for rel, blob in state.items()
             if (_cycle_index(rel) is None or _cycle_index(rel) <= snap_last)}
    return state, snap_last, path, rejections


def _has_cycle_feedback(root: Path, cycle: int) -> bool:
    d = root / "data/seeds"
    return d.exists() and any(
        p.name.startswith(f"feedback_cycle_{cycle:04d}")
        for p in d.glob("feedback_cycle_*.txt"))


def _clone_state_complete(last: int) -> bool:
    """Existence-level check, used by the init() guard: right cycle
    count, required files present, feedback for every cycle, AND the
    clone's owner marker naming this trajectory. (resume() applies the
    stronger byte-for-byte check against a validated snapshot.)"""
    if _clone_cycle_count() != last + 1:
        return False
    if not _clone_owner_ok():
        return False
    if not all((REPO / rel).exists()
               for rel in _required_state_files(last)):
        return False
    if ARM == "B3":
        return True  # B3 writes no feedback seeds by construction.
    return all(_has_cycle_feedback(REPO, i) for i in range(last + 1))


def _clone_matches_state(state: dict[str, bytes], last: int) -> bool:
    """Byte-for-byte check: every file in the snapshot is present in
    the clone with identical content, and the cycle count agrees.
    This is what 'already-restored' requires — a matching cycle count
    alone proves nothing about WHICH trajectory the clone holds."""
    if _clone_cycle_count() != last + 1:
        return False
    for rel, blob in state.items():
        p = REPO / rel
        if not p.exists() or p.read_bytes() != blob:
            return False
    return True


def _cleanup_clone_state(state: dict[str, bytes]) -> None:
    """Delete EVERY per-cycle feedback/corpus file in the clone that
    is not part of the state being restored — any cycle index, not
    just beyond the last completed one. Files from a foreign or
    interrupted run left alongside the restored ones would otherwise
    be globbed into the next cycle's training/feedback inputs."""
    for sub, pat in (("data/seeds", "feedback_cycle_*.txt"),
                     ("data/corpus", "corpus_cycle_*.jsonl")):
        d = REPO / sub
        if d.exists():
            for p in d.glob(pat):
                if str(p.relative_to(REPO)) not in state:
                    p.unlink()


def snapshot() -> None:
    """Copy the clone's trajectory state into OUT/state/ after a cycle.

    Run after stage 3. Unlike the notebook checkpoint zip (end-of-run in
    practice), this lands in OUT every cycle, so any per-cycle sync of OUT
    — Drive folder, pod disk, scp — carries enough state to resume.

    The snapshot is ATOMIC and REFUSES inconsistent state: if the
    clone's cycle_count is not exactly last_completed + 1, the state is
    mid-cycle or divergent, and overwriting the last good snapshot with
    it would destroy the only resumable copy — so snapshot exits
    non-zero and leaves OUT/state/ untouched. Otherwise the new
    snapshot is built in OUT/state.tmp/ (snapshot_meta.json, carrying a
    sha256 per file, written last) and swapped into place, with the
    previous snapshot kept as OUT/state.prev/ until the swap completes;
    resume() accepts state.prev/ as a fallback source.
    """
    summary = load_summary()
    completed = _completed_cycles(summary)
    if not completed:
        log.warning("snapshot: no completed cycles yet — nothing to snapshot")
        return
    last = completed[-1]
    cc = _clone_cycle_count()
    if cc != last + 1:
        raise SystemExit(
            f"snapshot: clone cycle_count={cc} but the last completed "
            f"cycle is {last} (expected cycle_count {last + 1}) — the "
            "clone state is mid-cycle or divergent; refusing to "
            "overwrite the last good snapshot in OUT/state/")
    _write_owner()
    rels: list[str] = [rel for rel in _expected_state_files(last)
                       if (REPO / rel).exists()]
    # Feedback files: names vary by injection mode, so glob them; keep
    # only cycles <= last (a partial file from an interrupted later cycle
    # must not leak into the snapshot).
    seeds_dir = REPO / "data/seeds"
    if seeds_dir.exists():
        for p in sorted(seeds_dir.glob("feedback_cycle_*.txt")):
            rel = f"data/seeds/{p.name}"
            idx = _cycle_index(rel)
            if idx is not None and idx <= last:
                rels.append(rel)
    if ARM == "B3":
        # B3 only: snapshot the adapter weights — they generate the
        # next cycle's corpus, so a resume without them would
        # silently fall back to the base model (run_direct_corpus
        # refuses that case, but the snapshot is what makes a
        # cross-pod B3 resume possible at all).
        ckpt_dir = REPO / "data/model/checkpoint"
        if ckpt_dir.exists():
            for p in sorted(ckpt_dir.iterdir()):
                if p.is_file():
                    rels.append(f"data/model/checkpoint/{p.name}")
    tmp = OUT / "state.tmp"
    prev = OUT / "state.prev"
    dst = OUT / "state"
    if tmp.exists():
        shutil.rmtree(tmp)
    # Files that change every cycle are always re-copied. Anything else
    # (earlier cycles' corpus files above all) is skipped when the live
    # snapshot already holds the identical file — same size and mtime —
    # by hardlinking it into the new snapshot instead of copying the
    # bytes again; on a Drive/FUSE-backed OUT every re-copy is a
    # re-upload. copy2 preserves the source mtime, so an unchanged file
    # keeps comparing equal on later snapshots. Either way, the sha256
    # recorded in snapshot_meta.json covers EVERY snapshotted file.
    _always_copy = {
        "data/ouro_state.json",
        "data/gate_decisions.jsonl",
        "data/model/checkpoint/checkpoint_metadata.json",
    }
    copied: list[str] = []
    hashes: dict[str, str] = {}
    for rel in rels:
        src = REPO / rel
        d = tmp / rel
        d.parent.mkdir(parents=True, exist_ok=True)
        linked = False
        if rel not in _always_copy:
            live = dst / rel
            if live.exists():
                s_src, s_live = src.stat(), live.stat()
                if s_src.st_size == s_live.st_size and \
                        s_src.st_mtime_ns == s_live.st_mtime_ns:
                    try:
                        os.link(live, d)
                        linked = True
                    except OSError:
                        linked = False
        if not linked:
            shutil.copy2(src, d)
        copied.append(rel)
        hashes[rel] = _sha256(d.read_bytes())
    meta = {
        "arm": ARM, "rng_seed": RNG_SEED, "last_completed_cycle": last,
        "clone_cycle_count": cc, "files": copied, "sha256": hashes,
        "written_at": time.time(),
    }
    (tmp / "snapshot_meta.json").write_text(json.dumps(meta, indent=2))
    if prev.exists():
        shutil.rmtree(prev)
    if dst.exists():
        os.rename(dst, prev)
    os.rename(tmp, dst)
    if prev.exists():
        shutil.rmtree(prev)
    log.info("SNAPSHOT DONE — %d state files for cycles 0..%d in %s",
             len(copied), last, dst)


def resume(dry_run: bool = False) -> dict:
    """Restore trajectory state into the clone so the next stage 1 starts
    at cycle (last completed + 1) instead of restarting from cycle 0.

    The state source must PROVE it belongs to this trajectory: a dir
    snapshot's snapshot_meta.json and a zip's inner run/r1_params.json
    must name this arm+seed, the snapshot must be internally consistent
    (for dir snapshots, every file's sha256 must match its meta), and
    its cycle coverage decides where the run resumes from. If the
    summary claims cycles beyond the snapshot (stage 3 marked a cycle
    complete, then the box died before snapshot ran), those records are
    moved to the summary's `abandoned_cycles` list — never deleted —
    and the run resumes from the snapshot.
    """
    summary = load_summary()
    completed = _completed_cycles(summary)
    report: dict = {"arm": ARM, "rng_seed": RNG_SEED,
                    "completed_cycles": completed,
                    "clone_cycle_count_before": _clone_cycle_count()}
    if not completed:
        report["status"] = "nothing-to-resume"
        log.info("resume: no completed cycles in summary — nothing to do")
        print(json.dumps(report, indent=2))
        return report
    # Strict identity check: both keys must be present and match. (The
    # driver always writes them, so an absent key means a foreign or
    # hand-edited summary — not something to guess around.)
    sum_seed = summary.get("rng_seed")
    try:
        sum_seed_ok = sum_seed is not None and int(sum_seed) == RNG_SEED
    except (ValueError, TypeError):
        sum_seed_ok = False
    if summary.get("arm") != ARM or not sum_seed_ok:
        raise SystemExit(
            f"resume: summary is arm={summary.get('arm')} "
            f"seed={summary.get('rng_seed')} but r1_params.json says "
            f"arm={ARM} seed={RNG_SEED} — refusing to mix trajectories")
    summary_last = completed[-1]
    found = _find_valid_source(summary_last)
    if found is None:
        # No validated snapshot anywhere. Last resort: the clone itself
        # already holds this trajectory's state (e.g. a notebook run
        # whose OUT was never snapshotted by this driver). Adopt it —
        # write the owner marker so future checks can verify it — but
        # only on the full existence check against the summary's arm
        # and seed, which were verified above.
        if _clone_cycle_count() == summary_last + 1 and \
                all((REPO / rel).exists()
                    for rel in _required_state_files(summary_last)) and \
                all(_has_cycle_feedback(REPO, i)
                    for i in range(summary_last + 1)):
            if not dry_run:
                _write_owner()
            report["status"] = "already-restored"
            report["next_cycle"] = summary_last + 1
            report["note"] = ("no snapshot source in OUT; clone state "
                              "adopted on existence checks + owner "
                              "marker written")
            log.warning("resume: no snapshot source in OUT, but the "
                        "clone holds state through cycle %d — adopting "
                        "it", summary_last)
            print(json.dumps(report, indent=2))
            return report
        raise SystemExit(
            f"resume: summary holds {len(completed)} completed cycle(s) "
            f"(0..{summary_last}) but no state source in OUT validated "
            f"for arm={ARM} seed={RNG_SEED} (looked at {OUT}/state/, "
            f"{OUT}/state.prev/, {OUT}/*checkpoint*.zip) and the clone "
            "does not hold the state either. The loose OUT files "
            "(summary/params/logs) never contained the clone's data/ "
            "state, so this trajectory cannot be resumed faithfully; "
            "it would have to restart from cycle 0.")
    state, last, source_path, _rejections = found
    report["source"] = str(source_path)
    report["files"] = sorted(state)
    # Roll the summary back to the snapshot if it claims more: stage 3
    # writes `complete: true` BEFORE the runner snapshots, so a death
    # in that window leaves the summary one cycle ahead of the newest
    # state. The extra records are preserved in `abandoned_cycles`.
    abandoned = [r for r in summary.get("cycles", [])
                 if int(r["cycle"]) > last and r.get("complete")]
    if abandoned:
        report["abandoned_cycles"] = [int(r["cycle"]) for r in abandoned]
        if not dry_run:
            summary.setdefault("abandoned_cycles", []).extend(abandoned)
            summary["cycles"] = [
                r for r in summary["cycles"]
                if not (int(r["cycle"]) > last and r.get("complete"))]
            save_summary(summary)
            log.warning("resume: summary claimed cycle(s) %s complete "
                        "beyond the snapshot's cycle %d — moved to "
                        "abandoned_cycles; resuming from the snapshot",
                        report["abandoned_cycles"], last)
    if _clone_matches_state(state, last):
        # Already restored — byte-for-byte identical to the validated
        # snapshot. Still run the foreign-file cleanup: partial files
        # from an interrupted later cycle must not survive just
        # because the restore itself is a no-op.
        if not dry_run:
            _cleanup_clone_state(state)
            _write_owner()
        report["status"] = "already-restored"
        report["next_cycle"] = last + 1
        log.info("resume: clone already holds the snapshot state "
                 "through cycle %d", last)
        print(json.dumps(report, indent=2))
        return report
    if dry_run:
        report["status"] = "would-resume"
        report["next_cycle"] = last + 1
        log.info("resume (dry-run): would restore %d files from %s; "
                 "next cycle = %d", len(state), source_path, last + 1)
        print(json.dumps(report, indent=2))
        return report
    _cleanup_clone_state(state)
    for rel, blob in state.items():
        d = REPO / rel
        d.parent.mkdir(parents=True, exist_ok=True)
        d.write_bytes(blob)
    _write_owner()
    if not (OUT / "run_state.json").exists():
        human = sorted((REPO / "data/seeds").glob("seed*.txt"))
        (OUT / "run_state.json").write_text(json.dumps({
            "initialized": True, "arm": ARM, "rng_seed": RNG_SEED,
            "seed_files": [p.name for p in human],
        }, indent=2))
    report["status"] = "resumed"
    report["next_cycle"] = last + 1
    report["clone_cycle_count_after"] = _clone_cycle_count()
    log.info("RESUME DONE — restored %d files from %s; clone cycle_count=%d, "
             "next cycle = %d", len(state), source_path,
             report["clone_cycle_count_after"], last + 1)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "resume":
        resume(dry_run="--dry-run" in sys.argv[2:])
    elif cmd == "snapshot":
        snapshot()
    else:
        {"init": init, "stage1": stage1, "stage2": stage2,
         "stage3": stage3}[cmd]()
