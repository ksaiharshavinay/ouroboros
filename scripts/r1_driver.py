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
        # Trap: OUT survived a dead runtime (or was restored from a
        # snapshot) but the clone is a fresh checkout — run_state says
        # "initialized" while the clone's cycle counter is 0 and the
        # summary already holds completed cycles. Silently no-oping here
        # would restart the trajectory at cycle 0 and overwrite the
        # completed cycles' records, so restore the trajectory instead.
        completed = _completed_cycles(load_summary())
        if completed and _clone_cycle_count() == 0:
            log.warning(
                "init: OUT holds %d completed cycle(s) but the clone is "
                "fresh — resuming instead of resetting", len(completed))
            resume()
        else:
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
    frozen_path = REPO / "data/corpus/frozen_eval_groups.json"
    if frozen_path.exists():
        rec["frozen_eval_groups"] = json.loads(frozen_path.read_text())
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
    return state


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
            if rel.startswith("data/") and not rel.endswith("/"):
                state[rel] = z.read(name)
    return state


def _find_state_source() -> tuple[str, Path] | None:
    if (OUT / "state" / "data/ouro_state.json").exists():
        return ("dir", OUT / "state")
    zips = sorted(OUT.glob("*checkpoint*.zip"),
                  key=lambda p: p.stat().st_mtime)
    if zips:
        return ("zip", zips[-1])
    return None


def _read_state(source: tuple[str, Path]) -> dict[str, bytes]:
    kind, path = source
    return _read_state_dir(path) if kind == "dir" else _read_state_zip(path)


def _has_cycle_feedback(root: Path, cycle: int) -> bool:
    d = root / "data/seeds"
    return d.exists() and any(
        p.name.startswith(f"feedback_cycle_{cycle:04d}")
        for p in d.glob("feedback_cycle_*.txt"))


def _clone_state_complete(last: int) -> bool:
    if _clone_cycle_count() != last + 1:
        return False
    if not all((REPO / rel).exists()
               for rel in _required_state_files(last)):
        return False
    return all(_has_cycle_feedback(REPO, i) for i in range(last + 1))


def snapshot() -> None:
    """Copy the clone's trajectory state into OUT/state/ after a cycle.

    Run after stage 3. Unlike the notebook checkpoint zip (end-of-run in
    practice), this lands in OUT every cycle, so any per-cycle sync of OUT
    — Drive folder, pod disk, scp — carries enough state to resume.
    """
    summary = load_summary()
    completed = _completed_cycles(summary)
    if not completed:
        log.warning("snapshot: no completed cycles yet — nothing to snapshot")
        return
    last = completed[-1]
    cc = _clone_cycle_count()
    if cc != last + 1:
        log.warning(
            "snapshot: clone cycle_count=%d, last completed cycle=%d — "
            "state may be mid-cycle; snapshotting anyway", cc, last)
    dst = OUT / "state"
    copied: list[str] = []
    for rel in _expected_state_files(last):
        src = REPO / rel
        if src.exists():
            d = dst / rel
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, d)
            copied.append(rel)
    # Feedback files: names vary by injection mode, so glob them; keep
    # only cycles <= last (a partial file from an interrupted later cycle
    # must not leak into the snapshot).
    seeds_dir = REPO / "data/seeds"
    if seeds_dir.exists():
        for p in sorted(seeds_dir.glob("feedback_cycle_*.txt")):
            rel = f"data/seeds/{p.name}"
            idx = _cycle_index(rel)
            if idx is not None and idx <= last:
                d = dst / rel
                d.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(p, d)
                copied.append(rel)
    # Prune state files from cycles beyond `last` (stale leftovers if OUT
    # was reused after a longer snapshot).
    for sub, pat in (("data/seeds", "feedback_cycle_*.txt"),
                     ("data/corpus", "corpus_cycle_*.jsonl")):
        d = dst / sub
        if d.exists():
            for p in d.glob(pat):
                idx = _cycle_index(str(p.relative_to(dst)))
                if idx is not None and idx > last:
                    p.unlink()
    meta = {
        "arm": ARM, "rng_seed": RNG_SEED, "last_completed_cycle": last,
        "clone_cycle_count": cc, "files": copied, "written_at": time.time(),
    }
    (dst / "snapshot_meta.json").write_text(json.dumps(meta, indent=2))
    log.info("SNAPSHOT DONE — %d state files for cycles 0..%d in %s",
             len(copied), last, dst)


def resume(dry_run: bool = False) -> dict:
    """Restore trajectory state into the clone so the next stage 1 starts
    at cycle (last completed + 1) instead of restarting from cycle 0."""
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
    last = completed[-1]
    if summary.get("arm") not in (None, ARM) or \
            int(summary.get("rng_seed", RNG_SEED)) != RNG_SEED:
        raise SystemExit(
            f"resume: summary is arm={summary.get('arm')} "
            f"seed={summary.get('rng_seed')} but r1_params.json says "
            f"arm={ARM} seed={RNG_SEED} — refusing to mix trajectories")
    if _clone_state_complete(last):
        report["status"] = "already-restored"
        log.info("resume: clone already holds state through cycle %d", last)
        print(json.dumps(report, indent=2))
        return report
    source = _find_state_source()
    if source is None:
        raise SystemExit(
            f"resume: summary holds {len(completed)} completed cycle(s) "
            f"(0..{last}) but OUT has no trajectory state snapshot "
            f"(looked for {OUT}/state/ and {OUT}/*checkpoint*.zip). "
            "Missing state: " + ", ".join(_required_state_files(last)) +
            ", plus the per-cycle feedback seed files under data/seeds/ "
            "— the loose OUT files (summary/params/logs) never contained "
            "the clone's data/ state, so this trajectory cannot be resumed "
            "faithfully; it would have to restart from cycle 0.")
    state = _read_state(source)
    # Drop per-cycle files from cycles beyond `last` — they belong to an
    # interrupted later cycle and must not leak into the restored state.
    state = {rel: blob for rel, blob in state.items()
             if (_cycle_index(rel) is None or _cycle_index(rel) <= last)}
    # The watch's per-cycle Drive sync also drops a loose copy of the
    # frozen groups directly in OUT; accept it as a fallback for that file.
    frozen_rel = "data/corpus/frozen_eval_groups.json"
    if frozen_rel not in state and (OUT / "frozen_eval_groups.json").exists():
        state[frozen_rel] = (OUT / "frozen_eval_groups.json").read_bytes()
    problems: list[str] = []
    raw = state.get("data/ouro_state.json")
    snap_state: dict = {}
    if raw is None:
        problems.append("data/ouro_state.json not in snapshot")
    else:
        try:
            snap_state = json.loads(raw)
            snap_cc = int(snap_state.get("cycle_count", -1))
        except (json.JSONDecodeError, ValueError, TypeError):
            snap_cc = -1
        if snap_cc != last + 1:
            problems.append(
                f"snapshot ouro_state cycle_count={snap_cc}, expected "
                f"{last + 1} (last completed cycle {last} + 1)")
    for rel in _required_state_files(last):
        if rel not in state:
            problems.append(f"missing {rel}")
    for i in range(last + 1):
        if not any(rel.startswith(f"data/seeds/feedback_cycle_{i:04d}")
                   for rel in state):
            problems.append(f"missing feedback seed file(s) for cycle {i}")
    # The snapshot's own ouro_state names the exact seed file(s) each
    # cycle produced; every one of them must be in the snapshot.
    for cyc in snap_state.get("cycles", []):
        if int(cyc.get("cycle", 0)) > last:
            continue
        named = cyc.get("seed_files") or ([cyc["seed_file"]]
                                          if cyc.get("seed_file") else [])
        for rel in named:
            if rel not in state:
                problems.append(f"missing {rel} (named in ouro_state)")
    if problems:
        raise SystemExit(
            f"resume: state source {source[1]} is incomplete for resuming "
            f"after cycle {last}: " + "; ".join(problems))
    report["source"] = str(source[1])
    report["files"] = sorted(state)
    if dry_run:
        report["status"] = "would-resume"
        report["next_cycle"] = last + 1
        log.info("resume (dry-run): would restore %d files from %s; "
                 "next cycle = %d", len(state), source[1], last + 1)
        print(json.dumps(report, indent=2))
        return report
    for rel, blob in state.items():
        d = REPO / rel
        d.parent.mkdir(parents=True, exist_ok=True)
        d.write_bytes(blob)
    # Defensive cleanup: drop partial-cycle leftovers beyond `last` so a
    # re-run of the interrupted cycle starts from exactly the snapshot.
    for sub, pat in (("data/seeds", "feedback_cycle_*.txt"),
                     ("data/corpus", "corpus_cycle_*.jsonl")):
        d = REPO / sub
        if d.exists():
            for p in d.glob(pat):
                idx = _cycle_index(str(p.relative_to(REPO)))
                if idx is not None and idx > last:
                    p.unlink()
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
             "next cycle = %d", len(state), source[1],
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
