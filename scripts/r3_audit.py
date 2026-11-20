#!/usr/bin/env python3
# OUROBOROS — scripts/r3_audit.py
# Copyright (c) 2026 Kudikala Sai Harsha Vinay. All rights reserved.
# PATENT PENDING
"""Offline gate audit for a completed B5 run (Run Spec R3 §6).

The live gate can never use its own outcome signal: inside a
feedback stage the prediction is made after the run it describes, so
no later outcomes exist yet and the signal is UNAVAILABLE. The audit
is where that signal finally gets exercised — offline, after the
trajectory completes:

  1. POOL: sample AUDIT_CANDIDATES_PER_CYCLE fresh candidates per
     checkpoint for cycles 0-3, using the driver's feedback prompt
     and the pipeline's inference settings (or read them from
     --candidates-file).
  2. LABELS: a candidate is outcome-positive iff >= OUTCOME_THRESHOLD
     (20%) of its salient terms (ouro/gate.py's extractor) appear in
     the trajectory's LATER-cycle corpora.
  3. DECISIONS: each pool candidate is scored by the gate under
     AS-RUN conditions — for a candidate of cycle c the gate sees
     only corpora 0..c and the seed set as it stood at cycle c — so
     its outcome signal is UNAVAILABLE exactly as it was live. Gate
     precision/recall then measure whether the gate's live decisions
     predict outcomes it could not see.
  4. LOGGED DECISIONS: the run's gate_decisions.jsonl gives the
     as-run accept rate and per-signal UNAVAILABLE rates, and is
     counterfactually rescored under the §4 ablated weightings
     (consistency-only, no-outcome): would a different weighting have
     flipped any live decision? Vetoed candidates stay rejected in
     every configuration — the ablations move weights, not vetoes.

Outputs audit_report.json and audit_report.md in the run's OUT
directory (or --report-dir).

Live sampling needs per-cycle adapter checkpoints inside OUT
(adapters/cycle_NN/, checkpoint_cycle_NN/, or the snapshot's final
checkpoint for the last cycle). The driver's snapshots deliberately
exclude adapter weights, so on a copied-off OUT the supported route
is --candidates-file with candidates sampled on the pod; cycles
without checkpoints are reported as skipped, never silently dropped.

  python3 scripts/r3_audit.py --out /workspace/r3_b5_rep1 \
      --candidates-file pool.json
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

AUDIT_CYCLES = (0, 1, 2, 3)
CANDIDATES_PER_CYCLE = 18
OUTCOME_THRESHOLD = 0.20
# The driver's stage-3 feedback prompt (r1_driver.PROMPT).
PROMPT = "Summarize the key predictions from this cycle."

SIGNAL_NAMES = ("outcome_verification", "diversity", "consistency",
                "provenance")
DEFAULT_WEIGHTS = {"outcome_verification": 0.40, "diversity": 0.25,
                   "consistency": 0.20, "provenance": 0.15}
ABLATIONS = {
    "consistency_only": {"outcome_verification": 0.0, "diversity": 0.0,
                         "consistency": 0.20, "provenance": 0.0},
    "no_outcome": {"outcome_verification": 0.0, "diversity": 0.25,
                   "consistency": 0.20, "provenance": 0.15},
}

_CORPUS_RE = re.compile(r"corpus_cycle_(\d+)\.jsonl$")
_FEEDBACK_RE = re.compile(r"feedback_cycle_(\d+)")


# ---------------------------------------------------------------- run state
class RunState:
    """A completed run's OUT directory, located tolerantly.

    Corpora/seeds/gate log normally live under OUT/state/data/ (the
    driver snapshot); loose OUT/data/ and a recursive search are
    fallbacks so zips extracted by hand still audit.
    """

    def __init__(self, out: Path):
        self.out = Path(out)
        self.params = self._load_params()
        self.corpora = self._find_corpora()
        self.seeds_dir = self._find_dir("seeds")
        self.gate_log = self._load_gate_log()

    def _load_params(self) -> dict:
        p = self.out / "r1_params.json"
        if p.exists():
            try:
                return json.loads(p.read_text())
            except json.JSONDecodeError:
                pass
        return {}

    def _data_roots(self) -> list[Path]:
        roots = [self.out / "state" / "data", self.out / "data",
                 self.out / "state", self.out]
        return [r for r in roots if r.exists()]

    def _find_corpora(self) -> dict[int, Path]:
        found: dict[int, Path] = {}
        for root in self._data_roots():
            corpus_dir = root / "corpus" if (root / "corpus").exists() \
                else root
            for p in sorted(corpus_dir.glob("corpus_cycle_*.jsonl")):
                m = _CORPUS_RE.search(p.name)
                if m:
                    found.setdefault(int(m.group(1)), p)
        if not found:
            for p in sorted(self.out.rglob("corpus_cycle_*.jsonl")):
                m = _CORPUS_RE.search(p.name)
                if m:
                    found.setdefault(int(m.group(1)), p)
        return found

    def _find_dir(self, name: str) -> Path | None:
        for root in self._data_roots():
            d = root / name
            if d.exists():
                return d
        hits = [p for p in self.out.rglob(name) if p.is_dir()]
        return hits[0] if hits else None

    def _load_gate_log(self) -> list[dict]:
        path = None
        for root in self._data_roots():
            cand = root / "gate_decisions.jsonl"
            if cand.exists():
                path = cand
                break
        if path is None:
            hits = sorted(self.out.rglob("gate_decisions.jsonl"))
            path = hits[0] if hits else None
        if path is None:
            return []
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows

    def corpus_texts(self) -> dict[int, str]:
        texts: dict[int, str] = {}
        for cycle, path in self.corpora.items():
            parts = []
            for line in path.read_text(encoding="utf-8",
                                      errors="replace").splitlines():
                try:
                    obj = json.loads(line)
                    parts.append(obj.get("content") or "")
                except (json.JSONDecodeError, AttributeError):
                    continue
            texts[cycle] = "\n".join(parts)
        return texts


# ---------------------------------------------------------------- labelling
def label_candidate(text: str, cycle: int,
                    corpus_texts: dict[int, str]) -> dict:
    """Outcome label from LATER-cycle corpora (spec §6).

    Mirrors gate._signal_outcome's hit rule (a salient term counts
    when it appears as a substring of the joined later text) over
    cycles strictly after the candidate's own.
    """
    from ouro.gate import _salient_terms

    terms = _salient_terms(text)
    later = "\n".join(t for c, t in sorted(corpus_texts.items())
                      if c > cycle).lower()
    if not terms:
        return {"label": None, "salient_terms": 0, "hits": 0,
                "hit_fraction": None,
                "reason": "candidate has no checkable salient terms"}
    if not later:
        return {"label": None, "salient_terms": len(terms), "hits": 0,
                "hit_fraction": None,
                "reason": "no later-cycle corpora in this run"}
    hits = sum(1 for term in terms if term in later)
    frac = hits / len(terms)
    return {"label": frac >= OUTCOME_THRESHOLD,
            "salient_terms": len(terms), "hits": hits,
            "hit_fraction": round(frac, 4), "reason": ""}


# ------------------------------------------------------- as-run gate views
def _asrun_views(state: RunState, cycle: int,
                 tmp: Path) -> tuple[Path, Path]:
    """Corpus/seed views as the live gate saw them at `cycle`.

    Corpus dir: only corpus files 0..cycle (so no later outcomes
    exist and the outcome signal is UNAVAILABLE, as-run). Seed dir:
    the human seeds plus feedback seeds from cycles < cycle (the
    feedback a cycle produced only joins the seed set afterwards).
    """
    corpus_view = tmp / f"corpus_asof_{cycle}"
    corpus_view.mkdir(parents=True, exist_ok=True)
    for c, path in state.corpora.items():
        if c <= cycle:
            link = corpus_view / path.name
            if not link.exists():
                link.symlink_to(path.resolve())
    seed_view = tmp / f"seeds_asof_{cycle}"
    seed_view.mkdir(parents=True, exist_ok=True)
    if state.seeds_dir is not None:
        for p in sorted(state.seeds_dir.iterdir()):
            m = _FEEDBACK_RE.match(p.name)
            if m and int(m.group(1)) >= cycle:
                continue
            link = seed_view / p.name
            if not link.exists():
                link.symlink_to(p.resolve())
    return corpus_view, seed_view


def _gate_config(state: RunState, corpus_view: Path, seed_view: Path):
    from ouro.config import OuroConfig

    cfg = OuroConfig()
    cfg.corpus_dir = corpus_view
    cfg.seed_docs_dir = seed_view
    cfg.feedback_gate_enabled = True
    weights = (state.params.get("gate") or {}).get("weights") or {}
    for key, attr in (("outcome", "gate_weight_outcome"),
                      ("diversity", "gate_weight_diversity"),
                      ("consistency", "gate_weight_consistency"),
                      ("provenance", "gate_weight_provenance")):
        if key in weights:
            setattr(cfg, attr, float(weights[key]))
    if (state.params.get("gate") or {}).get("vetoes") is False:
        cfg.gate_vetoes_enabled = False
    base = state.params.get("train_model")
    if base:
        cfg.train_base_model = base
    return cfg


# Candidates per gate evaluation, matching the live stage: the
# driver's feedback stage samples feedback_num_candidates (3) per
# prompt, so the consistency signal's peer set is 2, not 17. Scoring
# all 18 pool candidates in one evaluate() call would compute a
# different, lower-variance consistency statistic than the live one.
LIVE_GROUP_SIZE = 3


def evaluate_pool(state: RunState, candidates: list[dict],
                  tmp: Path) -> list[dict]:
    """Gate-decide every pool candidate under as-run conditions."""
    from ouro.gate import FeedbackGate

    by_cycle: dict[int, list[dict]] = {}
    for cand in candidates:
        by_cycle.setdefault(int(cand["cycle"]), []).append(cand)
    decided: list[dict] = []
    for cycle, group in sorted(by_cycle.items()):
        corpus_view, seed_view = _asrun_views(state, cycle, tmp)
        gate = FeedbackGate(_gate_config(state, corpus_view, seed_view))
        prepared = []
        for cand in group:
            c = dict(cand)
            c.setdefault("prompt", PROMPT)
            c.setdefault("cycle", cycle)
            # Always the LOCAL corpus file for this cycle: a candidate
            # sampled on the pod carries the pod's path, which does
            # not exist offline and would silently dock the
            # provenance signal's corpus_on_disk check.
            c["corpus_path"] = str(state.corpora.get(cycle, ""))
            prepared.append(c)
        for start in range(0, len(prepared), LIVE_GROUP_SIZE):
            chunk = prepared[start:start + LIVE_GROUP_SIZE]
            for cand, decision in zip(chunk, gate.evaluate(chunk)):
                decided.append({**cand, "decision": decision})
    return decided


# ---------------------------------------------------------------- sampling
def _find_adapter_dir(state: RunState, cycle: int) -> Path | None:
    out = state.out
    for cand in (out / "adapters" / f"cycle_{cycle:02d}",
                 out / f"checkpoint_cycle_{cycle:02d}"):
        if (cand / "adapter_model.safetensors").exists():
            return cand
    # The snapshot's checkpoint dir holds only the FINAL cycle's
    # adapter — using it for any earlier cycle would sample every
    # cycle from the wrong checkpoint.
    last_cycle = max(state.corpora) if state.corpora else None
    if cycle == last_cycle:
        cand = out / "state" / "data" / "model" / "checkpoint"
        if (cand / "adapter_model.safetensors").exists():
            return cand
    return None


def _live_generate_fn(state: RunState, cycle: int, corpus_path: Path,
                      tmp: Path) -> Callable[[str], dict] | None:
    """A generate_fn sampling from cycle's adapter, or None.

    Uses the pipeline's own inference path (settings unchanged) with
    the corpus excerpt as context, exactly as stage 3 samples.
    """
    adapter = _find_adapter_dir(state, cycle)
    if adapter is None:
        return None
    from ouro.config import OuroConfig
    from ouro.pipeline import OuroPipeline

    model_dir = tmp / f"model_{cycle}"
    (model_dir / "checkpoint").mkdir(parents=True, exist_ok=True)
    for p in adapter.iterdir():
        link = model_dir / "checkpoint" / p.name
        if not link.exists():
            link.symlink_to(p.resolve())
    cfg = OuroConfig()
    cfg.model_dir = model_dir
    if state.params.get("train_model"):
        cfg.train_base_model = state.params["train_model"]
    pipeline = OuroPipeline(cfg)
    sim_result = {"corpus_path": str(corpus_path)}
    context = pipeline._load_corpus_context(sim_result)

    def generate(_prompt: str) -> dict:
        meta = pipeline.infer.predict_with_metadata(PROMPT,
                                                    context=context)
        return {"text": meta["text"], "backend": meta["backend"],
                "checkpoint_sha256": meta["checkpoint_sha256"]}

    return generate


def sample_candidates(
    state: RunState,
    cycles: tuple[int, ...] = AUDIT_CYCLES,
    per_cycle: int = CANDIDATES_PER_CYCLE,
    generate_fn: Callable[[int, str], Any] | None = None,
    tmp: Path | None = None,
) -> tuple[list[dict], list[dict]]:
    """Sample the audit pool; returns (candidates, skipped_cycles).

    generate_fn(cycle, prompt) may return a string or a metadata
    dict; it is injectable so the audit is testable against
    fabricated run data. Without it, per-cycle adapters are located
    in OUT and sampled live; cycles without one are skipped with the
    reason recorded.
    """
    candidates: list[dict] = []
    skipped: list[dict] = []
    tmp = tmp or Path(tempfile.mkdtemp(prefix="r3_audit_sample_"))
    for cycle in cycles:
        gen = generate_fn
        if gen is None:
            live = _live_generate_fn(
                state, cycle, state.corpora.get(cycle, Path("")),
                tmp)
            if live is None:
                skipped.append({
                    "cycle": cycle,
                    "reason": "no adapter checkpoint for this cycle "
                              "inside OUT (snapshots exclude adapter "
                              "weights) — sample on the pod and pass "
                              "--candidates-file"})
                continue
            gen = lambda _cycle, _prompt, _live=live: _live(_prompt)
        for _ in range(per_cycle):
            result = gen(cycle, PROMPT)
            if isinstance(result, dict):
                cand = {"text": result.get("text", ""),
                        "backend": result.get("backend"),
                        "checkpoint_sha256":
                            result.get("checkpoint_sha256")}
            else:
                cand = {"text": str(result)}
            cand.update({"cycle": cycle, "prompt": PROMPT,
                         "corpus_path": str(state.corpora.get(cycle, ""))})
            candidates.append(cand)
    return candidates, skipped


# ---------------------------------------------------------------- rescoring
def rescore_decision(decision: dict, weights: dict[str, float],
                     threshold: float = 0.50) -> bool:
    """Would `decision` be accepted under `weights`?

    Recomputed from the logged per-signal values with the gate's own
    rule (weighted mean over AVAILABLE signals). A vetoed candidate
    stays rejected in every configuration.
    """
    if decision.get("veto"):
        return False
    if decision.get("bypassed"):
        return True
    signals = decision.get("signals") or {}
    available = [(weights[name], signals[name].get("value"))
                 for name in SIGNAL_NAMES
                 if name in signals
                 and signals[name].get("value") is not None]
    if not available:
        return False
    total_w = sum(w for w, _ in available)
    if total_w <= 0:
        return False
    score = sum(w * v for w, v in available) / total_w
    return score >= threshold


def _unavailable_rates(decisions: list[dict]) -> dict[str, float | None]:
    rates: dict[str, float | None] = {}
    for name in SIGNAL_NAMES:
        seen = [d for d in decisions
                if name in (d.get("signals") or {})]
        if not seen:
            rates[name] = None
        else:
            un = sum(1 for d in seen
                     if d["signals"][name].get("status") == "unavailable")
            rates[name] = round(un / len(seen), 4)
    return rates


# ---------------------------------------------------------------- report
def build_report(state: RunState, candidates: list[dict],
                 skipped: list[dict]) -> dict:
    corpus_texts = state.corpus_texts()
    with tempfile.TemporaryDirectory(prefix="r3_audit_") as td:
        decided = evaluate_pool(state, candidates, Path(td))
    pool_rows = []
    tp = fp = fn = tn = unlabelled = 0
    for row in decided:
        lab = label_candidate(row["text"], int(row["cycle"]),
                              corpus_texts)
        accepted = bool(row["decision"]["accepted"])
        label = lab["label"]
        if label is None:
            unlabelled += 1
        elif accepted and label:
            tp += 1
        elif accepted and not label:
            fp += 1
        elif not accepted and label:
            fn += 1
        else:
            tn += 1
        pool_rows.append({
            "cycle": int(row["cycle"]),
            "accepted": accepted,
            "weighted_score": row["decision"]["weighted_score"],
            "veto": row["decision"]["veto"],
            "signals": {k: v.get("status")
                        for k, v in row["decision"]["signals"].items()},
            "label": label,
            "hit_fraction": lab["hit_fraction"],
            "text_preview": row["text"][:120],
        })
    labelled = tp + fp + fn + tn
    accepted_total = sum(1 for r in pool_rows if r["accepted"])
    pool = {
        "candidates": len(pool_rows),
        "labelled": labelled,
        "unlabelled": unlabelled,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "recall": round(tp / (tp + fn), 4) if tp + fn else None,
        # Accepted counts directly: unlabelled candidates (no later
        # corpora, or no salient terms) still have gate decisions and
        # belong in the accept rate.
        "accept_rate": round(accepted_total / len(pool_rows), 4)
                       if pool_rows else None,
        "signal_unavailable_rates": _unavailable_rates(
            [r["decision"] for r in decided]),
        "rows": pool_rows,
    }

    logged = state.gate_log
    asrun_weights = dict(DEFAULT_WEIGHTS)
    for key, name in (("outcome", "outcome_verification"),
                      ("diversity", "diversity"),
                      ("consistency", "consistency"),
                      ("provenance", "provenance")):
        w = (state.params.get("gate") or {}).get("weights") or {}
        if key in w:
            asrun_weights[name] = float(w[key])
    # Self-check first: rescoring with the as-run weights must
    # reproduce the logged decisions (0 flips). It is included in the
    # counterfactual table so a broken rescore cannot hide. Caveat:
    # logged signal values are rounded to 4 decimals while the gate
    # scores raw values, so a decision within ~1e-4 of the threshold
    # could flip spuriously here.
    counterfactual: dict[str, Any] = {}
    for name, weights in [("as_run_replay", asrun_weights),
                          *ABLATIONS.items()]:
        flips_to_accept = flips_to_reject = accepts = 0
        for d in logged:
            now = rescore_decision(d, weights,
                                   float(d.get("threshold", 0.50)))
            was = bool(d.get("accepted"))
            accepts += now
            if now and not was:
                flips_to_accept += 1
            elif was and not now:
                flips_to_reject += 1
        counterfactual[name] = {
            "weights": weights, "accepts": accepts,
            "accept_rate": round(accepts / len(logged), 4)
                           if logged else None,
            "flips_to_accept": flips_to_accept,
            "flips_to_reject": flips_to_reject,
        }
    outcome_all_unavailable = bool(logged) and all(
        (d.get("signals") or {}).get("outcome_verification", {})
        .get("status") == "unavailable" for d in logged)
    logged_block = {
        "decisions": len(logged),
        "accept_rate": round(
            sum(1 for d in logged if d.get("accepted")) / len(logged), 4)
            if logged else None,
        "signal_unavailable_rates": _unavailable_rates(logged),
        "asrun_weights": asrun_weights,
        "counterfactual": counterfactual,
        # When the outcome signal was UNAVAILABLE for every logged
        # decision (the normal case for this driver), the no_outcome
        # ablation is mathematically identical to as-run — its 0
        # flips are a tautology, not a finding.
        "note": ("no_outcome is identical to as-run for this run: "
                 "the outcome signal was UNAVAILABLE for every "
                 "logged decision")
                if outcome_all_unavailable else None,
    }
    return {
        "run_out": str(state.out),
        "arm": state.params.get("arm"),
        "rng_seed": state.params.get("rng_seed"),
        "outcome_threshold": OUTCOME_THRESHOLD,
        "audit_cycles": list(AUDIT_CYCLES),
        "candidates_per_cycle": CANDIDATES_PER_CYCLE,
        "skipped_cycles": skipped,
        "pool": pool,
        "logged": logged_block,
    }


def render_markdown(report: dict) -> str:
    pool, logged = report["pool"], report["logged"]
    lines = [
        f"# Gate audit — {report['run_out']}",
        "",
        f"Arm {report['arm']}, seed {report['rng_seed']}. "
        f"Outcome-positive = >= {report['outcome_threshold']:.0%} of a "
        "candidate's salient terms appearing in later-cycle corpora.",
        "",
        "## Audit pool (fresh candidates, gate decided as-run)",
        "",
        f"- candidates: {pool['candidates']} "
        f"(labelled {pool['labelled']}, unlabelled {pool['unlabelled']})",
        f"- TP {pool['tp']} / FP {pool['fp']} / FN {pool['fn']} / "
        f"TN {pool['tn']}",
        f"- precision: {pool['precision']}",
        f"- recall: {pool['recall']}",
        f"- accept rate: {pool['accept_rate']}",
        f"- per-signal UNAVAILABLE rates: "
        f"{pool['signal_unavailable_rates']}",
    ]
    if report["skipped_cycles"]:
        lines += ["", "Skipped cycles:"]
        for s in report["skipped_cycles"]:
            lines.append(f"- cycle {s['cycle']}: {s['reason']}")
    lines += [
        "",
        "## Logged live decisions",
        "",
        f"- decisions: {logged['decisions']}",
        f"- accept rate: {logged['accept_rate']}",
        f"- per-signal UNAVAILABLE rates: "
        f"{logged['signal_unavailable_rates']}",
        "",
        "## Counterfactual rescoring of logged decisions",
        "",
    ]
    for name, cf in logged["counterfactual"].items():
        lines.append(
            f"- {name}: accepts {cf['accepts']} "
            f"(rate {cf['accept_rate']}); flips "
            f"{cf['flips_to_accept']} to accept, "
            f"{cf['flips_to_reject']} to reject vs as-run")
    if logged.get("note"):
        lines += ["", f"Note: {logged['note']}."]
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True,
                    help="a completed B5 run's OUT directory")
    ap.add_argument("--candidates-file", type=Path, default=None,
                    help="JSON list of pre-sampled candidates "
                         "({cycle, text, ...}); skips live sampling")
    ap.add_argument("--report-dir", type=Path, default=None,
                    help="where to write the reports (default: --out)")
    args = ap.parse_args(argv)

    state = RunState(args.out)
    if args.candidates_file:
        candidates = json.loads(args.candidates_file.read_text())
        skipped: list[dict] = []
    else:
        candidates, skipped = sample_candidates(state)
    report = build_report(state, candidates, skipped)
    report_dir = args.report_dir or args.out
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "audit_report.json").write_text(
        json.dumps(report, indent=2))
    (report_dir / "audit_report.md").write_text(render_markdown(report))
    print(f"[r3_audit] pool: {report['pool']['candidates']} candidates, "
          f"precision={report['pool']['precision']} "
          f"recall={report['pool']['recall']}; "
          f"logged: {report['logged']['decisions']} decisions")
    print(f"[r3_audit] reports in {report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
