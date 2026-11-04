# OUROBOROS — ouro/gate.py
# Copyright (c) 2026 Kudikala Sai Harsha Vinay. All rights reserved.
# PATENT PENDING
#
# Phase 2 feedback gate (brainstorm §4.1 C3, replacing "Algorithm 7").
#
# The old gate scored predictions by token log-probabilities the shipped
# adapters cannot produce, was never called by the pipeline, and fell
# back to injecting everything when nothing passed. This gate instead
# scores every candidate prediction on four measurable signals and
# applies a config-driven weighted threshold:
#
#   (a) outcome_verification — delayed-outcome verification. The
#       candidate's salient terms are checked against corpus turns the
#       simulation recorded LATER than the candidate's provenance point
#       (a later cycle, or a later round when the candidate declares
#       one). This is the only signal that injects information from
#       outside the generating model. When no later outcomes exist —
#       the normal case inside a single feedback stage, where the
#       prediction was made after the run it describes — the signal is
#       reported UNAVAILABLE, never faked.
#   (b) diversity — novelty of the candidate's vocabulary against the
#       existing seed set (1 - max token Jaccard), so re-injecting what
#       the swarm already has scores low.
#   (c) consistency — self-consistency: mean token Jaccard against the
#       other candidates sampled for the same prompt. Needs >= 2
#       candidates per prompt, else UNAVAILABLE. (The old logprob
#       "confidence" is deliberately not reused: the brainstorm's
#       Finding 4 shows it is not a measure of agreement or truth.)
#   (d) provenance — completeness of the candidate's lineage record
#       (cycle, corpus file on disk, generator identity, prompt).
#
# Hard vetoes run before scoring: empty/degenerate text (repetition
# collapse is the failure mode this whole design exists to catch) and
# inference-error strings never become seeds. An empty accept set is a
# legitimate result — there is no inject-everything fallback.
#
# Every decision is returned as a record with per-signal values (or
# UNAVAILABLE), the weighted score, and the reason; the feedback module
# persists them to the gate audit log and the cycle state.

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SIGNAL_OUTCOME = "outcome_verification"
SIGNAL_DIVERSITY = "diversity"
SIGNAL_CONSISTENCY = "consistency"
SIGNAL_PROVENANCE = "provenance"

_OK = "ok"
_UNAVAILABLE = "unavailable"

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "will", "would", "this", "that", "these",
    "those", "it", "its", "as", "at", "by", "from", "be", "been", "has",
    "have", "had", "not", "but", "we", "they", "their", "there", "here",
    "which", "who", "what", "when", "where", "how", "why", "can", "could",
    "should", "may", "might", "than", "then", "into", "over", "under",
    "about", "between", "through", "during", "before", "after", "more",
    "most", "other", "some", "such", "only", "also", "just", "like",
    "likely", "cycle", "simulation", "predictions", "prediction",
    "summary", "key", "based", "using", "used", "use",
}

_TOKEN_RE = re.compile(r"[a-z0-9']+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_CORPUS_FILE_RE = re.compile(r"corpus_cycle_(\d+)\.jsonl$")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _distinct_ratio(text: str) -> float:
    toks = _tokens(text)
    if not toks:
        return 0.0
    return len(set(toks)) / len(toks)


def _salient_terms(text: str, cap: int = 25) -> list[str]:
    """Terms a prediction can later be checked against outcomes for:
    hashtags plus distinctive content words."""
    terms: list[str] = [h.lower() for h in _HASHTAG_RE.findall(text)]
    seen = set(terms)
    for tok in _tokens(text):
        if len(tok) >= 6 and tok not in _STOPWORDS and tok not in seen:
            seen.add(tok)
            terms.append(tok)
        if len(terms) >= cap:
            break
    return terms


class FeedbackGate:
    """Multi-signal gate over inference-prediction candidates."""

    def __init__(self, config: Any):
        self.config = config

    # ------------------------------------------------------------------
    # Reference data (loaded from config paths)
    # ------------------------------------------------------------------

    def _seed_texts(self) -> list[str]:
        seed_dir = Path(self.config.seed_docs_dir)
        texts = []
        if seed_dir.exists():
            for path in sorted(seed_dir.glob("*.txt")) + sorted(seed_dir.glob("*.md")):
                try:
                    texts.append(path.read_text(encoding="utf-8", errors="replace"))
                except OSError:
                    continue
        return texts

    def _corpus_turns(self) -> list[dict[str, Any]]:
        """All exported corpus turns, each tagged with its cycle index."""
        corpus_dir = Path(self.config.corpus_dir)
        turns: list[dict[str, Any]] = []
        if not corpus_dir.exists():
            return turns
        for path in sorted(corpus_dir.glob("corpus_cycle_*.jsonl")):
            m = _CORPUS_FILE_RE.search(path.name)
            cycle = int(m.group(1)) if m else -1
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    turn = json.loads(line)
                except json.JSONDecodeError:
                    continue
                turn["_cycle"] = cycle
                turns.append(turn)
        return turns

    # ------------------------------------------------------------------
    # Signals — each returns (value | None, detail str)
    # ------------------------------------------------------------------

    def _signal_outcome(
        self, candidate: dict[str, Any], corpus_turns: list[dict[str, Any]]
    ) -> tuple[float | None, str]:
        cycle = candidate.get("cycle")
        if cycle is None:
            return None, "candidate carries no cycle provenance"
        source_round = candidate.get("source_round")
        later = [
            t for t in corpus_turns
            if t["_cycle"] > cycle
            or (source_round is not None and t["_cycle"] == cycle
                and (t.get("round") or 0) > source_round)
        ]
        if not later:
            return None, "no outcomes recorded after the candidate's provenance point"
        terms = _salient_terms(candidate.get("text", ""))
        if not terms:
            return None, "candidate has no checkable salient terms"
        later_text = "\n".join((t.get("content") or "") for t in later).lower()
        hits = sum(1 for term in terms if term in later_text)
        return hits / len(terms), (
            f"{hits}/{len(terms)} salient terms appear in {len(later)} later turns"
        )

    def _signal_diversity(
        self, candidate: dict[str, Any], seed_token_sets: list[set[str]]
    ) -> tuple[float | None, str]:
        if not seed_token_sets:
            return None, "no existing seed set to compare against"
        cand = set(_tokens(candidate.get("text", "")))
        if not cand:
            return None, "candidate has no tokens"
        max_sim = max(_jaccard(cand, s) for s in seed_token_sets)
        return 1.0 - max_sim, f"max seed Jaccard {max_sim:.3f} over {len(seed_token_sets)} seeds"

    def _signal_consistency(
        self, candidate: dict[str, Any], peers: list[dict[str, Any]]
    ) -> tuple[float | None, str]:
        if not peers:
            return None, "single candidate for its prompt; no resamples to agree with"
        cand = set(_tokens(candidate.get("text", "")))
        sims = [_jaccard(cand, set(_tokens(p.get("text", "")))) for p in peers]
        return sum(sims) / len(sims), f"mean Jaccard {sum(sims)/len(sims):.3f} over {len(peers)} resamples"

    def _signal_provenance(
        self, candidate: dict[str, Any]
    ) -> tuple[float | None, str]:
        checks = {
            "cycle": candidate.get("cycle") is not None,
            "corpus_on_disk": bool(candidate.get("corpus_path"))
            and Path(str(candidate["corpus_path"])).exists(),
            "generator_identity": bool(
                candidate.get("checkpoint_sha256") or candidate.get("backend")
            ),
            "prompt": bool((candidate.get("prompt") or "").strip()),
        }
        value = sum(checks.values()) / len(checks)
        missing = [k for k, ok in checks.items() if not ok]
        detail = "complete" if not missing else f"missing: {', '.join(missing)}"
        return value, detail

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self, candidates: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Score every candidate and return one decision record per
        candidate (same order). When the gate is disabled in config,
        every candidate is accepted with a bypass record — the paper's
        ungated baseline condition.
        """
        cfg = self.config
        threshold = float(getattr(cfg, "gate_accept_threshold", 0.50))

        if not getattr(cfg, "feedback_gate_enabled", True):
            return [
                {
                    "index": i,
                    "accepted": True,
                    "bypassed": True,
                    "reason": "gate disabled in config (uniform feedback baseline)",
                    "weighted_score": None,
                    "threshold": threshold,
                    "veto": None,
                    "signals": {},
                    "text_preview": (c.get("text") or "")[:120],
                }
                for i, c in enumerate(candidates)
            ]

        seed_token_sets = [
            set(_tokens(t)) for t in self._seed_texts()
        ]
        seed_token_sets = [s for s in seed_token_sets if s]
        corpus_turns = self._corpus_turns()
        weights = {
            SIGNAL_OUTCOME: float(getattr(cfg, "gate_weight_outcome", 0.40)),
            SIGNAL_DIVERSITY: float(getattr(cfg, "gate_weight_diversity", 0.25)),
            SIGNAL_CONSISTENCY: float(getattr(cfg, "gate_weight_consistency", 0.20)),
            SIGNAL_PROVENANCE: float(getattr(cfg, "gate_weight_provenance", 0.15)),
        }
        min_distinct = float(getattr(cfg, "gate_min_distinct_ratio", 0.35))
        min_chars = int(getattr(cfg, "gate_min_chars", 10))

        decisions: list[dict[str, Any]] = []
        for i, cand in enumerate(candidates):
            text = cand.get("text") or ""
            peers = [
                c for j, c in enumerate(candidates)
                if j != i and (c.get("prompt") or "") == (cand.get("prompt") or "")
            ]

            veto = None
            if text.strip().startswith("[Inference error"):
                veto = "inference_error: candidate is an error string, not a prediction"
            elif len(text.strip()) < min_chars:
                veto = f"too_short: {len(text.strip())} chars < gate_min_chars {min_chars}"
            elif _distinct_ratio(text) < min_distinct:
                veto = (
                    f"degenerate_repetition: distinct token ratio "
                    f"{_distinct_ratio(text):.3f} < {min_distinct}"
                )

            raw_signals = {
                SIGNAL_OUTCOME: self._signal_outcome(cand, corpus_turns),
                SIGNAL_DIVERSITY: self._signal_diversity(cand, seed_token_sets),
                SIGNAL_CONSISTENCY: self._signal_consistency(cand, peers),
                SIGNAL_PROVENANCE: self._signal_provenance(cand),
            }
            signals = {
                name: {
                    "value": (round(v, 4) if v is not None else None),
                    "status": _OK if v is not None else _UNAVAILABLE,
                    "detail": detail,
                }
                for name, (v, detail) in raw_signals.items()
            }

            available = [
                (weights[name], raw_signals[name][0])
                for name in raw_signals
                if raw_signals[name][0] is not None
            ]
            if available:
                total_w = sum(w for w, _ in available)
                score = sum(w * v for w, v in available) / total_w if total_w > 0 else 0.0
            else:
                score = None

            if veto is not None:
                accepted, reason = False, f"vetoed: {veto}"
            elif score is None:
                accepted = False
                reason = "rejected: no measurable signals (an empty accept set is a result)"
            elif score >= threshold:
                accepted = True
                reason = f"weighted score {score:.3f} >= threshold {threshold}"
            else:
                accepted = False
                reason = f"weighted score {score:.3f} < threshold {threshold}"

            decisions.append({
                "index": i,
                "accepted": accepted,
                "bypassed": False,
                "reason": reason,
                "weighted_score": (round(score, 4) if score is not None else None),
                "threshold": threshold,
                "veto": veto,
                "signals": signals,
                "text_preview": text[:120],
            })
            logger.info(
                "[gate] candidate %d: %s — %s (score=%s)",
                i, "ACCEPT" if accepted else "REJECT", reason,
                decisions[-1]["weighted_score"],
            )
        return decisions
