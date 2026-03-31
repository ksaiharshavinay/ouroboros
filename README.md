# OUROBOROS

> *The serpent that consumes its own tail — endlessly regenerating.*

**PATENT PENDING** | © 2026 Sai Harsha Vinaykudikala. All rights reserved.

---

## What is OUROBOROS?

OUROBOROS is a self-improving AI system built on a **recursive feedback loop**. It uses a swarm simulation to generate a training corpus, trains a model on that corpus, runs inference, and feeds the model's own predictions back as new simulation seeds — closing the loop.

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   Seed Docs ──► Swarm Simulation ──► Corpus                 │
│       ▲               (OASIS)          │                    │
│       │                                ▼                    │
│   Feedback         AutoResearch ◄── Training Data           │
│   Seeds ◄────  Inference Engine ◄── Trained Model           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

The core novel mechanism: **the model's own prediction outputs feed back as new simulation seeds**, creating a closed loop that neither the swarm engine nor the training loop achieves independently.

---

## Tech Stack

| Component | Library | License |
|-----------|---------|---------|
| Swarm simulation | OASIS (camel-ai) | Apache 2.0 |
| Knowledge graph | KuzuDB 0.7.x | MIT |
| LLM inference | Anthropic SDK | MIT |
| Training loop | Custom (AutoResearch-inspired) | Proprietary |

---

## Architecture

### `ouro/` — Core pipeline
- **`config.py`** — Centralised configuration (`OuroConfig`)
- **`pipeline.py`** — Orchestrates all three stages (`OuroPipeline`)
- **`feedback.py`** — Recursive feedback loop, persisted to disk (`OuroFeedbackModule`)
- **`infer.py`** — Inference engine with tiktoken tokenisation (`OuroInferenceEngine`)
- **`claude_adapter.py`** — Drop-in adapter wrapping the Anthropic SDK behind an OpenAI-compatible interface

### `swarm/` — Swarm simulation engine
- **`generator.py`** — Full pipeline: graph → personas → OASIS → corpus → report
- **`graph.py`** — KuzuDB knowledge graph from seed documents
- **`personas.py`** — LLM-generated agent personas
- **`report.py`** — ReACT-style prediction report generator
- **`base.py`** — Abstract interface for swarm backends

### `autoresearch/` — Autonomous training
- **`prepare.py`** — Converts OASIS corpus to training/eval JSONL
- **`train.py`** — AutoResearch training loop

### `scripts/` — Run the pipeline step by step
```bash
python3 scripts/1_run_swarm.py       # Stage 1: swarm simulation
python3 scripts/2_run_research.py    # Stage 2: autonomous training
python3 scripts/3_infer.py           # Stage 3: inference + feedback
```

---

## Quickstart

### Requirements

- Python 3.10–3.12 (3.12 recommended)
- An [Anthropic API key](https://console.anthropic.com/settings/keys) with available credits

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/ouroboros.git
cd ouroboros
```

### 2. Install dependencies

```bash
python3 -m pip install anthropic "kuzu==0.7.1" tiktoken openai
```

> Note: `camel-oasis` (the full swarm engine) requires Python < 3.12. On Python 3.12 the pipeline
> automatically falls back to a stub simulation. Full OASIS support requires Python 3.10 or 3.11.

### 3. Configure environment

```bash
cp .env.example .env
# Open .env and set your ANTHROPIC_API_KEY
```

Your `.env` should look like:

```
ANTHROPIC_API_KEY=sk-ant-api03-...
```

### 4. Add a seed document

Drop any `.txt` or `.md` file into `data/seeds/`. This is the starting knowledge the swarm will simulate around. Example:

```bash
cat > data/seeds/seed1.txt << 'EOF'
Title: The Rise of AI Agents in Social Networks — Trends and Predictions (2025)

AI agents are rapidly transforming how people interact on social platforms. Algorithmic
recommendation systems optimize for engagement at the expense of accuracy and wellbeing.
Echo chambers are deepening political polarization. AI-generated misinformation now
accounts for ~15% of viral political content. Decentralized platforms (Mastodon, Bluesky)
are growing as trust in incumbents erodes. Regulators (EU DSA, FTC) are pushing for
algorithmic transparency. The key actors — Meta, ByteDance, X, and independent researchers
— are locked in an arms race between AI content generation and AI content detection.
EOF
```

### 5. Run the full loop

```bash
# Export your API key (or load from .env with a tool like direnv)
export ANTHROPIC_API_KEY=sk-ant-api03-...

# Stage 1: Swarm simulation — extracts entities, generates personas, runs simulation
python3 scripts/1_run_swarm.py --agents 10 --rounds 5

# Stage 2: AutoResearch — prepares corpus and runs training loop
python3 scripts/2_run_research.py

# Stage 3: Inference + feedback — generates prediction and writes it back as next seed
python3 scripts/3_infer.py --from-disk
```

### 6. Repeat (self-reinforcing loop)

Each run of Stage 1 automatically picks up `data/seeds/feedback_cycle_NNNN.txt` written by the previous Stage 3. The loop is self-reinforcing — each cycle's predictions shape the next cycle's simulation.

```bash
# Cycle 2
python3 scripts/1_run_swarm.py --agents 10 --rounds 5
python3 scripts/2_run_research.py
python3 scripts/3_infer.py --from-disk

# Cycle 3, 4, ... same commands
```

---

## Sample Run

Below is a real end-to-end run using `data/seeds/seed1.txt` (the AI in social networks seed above).

### Stage 1 output
```
=== Swarm Complete ===
  Agents:     10
  Turns:      50
  Corpus:     data/corpus/corpus_cycle_0000.jsonl
  Cycle:      0

  Report preview:
  # Social Simulation Analysis Report

  ## 1. Observe: Dominant Themes
  Initial Universal Engagement (Round 0): All 10 agents simultaneously observe trends,
  indicating synchronized information flow typical of breaking-news events...
```

### Stage 2 output
```
INFO: Wrote 45 examples to data/corpus/train.jsonl
INFO: Wrote 4 examples to data/corpus/eval.jsonl
INFO: Corpus prepared: 45 train / 4 eval
=== Training Complete === Return code: 0
```

### Stage 3 output (written to `data/seeds/feedback_cycle_0000.txt`)
```
# Forward-Looking Report: AI Agents in Social Networks — 2026 Outlook

The simulation revealed sustained influence concentration among two dominant actors
(Agents 4 and 9), each maintaining 0.40 influence across all cycles. This persistent
duopoly suggests incumbent platforms are consolidating power rather than fragmenting.

Prediction: Major platforms will implement transparency measures that satisfy regulators
while preserving core engagement-optimization logic in hidden layers — creating the
appearance of reform while maintaining algorithmic dominance.
```

This output becomes the seed for Cycle 1, and so on.

---

## CLI Reference

### `scripts/1_run_swarm.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--agents N` | 500 | Number of simulated agents |
| `--rounds N` | 30 | Simulation rounds per cycle |
| `--cycle N` | 0 | Override cycle number |

### `scripts/3_infer.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--prompt TEXT` | Built-in summary prompt | Custom inference prompt |
| `--from-disk` | off | Load last sim result from disk and write feedback seed |

---

## Data Directory Layout

```
data/
  seeds/                        # Input seed documents (you add these)
    seed1.txt                   # Your starting knowledge document
    feedback_cycle_0000.txt     # Auto-generated by Stage 3, Cycle 0
    feedback_cycle_0001.txt     # Auto-generated by Stage 3, Cycle 1
    ...
  corpus/
    corpus_cycle_0000.jsonl     # Swarm interaction corpus, Cycle 0
    train.jsonl                 # Training split
    eval.jsonl                  # Eval split
  model/
    checkpoint                  # Training metadata
  kuzu_graph/                   # KuzuDB knowledge graph (auto-managed)
  ouro_state.json               # Feedback loop state (cycle count, history)
  last_cycle.json               # Last cycle metadata (used by --from-disk)
```

---

## Configuration

All settings live in `ouro/config.py` as an `OuroConfig` dataclass. Key fields:

| Field | Default | Description |
|-------|---------|-------------|
| `llm_model` | `claude-haiku-4-5` | Claude model to use |
| `llm_api_key_env` | `ANTHROPIC_API_KEY` | Environment variable name for the API key |
| `num_agents` | 500 | Default agent count (override with `--agents`) |
| `simulation_rounds` | 30 | Default rounds (override with `--rounds`) |
| `max_feedback_cycles` | 10 | Maximum auto-feedback cycles |

To use a more powerful model, change `llm_model` in `config.py`:

```python
llm_model: str = "claude-opus-4-6"   # Most capable
llm_model: str = "claude-sonnet-4-6" # Balanced
llm_model: str = "claude-haiku-4-5"  # Fastest / cheapest (default)
```

---

## Contributing

See `CONTRIBUTING.md`.

---

## License

Proprietary — All Rights Reserved. Patent Pending.
See `LICENSE-FUTURE.md` for the planned post-patent Apache 2.0 release.
