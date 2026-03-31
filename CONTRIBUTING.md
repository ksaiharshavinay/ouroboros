# Contributing to OUROBOROS

**Note:** This repository is currently private and patent-pending. Contributions
are by invitation only. This guide is intended for trusted collaborators.

---

## Setup

```bash
# Clone the private repo
git clone <repo-url>
cd ouroboros

# Install dependencies
pip install -e ".[dev,swarm]"

# Configure environment
cp .env.example .env
# Add your OPENAI_API_KEY to .env

# Verify setup
pytest tests/ -v
```

---

## Running the Pipeline

```bash
# Add at least one seed document
echo "AI agents are transforming social discourse." > data/seeds/seed1.txt

# Run each stage in order
python scripts/1_run_swarm.py
python scripts/2_run_research.py
python scripts/3_infer.py
```

---

## Pull Request Checklist

- [ ] All existing tests pass (`pytest tests/ -v`)
- [ ] Coverage stays ≥ 80% (`pytest --cov=ouro --cov=swarm --cov-report=term`)
- [ ] No new AGPL dependencies introduced
- [ ] All new modules have `# PATENT PENDING` header
- [ ] Type hints added to all public functions
- [ ] Docstrings updated
- [ ] `NOTICE` updated if a new third-party library is added

---

## Contributor License Agreement (CLA)

Once OUROBOROS is open-sourced (post-patent), contributors will be required
to sign a CLA confirming that:

1. Their contributions are their own original work
2. They grant the project a perpetual, royalty-free license to their contributions
3. They understand the patent pending status of the core innovation

---

## Code Style

- Python 3.10+, type hints throughout
- `from __future__ import annotations` in all modules
- Line length: 100
- All logging via `logging.getLogger(__name__)` — no bare `print()` in library code
