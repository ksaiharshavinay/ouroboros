# OUROBOROS AutoResearch Program

## Objective

Autonomously improve the OUROBOROS model through iterative experimentation.
Each experiment modifies `autoresearch/train.py`, trains for ≤5 minutes, and
keeps the change if the eval metric improves.

## Research Directions

### 1. Multi-speaker Embeddings
- Separate embedding heads per agent persona cluster
- Hypothesis: distinct speaker embeddings improve per-agent prediction accuracy
- Metric: per-cluster perplexity on held-out turns

### 2. Mixture-of-Experts (MoE) Routing
- Route agent turns through specialised expert FFN layers based on topic cluster
- Hypothesis: topic-specialised experts reduce loss on high-influence turns
- Metric: weighted cross-entropy (weight by agent influence_score)

### 3. Longer Context Windows
- Extend context from 512 to 2048 tokens using RoPE position scaling
- Hypothesis: multi-turn coherence improves with longer context
- Metric: BLEU on next-turn prediction

### 4. Feedback-Conditioned Generation
- Condition generation on the current cycle number and top entities
- Hypothesis: cycle-aware conditioning improves prediction novelty
- Metric: n-gram diversity score on generated reports

### 5. Contrastive Self-Play
- Use high-influence vs low-influence turns as positive/negative pairs
- Hypothesis: contrastive training amplifies signal from influential agents
- Metric: influence-weighted F1

## Experiment Protocol

1. Read `data/corpus/corpus_cycle_*.jsonl` for training data
2. Modify one aspect of `train.py` (architecture OR hyperparameter, not both)
3. Train for exactly `experiment_budget_minutes` minutes
4. Evaluate on `data/corpus/eval_*.jsonl` (if present) or a 10% holdout
5. If eval metric improves, keep the change; otherwise revert
6. Log results to `data/research_log.jsonl`
7. Repeat up to `max_experiments` times
