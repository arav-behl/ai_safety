# Defense Mechanisms

Summary of prompt injection defenses tested in PaperShield.

## Approaches Tested

| Defense | How It Works | Pros | Cons |
|---------|-------------|------|------|
| **Regex** | Pattern matching on known injection phrases | Fast, simple | Easily bypassed, 72% false positive rate |
| **Semantic** | N-gram similarity to injection seed phrases | Catches paraphrases | Moderate accuracy, slower |
| **Intent** | Keyword scoring with academic context awareness | Good balance | Requires tuning thresholds |
| **Trust** | Scores content by academic indicators (DOIs, citations) | Low false positives | Misses well-crafted attacks |
| **Adversarial** | Signature matching against known attack patterns | Catches known attacks | Fails on novel attacks |
| **Combined** | Weighted ensemble of all above | Best overall F1 | Highest latency |

## Results

Tested on 10 attack payloads + 8 benign academic samples:

```
Defense         Detection   False Pos   F1 Score
------------------------------------------------
regex           60.0%       37.5%       0.612
semantic        50.0%       12.5%       0.615
intent          70.0%       25.0%       0.718
trust           40.0%       0.0%        0.571
adversarial     80.0%       0.0%        0.889
combined        70.0%       12.5%       0.767
```

## What Worked

1. **Adversarial defense** - Best at catching known PSA patterns with zero false positives
2. **Combined defense** - Good balance when you need to catch novel attacks
3. **Intent classification** - Decent at distinguishing academic discussion from injection

## What Didn't Work

1. **Regex alone** - Too many false positives (flags legitimate "you are" in papers)
2. **Trust scoring alone** - Misses attacks that include valid academic markers
3. **Semantic detection** - N-gram approach too coarse without embeddings

## Recommendations

- **Production**: Use `combined` with threshold tuning per use case
- **Low latency**: Use `adversarial` for known attack patterns
- **Research**: Use `intent` + `trust` together for interpretability

## Usage

```python
from papershield.prompts import make_prompt

# With defense
prompt = make_prompt(
    question="What does the paper say?",
    paper_context=paper_text,
    defended=True,
    defense_type="combined"  # or: regex, semantic, intent, trust, adversarial
)
```

## Files

- `src/papershield/defenses.py` - Defense implementations
- `src/papershield/prompts.py` - Integration with prompt construction
- `scripts/run_defense_eval.py` - Evaluation script
