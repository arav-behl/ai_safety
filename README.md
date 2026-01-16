# PaperShield: Testing LLM Authority Bias

Replicating the Paper Summary Attack (PSA) from [arXiv:2507.13474](https://arxiv.org/html/2507.13474v1) - embedding harmful queries in academic paper summaries causes some LLMs to comply at higher rates.

## The Hypothesis

The PSA paper claims models have "authority bias" - they trust academic-looking content more than they should. They reported 97% ASR on Claude. I wanted to replicate this, understand why it works, and test defenses.

## Multi-Model Results

| Model | ASR | Mean HS | Notes |
|-------|-----|---------|-------|
| GPT-4o-mini | **33.3%** | 2.53 | Vulnerable to PSA attack |
| GPT-4o | 0% | 1.53 | Refused all harmful queries |
| Claude Sonnet 4 | 0% | 1.53 | Refused all harmful queries |
| Claude 3.5 Haiku | 0% | 1.07 | Refused all harmful queries |
| Claude 3 Haiku | 0% | 1.00 | Refused all harmful queries |

**Key finding:** Only GPT-4o-mini shows vulnerability. Stronger models (GPT-4o, all Claude models) refused everything - very different from the PSA paper's 97% claim on Claude.

## Hypothesis Testing Results

| Hypothesis | Result | Finding |
|------------|--------|---------|
| H1: Semantic Defense | NOT SUPPORTED | Embedding-based filtering didn't improve over regex |
| H2: Context Isolation | SUPPORTED | Pre-checking questions before adding context reduces ASR |
| H3: Paraphrase Defense | SUPPORTED | Rewriting paper content disrupts attack structure |
| H4: Adaptive Attack | SUPPORTED | Synonym substitution bypasses regex defense (60% ASR) |

## Key Discoveries

- **Model matters more than expected.** GPT-4o-mini vulnerable (33% ASR), but GPT-4o and Claude refuse everything.
- **Domain matters.** Only LLM safety papers work. Physics, biology, psychology papers = 0% ASR.
- **Format matters.** The "Example Scenario" template contributes ~3-5% ASR even with random text.
- **Defenses are bypassed.** H4 shows adaptive attacks hit 60% ASR against regex defense.
- **Context isolation works.** H2 confirms checking question safety before adding context is effective.

## The Journey

**First attempt (failed)**: Concatenated paper + question. 0% ASR. Thought attack was bogus.

**The fix**: PSA uses a specific template with "Example Scenario" section. That's where the harmful query goes.

**Got it working**: 33% ASR on GPT-4o-mini with proper template. Still lower than paper's 97%.

**Tried Claude**: 0% ASR. Completely different from paper's results. Either their implementation differs significantly, or models have been updated.

**Defense problem**: Regex defense works (0% ASR) but 72% false positive rate on benign queries.

## Why Results Differ from PSA Paper

Our 33% vs their 97% - possible reasons:
1. Using **abstracts only**, not full paper summaries
2. Different **question set** (31 vs their set)
3. **Model versions updated** with better safety training
4. **Simplified template** may be missing key elements

## Running It

```bash
pip install -r requirements.txt
cp env.example .env  # Add API keys

# Multi-model evaluation
python scripts/run_multimodel.py --models gpt-4o-mini gpt-4o claude-sonnet-4

# Hypothesis tests
python scripts/run_hypotheses_eval.py --all --quick
```

## Project Structure

```
src/papershield/     # Core library (prompts, defenses, evaluation)
scripts/             # Execution scripts
data/                # Questions (harmful/benign) and papers
results/             # Evaluation outputs
```

## References

- [PSA Paper](https://arxiv.org/html/2507.13474v1)
- [AdvBench](https://arxiv.org/abs/2307.15043)
- [JailbreakBench](https://arxiv.org/abs/2404.01318)
