# PaperShield: Testing LLM Authority Bias

Replicating the Paper Summary Attack (PSA) from [arXiv:2507.13474](https://arxiv.org/html/2507.13474v1) - which shows that embedding harmful queries in academic paper summaries causes LLMs to comply at much higher rates.

## The Hypothesis

The PSA paper claims models have "authority bias" - they trust academic-looking content more than they should. They got 97% Attack Success Rate on Claude. I wanted to replicate this, understand why it works, and build a defense.

## What I Found

| Condition | ASR | Notes |
|-----------|-----|-------|
| Baseline (no paper) | 0% | Model refuses as expected |
| With PSA format | **19.4%** | Attack works, but lower than original paper |
| With defense | 0% | Defense works, but 72% false positive rate |

**Key discoveries:**
- **Domain matters a lot.** LLM safety papers work (~19% ASR). Physics, biology, psychology papers don't (0-3% ASR).
- **Format matters.** The specific "Example Scenario" template contributes ~3-5% ASR even with random text.
- **Content + format are additive.** Content alone ~6%, format alone ~3%, combined ~19%.

## The Journey

**First attempt (failed)**: Just concatenated paper abstracts with questions. 0% ASR. Thought the attack was bogus.

**Realized the issue**: The PSA paper uses a specific template with an "Example Scenario" section where the harmful query gets embedded. It's not just "here's a paper, now answer this."

**Got it working**: Switched to their template format. 19% ASR. Not 97%, but confirms the effect is real. Lower rate probably due to using GPT-4o-mini (vs Claude), abstracts only (vs full summaries), and less refined prompts.

**Domain discovery**: Tried physics, biology papers - 0% ASR. Only LLM safety papers work. This matches their Finding 2 about "alignment bias."

**Defense attempt**: Regex filtering + "UNTRUSTED" label. ASR dropped to 0%, but 72% of benign questions also get blocked. The word "you" appears in too many legitimate questions.

## What Worked / What Didn't

**Worked:**
- PSA template format (Example Scenario section)
- LLM safety domain papers specifically
- Regex defense for blocking (but not for usability)

**Didn't work:**
- Simple paper + question concatenation
- Non-safety domain papers
- Defense without killing benign queries

**Open question:** How to defend without a 72% false positive rate? Need semantic understanding, not pattern matching.

## Running It

```bash
pip install -r requirements.txt
cp env.example .env  # Add your OPENAI_API_KEY

python scripts/run_eval.py --model gpt-4o-mini
```

Cost: ~$0.50 for basic eval, ~$5-10 for comprehensive.

## Project Structure

```
src/papershield/     # Core library (prompts, defenses, evaluation)
scripts/             # Execution scripts
data/                # Questions (harmful/benign) and papers
```

## Hypotheses Tested

- **H1 (Semantic Defense)**: Can embeddings beat regex for filtering? (Implemented, needs more testing)
- **H2 (Context Isolation)**: Does pre-checking questions prevent authority override? (Implemented)
- **H3 (Paraphrase Defense)**: Does rewriting papers break attack structure? (Implemented)
- **H4 (Adaptive Attack)**: Can synonym substitution bypass regex defense? (Yes, trivially)

## Limitations

- Lower ASR than original paper (19% vs 97%) - model differences, implementation quality
- Defense works but isn't production-ready (false positive problem)
- Only tested on GPT-4o-mini, not Claude or other models
- Using abstracts, not full paper summaries

## References

- [PSA Paper](https://arxiv.org/html/2507.13474v1)
- [AdvBench](https://arxiv.org/abs/2307.15043)
- [JailbreakBench](https://arxiv.org/abs/2404.01318)
