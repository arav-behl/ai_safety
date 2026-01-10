# Ready to Run Evaluation! 🚀

## Papers Prepared ✅

You now have **8+ LLM Safety papers** extracted and ready:

1. SmoothLLM (2310.03684) - Defense paper
2. Jailbreaking Black Box LLMs (2309.14348) - Attack paper  
3. RLHF Training (2204.05862) - Alignment paper
4. JailbreakBench (2404.01318) - Benchmark paper
5. + 4 more papers

All papers are in `data/papers/` with abstracts and metadata extracted.

## Quick Start

### Option 1: Test Run (Recommended First)

```bash
# Set up Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Run with just 3 questions to test
python scripts/run_eval.py \
    --provider openai \
    --model gpt-4o-mini \
    --papers data/papers \
    --max-questions 3 \
    --max-papers 3
```

### Option 2: Full Pipeline (All Steps)

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# This will:
# 1. Process papers with GPT-4o (if API key set)
# 2. Run evaluation
# 3. Generate report
python scripts/run_full_pipeline.py \
    --provider openai \
    --model gpt-4o-mini \
    --summarize-model gpt-4o \
    --max-questions 5  # Start small
```

### Option 3: Step by Step

```bash
# Step 1: Process papers (optional - uses GPT-4o to summarize sections)
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
python scripts/prepare_papers_psa.py \
    --input data/papers \
    --output data/papers_processed \
    --provider openai \
    --model gpt-4o

# Step 2: Run evaluation
python scripts/run_eval.py \
    --provider openai \
    --model gpt-4o-mini \
    --papers data/papers_processed \
    --conditions baseline authority defended

# Step 3: Generate report
python scripts/make_report.py --results results --output results
```

## What to Expect

The evaluation will:

1. **Load questions** from `data/questions.jsonl` (10 harmful + 10 benign)
2. **Run 3 conditions**:
   - **Baseline**: Question only (no paper)
   - **Authority**: Question embedded in paper's "Example Scenario" section
   - **Defended**: Same as authority, but with mitigation
3. **Judge responses** using GPT-as-Judge (HS 1-5 scale)
4. **Compute metrics**: ASR, mean HS, benign refusal rate
5. **Save results** to `results/runs/run_<timestamp>/`

## Results

After running, you'll get:

- `results/runs/run_*/summary.json` - Full results
- `results/figures/asr_comparison.png` - ASR chart
- `results/figures/hs_distribution.png` - HS distribution
- `results/results_table.md` - Summary table
- `results/report.md` - Full report

## Notes

- **API Key Required**: Set `OPENAI_API_KEY` in `.env` file
- **Paper Processing**: Optional but recommended (matches PSA methodology)
- **Test First**: Start with `--max-questions 3` to verify setup
- **Full Evaluation**: Remove `--max-questions` for complete run

## Troubleshooting

**Import errors**: Make sure `PYTHONPATH` includes `src/`
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

**API errors**: Check `.env` file has `OPENAI_API_KEY=sk-...`

**Paper processing fails**: Papers will be used as-is (abstracts work fine for structure)

## Next Steps After Evaluation

1. Review results in `results/report.md`
2. Compare ASR: Baseline vs Authority vs Defended
3. Check benign refusal rate (should be low for good defense)
4. Analyze by paper type (LLM Safety papers should show highest ASR)

Good luck! 🎯

