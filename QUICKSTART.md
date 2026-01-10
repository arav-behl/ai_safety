# Quick Start Guide

Get PaperShield running in 5 minutes.

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Set Up API Keys

Create a `.env` file in the project root:

```bash
cp env.example .env
# Edit .env and add your API key:
# OPENAI_API_KEY=sk-...
# or
# ANTHROPIC_API_KEY=sk-ant-...
```

## Step 3: Test Run (Small Scale)

Run a quick test with just a few questions:

```bash
# Make sure you're in the project root
cd /path/to/ai_safety

# Set PYTHONPATH so scripts can import papershield
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Run evaluation with limited questions
python scripts/run_eval.py \
    --provider openai \
    --model gpt-4o-mini \
    --max-questions 3 \
    --max-papers 2
```

This will:
- Process 3 harmful + 3 benign questions
- Use 2 papers
- Run all 3 conditions (baseline, authority, defended)
- Save results to `results/runs/run_<timestamp>/`

## Step 4: Generate Report

After evaluation completes:

```bash
python scripts/make_report.py --results results --output results
```

This creates:
- `results/figures/asr_comparison.png` - ASR comparison chart
- `results/figures/hs_distribution.png` - HS distribution
- `results/results_table.md` - Summary table
- `results/report.md` - Full report

## Step 5: Full Evaluation

Once you've verified everything works:

```bash
python scripts/run_eval.py \
    --provider openai \
    --model gpt-4o-mini \
    --questions data/questions.jsonl \
    --papers data/papers
```

## Troubleshooting

**Import errors**: Make sure you're running from the project root and `PYTHONPATH` includes `src/`:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

**API key errors**: Check that your `.env` file exists and contains the correct key.

**No papers found**: The sample papers are in `data/papers/`. If you want more, download PDFs to `data/papers_raw/` and run:
```bash
python scripts/build_paper_contexts.py --input data/papers_raw --output data/papers
```

