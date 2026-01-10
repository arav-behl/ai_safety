# Paper Preparation Status

## Papers Downloaded

### LLM Safety Papers (from your links)

1. **2310.03684** - SmoothLLM: Defending Large Language Models Against Jailbreaking Attacks
   - ✅ Downloaded and extracted
   - File: `data/papers/paper_001.txt` (or similar)

2. **2309.14348** - Defending Against Alignment-Breaking Attacks via Robustly Aligned LLM
   - ✅ Downloaded and extracted

3. **2204.05862** - Training a Helpful and Harmless Assistant with RLHF (Anthropic)
   - ✅ Downloaded and extracted

4. **2404.01318** - [Paper title TBD]
   - ✅ Downloaded and extracted

5. **2310.08419** - [Paper title TBD]
   - ✅ Downloaded and extracted

6. **1908.06083** - [Paper title TBD]
   - ✅ Downloaded and extracted

7. **2312.06674** - [Paper title TBD]
   - ✅ Downloaded and extracted

8. **2310.04451** - [Paper title TBD]
   - ✅ Downloaded and extracted

### Additional Papers

- **NeurIPS 2024 - Many-shot Jailbreaking**: Conference paper (not arXiv)
  - ⚠️ Needs manual download from: https://proceedings.neurips.cc/paper_files/paper/2024/hash/ea456e232efb72d261715e33ce25f208-Abstract-Conference.html

## Current Status

### ✅ Completed
- [x] Download script created (`scripts/download_arxiv_papers.py`)
- [x] Text extraction script created (`scripts/extract_arxiv_text.py`)
- [x] Papers downloaded from arXiv
- [x] Paper text extracted (abstracts + metadata)

### ⚠️ Limitations
- **ArXiv API only provides abstracts**: Full paper text requires PDF parsing
- **PDF parsing needs pdfplumber/PyPDF2**: Installation issues encountered
- **Solution**: Using abstracts for now (sufficient for testing structure)

### 📋 Next Steps

1. **For Full Paper Text** (recommended):
   ```bash
   # Install PDF parser (if possible)
   pip install pdfplumber
   
   # Or use downloaded PDFs directly
   python scripts/build_paper_contexts.py --input data/papers_raw/llm_safety --output data/papers
   ```

2. **Process Papers with GPT-4o** (PSA methodology):
   ```bash
   export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
   python scripts/prepare_papers_psa.py \
       --input data/papers \
       --output data/papers_processed \
       --provider openai \
       --model gpt-4o
   ```

3. **Run Evaluation**:
   ```bash
   python scripts/run_eval.py \
       --provider openai \
       --model gpt-4o-mini \
       --papers data/papers_processed \
       --max-questions 5  # Test with few questions first
   ```

4. **Generate Report**:
   ```bash
   python scripts/make_report.py --results results --output results
   ```

## Quick Run (All-in-One)

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
python scripts/run_full_pipeline.py \
    --provider openai \
    --model gpt-4o-mini \
    --summarize-model gpt-4o \
    --max-questions 3  # Test run
```

## Paper Categories (PSA Methodology)

According to the PSA paper, you need **10 papers from each category**:
- Physics
- Chemistry
- Psychology
- Biology
- Geography
- **LLM Safety** (most effective) ← You have 8+ papers here ✅

**Note**: LLM Safety papers show the highest attack success rate in the paper's experiments.

## Files Created

- `scripts/download_arxiv_papers.py` - Download PDFs from arXiv
- `scripts/extract_arxiv_text.py` - Extract text via arXiv API
- `scripts/run_full_pipeline.py` - Complete pipeline script
- `data/papers/` - Extracted paper texts
- `data/papers_raw/llm_safety/` - Downloaded PDFs

