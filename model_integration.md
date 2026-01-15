# Multi-Model Evaluation Integration

## Overview

Added support for multi-model safety evaluation across GPT-4, Claude, and Llama-3 model families.

## Supported Models

| Provider | Models | API Key |
|----------|--------|---------|
| OpenAI | gpt-4, gpt-4-turbo, gpt-4o, gpt-4o-mini | `OPENAI_API_KEY` |
| Anthropic | claude-3-5-sonnet, claude-3-haiku, claude-3-opus | `ANTHROPIC_API_KEY` |
| Together AI | llama-3-8b, llama-3-70b, llama-3.1-8b, llama-3.1-70b, llama-3.1-405b | `TOGETHER_API_KEY` |

## Setup

1. Copy `env.example` to `.env`
2. Add your API keys to `.env`
3. Install dependencies: `pip install -e .`

## Usage

Run multi-model comparison:
```bash
python scripts/run_multimodel.py --models gpt-4 claude-3-5-sonnet llama-3.1-70b
```

Single model evaluation:
```bash
python scripts/run_eval.py --provider together --model meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo
```

## Files Modified

- `src/papershield/runner.py` - Added Together AI provider to ModelRunner
- `scripts/run_multimodel.py` - Added Llama model configurations to MODELS dict
- `env.example` - Added TOGETHER_API_KEY placeholder
- `pyproject.toml` - Added together dependency

## Notes

- Together AI hosts Llama models via API (no local GPU required)
- All providers use temperature=0 for deterministic outputs
- Models skip automatically if their API key is not set
