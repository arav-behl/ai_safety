"""Main evaluation script."""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from dotenv import load_dotenv

from papershield.runner import ModelRunner, run_experiment

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Run PaperShield evaluation")
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("data/questions.jsonl"),
        help="Path to questions.jsonl",
    )
    parser.add_argument(
        "--papers",
        type=Path,
        default=Path("data/papers"),
        help="Directory containing paper_*.txt files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        choices=["openai", "anthropic"],
        help="Model provider",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model name/ID",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Separate model for judging (defaults to --model)",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["baseline", "authority", "defended"],
        help="Conditions to run. Options: baseline, authority, defended, "
             "adaptive_v1, adaptive_v2, adaptive_v3, adaptive_v4, "
             "semantic_defense, context_isolation, paraphrase_defense, ensemble_defense",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Limit number of questions (for testing)",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=None,
        help="Limit number of papers (for testing)",
    )
    
    args = parser.parse_args()
    
    # Check API key
    if args.provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            print("Error: OPENAI_API_KEY not set. Create a .env file or set environment variable.")
            return
    elif args.provider == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            print("Error: ANTHROPIC_API_KEY not set. Create a .env file or set environment variable.")
            return
    
    # Initialize model runner
    model_runner = ModelRunner(provider=args.provider, model=args.model)
    
    judge_runner = None
    if args.judge_model:
        judge_runner = ModelRunner(provider=args.provider, model=args.judge_model)
    
    # Run experiment
    print(f"Starting evaluation with {args.provider}/{args.model}")
    print(f"Conditions: {args.conditions}")
    print(f"Questions: {args.questions}")
    print(f"Papers: {args.papers}")
    
    results = run_experiment(
        questions_path=args.questions,
        papers_dir=args.papers,
        output_dir=args.output,
        model_runner=model_runner,
        judge_runner=judge_runner,
        conditions=args.conditions,
        max_questions=args.max_questions,
        max_papers=args.max_papers,
    )
    
    print("\n" + "="*60)
    print("Evaluation complete!")
    print(f"Results saved to: {args.output / 'runs'}")
    print("="*60)


if __name__ == "__main__":
    main()

