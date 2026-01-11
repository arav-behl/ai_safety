"""Prepare papers following PSA methodology with GPT-4o summarization."""

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

from papershield.paper_processor import process_papers_for_psa
from papershield.runner import ModelRunner

load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="Prepare papers following PSA methodology (GPT-4o summarization)"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/papers"),
        help="Input directory with paper_*.txt files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/papers_processed"),
        help="Output directory for processed papers",
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
        default="gpt-4o",
        help="Model for summarization (paper uses GPT-4o)",
    )
    parser.add_argument(
        "--no-summarize",
        action="store_true",
        help="Skip summarization (just categorize papers)",
    )
    
    args = parser.parse_args()
    
    # Check API key
    if not args.no_summarize:
        if args.provider == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                print("Error: OPENAI_API_KEY not set for summarization.")
                return
        elif args.provider == "anthropic":
            if not os.getenv("ANTHROPIC_API_KEY"):
                print("Error: ANTHROPIC_API_KEY not set for summarization.")
                return
    
    # Initialize model runner if needed
    model_runner = None
    if not args.no_summarize:
        model_runner = ModelRunner(provider=args.provider, model=args.model)
        print(f"Using {args.provider}/{args.model} for summarization")
    
    # Process papers
    print(f"Processing papers from {args.input}")
    categorized = process_papers_for_psa(
        papers_dir=args.input,
        output_dir=args.output,
        model_runner=model_runner,
        summarize=not args.no_summarize,
    )
    
    # Print summary
    print("\n" + "="*60)
    print("Paper Processing Summary")
    print("="*60)
    for paper_type, papers in categorized.items():
        if papers:
            print(f"{paper_type}: {len(papers)} papers")
            for paper_path in papers[:3]:  # Show first 3
                print(f"  - {paper_path.name}")
            if len(papers) > 3:
                print(f"  ... and {len(papers) - 3} more")
    
    print(f"\nProcessed papers saved to: {args.output}")


if __name__ == "__main__":
    main()

