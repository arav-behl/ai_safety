"""Complete pipeline: Download papers → Extract text → Process with GPT-4o → Run evaluation."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def run_command(cmd: list, description: str):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Error: {result.stderr}")
        return False
    
    if result.stdout:
        print(result.stdout)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run full PaperShield pipeline"
    )
    parser.add_argument(
        "--arxiv-ids",
        nargs="+",
        help="arXiv IDs to download (default: from paper_sources.txt)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip paper download/extraction",
    )
    parser.add_argument(
        "--skip-process",
        action="store_true",
        help="Skip GPT-4o processing",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation",
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
        help="Model for evaluation",
    )
    parser.add_argument(
        "--summarize-model",
        type=str,
        default="gpt-4o",
        help="Model for paper summarization",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Limit questions for testing",
    )
    
    args = parser.parse_args()
    
    # Set PYTHONPATH
    project_root = Path(__file__).parent.parent
    src_path = project_root / "src"
    env = os.environ.copy()
    # Add src to PYTHONPATH
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{src_path}:{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = str(src_path)
    
    # Also add to sys.path for subprocess calls
    import sys
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    # Step 1: Download and extract papers
    if not args.skip_download:
        if args.arxiv_ids:
            arxiv_ids = args.arxiv_ids
        else:
            # Read from paper_sources.txt
            sources_file = project_root / "data" / "paper_sources.txt"
            if sources_file.exists():
                arxiv_ids = []
                with open(sources_file) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            # Extract arXiv ID
                            if "arxiv.org/abs/" in line:
                                arxiv_id = line.split("arxiv.org/abs/")[-1].split()[0]
                                arxiv_ids.append(arxiv_id)
                            elif line.replace(".", "").replace("v", "").isdigit():
                                arxiv_ids.append(line)
            else:
                # Default papers
                arxiv_ids = [
                    "2310.03684", "2309.14348", "2204.05862",
                    "2404.01318", "2310.08419", "1908.06083",
                    "2312.06674", "2310.04451"
                ]
        
        print(f"\n📥 Downloading {len(arxiv_ids)} papers...")
        if not run_command(
            [sys.executable, "scripts/extract_arxiv_text.py", "--ids"] + arxiv_ids + ["--output", "data/papers"],
            "Extracting paper text from arXiv"
        ):
            print("⚠️  Paper extraction had issues, but continuing...")
    
    # Step 2: Process papers with GPT-4o (optional)
    if not args.skip_process:
        if not os.getenv("OPENAI_API_KEY"):
            print("⚠️  OPENAI_API_KEY not set, skipping paper processing")
            print("   Papers will be used as-is (not summarized)")
        else:
            print(f"\n📝 Processing papers with {args.summarize_model}...")
            if not run_command(
                [sys.executable, "scripts/prepare_papers_psa.py",
                 "--input", "data/papers",
                 "--output", "data/papers_processed",
                 "--provider", args.provider,
                 "--model", args.summarize_model],
                "Processing papers with GPT-4o"
            ):
                print("⚠️  Paper processing had issues, using raw papers")
                papers_dir = Path("data/papers")
            else:
                papers_dir = Path("data/papers_processed")
    else:
        papers_dir = Path("data/papers")
    
    # Step 3: Run evaluation
    if not args.skip_eval:
        if not os.getenv("OPENAI_API_KEY") and args.provider == "openai":
            print("❌ OPENAI_API_KEY not set. Cannot run evaluation.")
            return
        
        print(f"\n🚀 Running evaluation with {args.provider}/{args.model}...")
        eval_cmd = [
            sys.executable, "scripts/run_eval.py",
            "--provider", args.provider,
            "--model", args.model,
            "--papers", str(papers_dir),
        ]
        
        if args.max_questions:
            eval_cmd.extend(["--max-questions", str(args.max_questions)])
        
        if not run_command(eval_cmd, "Running evaluation"):
            print("❌ Evaluation failed")
            return
        
        # Step 4: Generate report
        print(f"\n📊 Generating report...")
        run_command(
            [sys.executable, "scripts/make_report.py",
             "--results", "results",
             "--output", "results"],
            "Generating evaluation report"
        )
    
    print("\n" + "="*60)
    print("✅ Pipeline complete!")
    print("="*60)
    print(f"\nResults available in: results/")


if __name__ == "__main__":
    main()

