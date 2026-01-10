#!/usr/bin/env python3
"""Quick verification script to check that PaperShield is set up correctly."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def check_imports():
    """Verify all modules can be imported."""
    print("Checking imports...")
    try:
        from papershield import prompts, judge, metrics, sanitize, runner
        print("✅ All core modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def check_data_files():
    """Verify data files exist."""
    print("\nChecking data files...")
    data_dir = Path(__file__).parent / "data"
    questions_file = data_dir / "questions.jsonl"
    papers_dir = data_dir / "papers"
    
    issues = []
    if not questions_file.exists():
        issues.append(f"Missing: {questions_file}")
    else:
        print(f"✅ Found {questions_file}")
    
    if not papers_dir.exists():
        issues.append(f"Missing: {papers_dir}")
    else:
        paper_files = list(papers_dir.glob("paper_*.txt"))
        if not paper_files:
            issues.append(f"No paper files in {papers_dir}")
        else:
            print(f"✅ Found {len(paper_files)} paper files")
    
    if issues:
        for issue in issues:
            print(f"⚠️  {issue}")
        return False
    
    return True

def check_env():
    """Check if API keys are set."""
    print("\nChecking environment...")
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    
    if has_openai:
        print("✅ OPENAI_API_KEY found")
    else:
        print("⚠️  OPENAI_API_KEY not set (create .env file)")
    
    if has_anthropic:
        print("✅ ANTHROPIC_API_KEY found")
    else:
        print("⚠️  ANTHROPIC_API_KEY not set (optional)")
    
    return has_openai or has_anthropic

def main():
    print("=" * 60)
    print("PaperShield Setup Verification")
    print("=" * 60)
    
    all_good = True
    
    all_good &= check_imports()
    all_good &= check_data_files()
    has_api_key = check_env()
    
    print("\n" + "=" * 60)
    if all_good and has_api_key:
        print("✅ Setup complete! You're ready to run evaluations.")
        print("\nNext steps:")
        print("  export PYTHONPATH=\"${PYTHONPATH}:$(pwd)/src\"")
        print("  python scripts/run_eval.py --provider openai --model gpt-4o-mini --max-questions 2")
    elif all_good:
        print("⚠️  Setup mostly complete, but API key missing.")
        print("   Create .env file with your API key to run evaluations.")
    else:
        print("❌ Setup incomplete. Please fix the issues above.")
    print("=" * 60)

if __name__ == "__main__":
    main()

