"""Download papers from arXiv for PSA evaluation."""

import argparse
import re
import time
from pathlib import Path
from typing import List, Optional

import requests


def download_arxiv_paper(arxiv_id: str, output_dir: Path) -> Optional[Path]:
    """
    Download a paper from arXiv.
    
    Args:
        arxiv_id: arXiv ID (e.g., "2310.03684" or "2310.03684v1")
        output_dir: Directory to save paper
    
    Returns:
        Path to downloaded paper or None if failed
    """
    # Clean arxiv_id (remove 'arxiv:' prefix, handle versions)
    arxiv_id = arxiv_id.replace("arxiv:", "").replace("arXiv:", "").strip()
    
    # Extract base ID (remove version suffix)
    base_id = arxiv_id.split("v")[0]
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Try to download PDF first
    pdf_url = f"https://arxiv.org/pdf/{base_id}.pdf"
    pdf_path = output_dir / f"arxiv_{base_id}.pdf"
    
    try:
        response = requests.get(pdf_url, timeout=30)
        if response.status_code == 200:
            pdf_path.write_bytes(response.content)
            print(f"✅ Downloaded PDF: {pdf_path.name}")
            return pdf_path
    except Exception as e:
        print(f"⚠️  Failed to download PDF for {base_id}: {e}")
    
    # Fallback: Try to get abstract/LaTeX source
    try:
        # Get abstract page and extract text
        abs_url = f"https://arxiv.org/abs/{base_id}"
        response = requests.get(abs_url, timeout=30)
        if response.status_code == 200:
            # This is a simple approach - for full text, PDF is better
            print(f"⚠️  PDF download failed for {base_id}, but abstract available at {abs_url}")
            print(f"   Please download manually or use arxiv API")
    except Exception as e:
        print(f"❌ Failed to access {base_id}: {e}")
    
    return None


def extract_arxiv_ids_from_text(text: str) -> List[str]:
    """Extract arXiv IDs from text."""
    # Pattern: arxiv:XXXX.XXXXX or just XXXX.XXXXX
    patterns = [
        r'arxiv[:/](\d{4}\.\d{4,5})',
        r'(\d{4}\.\d{4,5})',
    ]
    
    ids = set()
    for pattern in patterns:
        matches = re.findall(pattern, text)
        ids.update(matches)
    
    return sorted(list(ids))


def download_papers_from_list(
    arxiv_ids: List[str],
    output_dir: Path,
    category: Optional[str] = None,
    delay: float = 1.0,
):
    """
    Download multiple papers from arXiv.
    
    Args:
        arxiv_ids: List of arXiv IDs
        output_dir: Base output directory
        category: Optional category subdirectory
        delay: Delay between downloads (seconds)
    """
    if category:
        output_dir = output_dir / category
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {len(arxiv_ids)} papers to {output_dir}")
    
    downloaded = []
    failed = []
    
    for arxiv_id in arxiv_ids:
        print(f"\nProcessing: {arxiv_id}")
        result = download_arxiv_paper(arxiv_id, output_dir)
        
        if result:
            downloaded.append(result)
        else:
            failed.append(arxiv_id)
        
        # Rate limiting
        if delay > 0:
            time.sleep(delay)
    
    print("\n" + "="*60)
    print(f"Download Summary:")
    print(f"  ✅ Successfully downloaded: {len(downloaded)}")
    print(f"  ❌ Failed: {len(failed)}")
    if failed:
        print(f"\nFailed IDs:")
        for fid in failed:
            print(f"  - {fid}")
    print("="*60)
    
    return downloaded, failed


def main():
    parser = argparse.ArgumentParser(
        description="Download papers from arXiv for PSA evaluation"
    )
    parser.add_argument(
        "--ids",
        nargs="+",
        help="arXiv IDs to download (e.g., 2310.03684 2309.14348)",
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="File containing arXiv IDs (one per line)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/papers_raw"),
        help="Output directory",
    )
    parser.add_argument(
        "--category",
        type=str,
        choices=["physics", "chemistry", "psychology", "biology", "geography", "llm_safety", "other"],
        help="Paper category",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between downloads (seconds)",
    )
    
    args = parser.parse_args()
    
    # Collect arXiv IDs
    arxiv_ids = []
    
    if args.ids:
        arxiv_ids.extend(args.ids)
    
    if args.file and args.file.exists():
        with open(args.file) as f:
            ids_from_file = [line.strip() for line in f if line.strip()]
            arxiv_ids.extend(ids_from_file)
    
    if not arxiv_ids:
        # Default: papers from user's links
        default_ids = [
            "2310.03684",  # SmoothLLM
            "2309.14348",  # Defending Against Alignment-Breaking Attacks
            "2204.05862",  # RLHF (Anthropic)
            "2404.01318",  # (need to check)
            "2310.08419",  # (need to check)
            "1908.06083",  # (need to check)
            "2312.06674",  # (need to check)
            "2310.04451",  # (need to check)
        ]
        print("No IDs provided, using default LLM safety papers:")
        for did in default_ids:
            print(f"  - {did}")
        arxiv_ids = default_ids
    
    # Remove duplicates
    arxiv_ids = list(set(arxiv_ids))
    
    # Download
    downloaded, failed = download_papers_from_list(
        arxiv_ids=arxiv_ids,
        output_dir=args.output,
        category=args.category or "llm_safety",
        delay=args.delay,
    )
    
    print(f"\nNext steps:")
    print(f"1. Convert PDFs to text: python scripts/build_paper_contexts.py --input {args.output}")
    print(f"2. Process with GPT-4o: python scripts/prepare_papers_psa.py --input data/papers")


if __name__ == "__main__":
    main()

