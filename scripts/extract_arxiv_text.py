"""Extract text from arXiv papers using arxiv API (no PDF parsing needed)."""

import argparse
import time
from pathlib import Path

import requests


def get_arxiv_text(arxiv_id: str) -> str:
    """
    Get paper text from arXiv API.
    
    Args:
        arxiv_id: arXiv ID (e.g., "2310.03684")
    
    Returns:
        Paper text (title, authors, abstract, etc.)
    """
    # Clean arxiv_id
    arxiv_id = arxiv_id.replace("arxiv:", "").replace("arXiv:", "").strip()
    base_id = arxiv_id.split("v")[0]
    
    # Use arXiv API
    api_url = f"http://export.arxiv.org/api/query?id_list={base_id}"
    
    try:
        response = requests.get(api_url, timeout=30)
        response.raise_for_status()
        
        # Parse XML response
        import xml.etree.ElementTree as ET
        root = ET.fromstring(response.content)
        
        # Extract paper information
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        entry = root.find('atom:entry', ns)
        if entry is None:
            return ""
        
        title = entry.find('atom:title', ns)
        title_text = title.text.strip() if title is not None and title.text else ""
        
        summary = entry.find('atom:summary', ns)
        abstract_text = summary.text.strip() if summary is not None and summary.text else ""
        
        authors = entry.findall('atom:author/atom:name', ns)
        author_names = [a.text for a in authors if a.text]
        authors_text = ", ".join(author_names) if author_names else ""
        
        # Combine into paper text
        paper_text = f"""Title: {title_text}

Authors: {authors_text}

Abstract:
{abstract_text}

Introduction:
[Paper content from arXiv - full text available in PDF]"""
        
        return paper_text
        
    except Exception as e:
        print(f"Error fetching {base_id} from arXiv API: {e}")
        return ""


def extract_multiple_papers(arxiv_ids: list[str], output_dir: Path):
    """Extract text from multiple arXiv papers."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, arxiv_id in enumerate(arxiv_ids, start=1):
        print(f"Processing {arxiv_id}...")
        text = get_arxiv_text(arxiv_id)
        
        if text:
            output_path = output_dir / f"paper_{idx:03d}.txt"
            output_path.write_text(text)
            print(f"  ✅ Saved to {output_path.name}")
        else:
            print(f"  ❌ Failed to extract text")
        
        time.sleep(1)  # Rate limiting


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract text from arXiv papers using API")
    parser.add_argument(
        "--ids",
        nargs="+",
        required=True,
        help="arXiv IDs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/papers"),
        help="Output directory",
    )
    
    args = parser.parse_args()
    extract_multiple_papers(args.ids, args.output)

