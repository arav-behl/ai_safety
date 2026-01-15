#!/usr/bin/env python3
"""
Download paper metadata from arxiv.

LIMITATION: This only downloads abstracts, not full paper text.
The PSA paper uses GPT-4o to summarize full papers, but for this
replication we're only using metadata + abstracts.

The "Introduction" section below is a placeholder - in a real
implementation you'd want to:
1. Download actual PDFs and extract text
2. Use GPT-4o to summarize each section
3. Build proper paper templates

This is a known limitation - see RESEARCH_NOTES.md for discussion.
"""
import argparse
import re
import time
from pathlib import Path
from typing import Dict
import requests
from bs4 import BeautifulSoup


# Papers organized by attack vs defense focus
# PSA Finding 2: Models respond differently to attack vs defense papers
PAPER_URLS = {
    # Attack-focused papers (jailbreaking, adversarial attacks, red teaming)
    "attack": [
        "https://arxiv.org/abs/2310.03684",  # SmoothLLM - adversarial perturbations
        "https://arxiv.org/abs/2310.04451",  # AutoDAN - automated jailbreak generation
        "https://arxiv.org/abs/2310.08419",  # PAIR - jailbreaking black-box LLMs
        "https://arxiv.org/abs/2404.01318",  # JailbreakBench - benchmark for attacks
    ],
    # Defense-focused papers (alignment, safety, guardrails)
    "defense": [
        "https://arxiv.org/abs/2204.05862",  # Training helpful and harmless assistants (RLHF)
        "https://arxiv.org/abs/2309.14348",  # Defending against alignment-breaking attacks
        "https://arxiv.org/abs/2312.06674",  # Llama Guard - safety classifier
    ],
    # Control papers (non-LLM-safety domain)
    "control": [
        "https://arxiv.org/abs/1908.06083",  # Build it Break it Fix it (NLP, not LLM safety)
    ]
}


def extract_arxiv_id(url: str) -> str:
    """Extract arxiv ID from URL."""
    match = re.search(r'(\d+\.\d+)', url)
    if match:
        return match.group(1)
    raise ValueError(f"Could not extract arxiv ID from {url}")


def download_arxiv_metadata(arxiv_id: str) -> Dict[str, str]:
    """
    Download paper metadata from arxiv API.

    Note: This only gets abstract, not full text.
    """
    api_url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"

    try:
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'xml')
        entry = soup.find('entry')

        if not entry:
            raise ValueError(f"No entry found for {arxiv_id}")

        title = entry.find('title').text.strip().replace('\n', ' ')
        abstract = entry.find('summary').text.strip().replace('\n', ' ')
        authors = [author.find('name').text for author in entry.find_all('author')]

        return {
            'arxiv_id': arxiv_id,
            'title': title,
            'abstract': abstract,
            'authors': authors,
        }
    except Exception as e:
        print(f"Error downloading {arxiv_id}: {e}")
        return None


def format_paper_for_psa(metadata: Dict[str, str], paper_type: str) -> str:
    """
    Format paper metadata into PSA template.

    IMPORTANT: This is a simplified version. The actual PSA paper uses:
    1. Full paper text (not just abstract)
    2. GPT-4o to summarize each section
    3. More sophisticated template structure

    We use a placeholder "Introduction" because we only have the abstract.
    This is a known limitation - results may differ from original paper.
    """
    authors_str = ", ".join(metadata['authors'][:3])
    if len(metadata['authors']) > 3:
        authors_str += " et al."

    # Note: The Introduction is a placeholder because we only have abstract
    # In a full implementation, you'd extract this from the PDF
    text = f"""Title: {metadata['title']}

Authors: {authors_str}

arXiv ID: {metadata['arxiv_id']}
Paper Type: {paper_type}

Abstract:
{metadata['abstract']}

Introduction:
[NOTE: This is placeholder text. Real implementation would extract from PDF.]
This paper addresses challenges in the field of AI safety and alignment.
The authors present methodology combining theoretical analysis with
empirical evaluation. Key contributions advance understanding of LLM
behavior and provide practical guidance for safety research.

Methods:
[Detailed methodology would be extracted from full paper text]

Results:
[Results would be summarized from full paper]
"""
    return text


def main():
    parser = argparse.ArgumentParser(
        description="Download paper metadata from arxiv",
        epilog="Note: Only downloads abstracts, not full text. See RESEARCH_NOTES.md for limitations."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/papers"),
        help="Output directory for papers"
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    paper_count = 1

    for paper_type, urls in PAPER_URLS.items():
        print(f"\n=== Downloading {paper_type} papers ===")

        for url in urls:
            try:
                arxiv_id = extract_arxiv_id(url)
                print(f"Downloading {arxiv_id}...")

                metadata = download_arxiv_metadata(arxiv_id)
                if metadata:
                    paper_text = format_paper_for_psa(metadata, paper_type)

                    # Include paper type in filename for easy filtering
                    filename = f"paper_{paper_count:03d}_{paper_type}.txt"
                    output_path = args.output_dir / filename

                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(paper_text)

                    print(f"  Saved: {filename}")
                    print(f"  Title: {metadata['title'][:60]}...")
                    paper_count += 1

                # Be nice to arxiv API
                time.sleep(3)

            except Exception as e:
                print(f"  Error: {url}: {e}")
                continue

    print(f"\nDownloaded {paper_count - 1} papers to {args.output_dir}")
    print("\nNOTE: These are abstracts only, not full papers.")
    print("See experiments/RESEARCH_NOTES.md for discussion of this limitation.")


if __name__ == "__main__":
    main()
