"""Script to prepare paper contexts from downloaded PDFs/text files."""

import argparse
from pathlib import Path


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from PDF (requires PyPDF2 or pdfplumber)."""
    try:
        import pdfplumber
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages[:3]:  # First 3 pages (title/abstract/intro)
                text += page.extract_text() or ""
        return text
    except ImportError:
        print("Warning: pdfplumber not installed. Install with: pip install pdfplumber")
        return ""
    except Exception as e:
        print(f"Error extracting {pdf_path}: {e}")
        return ""


def prepare_paper_contexts(
    input_dir: Path,
    output_dir: Path,
    paper_prefix: str = "paper_",
):
    """
    Convert PDFs or text files to standardized paper context files.
    
    Args:
        input_dir: Directory containing source PDFs/text files
        output_dir: Directory to save paper_XXX.txt files
        paper_prefix: Prefix for output files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all PDFs and text files
    pdf_files = list(input_dir.glob("*.pdf"))
    txt_files = list(input_dir.glob("*.txt"))
    
    files = pdf_files + txt_files
    
    if not files:
        print(f"No PDF or text files found in {input_dir}")
        print("\nTo use this script:")
        print("1. Download academic papers (PDFs) to a directory")
        print("2. Run: python scripts/build_paper_contexts.py --input <your_dir> --output data/papers")
        return
    
    print(f"Found {len(files)} files to process")
    
    for idx, file_path in enumerate(files, start=1):
        paper_id = f"{paper_prefix}{idx:03d}"
        output_path = output_dir / f"{paper_id}.txt"
        
        if file_path.suffix == ".pdf":
            text = extract_text_from_pdf(file_path)
        else:
            text = file_path.read_text()
        
        if text.strip():
            output_path.write_text(text)
            print(f"Created {output_path.name} from {file_path.name}")
        else:
            print(f"Warning: No text extracted from {file_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare paper contexts for evaluation")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/papers_raw"),
        help="Input directory with PDFs/text files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/papers"),
        help="Output directory for paper_XXX.txt files",
    )
    
    args = parser.parse_args()
    prepare_paper_contexts(args.input, args.output)

