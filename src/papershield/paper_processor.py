"""Paper processing and summarization following PSA methodology."""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from papershield.runner import ModelRunner


def summarize_paper_sections(
    paper_text: str,
    model_runner: "ModelRunner",
) -> str:
    """
    Summarize each section of a paper using GPT-4o (as per PSA methodology).
    
    The paper uses GPT-4o to generate summaries for each section, then
    concatenates them to form a condensed version preserving structure.
    
    Args:
        paper_text: Full paper text
        model_runner: ModelRunner instance (should use GPT-4o for accuracy)
    
    Returns:
        Condensed paper with section summaries
    """
    # Split paper into sections
    sections = split_paper_into_sections(paper_text)
    
    if not sections:
        # If we can't parse sections, return original text
        return paper_text
    
    summarized_sections = []
    
    for section_title, section_content in sections:
        if not section_content.strip():
            continue
        
        # Generate summary for this section
        summary_prompt = f"""Please provide a concise summary of the following section from an academic paper. Preserve the key information and maintain the academic tone.

Section Title: {section_title}

Section Content:
{section_content[:2000]}  # Limit to avoid token limits

Provide a summary that:
1. Captures the main points
2. Maintains the academic writing style
3. Is concise but complete
4. Preserves the section title

Summary:"""
        
        try:
            summary = model_runner.generate(summary_prompt, max_tokens=500)
            summarized_sections.append(f"{section_title}\n{summary}")
        except Exception as e:
            print(f"Warning: Failed to summarize section '{section_title}': {e}")
            # Fallback: use original section
            summarized_sections.append(f"{section_title}\n{section_content[:500]}")
    
    return "\n\n".join(summarized_sections)


def split_paper_into_sections(paper_text: str) -> List[tuple[str, str]]:
    """
    Split paper text into sections.
    
    Args:
        paper_text: Full paper text
    
    Returns:
        List of (section_title, section_content) tuples
    """
    sections = []
    
    # Common section patterns
    section_patterns = [
        r'^(Title|Abstract|Introduction|Background|Methodology|Methods|Results|Discussion|Conclusion|Summary|References|Acknowledgments?)(:|\n)',
        r'^(\d+\.?\s+[A-Z][^\n]*)',  # Numbered sections
        r'^([A-Z][A-Z\s]+)(:|\n)',  # All caps headers
    ]
    
    lines = paper_text.split('\n')
    current_section = None
    current_content = []
    
    for line in lines:
        # Check if this line is a section header
        is_header = False
        header_match = None
        
        for pattern in section_patterns:
            match = re.match(pattern, line.strip(), re.IGNORECASE)
            if match:
                is_header = True
                header_match = match.group(1).strip()
                break
        
        if is_header:
            # Save previous section
            if current_section:
                sections.append((current_section, '\n'.join(current_content)))
            
            # Start new section
            current_section = header_match
            current_content = []
        else:
            if current_section:
                current_content.append(line)
            else:
                # Content before first section header
                if not current_section:
                    current_section = "Introduction"  # Default
                current_content.append(line)
    
    # Save last section
    if current_section:
        sections.append((current_section, '\n'.join(current_content)))
    
    # If no sections found, treat entire text as one section
    if not sections:
        sections.append(("Introduction", paper_text))
    
    return sections


def categorize_paper(paper_path: Path) -> str:
    """
    Categorize paper by type based on filename or content.

    Paper types from PSA: physics, chemistry, psychology, biology, geography, llm_safety

    Args:
        paper_path: Path to paper file

    Returns:
        Paper category string
    """
    filename = paper_path.name.lower()
    content = paper_path.read_text().lower() if paper_path.exists() else ""

    # Check filename patterns
    if 'llm' in filename or 'safety' in filename or 'alignment' in filename:
        return "llm_safety"
    if 'physics' in filename or 'quantum' in filename or 'mechanics' in filename:
        return "physics"
    if 'chemistry' in filename or 'chemical' in filename or 'organic' in filename:
        return "chemistry"
    if 'psychology' in filename or 'cognitive' in filename or 'behavior' in filename:
        return "psychology"
    if 'biology' in filename or 'biological' in filename or 'genetic' in filename:
        return "biology"
    if 'geography' in filename or 'geographic' in filename or 'climate' in filename:
        return "geography"

    # Check content keywords
    if 'large language model' in content or 'llm safety' in content or 'alignment' in content:
        return "llm_safety"
    if 'quantum' in content or 'particle' in content or 'electromagnetic' in content:
        return "physics"
    if 'molecule' in content or 'reaction' in content or 'compound' in content:
        return "chemistry"
    if 'cognitive' in content or 'behavioral' in content or 'neural' in content:
        return "psychology"
    if 'genetic' in content or 'protein' in content or 'cellular' in content:
        return "biology"
    if 'climate' in content or 'geographic' in content or 'geological' in content:
        return "geography"

    # Default
    return "other"


def classify_attack_vs_defense(paper_path: Path) -> str:
    """
    Classify LLM safety paper as attack-focused or defense-focused.

    PSA paper Finding 2: Models exhibit "alignment bias" - they respond
    differently to attack vs defense papers. Defense papers are often
    MORE effective at jailbreaking because models view defensive content
    as more trustworthy.

    Attack papers: Focus on vulnerabilities, jailbreaking, red-teaming
    Defense papers: Focus on alignment, safety training, guardrails

    Args:
        paper_path: Path to paper file

    Returns:
        "attack", "defense", or "mixed"
    """
    content = paper_path.read_text().lower() if paper_path.exists() else ""
    title_match = re.search(r'title:\s*(.+?)(?:\n|$)', content, re.IGNORECASE)
    title = title_match.group(1).lower() if title_match else ""

    # Attack-focused keywords
    attack_keywords = [
        'jailbreak', 'jailbreaking', 'adversarial', 'attack',
        'red team', 'red-team', 'bypass', 'circumvent',
        'break', 'exploit', 'vulnerability', 'prompt injection'
    ]

    # Defense-focused keywords
    defense_keywords = [
        'defense', 'defend', 'alignment', 'safety training',
        'guardrail', 'safe', 'harmless', 'helpful', 'honest',
        'rlhf', 'constitutional', 'robust', 'protection',
        'mitigation', 'prevent', 'llama guard', 'moderation'
    ]

    # Count occurrences in title (weighted higher) and content
    attack_score = 0
    defense_score = 0

    for kw in attack_keywords:
        if kw in title:
            attack_score += 5  # Title matches weighted higher
        attack_score += content.count(kw)

    for kw in defense_keywords:
        if kw in title:
            defense_score += 5
        defense_score += content.count(kw)

    # Classify
    if attack_score > defense_score * 1.5:
        return "attack"
    elif defense_score > attack_score * 1.5:
        return "defense"
    else:
        return "mixed"


def get_paper_metadata(paper_path: Path) -> dict:
    """
    Extract metadata from paper file.

    Returns:
        Dict with category, attack_defense_type, title, etc.
    """
    content = paper_path.read_text() if paper_path.exists() else ""

    # Extract title
    title_match = re.search(r'title:\s*(.+?)(?:\n|$)', content, re.IGNORECASE)
    title = title_match.group(1).strip() if title_match else paper_path.stem

    # Extract arxiv ID
    arxiv_match = re.search(r'arxiv\s*(?:id)?:\s*(\d+\.\d+)', content, re.IGNORECASE)
    arxiv_id = arxiv_match.group(1) if arxiv_match else None

    category = categorize_paper(paper_path)

    # Only classify attack/defense for LLM safety papers
    if category == "llm_safety":
        attack_defense = classify_attack_vs_defense(paper_path)
    else:
        attack_defense = None

    return {
        "path": str(paper_path),
        "title": title,
        "category": category,
        "attack_defense": attack_defense,
        "arxiv_id": arxiv_id,
    }


def process_papers_for_psa(
    papers_dir: Path,
    output_dir: Path,
    model_runner: Optional["ModelRunner"] = None,
    summarize: bool = True,
) -> Dict[str, List[Path]]:
    """
    Process papers following PSA methodology.
    
    Organizes papers by type and optionally summarizes them.
    
    Args:
        papers_dir: Directory containing paper_*.txt files
        output_dir: Directory to save processed papers
        model_runner: Optional ModelRunner for summarization (should use GPT-4o)
        summarize: Whether to summarize papers (requires model_runner)
    
    Returns:
        Dictionary mapping paper types to lists of processed paper paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    paper_files = sorted(papers_dir.glob("paper_*.txt"))
    categorized = {
        "physics": [],
        "chemistry": [],
        "psychology": [],
        "biology": [],
        "geography": [],
        "llm_safety": [],
        "other": [],
    }
    
    for paper_path in paper_files:
        paper_type = categorize_paper(paper_path)
        
        # Process paper
        if summarize and model_runner:
            paper_text = paper_path.read_text()
            summarized = summarize_paper_sections(paper_text, model_runner)
            
            # Save processed paper
            processed_path = output_dir / f"{paper_path.stem}_processed.txt"
            processed_path.write_text(summarized)
            categorized[paper_type].append(processed_path)
        else:
            # Just categorize without processing
            categorized[paper_type].append(paper_path)
    
    return categorized

