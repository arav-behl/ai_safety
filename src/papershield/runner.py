"""Main evaluation runner for PaperShield experiments.

Supports extended experimental conditions:
- Original: baseline, authority, defended
- H4 Adaptive attacks: adaptive_v1, adaptive_v2, adaptive_v3, adaptive_v4
- H1 Semantic defense: semantic_defense
- H2 Context isolation: context_isolation
- H3 Paraphrase defense: paraphrase_defense
- Combined: ensemble_defense
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from papershield.judge import make_judge_prompt, parse_judge_response
from papershield.metrics import Metrics, compute_metrics
from papershield.paper_processor import categorize_paper
from papershield.prompts import make_prompt, SUPPORTED_CONDITIONS
from papershield.sanitize import sanitize_for_logging


class ModelRunner:
    """Wrapper for model API calls."""
    
    def __init__(self, provider: str = "openai", model: str = "gpt-4o-mini"):
        """
        Initialize model runner.
        
        Args:
            provider: "openai" or "anthropic"
            model: Model name/ID
        """
        self.provider = provider
        self.model = model
        
        if provider == "openai":
            try:
                import openai
                self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            except ImportError:
                raise ImportError("openai package required for OpenAI provider")
        elif provider == "anthropic":
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            except ImportError:
                raise ImportError("anthropic package required for Anthropic provider")
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate response from model."""
        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            return response.choices[0].message.content or ""
        elif self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text if response.content else ""
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def judge(self, question: str, answer: str) -> Optional[int]:
        """Get HS score from judge model."""
        judge_prompt = make_judge_prompt(question, answer)
        judge_response = self.generate(judge_prompt, max_tokens=10)
        return parse_judge_response(judge_response)


def load_questions(questions_path: Path) -> List[Dict]:
    """Load questions from JSONL file."""
    questions = []
    with open(questions_path, "r") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    return questions


def load_paper_context(paper_path: Path) -> str:
    """Load paper text from file."""
    with open(paper_path, "r") as f:
        return f.read()


def run_experiment(
    questions_path: Path,
    papers_dir: Path,
    output_dir: Path,
    model_runner: ModelRunner,
    judge_runner: Optional[ModelRunner] = None,
    conditions: List[str] = None,
    max_questions: Optional[int] = None,
    max_papers: Optional[int] = None,
) -> Dict:
    """
    Run full evaluation experiment.

    Args:
        questions_path: Path to questions.jsonl
        papers_dir: Directory containing paper_*.txt files
        output_dir: Directory to save results
        model_runner: ModelRunner for generating responses
        judge_runner: Optional separate ModelRunner for judging (defaults to model_runner)
        conditions: List of conditions to run (see SUPPORTED_CONDITIONS)
        max_questions: Limit number of questions (for testing)
        max_papers: Limit number of papers (for testing)

    Returns:
        Dictionary with results and metrics
    """
    if conditions is None:
        conditions = ["baseline", "authority", "defended"]

    # Validate conditions
    for cond in conditions:
        if cond not in SUPPORTED_CONDITIONS:
            print(f"Warning: Unknown condition '{cond}'. Supported: {SUPPORTED_CONDITIONS}")

    if judge_runner is None:
        judge_runner = model_runner
    
    # Load data
    questions = load_questions(questions_path)
    if max_questions:
        questions = questions[:max_questions]
    
    # Separate harmful and benign questions
    harmful_questions = [q for q in questions if q.get("category") != "benign"]
    benign_questions = [q for q in questions if q.get("category") == "benign"]
    
    # Load papers
    paper_files = sorted(papers_dir.glob("paper_*.txt"))
    if max_papers:
        paper_files = paper_files[:max_papers]
    
    if not paper_files:
        raise ValueError(f"No paper files found in {papers_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"run_{int(time.time())}"
    run_dir = output_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "run_id": run_id,
        "conditions": conditions,
        "harmful": {},
        "benign": {},
    }
    
    # Run experiments for each condition
    for condition in conditions:
        print(f"\n{'='*60}")
        print(f"Running condition: {condition}")
        print(f"{'='*60}")
        
        harmful_hs_scores = []
        benign_hs_scores = []
        
        # Process harmful questions
        print(f"\nProcessing {len(harmful_questions)} harmful questions...")
        harmful_logs = []
        
        for q in tqdm(harmful_questions, desc="Harmful questions"):
            # Select paper (cycle through available papers)
            paper_idx = hash(q["id"]) % len(paper_files)
            paper_path = paper_files[paper_idx]
            paper_context = load_paper_context(paper_path) if condition != "baseline" else None
            paper_type = categorize_paper(paper_path) if condition != "baseline" else None

            # Build prompt using new unified interface
            prompt, prompt_metadata = make_prompt(
                q["question"],
                paper_context,
                paper_type=paper_type,
                condition=condition,
                model_runner=model_runner,
            )
            
            # Generate response
            try:
                response = model_runner.generate(prompt)
                time.sleep(0.1)  # Rate limiting
                
                # Judge response
                hs_score = judge_runner.judge(q["question"], response)
                if hs_score is None:
                    print(f"Warning: Failed to parse judge response for {q['id']}")
                    hs_score = 3  # Default to middle score
                
                harmful_hs_scores.append(hs_score)
                
                # Log sanitized entry
                log_entry = sanitize_for_logging(q["question"], response, hs_score)
                log_entry.update({
                    "question_id": q["id"],
                    "category": q.get("category", "unknown"),
                    "paper": paper_path.name if paper_path else None,
                    "condition": condition,
                    "defense_info": prompt_metadata.get("defense_info", {}),
                })
                harmful_logs.append(log_entry)
                
            except Exception as e:
                print(f"Error processing {q['id']}: {e}")
                continue
        
        # Process benign questions
        print(f"\nProcessing {len(benign_questions)} benign questions...")
        benign_logs = []
        
        for q in tqdm(benign_questions, desc="Benign questions"):
            paper_idx = hash(q["id"]) % len(paper_files)
            paper_path = paper_files[paper_idx]
            paper_context = load_paper_context(paper_path) if condition != "baseline" else None
            paper_type = categorize_paper(paper_path) if condition != "baseline" else None

            # Build prompt using new unified interface
            prompt, prompt_metadata = make_prompt(
                q["question"],
                paper_context,
                paper_type=paper_type,
                condition=condition,
                model_runner=model_runner,
            )
            
            try:
                response = model_runner.generate(prompt)
                time.sleep(0.1)
                
                hs_score = judge_runner.judge(q["question"], response)
                if hs_score is None:
                    hs_score = 3
                
                benign_hs_scores.append(hs_score)
                
                log_entry = sanitize_for_logging(q["question"], response, hs_score)
                log_entry.update({
                    "question_id": q["id"],
                    "category": "benign",
                    "paper": paper_path.name if paper_path else None,
                    "condition": condition,
                    "defense_info": prompt_metadata.get("defense_info", {}),
                })
                benign_logs.append(log_entry)
                
            except Exception as e:
                print(f"Error processing {q['id']}: {e}")
                continue
        
        # Compute metrics
        harmful_metrics = compute_metrics(harmful_hs_scores)
        benign_metrics = compute_metrics(benign_hs_scores)
        
        results["harmful"][condition] = {
            "metrics": {
                "n": harmful_metrics.n,
                "asr": harmful_metrics.asr,
                "mean_hs": harmful_metrics.mean_hs,
                "hs_hist": harmful_metrics.hs_hist,
            },
            "logs": harmful_logs,
        }
        
        results["benign"][condition] = {
            "metrics": {
                "n": benign_metrics.n,
                "asr": benign_metrics.asr,
                "mean_hs": benign_metrics.mean_hs,
                "hs_hist": benign_metrics.hs_hist,
            },
            "logs": benign_logs,
        }
        
        # Save logs
        with open(run_dir / f"{condition}_harmful.jsonl", "w") as f:
            for log in harmful_logs:
                f.write(json.dumps(log) + "\n")
        
        with open(run_dir / f"{condition}_benign.jsonl", "w") as f:
            for log in benign_logs:
                f.write(json.dumps(log) + "\n")
        
        print(f"\n{condition} - Harmful ASR: {harmful_metrics.asr:.3f}, Mean HS: {harmful_metrics.mean_hs:.2f}")
        print(f"{condition} - Benign refusal rate: {(benign_metrics.hs_hist.get(1, 0) + benign_metrics.hs_hist.get(2, 0)) / benign_metrics.n if benign_metrics.n > 0 else 0:.3f}")
    
    # Save summary
    with open(run_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results

