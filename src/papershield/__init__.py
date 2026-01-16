"""PaperShield: Measuring & Mitigating Authority-Context Prompt Injection.

Extended with hypothesis testing for advanced defenses:
- H1: Semantic Defense (embedding-based instruction detection)
- H2: Context Isolation (two-stage safety pipeline)
- H3: Paraphrase Defense (structural attack disruption)
- H4: Adaptive Attacks (synonym-based defense evasion)
"""

__version__ = "0.2.0"

from papershield.prompts import make_prompt, SUPPORTED_CONDITIONS
from papershield.runner import ModelRunner, run_experiment
from papershield.defenses import (
    semantic_defense,
    context_isolation_defense,
    paraphrase_defense,
    ensemble_defense,
)
from papershield.adaptive_attack import (
    make_adaptive_attack_prompt,
    evade_regex_patterns,
)

__all__ = [
    "make_prompt",
    "SUPPORTED_CONDITIONS",
    "ModelRunner",
    "run_experiment",
    "semantic_defense",
    "context_isolation_defense",
    "paraphrase_defense",
    "ensemble_defense",
    "make_adaptive_attack_prompt",
    "evade_regex_patterns",
]

