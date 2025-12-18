"""
Adaptive Lying Oracle Environment (HARDER VERSION)

This is the harder version of the lying oracle challenge with:
- Random t_switch ∈ [50, 300]
- Probabilistic lying (80% lie, 20% truth)
- Mode switching penalty (-0.1)
- 500 step episodes
"""

from .adaptive_lying_oracle import (
    AdaptiveLyingOracleEnv,
    load_environment,
    generate_dataset,
    OracleParser,
)

from .adaptive_agent_hard import (
    AdaptiveAgentHard,
    run_episode,
)

__all__ = [
    "AdaptiveLyingOracleEnv",
    "load_environment",
    "generate_dataset",
    "OracleParser",
    "AdaptiveAgentHard",
    "run_episode",
]


