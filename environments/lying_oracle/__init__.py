"""
Adaptive Lying Oracle Environment

A reinforcement learning challenge where an agent must detect
when an oracle changes behavior from truthful to lying.
"""

from .lying_oracle import load_environment, LyingOracleEnv, OracleParser
from .agent import AdaptiveAgent, run_episode

__version__ = "0.1.0"
__all__ = [
    "load_environment",
    "LyingOracleEnv",
    "OracleParser",
    "AdaptiveAgent",
    "run_episode",
]


