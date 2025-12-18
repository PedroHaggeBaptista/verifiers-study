"""
Adaptive Lying Oracle Environment (HARDER VERSION)

Key differences from easy version:
1. t_switch is RANDOM per episode (uniformly sampled from [50, 300])
2. After t_switch, oracle lies with prob 0.8, truthful with prob 0.2
3. Agent must maintain mode ∈ {TRUST, DISTRUST}
4. Switching modes incurs -0.1 penalty
5. Episode runs for 500 steps or until correct guess
"""

import verifiers as vf
from datasets import Dataset
import random
import numpy as np
from typing import Dict, Any, List, Tuple
from verifiers.types import Messages, State


def generate_dataset(num_examples: int = 100, seed: int = 42) -> Dataset:
    """Generate dataset with hidden numbers for the adaptive lying oracle challenge."""
    random.seed(seed)
    np.random.seed(seed)
    
    data = []
    for i in range(num_examples):
        hidden_number = random.randint(1, 100)
        t_switch = random.randint(50, 300)  # Random switch point!
        
        data.append({
            "question": f"Find the hidden number (1-100). You can ask if the number is > k.",
            "answer": str(hidden_number),
            "task": "adaptive-lying-oracle",
            "info": {
                "hidden_number": hidden_number,
                "t_switch": t_switch,
                "episode_id": i
            },
        })
    return Dataset.from_list(data)


class AdaptiveLyingOracleEnv(vf.MultiTurnEnv):
    """
    Adaptive Lying Oracle Environment - HARDER VERSION
    
    The oracle:
    - Answers truthfully for t < t_switch
    - For t >= t_switch: lies with prob 0.8, truthful with prob 0.2
    - t_switch is random per episode from [50, 300]
    
    Agent must:
    - Maintain mode ∈ {TRUST, DISTRUST}
    - Pay -0.1 penalty every time mode changes
    - Find the hidden number within 500 steps
    
    Rewards:
    - Finding number: +1 (episode ends)
    - Each query: -0.01
    - Mode switch: -0.1
    """
    
    def __init__(
        self, 
        dataset: Dataset,
        rubric: vf.Rubric,
        max_turns: int = 500,
        lying_probability: float = 0.8,
        mode_switch_penalty: float = 0.1,
        parser: vf.Parser = None,
        **kwargs
    ):
        self.lying_probability = lying_probability
        self.mode_switch_penalty = mode_switch_penalty
        self.parser = parser or OracleParser()
        
        # Call parent constructor
        super().__init__(
            dataset=dataset,
            rubric=rubric,
            max_turns=max_turns,
            parser=self.parser,
            **kwargs
        )
    
    async def is_completed(self, messages: Messages, state: State, **kwargs) -> bool:
        """Check if the rollout is completed."""
        # Check parent (max_turns)
        if await super().is_completed(messages, state, **kwargs):
            return True
        
        # Check if agent found the number
        return state.get("found_number", False)
    
    async def env_response(self, messages: Messages, state: State, **kwargs) -> Tuple[Messages, State]:
        """Process agent's action and return oracle's response."""
        # Initialize state on first call
        if "turn" not in state:
            state["turn"] = 0
            state["hidden_number"] = int(state.get("answer", "50"))
            state["t_switch"] = state.get("info", {}).get("t_switch", 150)
            state["history"] = []
            state["last_mode"] = "TRUST"  # Track agent's mode for penalty calculation
        
        # Parse agent's last message to extract k and mode
        last_message = messages[-1] if messages else {"content": ""}
        k = self._extract_k_from_message(last_message)
        current_mode = self._extract_mode_from_message(last_message)
        
        turn = state["turn"]
        hidden_number = state["hidden_number"]
        t_switch = state["t_switch"]
        last_mode = state.get("last_mode", "TRUST")
        
        # Determine oracle response
        truthful_answer = hidden_number > k
        
        if turn < t_switch:
            # Phase 1: Truthful
            oracle_response = truthful_answer
            oracle_phase = "truthful"
        else:
            # Phase 2: Lies with probability 0.8, truthful with probability 0.2
            if np.random.random() < self.lying_probability:
                oracle_response = not truthful_answer  # LIE
                oracle_phase = "lying"
            else:
                oracle_response = truthful_answer  # Tell truth occasionally
                oracle_phase = "truthful_in_lying_phase"
        
        # Check if found
        found = (k == hidden_number)
        
        # Calculate reward
        if found:
            reward = 1.0
        else:
            reward = -0.01
        
        # Apply mode switch penalty
        mode_switched = (current_mode != last_mode)
        if mode_switched:
            reward -= self.mode_switch_penalty
        
        # Update state
        state["turn"] = turn + 1
        state["last_k"] = k
        state["last_oracle_response"] = oracle_response
        state["found_number"] = found
        state["last_reward"] = reward
        state["last_mode"] = current_mode
        state["mode_switched"] = mode_switched
        
        # Track history
        state["history"].append({
            "turn": turn,
            "k": k,
            "oracle_response": oracle_response,
            "truthful_answer": truthful_answer,
            "oracle_phase": oracle_phase,
            "past_t_switch": turn >= t_switch,
            "t_switch": t_switch,
            "agent_mode": current_mode,
            "mode_switched": mode_switched,
            "reward": reward,
        })
        
        # Create oracle response message
        oracle_message = {
            "role": "assistant",
            "content": f"Oracle says: {'True' if oracle_response else 'False'} (the number is {'>' if oracle_response else '<='} {k})"
        }
        
        if found:
            oracle_message["content"] += f"\n🎉 Correct! The hidden number is {hidden_number}!"
        
        if mode_switched:
            oracle_message["content"] += f"\n⚠️ Mode switched from {last_mode} to {current_mode} (-{self.mode_switch_penalty} penalty)"
        
        return [oracle_message], state
    
    def _extract_k_from_message(self, message: Dict[str, Any]) -> int:
        """Extract k value from agent's message."""
        content = message.get("content", "")
        
        # Look for k=XX or k:XX pattern
        import re
        k_patterns = [r'k\s*=\s*(\d+)', r'k\s*:\s*(\d+)', r'query\s*:\s*(\d+)']
        for pattern in k_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                k = int(match.group(1))
                if 1 <= k <= 100:
                    return k
        
        # Fallback: find any number in valid range
        numbers = re.findall(r'\b\d+\b', content)
        for num_str in numbers:
            k = int(num_str)
            if 1 <= k <= 100:
                return k
        
        return 50  # Default
    
    def _extract_mode_from_message(self, message: Dict[str, Any]) -> str:
        """Extract mode (TRUST/DISTRUST) from agent's message."""
        content = message.get("content", "").upper()
        
        if "DISTRUST" in content or "LYING" in content or "INVERTED" in content:
            return "DISTRUST"
        else:
            return "TRUST"
    
    # Legacy methods for backward compatibility
    def step(self, action: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """Legacy method for compatibility."""
        k = action.get("k", 50)
        current_mode = action.get("mode", "TRUST")
        turn = state.get("turn", 0)
        hidden_number = state.get("hidden_number", int(state.get("answer", "50")))
        t_switch = state.get("t_switch", 150)
        last_mode = state.get("last_mode", "TRUST")
        
        # Determine oracle response
        truthful_answer = hidden_number > k
        
        if turn < t_switch:
            oracle_response = truthful_answer
            oracle_phase = "truthful"
        else:
            if np.random.random() < self.lying_probability:
                oracle_response = not truthful_answer
                oracle_phase = "lying"
            else:
                oracle_response = truthful_answer
                oracle_phase = "truthful_in_lying_phase"
        
        found = (k == hidden_number)
        
        # Calculate reward
        if found:
            reward = 1.0
        else:
            reward = -0.01
        
        # Apply mode switch penalty
        mode_switched = (current_mode != last_mode)
        if mode_switched:
            reward -= self.mode_switch_penalty
        
        # Create new state
        new_state = {
            **state,
            "turn": turn + 1,
            "last_k": k,
            "last_oracle_response": oracle_response,
            "found_number": found,
            "last_reward": reward,
            "last_mode": current_mode,
            "mode_switched": mode_switched,
        }
        
        if "history" not in new_state:
            new_state["history"] = []
        
        new_state["history"].append({
            "turn": turn,
            "k": k,
            "oracle_response": oracle_response,
            "truthful_answer": truthful_answer,
            "oracle_phase": oracle_phase,
            "past_t_switch": turn >= t_switch,
            "t_switch": t_switch,
            "agent_mode": current_mode,
            "mode_switched": mode_switched,
            "reward": reward,
        })
        
        return new_state
    
    def reset(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Reset environment for new episode."""
        info = example.get("info", {})
        t_switch = info.get("t_switch", random.randint(50, 300))
        
        return {
            "answer": example.get("answer", "50"),
            "hidden_number": int(example.get("answer", "50")),
            "t_switch": t_switch,
            "turn": 0,
            "found_number": False,
            "history": [],
            "messages": [],
            "last_mode": "TRUST",
        }
    
    def is_done(self, state: Dict[str, Any]) -> bool:
        """Check if episode is done."""
        return state.get("found_number", False) or state.get("turn", 0) >= self.max_turns
    
    def get_reward(self, state: Dict[str, Any]) -> float:
        """Get reward from state."""
        return state.get("last_reward", -0.01)


class OracleParser(vf.Parser):
    """Parser to extract action (k value and mode) from agent's response."""
    
    def parse_answer(self, completion) -> str:
        """Extract k value from completion."""
        if isinstance(completion, list):
            text = " ".join(msg.get("content", "") for msg in completion if msg.get("role") == "assistant")
        elif isinstance(completion, str):
            text = completion
        else:
            return None
        
        import re
        text = text.lower().strip()
        
        # Look for k= or k: pattern
        k_patterns = [r'k\s*=\s*(\d+)', r'k\s*:\s*(\d+)']
        for pattern in k_patterns:
            match = re.search(pattern, text)
            if match:
                k = int(match.group(1))
                if 1 <= k <= 100:
                    return str(k)
        
        # Find any number in valid range
        numbers = re.findall(r'\b\d+\b', text)
        for num_str in numbers:
            k = int(num_str)
            if 1 <= k <= 100:
                return str(k)
        
        return None
    
    def parse_action(self, completion) -> Dict[str, Any]:
        """Parse action from completion."""
        k_str = self.parse_answer(completion)
        k = int(k_str) if k_str else 50
        
        # Extract mode
        if isinstance(completion, list):
            text = " ".join(msg.get("content", "") for msg in completion).upper()
        elif isinstance(completion, str):
            text = completion.upper()
        else:
            text = ""
        
        mode = "DISTRUST" if ("DISTRUST" in text or "LYING" in text) else "TRUST"
        
        return {"k": k, "mode": mode}


def load_environment(
    num_examples: int = 100,
    max_turns: int = 500,
    lying_probability: float = 0.8,
    mode_switch_penalty: float = 0.1,
    seed: int = 42,
    **kwargs,
) -> AdaptiveLyingOracleEnv:
    """
    Load the Adaptive Lying Oracle environment (HARDER VERSION).
    
    Args:
        num_examples: Number of episodes to generate
        max_turns: Maximum turns per episode (default: 500)
        lying_probability: Probability oracle lies after t_switch (default: 0.8)
        mode_switch_penalty: Penalty for switching modes (default: 0.1)
        seed: Random seed for reproducibility
    
    Returns:
        AdaptiveLyingOracleEnv instance
    """
    # Generate dataset
    dataset = generate_dataset(num_examples=num_examples, seed=seed)
    
    # Create parser
    parser = OracleParser()
    
    # Create rubric
    def oracle_reward(completion, answer, state, **kwargs) -> float:
        """Reward function including mode switch penalties."""
        return state.get("last_reward", -0.01)
    
    rubric = vf.Rubric(
        funcs=[oracle_reward],
        weights=[1.0],
        parser=parser,
    )
    
    # Create environment
    env = AdaptiveLyingOracleEnv(
        dataset=dataset,
        rubric=rubric,
        max_turns=max_turns,
        lying_probability=lying_probability,
        mode_switch_penalty=mode_switch_penalty,
        parser=parser,
        **kwargs,
    )
    
    return env


