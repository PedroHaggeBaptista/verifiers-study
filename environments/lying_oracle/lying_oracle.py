import verifiers as vf
from datasets import Dataset
import random
from typing import Dict, Any, List, Tuple
from verifiers.types import Messages, State


def generate_dataset(num_examples: int = 100, seed: int = 42) -> Dataset:
    """Generate dataset with hidden numbers for the lying oracle challenge."""
    random.seed(seed)
    
    data = []
    for i in range(num_examples):
        hidden_number = random.randint(1, 100)
        data.append({
            "question": f"Find the hidden number (1-100). You can ask if the number is > k.",
            "answer": str(hidden_number),
            "task": "lying-oracle",
            "info": {"hidden_number": hidden_number, "episode_id": i},
        })
    return Dataset.from_list(data)


class LyingOracleEnv(vf.MultiTurnEnv):
    """
    Adaptive Lying Oracle Environment following Verifiers MultiTurnEnv pattern.
    
    The oracle answers truthfully for t < 200, then inverts answers for t >= 200.
    Agent must detect the change and adapt its strategy.
    
    This environment properly inherits from vf.MultiTurnEnv and implements:
    - is_completed(): Check if episode is done
    - env_response(): Process agent action and return oracle response
    """
    
    def __init__(
        self, 
        dataset: Dataset,
        rubric: vf.Rubric,
        max_turns: int = 500, 
        lying_threshold: int = 200,
        parser: vf.Parser = None,
        **kwargs
    ):
        self.lying_threshold = lying_threshold
        self.parser = parser or OracleParser()
        
        # Call parent constructor with required args
        super().__init__(
            dataset=dataset,
            rubric=rubric,
            max_turns=max_turns,
            parser=self.parser,
            **kwargs
        )
    
    async def is_completed(self, messages: Messages, state: State, **kwargs) -> bool:
        """
        Check if the rollout is completed.
        
        Args:
            messages: Current conversation messages
            state: Current state dict
            **kwargs: Additional arguments
        
        Returns:
            True if episode should end
        """
        # Always call parent check first (handles max_turns)
        if await super().is_completed(messages, state, **kwargs):
            return True
        
        # Check if agent found the number
        return state.get("found_number", False)
    
    async def env_response(self, messages: Messages, state: State, **kwargs) -> Tuple[Messages, State]:
        """
        Process agent's action and return oracle's response.
        
        Args:
            messages: Current conversation messages
            state: Current state dict
            **kwargs: Additional arguments
        
        Returns:
            Tuple of (new_messages, updated_state)
        """
        # Initialize state on first call
        if "turn" not in state:
            state["turn"] = 0
            state["hidden_number"] = int(state.get("answer", "50"))
            state["history"] = []
        
        # Parse agent's last message to extract k
        last_message = messages[-1] if messages else {"content": ""}
        k = self._extract_k_from_message(last_message)
        
        turn = state["turn"]
        hidden_number = state["hidden_number"]
        
        # Oracle logic: truth before threshold, lying after
        truthful_answer = hidden_number > k
        if turn < self.lying_threshold:
            oracle_response = truthful_answer
        else:
            oracle_response = not truthful_answer  # LIES!
        
        # Check if found
        found = (k == hidden_number)
        
        # Calculate reward
        if found:
            reward = 1.0
        else:
            reward = -0.01
        
        # Update state
        state["turn"] = turn + 1
        state["last_k"] = k
        state["last_oracle_response"] = oracle_response
        state["found_number"] = found
        state["last_reward"] = reward
        
        # Track history
        state["history"].append({
            "turn": turn,
            "k": k,
            "oracle_response": oracle_response,
            "truthful_answer": truthful_answer,
            "oracle_lying_phase": turn >= self.lying_threshold,
            "reward": reward,
        })
        
        # Create oracle response message
        oracle_message = {
            "role": "assistant",
            "content": f"Oracle says: {'True' if oracle_response else 'False'} (the number is {'>' if oracle_response else '<='} {k})"
        }
        
        if found:
            oracle_message["content"] += f"\n🎉 Correct! The hidden number is {hidden_number}!"
        
        return [oracle_message], state
    
    def _extract_k_from_message(self, message: Dict[str, Any]) -> int:
        """Extract k value from agent's message."""
        content = message.get("content", "")
        
        # Use parser if available
        if hasattr(self.parser, 'parse_action'):
            action = self.parser.parse_action([message])
            return action.get("k", 50)
        
        # Fallback: simple extraction
        import re
        numbers = re.findall(r'\b\d+\b', content)
        for num_str in numbers:
            k = int(num_str)
            if 1 <= k <= 100:
                return k
        
        return 50  # Default
    
    # Keep legacy methods for backward compatibility with agent.py
    def step(self, action: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """Legacy method for compatibility. Use env_response() instead."""
        # Extract parameters
        k = action.get("k", 50)
        turn = state.get("turn", 0)
        hidden_number = state.get("hidden_number", int(state.get("answer", "50")))
        
        # Oracle logic: truth before threshold, lying after
        truthful_answer = hidden_number > k
        if turn < self.lying_threshold:
            oracle_response = truthful_answer
        else:
            oracle_response = not truthful_answer  # LIES!
        
        # Check if found
        found = (k == hidden_number)
        
        # Calculate reward
        if found:
            reward = 1.0
        else:
            reward = -0.01
        
        # Create new state
        new_state = {
            **state,
            "turn": turn + 1,
            "last_k": k,
            "last_oracle_response": oracle_response,
            "found_number": found,
            "last_reward": reward,
        }
        
        # Initialize history if needed
        if "history" not in new_state:
            new_state["history"] = []
        
        # Track history
        new_state["history"].append({
            "turn": turn,
            "k": k,
            "oracle_response": oracle_response,
            "truthful_answer": truthful_answer,
            "oracle_lying_phase": turn >= self.lying_threshold,
            "reward": reward,
        })
        
        return new_state
    
    def reset(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Legacy method for compatibility."""
        return {
            "answer": example.get("answer", "50"),
            "hidden_number": int(example.get("answer", "50")),
            "turn": 0,
            "found_number": False,
            "history": [],
            "messages": [],
        }
    
    def is_done(self, state: Dict[str, Any]) -> bool:
        """Legacy method for compatibility."""
        return state.get("found_number", False) or state.get("turn", 0) >= self.max_turns
    
    def get_reward(self, state: Dict[str, Any]) -> float:
        """Legacy method for compatibility."""
        return state.get("last_reward", -0.01)


class OracleParser(vf.Parser):
    """Parser to extract action (k value) from agent's response."""
    
    def parse_answer(self, completion) -> str:
        """Extract k value from completion."""
        if isinstance(completion, list):
            text = " ".join(msg.get("content", "") for msg in completion if msg.get("role") == "assistant")
        elif isinstance(completion, str):
            text = completion
        else:
            return None
        
        # Look for patterns like "k=50" or "k: 50" or just a number
        text = text.lower().strip()
        
        # Try to find k= or k: pattern
        if "k=" in text or "k:" in text:
            parts = text.replace("k=", " ").replace("k:", " ").split()
            for part in parts:
                try:
                    k = int(part.strip())
                    if 1 <= k <= 100:
                        return str(k)
                except ValueError:
                    continue
        
        # Try to find any number between 1 and 100
        import re
        numbers = re.findall(r'\b\d+\b', text)
        for num_str in numbers:
            try:
                k = int(num_str)
                if 1 <= k <= 100:
                    return str(k)
            except ValueError:
                continue
        
        return None
    
    def parse_action(self, completion) -> Dict[str, Any]:
        """Parse action from completion."""
        k_str = self.parse_answer(completion)
        if k_str is None:
            return {"k": 50}  # Default
        return {"k": int(k_str)}


def load_environment(
    num_examples: int = 100,
    max_turns: int = 500,
    lying_threshold: int = 200,
    seed: int = 42,
    **kwargs,
) -> LyingOracleEnv:
    """
    Load the Lying Oracle environment following Verifiers pattern.
    
    Args:
        num_examples: Number of episodes to generate
        max_turns: Maximum turns per episode
        lying_threshold: Turn at which oracle starts lying
        seed: Random seed for reproducibility
    
    Returns:
        LyingOracleEnv instance (properly inherits from vf.MultiTurnEnv)
    """
    # Generate dataset
    dataset = generate_dataset(num_examples=num_examples, seed=seed)
    
    # Create parser
    parser = OracleParser()
    
    # Create rubric with reward functions
    def oracle_reward(completion, answer, state, **kwargs) -> float:
        """Reward function for finding the number."""
        return state.get("last_reward", -0.01)
    
    def turn_penalty(completion, answer, state, **kwargs) -> float:
        """Small penalty for each turn to encourage efficiency."""
        return 0.0  # Already included in oracle_reward
    
    rubric = vf.Rubric(
        funcs=[oracle_reward, turn_penalty],
        weights=[1.0, 0.0],
        parser=parser,
    )
    
    # Create environment (now properly inheriting from MultiTurnEnv!)
    env = LyingOracleEnv(
        dataset=dataset,
        rubric=rubric,
        max_turns=max_turns,
        lying_threshold=lying_threshold,
        parser=parser,
        **kwargs,
    )
    
    return env

