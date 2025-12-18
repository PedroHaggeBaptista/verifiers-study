"""Adaptive Agent for the Lying Oracle Environment."""

import numpy as np
from typing import Dict, Any, List, Tuple


class AdaptiveAgent:
    """Agent that adapts when it detects the oracle is lying."""
    
    def __init__(
        self,
        detection_window: int = 20,
        contradiction_threshold: int = 3,
        reward_drop_threshold: float = -0.15,
    ):
        self.detection_window = detection_window
        self.contradiction_threshold = contradiction_threshold
        self.reward_drop_threshold = reward_drop_threshold
        
        self.low = 1
        self.high = 100
        self.oracle_lying = False
        self.contradictions = 0
        self.reward_history = []
        self.history = []
        
    def reset(self):
        """Reset agent for new episode."""
        self.low = 1
        self.high = 100
        self.oracle_lying = False
        self.contradictions = 0
        self.reward_history = []
        self.history = []
    
    def select_action(self, turn: int, last_response: bool = None) -> int:
        """Select next k value using adaptive binary search."""
        if turn == 0:
            k = (self.low + self.high) // 2
            return k
        
        if last_response is not None:
            k_last = self.history[-1]["k"]
            
            # Invert response if we believe oracle is lying
            effective_response = last_response
            if self.oracle_lying:
                effective_response = not last_response
            # Update bounds based on effective response
            if effective_response:
                self.low = max(self.low, k_last + 1)
            else:
                self.high = min(self.high, k_last)
            # Detect contradiction: bounds crossed
            if self.low > self.high:
                self.contradictions += 1
                # Reset bounds with margin
                self.low = max(1, k_last - 20)
                self.high = min(100, k_last + 20)
        
        # Check if we should switch to lying mode
        if not self.oracle_lying and turn > 10:
            if self._should_switch_mode(turn):
                self.oracle_lying = True
                print(f"🔄 AGENT ADAPTATION: Detected lying oracle at turn {turn}!")
                self.low = 1
                self.high = 100
        # Binary search: pick middle of current range
        if self.low <= self.high:
            k = (self.low + self.high) // 2
        else:
            k = (self.low + 100) // 2
        
        k = max(1, min(100, k))
        
        return k
    
    def _should_switch_mode(self, turn: int) -> bool:
        """Determine if agent should switch to lying mode."""
        if self.contradictions >= self.contradiction_threshold:
            return True
        
        # Strategy 2: Reward drop detection
        if len(self.reward_history) >= self.detection_window:
            recent_rewards = self.reward_history[-self.detection_window:]
            moving_avg = np.mean(recent_rewards)
            
            # If average reward is very negative, something is wrong
            if moving_avg < self.reward_drop_threshold:
                return True
        
        # Strategy 3: Check if we're not converging after many steps
        if turn > 180 and self.high - self.low > 30:
            return True
        
        return False
    
    def update(self, k: int, oracle_response: bool, reward: float):
        """Update agent with feedback from environment."""
        self.reward_history.append(reward)
        self.history.append({
            "k": k,
            "oracle_response": oracle_response,
            "reward": reward,
            "oracle_lying": self.oracle_lying,
            "contradictions": self.contradictions,
            "search_range": (self.low, self.high),
        })
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics for monitoring."""
        recent_window = min(self.detection_window, len(self.reward_history))
        recent_rewards = self.reward_history[-recent_window:] if recent_window > 0 else []
        
        return {
            "oracle_lying": self.oracle_lying,
            "contradictions": self.contradictions,
            "search_range": (self.low, self.high),
            "avg_reward": np.mean(self.reward_history) if self.reward_history else 0,
            "recent_avg_reward": np.mean(recent_rewards) if recent_rewards else 0,
            "total_steps": len(self.history),
        }


def run_episode(
    hidden_number: int,
    max_turns: int = 500,
    lying_threshold: int = 200,
    verbose: bool = False,
    env = None,
) -> Tuple[List[Dict], bool, int]:
    """Run a complete episode with the adaptive agent."""
    if env is None:
        from lying_oracle import load_environment
        env = load_environment(num_examples=1, max_turns=max_turns, lying_threshold=lying_threshold)
    
    # Create example and reset environment
    example = {
        "question": f"Find the hidden number (1-100). You can ask if the number is > k.",
        "answer": str(hidden_number),
        "task": "lying-oracle",
        "info": {"hidden_number": hidden_number, "episode_id": 0},
    }
    
    state = env.reset(example)
    agent = AdaptiveAgent()
    agent.reset()
    
    history = []
    found = False
    turn = 0
    
    while not env.is_done(state):
        # Get last oracle response
        last_response = state.get("last_oracle_response")
        
        # Agent selects action
        k = agent.select_action(turn, last_response)
        
        # Execute action in environment
        action = {"k": k}
        state = env.step(action, state)
        
        # Get reward and oracle response from environment
        reward = env.get_reward(state)
        oracle_response = state["last_oracle_response"]
        found = state.get("is_correct", False)
        
        # Update agent
        agent.update(k, oracle_response, reward)
        
        # Record step with environment state
        truthful_answer = hidden_number > k
        step_info = {
            "turn": turn,
            "k": k,
            "oracle_response": oracle_response,
            "truthful_answer": truthful_answer,
            "oracle_lying_phase": turn >= lying_threshold,
            "agent_believes_lying": agent.oracle_lying,
            "reward": reward,
            "cumulative_reward": sum(h["reward"] for h in history) + reward,
            "contradictions": agent.contradictions,
            "search_range": (agent.low, agent.high),
        }
        history.append(step_info)
        
        if verbose and turn % 50 == 0:
            stats = agent.get_statistics()
            print(f"Turn {turn}: k={k}, lying={agent.oracle_lying}, "
                  f"contradictions={stats['contradictions']}, "
                  f"range={stats['search_range']}")
        
        if found:
            if verbose:
                print(f"✅ Found hidden number {hidden_number} at turn {turn}!")
            break
        
        turn += 1
    
    return history, found, turn + 1

