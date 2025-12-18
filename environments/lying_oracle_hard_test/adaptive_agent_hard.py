"""
Adaptive Agent for the HARDER Lying Oracle Environment.

This agent:
1. Starts in TRUST mode with binary search
2. Tracks multiple statistics to detect lying
3. Switches to DISTRUST mode when evidence is strong
4. Avoids frequent mode switching due to -0.1 penalty
5. Uses confidence-based switching with hysteresis
"""

import numpy as np
from typing import Dict, Any, List, Tuple


class AdaptiveAgentHard:
    """
    An agent that carefully adapts when detecting the oracle is lying.
    
    Key differences from easy version:
    - Mode switching has a cost (-0.1), so we need confidence before switching
    - Oracle lies probabilistically (80% lie, 20% truth), making detection harder
    - Uses multiple detection signals with hysteresis to avoid rapid toggling
    - Maintains confidence score that must cross thresholds to switch modes
    """
    
    def __init__(
        self,
        detection_window: int = 30,
        trust_to_distrust_threshold: float = 0.7,  # High confidence needed to switch
        distrust_to_trust_threshold: float = 0.3,  # Hysteresis: harder to switch back
        min_observations: int = 20,  # Minimum observations before switching
    ):
        """
        Initialize adaptive agent.
        
        Args:
            detection_window: Window size for statistics
            trust_to_distrust_threshold: Confidence level to switch from TRUST to DISTRUST
            distrust_to_trust_threshold: Confidence level to switch back to TRUST
            min_observations: Minimum turns before considering mode switch
        """
        self.detection_window = detection_window
        self.trust_to_distrust_threshold = trust_to_distrust_threshold
        self.distrust_to_trust_threshold = distrust_to_trust_threshold
        self.min_observations = min_observations
        
        # Agent state
        self.mode = "TRUST"
        self.low = 1
        self.high = 100
        self.contradictions = 0
        self.reward_history = []
        self.history = []
        self.lying_confidence = 0.0  # Confidence that oracle is lying (0-1)
        
    def reset(self):
        """Reset agent for new episode."""
        self.mode = "TRUST"
        self.low = 1
        self.high = 100
        self.contradictions = 0
        self.reward_history = []
        self.history = []
        self.lying_confidence = 0.0
    
    def select_action(self, turn: int, last_response: bool = None) -> Tuple[int, str]:
        """
        Select next k value and mode using adaptive binary search.
        
        Args:
            turn: Current turn number
            last_response: Oracle's response to last query (True/False)
        
        Returns:
            Tuple of (k value, mode)
        """
        # First turn: start in the middle with TRUST mode
        if turn == 0:
            k = (self.low + self.high) // 2
            return k, self.mode
        
        # Update search space based on oracle response
        if last_response is not None:
            k_last = self.history[-1]["k"]
            
            # Interpret response based on current mode
            if self.mode == "TRUST":
                effective_response = last_response
            else:  # DISTRUST mode: assume oracle is lying
                effective_response = not last_response
            
            # Update bounds
            if effective_response:  # hidden_number > k
                new_low = max(self.low, k_last + 1)
                if new_low > self.high:
                    self.contradictions += 1
                    # Reset with margin
                    self.low = max(1, k_last - 15)
                    self.high = min(100, k_last + 15)
                else:
                    self.low = new_low
            else:  # hidden_number <= k
                new_high = min(self.high, k_last)
                if new_high < self.low:
                    self.contradictions += 1
                    # Reset with margin
                    self.low = max(1, k_last - 15)
                    self.high = min(100, k_last + 15)
                else:
                    self.high = new_high
        
        # Update lying confidence and check if we should switch modes
        if turn >= self.min_observations:
            self._update_lying_confidence(turn)
            self._check_mode_switch(turn)
        
        # Binary search: pick middle of current range
        if self.low <= self.high:
            k = (self.low + self.high) // 2
        else:
            # Recovery: random sample around last k
            k_last = self.history[-1]["k"] if self.history else 50
            k = max(1, min(100, k_last + np.random.randint(-10, 11)))
        
        # Ensure k is in valid range
        k = max(1, min(100, k))
        
        return k, self.mode
    
    def _update_lying_confidence(self, turn: int):
        """
        Update confidence that oracle is lying based on multiple signals.
        
        Signals:
        1. Contradiction rate
        2. Reward degradation
        3. Search space convergence failure
        4. Pattern of responses
        """
        if len(self.history) < self.min_observations:
            return
        
        # Signal 1: Contradiction rate
        recent_history = self.history[-self.detection_window:]
        contradiction_rate = sum(1 for h in recent_history if h.get("contradiction", False)) / len(recent_history)
        contradiction_signal = min(1.0, contradiction_rate * 5)  # Scale up
        
        # Signal 2: Reward degradation
        if len(self.reward_history) >= self.detection_window:
            recent_rewards = self.reward_history[-self.detection_window:]
            avg_reward = np.mean(recent_rewards)
            # If average reward is very negative, something is wrong
            reward_signal = max(0.0, min(1.0, (-avg_reward - 0.01) / 0.2))
        else:
            reward_signal = 0.0
        
        # Signal 3: Convergence failure
        range_size = self.high - self.low
        if turn > 100 and range_size > 40:
            convergence_signal = 0.8
        elif turn > 150 and range_size > 25:
            convergence_signal = 0.9
        else:
            convergence_signal = 0.0
        
        # Combine signals with weights
        new_confidence = (
            0.4 * contradiction_signal +
            0.3 * reward_signal +
            0.3 * convergence_signal
        )
        
        # Smooth confidence with exponential moving average
        alpha = 0.3  # Smoothing factor
        self.lying_confidence = alpha * new_confidence + (1 - alpha) * self.lying_confidence
    
    def _check_mode_switch(self, turn: int):
        """
        Check if agent should switch modes based on confidence.
        
        Uses hysteresis to avoid rapid toggling:
        - TRUST -> DISTRUST requires high confidence (0.7)
        - DISTRUST -> TRUST requires low confidence (0.3)
        """
        if self.mode == "TRUST":
            # Switch to DISTRUST if confidence is high enough
            if self.lying_confidence >= self.trust_to_distrust_threshold:
                print(f"🔄 AGENT SWITCH: TRUST → DISTRUST at turn {turn} "
                      f"(confidence: {self.lying_confidence:.2f})")
                self.mode = "DISTRUST"
                # Reset search space to explore with new interpretation
                self.low = 1
                self.high = 100
                self.contradictions = 0
        
        elif self.mode == "DISTRUST":
            # Switch back to TRUST if confidence drops low enough
            if self.lying_confidence <= self.distrust_to_trust_threshold:
                print(f"🔄 AGENT SWITCH: DISTRUST → TRUST at turn {turn} "
                      f"(confidence: {self.lying_confidence:.2f})")
                self.mode = "TRUST"
                # Reset search space
                self.low = 1
                self.high = 100
                self.contradictions = 0
    
    def update(self, k: int, oracle_response: bool, reward: float, mode_switched: bool):
        """
        Update agent with feedback from environment.
        
        Args:
            k: The k value queried
            oracle_response: Oracle's response
            reward: Reward received
            mode_switched: Whether mode switch occurred (from env perspective)
        """
        self.reward_history.append(reward)
        
        # Detect if this response caused a contradiction
        contradiction = False
        if len(self.history) > 0:
            if self.low > self.high:
                contradiction = True
        
        self.history.append({
            "k": k,
            "oracle_response": oracle_response,
            "reward": reward,
            "mode": self.mode,
            "lying_confidence": self.lying_confidence,
            "contradictions": self.contradictions,
            "search_range": (self.low, self.high),
            "mode_switched": mode_switched,
            "contradiction": contradiction,
        })
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics for monitoring."""
        recent_window = min(self.detection_window, len(self.reward_history))
        recent_rewards = self.reward_history[-recent_window:] if recent_window > 0 else []
        
        return {
            "mode": self.mode,
            "lying_confidence": self.lying_confidence,
            "contradictions": self.contradictions,
            "search_range": (self.low, self.high),
            "avg_reward": np.mean(self.reward_history) if self.reward_history else 0,
            "recent_avg_reward": np.mean(recent_rewards) if recent_rewards else 0,
            "total_steps": len(self.history),
        }


def run_episode(
    hidden_number: int,
    max_turns: int = 500,
    t_switch: int = None,
    verbose: bool = False,
    env = None,
) -> Tuple[List[Dict], bool, int]:
    """
    Run a complete episode with the adaptive agent.
    
    Args:
        hidden_number: The number to find
        max_turns: Maximum number of turns
        t_switch: Turn when oracle starts lying (random if None)
        verbose: Print debug information
        env: AdaptiveLyingOracleEnv instance
    
    Returns:
        Tuple of (history, success, turns_taken)
    """
    # Import here to avoid circular dependency
    if env is None:
        from adaptive_lying_oracle import load_environment
        env = load_environment(num_examples=1, max_turns=max_turns)
    
    # Create example
    if t_switch is None:
        t_switch = np.random.randint(50, 300)
    
    example = {
        "question": f"Find the hidden number (1-100). You can ask if the number is > k.",
        "answer": str(hidden_number),
        "task": "adaptive-lying-oracle",
        "info": {"hidden_number": hidden_number, "t_switch": t_switch, "episode_id": 0},
    }
    
    state = env.reset(example)
    agent = AdaptiveAgentHard()
    agent.reset()
    
    history = []
    found = False
    turn = 0
    
    if verbose:
        print(f"🎯 Starting episode: hidden_number={hidden_number}, t_switch={t_switch}")
    
    while not env.is_done(state):
        # Get last oracle response
        last_response = state.get("last_oracle_response")
        
        # Agent selects action
        k, mode = agent.select_action(turn, last_response)
        
        # Execute action in environment
        action = {"k": k, "mode": mode}
        state = env.step(action, state)
        
        # Get feedback from environment
        reward = env.get_reward(state)
        oracle_response = state["last_oracle_response"]
        found = state.get("found_number", False)
        mode_switched = state.get("mode_switched", False)
        
        # Update agent
        agent.update(k, oracle_response, reward, mode_switched)
        
        # Record step
        if state["history"]:
            env_step = state["history"][-1]
            step_info = {
                **env_step,
                "agent_mode": mode,
                "lying_confidence": agent.lying_confidence,
                "cumulative_reward": sum(h["reward"] for h in history) + reward,
            }
            history.append(step_info)
        
        if verbose and turn % 50 == 0:
            stats = agent.get_statistics()
            print(f"Turn {turn}: k={k}, mode={mode}, confidence={agent.lying_confidence:.2f}, "
                  f"contradictions={stats['contradictions']}, range={stats['search_range']}")
        
        if found:
            if verbose:
                print(f"✅ Found hidden number {hidden_number} at turn {turn}!")
            break
        
        turn += 1
    
    return history, found, turn + 1


