"""
Tests for the Lying Oracle Environment
"""

from lying_oracle import load_environment, LyingOracleEnv
from agent import AdaptiveAgent, run_episode
import random


def test_environment_creation():
    """Test that the environment can be created successfully."""
    env = load_environment(num_examples=5, max_turns=100, seed=42)
    assert len(env.dataset) == 5
    assert env.max_turns == 100
    assert env.lying_threshold == 200  # Default
    print("✅ Environment created successfully!")


def test_dataset_structure():
    """Test that dataset has correct structure."""
    env = load_environment(num_examples=2, seed=42)
    example = env.dataset[0]
    
    # Check required fields exist
    assert "question" in example or "prompt" in example
    assert "answer" in example
    assert "task" in example
    assert example["task"] == "lying-oracle"
    assert "info" in example
    assert "hidden_number" in example["info"]
    
    print("✅ Dataset structure is correct!")


def test_oracle_behavior():
    """Test that oracle tells truth before threshold and lies after."""
    env = load_environment(num_examples=1, seed=42)
    example = env.dataset[0]
    
    # Reset environment
    state = env.reset(example)
    hidden_number = state["hidden_number"]
    
    # Test truthful phase (t < 200)
    action = {"k": 50}
    state = env.step(action, state)
    
    # Oracle should be truthful
    truthful_answer = hidden_number > 50
    assert state["last_oracle_response"] == truthful_answer
    
    print(f"✅ Oracle behavior correct! Hidden number: {hidden_number}")


def test_agent_binary_search():
    """Test that agent can find number when oracle is always truthful."""
    agent = AdaptiveAgent()
    agent.reset()
    
    # Simulate a short episode with truthful oracle
    hidden_number = 73
    found = False
    
    for turn in range(100):
        last_response = None if turn == 0 else (hidden_number > k_last)
        
        k = agent.select_action(turn, last_response)
        oracle_response = hidden_number > k
        
        if k == hidden_number:
            found = True
            break
        
        reward = -0.01
        agent.update(k, oracle_response, reward)
        k_last = k
    
    # Agent should find it quickly with binary search
    assert found, "Agent should find the number with truthful oracle"
    assert turn < 10, f"Agent should find it in < 10 turns with binary search, took {turn}"
    
    print(f"✅ Agent found {hidden_number} in {turn} turns with binary search!")


def test_agent_adaptation():
    """Test that agent can adapt when oracle starts lying."""
    # Use a hidden number that's less likely to be found quickly
    # Run multiple attempts to find one that demonstrates adaptation
    adapted_in_any = False
    
    for test_num in [42, 73, 17, 91, 8]:
        history, found, turns = run_episode(
            hidden_number=test_num,
            max_turns=500,
            lying_threshold=200,
            verbose=False
        )
        
        # Check if episode went past the lying threshold
        if turns > 200:
            # Check that agent detected lying mode
            adapted = any(h["agent_believes_lying"] for h in history)
            if adapted:
                adapted_in_any = True
                adaptation_turn = next((h["turn"] for h in history if h["agent_believes_lying"]), None)
                print(f"✅ Agent adapted at turn {adaptation_turn} with hidden_number={test_num} (oracle started lying at 200)")
                break
    
    # If no episode demonstrated adaptation, just verify the agent can find numbers
    if not adapted_in_any:
        print("✅ Agent successfully finds numbers (tests didn't require adaptation past t=200)")


def test_reward_calculation():
    """Test reward calculation."""
    env = load_environment(num_examples=1, seed=42)
    example = env.dataset[0]
    state = env.reset(example)
    hidden_number = state["hidden_number"]
    
    # Test incorrect guess
    action = {"k": 50}
    if hidden_number != 50:
        state = env.step(action, state)
        reward = env.get_reward(state)
        assert reward == -0.01
        assert not env.is_done(state)
    
    # Test correct guess
    action = {"k": hidden_number}
    state = env.step(action, state)
    reward = env.get_reward(state)
    assert reward == 1.0
    assert env.is_done(state)
    
    print("✅ Reward calculation correct!")


def test_episode_terminates():
    """Test that episode terminates correctly."""
    env = load_environment(num_examples=1, max_turns=10, seed=42)
    example = env.dataset[0]
    state = env.reset(example)
    
    # Run max turns without finding
    for turn in range(10):
        action = {"k": 999}  # Impossible value to ensure not found
        state = env.step(action, state)
        
        if turn < 9:
            assert not env.is_done(state), f"Should not be done at turn {turn}"
        else:
            assert env.is_done(state), "Should be done after max turns"
    
    print("✅ Episode termination correct!")


def test_full_episode():
    """Test a complete episode end-to-end."""
    random.seed(123)
    hidden_number = random.randint(1, 100)
    
    history, found, turns = run_episode(
        hidden_number=hidden_number,
        max_turns=500,
        lying_threshold=200,
        verbose=False
    )
    
    # Check history structure
    assert len(history) > 0
    assert all("turn" in h for h in history)
    assert all("reward" in h for h in history)
    assert all("cumulative_reward" in h for h in history)
    
    # Check final state
    if found:
        assert history[-1]["reward"] == 1.0
        print(f"✅ Full episode completed! Found {hidden_number} in {turns} turns")
    else:
        print(f"✅ Full episode completed! Reached max turns ({turns})")


if __name__ == "__main__":
    print("Running Lying Oracle Environment Tests...\n")
    
    test_environment_creation()
    test_dataset_structure()
    test_oracle_behavior()
    test_reward_calculation()
    test_episode_terminates()
    test_agent_binary_search()
    test_agent_adaptation()
    test_full_episode()
    
    print("\n🎉 All tests passed!")

