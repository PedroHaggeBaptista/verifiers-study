"""
Tests for the Adaptive Lying Oracle Environment (HARDER VERSION)

Run with: python test_environment.py
"""

import numpy as np
from adaptive_lying_oracle import load_environment, generate_dataset
from adaptive_agent_hard import AdaptiveAgentHard, run_episode


def test_dataset_generation():
    """Test that dataset generates with random t_switch."""
    print("\n" + "="*70)
    print("TEST 1: Dataset Generation")
    print("="*70)
    
    dataset = generate_dataset(num_examples=10, seed=42)
    
    assert len(dataset) == 10, "Dataset should have 10 examples"
    
    t_switches = []
    for i in range(len(dataset)):
        example = dataset[i]
        assert "info" in example
        assert "hidden_number" in example["info"]
        assert "t_switch" in example["info"]
        
        hidden_number = example["info"]["hidden_number"]
        t_switch = example["info"]["t_switch"]
        
        assert 1 <= hidden_number <= 100, f"Hidden number {hidden_number} out of range"
        assert 50 <= t_switch <= 300, f"t_switch {t_switch} out of range [50, 300]"
        
        t_switches.append(t_switch)
    
    # Check that t_switches are diverse (not all the same)
    assert len(set(t_switches)) > 1, "t_switch should vary across episodes"
    
    print(f"✅ Generated {len(dataset)} examples")
    print(f"   Hidden numbers: {[dataset[i]['info']['hidden_number'] for i in range(5)]}...")
    print(f"   t_switch values: {t_switches[:5]}...")
    print(f"   t_switch range: [{min(t_switches)}, {max(t_switches)}]")


def test_environment_initialization():
    """Test environment initialization."""
    print("\n" + "="*70)
    print("TEST 2: Environment Initialization")
    print("="*70)
    
    env = load_environment(num_examples=5, max_turns=500, seed=42)
    
    assert env.max_turns == 500
    assert env.lying_probability == 0.8
    assert env.mode_switch_penalty == 0.1
    
    print("✅ Environment initialized correctly")
    print(f"   Max turns: {env.max_turns}")
    print(f"   Lying probability: {env.lying_probability}")
    print(f"   Mode switch penalty: {env.mode_switch_penalty}")


def test_truthful_phase():
    """Test that oracle is truthful before t_switch."""
    print("\n" + "="*70)
    print("TEST 3: Truthful Phase (t < t_switch)")
    print("="*70)
    
    env = load_environment(num_examples=1, max_turns=500, seed=42)
    
    example = {
        "answer": "67",
        "info": {"hidden_number": 67, "t_switch": 100, "episode_id": 0}
    }
    
    state = env.reset(example)
    
    # Test 10 queries before t_switch
    truthful_count = 0
    for i in range(10):
        k = 50 + i
        action = {"k": k, "mode": "TRUST"}
        state = env.step(action, state)
        
        oracle_response = state["last_oracle_response"]
        truthful_answer = 67 > k
        
        if oracle_response == truthful_answer:
            truthful_count += 1
    
    # Should be 100% truthful before t_switch
    assert truthful_count == 10, f"Oracle should be truthful before t_switch, got {truthful_count}/10"
    
    print(f"✅ Oracle was truthful in {truthful_count}/10 queries before t_switch")


def test_lying_phase():
    """Test that oracle lies probabilistically after t_switch."""
    print("\n" + "="*70)
    print("TEST 4: Lying Phase (t ≥ t_switch)")
    print("="*70)
    
    np.random.seed(42)
    env = load_environment(num_examples=1, max_turns=500, seed=42)
    
    example = {
        "answer": "67",
        "info": {"hidden_number": 67, "t_switch": 0, "episode_id": 0}  # Lies from start
    }
    
    state = env.reset(example)
    
    # Test 100 queries to measure lying rate
    lying_count = 0
    for i in range(100):
        k = 30 + (i % 40)  # Vary k
        action = {"k": k, "mode": "TRUST"}
        state = env.step(action, state)
        
        oracle_response = state["last_oracle_response"]
        truthful_answer = 67 > k
        
        if oracle_response != truthful_answer:
            lying_count += 1
    
    lying_rate = lying_count / 100
    
    # Should lie approximately 80% of the time (with some variance)
    assert 0.65 <= lying_rate <= 0.95, f"Lying rate should be ~0.8, got {lying_rate}"
    
    print(f"✅ Oracle lied in {lying_count}/100 queries (rate: {lying_rate:.2%})")
    print(f"   Expected: ~80%, Got: {lying_rate:.0%}")


def test_mode_switch_penalty():
    """Test that mode switching incurs penalty."""
    print("\n" + "="*70)
    print("TEST 5: Mode Switch Penalty")
    print("="*70)
    
    env = load_environment(num_examples=1, max_turns=500, seed=42)
    
    example = {
        "answer": "67",
        "info": {"hidden_number": 67, "t_switch": 100, "episode_id": 0}
    }
    
    state = env.reset(example)
    
    # Query 1: TRUST mode (no switch, first query)
    action1 = {"k": 50, "mode": "TRUST"}
    state = env.step(action1, state)
    reward1 = state["last_reward"]
    
    # Should be -0.01 (query penalty only, no switch on first turn)
    assert reward1 == -0.01, f"First query should have -0.01 reward, got {reward1}"
    
    # Query 2: Stay in TRUST mode (no switch)
    action2 = {"k": 60, "mode": "TRUST"}
    state = env.step(action2, state)
    reward2 = state["last_reward"]
    
    # Should be -0.01 (query penalty only)
    assert reward2 == -0.01, f"No-switch query should have -0.01 reward, got {reward2}"
    
    # Query 3: Switch to DISTRUST mode (should have penalty)
    action3 = {"k": 70, "mode": "DISTRUST"}
    state = env.step(action3, state)
    reward3 = state["last_reward"]
    mode_switched3 = state["mode_switched"]
    
    # Should be -0.11 (query penalty -0.01 + mode switch penalty -0.1)
    assert mode_switched3, "Mode switch should be detected"
    assert abs(reward3 - (-0.11)) < 0.001, f"Switch query should have -0.11 reward, got {reward3}"
    
    print(f"✅ Mode switch penalty working correctly")
    print(f"   First query (TRUST): {reward1:.3f}")
    print(f"   Stay TRUST: {reward2:.3f}")
    print(f"   Switch to DISTRUST: {reward3:.3f} (includes -0.1 penalty)")


def test_agent_basic():
    """Test that agent can run without errors."""
    print("\n" + "="*70)
    print("TEST 6: Agent Basic Functionality")
    print("="*70)
    
    agent = AdaptiveAgentHard()
    agent.reset()
    
    assert agent.mode == "TRUST", "Agent should start in TRUST mode"
    assert agent.lying_confidence == 0.0, "Initial confidence should be 0"
    
    # Simulate a few steps
    for i in range(10):
        k, mode = agent.select_action(i, last_response=True)
        assert 1 <= k <= 100, f"k should be in range [1, 100], got {k}"
        assert mode in ["TRUST", "DISTRUST"], f"Mode should be TRUST or DISTRUST, got {mode}"
        agent.update(k, True, -0.01, False)
    
    stats = agent.get_statistics()
    assert stats["total_steps"] == 10
    
    print(f"✅ Agent ran 10 steps successfully")
    print(f"   Mode: {agent.mode}")
    print(f"   Lying confidence: {agent.lying_confidence:.3f}")
    print(f"   Search range: {stats['search_range']}")


def test_full_episode():
    """Test a full episode with the agent."""
    print("\n" + "="*70)
    print("TEST 7: Full Episode")
    print("="*70)
    
    np.random.seed(42)
    env = load_environment(num_examples=1, max_turns=500, seed=42)
    
    hidden_number = 67
    t_switch = 150
    
    history, found, turns = run_episode(
        hidden_number=hidden_number,
        t_switch=t_switch,
        max_turns=500,
        verbose=False,
        env=env
    )
    
    assert len(history) > 0, "History should not be empty"
    assert turns <= 500, f"Episode should not exceed 500 turns, got {turns}"
    
    # Check that mode switches were recorded
    mode_switches = sum(1 for h in history if h.get("mode_switched", False))
    
    # Calculate final reward
    final_reward = sum(h["reward"] for h in history)
    
    print(f"✅ Episode completed successfully")
    print(f"   Hidden number: {hidden_number}")
    print(f"   t_switch: {t_switch}")
    print(f"   Found: {found}")
    print(f"   Turns: {turns}")
    print(f"   Mode switches: {mode_switches}")
    print(f"   Final reward: {final_reward:.2f}")
    
    # Check that agent adapted
    agent_modes = [h["agent_mode"] for h in history]
    used_distrust = "DISTRUST" in agent_modes
    
    if t_switch < turns:
        print(f"   Agent used DISTRUST mode: {used_distrust}")


def run_all_tests():
    """Run all tests."""
    print("\n" + "🧪 "*25)
    print("RUNNING ALL TESTS FOR ADAPTIVE LYING ORACLE (HARDER VERSION)")
    print("🧪 "*25)
    
    try:
        test_dataset_generation()
        test_environment_initialization()
        test_truthful_phase()
        test_lying_phase()
        test_mode_switch_penalty()
        test_agent_basic()
        test_full_episode()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70 + "\n")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        raise
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}\n")
        raise


if __name__ == "__main__":
    run_all_tests()

