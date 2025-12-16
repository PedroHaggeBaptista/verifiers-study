from word_count import load_environment

def test_environment_creation():
    """Test that the environment can be created successfully."""
    env = load_environment(num_examples=5, min_words=3, max_words=8, seed=42)
    assert len(env.dataset) == 5
    print("✅ Environment created successfully!")

def test_dataset_structure():
    """Test that dataset has correct structure."""
    env = load_environment(num_examples=2, seed=42)
    example = env.dataset[0]
    
    # Check required fields exist
    assert "question" in example or "prompt" in example
    assert "answer" in example
    assert "task" in example
    assert example["task"] == "word-count"
    
    print("✅ Dataset structure is correct!")

def test_parser():
    """Test the XML parser."""
    env = load_environment(num_examples=1, seed=42)
    
    # Test correct format
    correct_text = "<word_count>\n42\n</word_count>"
    parsed = env.parser.parse_answer([{"role": "assistant", "content": correct_text}])
    assert parsed == "42", f"Expected '42', got '{parsed}'"
    
    # Test incorrect format
    incorrect_text = "The answer is 42"
    parsed_incorrect = env.parser.parse_answer([{"role": "assistant", "content": incorrect_text}])
    assert parsed_incorrect is None or parsed_incorrect == "", "Parser should return None or empty for incorrect format"
    
    print("✅ Parser works correctly!")

if __name__ == "__main__":
    print("Running tests...\n")
    test_environment_creation()
    test_dataset_structure()
    test_parser()
    print("\n🎉 All tests passed!")