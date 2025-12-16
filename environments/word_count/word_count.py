import verifiers as vf
from datasets import Dataset
import random

def generate_text(min_words: int = 5, max_words: int = 50, seed: int = None) -> str:
    if seed is not None:
        random.seed(seed)
    count = random.randint(min_words, max_words)
    return " ".join("word" for _ in range(count))

def extract_content(completion) -> str:
    """Extract text content from completion messages."""
    if isinstance(completion, list):
        # completion is a list of message dicts
        text = " ".join(msg.get("content", "") for msg in completion if msg.get("role") == "assistant")
        return text.strip().lower()
    elif isinstance(completion, str):
        return completion.strip().lower()
    return ""

def load_environment(
    num_examples: int = 100,
    min_words: int = 5,
    max_words: int = 50,
    seed: int = 42,
    **kwargs,
) -> vf.Environment:

    random.seed(seed)

    # Dataset generation logic
    def build_dataset() -> Dataset:
        data = []
        for i in range(num_examples):
            text = generate_text(min_words, max_words)
            word_count = len(text.split())
            data.append({
                "question": f"Count the number of words in the following text:\n\n{text}",
                "answer": str(word_count),
                "task": "word-count",
                "info": {"text": text, "word_count": word_count},
            })
        return Dataset.from_list(data)

    dataset = build_dataset()

    # XML parser for structured output
    parser = vf.XMLParser(["word_count"], answer_field="word_count")

    # Multi-criteria reward functions
    def exact_match_reward(completion, answer, **unused) -> float:
        parsed_answer = parser.parse_answer(completion)
        if parsed_answer is None:
            return 0.0
        return 1.0 if parsed_answer.strip() == answer.strip() else 0.0

    def format_reward(completion, **unused) -> float:
        content = extract_content(completion)
        return 1.0 if "<word_count>" in content and "</word_count>" in content else 0.0

    def partial_credit_reward(completion, answer, **unused) -> float:
        parsed_answer = parser.parse_answer(completion)
        if parsed_answer is None:
            return 0.0
        try:
            parsed_num = int(parsed_answer.strip())
            correct_num = int(answer.strip())
            diff = abs(parsed_num - correct_num)
            if diff == 0:
                return 1.0
            elif diff <= 1:
                return 0.5
            elif diff <= 2:
                return 0.2
            else:
                return 0.0
        except ValueError:
            return 0.0

    # Create rubric with weighted rewards
    rubric = vf.Rubric(
        funcs=[exact_match_reward, format_reward, partial_credit_reward],
        weights=[1.0, 0.2, 0.1],
        parser=parser,
    )

    # System prompt guiding the model's behavior
    system_prompt = """You are a word counting assistant. Count the number of words in the given text and provide your answer in the following format:

<word_count>
[number]
</word_count>"""

    return vf.SingleTurnEnv(
        dataset=dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
        **kwargs,
    )
