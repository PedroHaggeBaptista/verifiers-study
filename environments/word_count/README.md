# word-count

### Overview
- **Environment ID**: `word-count`
- **Short description**: A simple environment for testing LLM ability to count words in text
- **Tags**: train, eval, counting, basic-reasoning

### Datasets
- **Primary dataset(s)**: Synthetically generated text with varying word counts
- **Source links**: Generated in code
- **Split sizes**: Configurable (default: 100 examples)

### Task
- **Type**: Single-turn
- **Parser**: XMLParser with `<word_count>` tags
- **Rubric overview**: 
  - Exact match reward (weight: 1.0)
  - Format adherence reward (weight: 0.2)
  - Partial credit for close answers (weight: 0.1)

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval word-count
```

Configure model and sampling:

```bash
uv run vf-eval word-count   -m gpt-4.1-mini   -n 20 -r 3 -t 1024 -T 0.7   -a '{"key": "value"}'  # env-specific args as JSON
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `num_examples` | int | `100` | Number of examples to generate |
| `min_words` | int | `5` | Minimum words per example |
| `max_words` | int | `50` | Maximum words per example |
| `seed` | int | `42` | Random seed for reproducibility |

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `reward` | Main scalar reward (weighted sum: exact_match × 1.0 + format × 0.2 + partial × 0.1) |
| `exact_match` | Binary indicator: 1.0 if answer exactly matches, 0.0 otherwise |
| `format` | Binary indicator: 1.0 if proper XML tags used, 0.0 otherwise |
| `partial_credit` | Graded reward: 1.0 (exact), 0.5 (±1 word), 0.2 (±2 words), 0.0 (>2 off) |

### Testing

Run unit tests:
```bash
cd environments/word_count
uv run test_environment.py
```

Or test from the root directory:
```bash
python test_evaluation.py
```

### Tests
To run the tests, run the following command:

```bash
uv run test_environment.py
```

![tests_run.png](./tests_run.png)