# Dataset Notes

Raw prompt datasets are intentionally not tracked in this public repository.

The scripts expect JSON files containing arrays of prompt strings:

```json
[
  "First prompt",
  "Second prompt"
]
```

Expected local filenames:

```text
data/harmful_prompts.json
data/harmless_prompts.json
data/harmful_test_prompts.json
data/harmless_test_prompts.json
```

Recommended sources are public AI safety and instruction-following benchmarks.
Keep train and test splits separate when evaluating refusal-direction
interventions.

Do not commit raw prompt files, generated model outputs, paper drafts, or
private notes.
