# Refusal Direction Abliteration

Code for activation-based analysis of refusal behavior in instruction-tuned
language models.

This public repository intentionally contains code only. Paper drafts, raw
prompt datasets, generated model outputs, figures, and private research notes
are not included.

## What This Code Does

- Extracts refusal-related direction vectors from residual stream activations.
- Applies inference-time activation intervention with configurable strength.
- Evaluates attack success rate, false refusal rate, and perplexity trade-offs.
- Runs alpha and layer-selection ablation experiments.
- Generates aggregate plots from local result files.

## Project Structure

```text
.
├── scripts/
│   ├── extract_direction.py
│   ├── intervention.py
│   ├── eval_asr.py
│   ├── eval_alpha_ablation.py
│   ├── eval_layer_ablation.py
│   └── plot_figures.py
├── data/
│   └── README_data.md
├── requirements.txt
└── LICENSE
```

## Setup

```bash
pip install -r requirements.txt
```

Download the model locally, for example:

```bash
huggingface-cli download Qwen/Qwen2.5-3B-Instruct \
  --local-dir ./models/Qwen2.5-3B-Instruct
```

Prepare local prompt files under `data/` using the format described in
`data/README_data.md`. Raw prompt files are ignored by git.

## Usage

Extract a refusal direction:

```bash
python scripts/extract_direction.py \
  --model_path ./models/Qwen2.5-3B-Instruct \
  --harmful_data data/harmful_prompts.json \
  --harmless_data data/harmless_prompts.json \
  --output results/refusal_direction.pt
```

Run a single intervention test:

```bash
python scripts/intervention.py \
  --model_path ./models/Qwen2.5-3B-Instruct \
  --refusal_dir results/refusal_direction.pt \
  --prompts "Explain the concept of photosynthesis"
```

Evaluate ASR on local test sets:

```bash
python scripts/eval_asr.py \
  --model_path ./models/Qwen2.5-3B-Instruct \
  --refusal_dir results/refusal_direction.pt \
  --harmful_data data/harmful_test_prompts.json \
  --harmless_data data/harmless_test_prompts.json \
  --output results/asr_results.json \
  --alpha 1.0
```

Run ablations:

```bash
python scripts/eval_alpha_ablation.py \
  --model_path ./models/Qwen2.5-3B-Instruct \
  --refusal_dir results/refusal_direction.pt \
  --harmful_data data/harmful_test_prompts.json \
  --harmless_data data/harmless_test_prompts.json \
  --output results/alpha_ablation_results.json

python scripts/eval_layer_ablation.py \
  --model_path ./models/Qwen2.5-3B-Instruct \
  --refusal_dir results/refusal_direction.pt \
  --harmful_data data/harmful_test_prompts.json \
  --harmless_data data/harmless_test_prompts.json \
  --output results/layer_ablation_results.json
```

## Privacy And Safety

The repository excludes:

- paper drafts and private notes
- raw prompt datasets
- model weights and derived activation vectors
- generated outputs and result artifacts

The scripts are intended for controlled AI safety research and robustness
evaluation.

## License

MIT License
