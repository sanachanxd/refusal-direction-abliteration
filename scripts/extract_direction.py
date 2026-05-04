"""
extract_refusal_direction.py
从有害/无害 prompt 的激活值差异中提取拒绝方向向量

参考：Arditi et al. "Refusal in Language Models Is Mediated by a Single Direction" (2024)

用法：
  python extract_direction.py --model_path ./models/Qwen2.5-3B-Instruct \
      --harmful_data data/harmful_prompts.json \
      --harmless_data data/harmless_prompts.json \
      --output results/refusal_direction.pt
"""

import argparse
import json
import os
import sys
import time
import logging
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Extract refusal direction from model activations")
    parser.add_argument("--model_path", required=True, help="Path to the HuggingFace model")
    parser.add_argument("--harmful_data", required=True, help="Path to harmful prompts JSON")
    parser.add_argument("--harmless_data", required=True, help="Path to harmless prompts JSON")
    parser.add_argument("--output", required=True, help="Output path for refusal_direction.pt")
    parser.add_argument("--device", default="auto", help="Device: auto, mps, cuda, cpu")
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--log_file", default=None, help="Log file path")
    return parser.parse_args()


def get_device(device_str):
    if device_str == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device_str


def get_dtype(dtype_str):
    return {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[dtype_str]


def setup_logging(log_file=None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=handlers)
    return logging.getLogger(__name__)


def load_prompts(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    return prompts


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise AttributeError(f"Cannot find layers in model: {type(model).__name__}")


def extract_activations(model, tokenizer, prompts, num_layers, device, label="unknown", logger=None):
    layers = get_layers(model)
    all_activations = []
    hook_handles = []
    layer_outputs = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                layer_outputs[layer_idx] = output[0].detach()
            else:
                layer_outputs[layer_idx] = output.detach()
        return hook_fn

    for i in range(num_layers):
        handle = layers[i].register_forward_hook(make_hook(i))
        hook_handles.append(handle)

    if logger:
        logger.info(f"Extracting [{label}] activations ({len(prompts)} prompts)...")

    start_time = time.time()

    for idx, prompt in enumerate(tqdm(prompts, desc=f"Extracting [{label}]")):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)

        layer_outputs.clear()
        with torch.no_grad():
            model(**inputs)

        last_token_acts = []
        for layer_idx in range(num_layers):
            act = layer_outputs[layer_idx][0, -1, :].cpu()
            last_token_acts.append(act)

        stacked = torch.stack(last_token_acts)
        all_activations.append(stacked)

    for h in hook_handles:
        h.remove()

    elapsed = time.time() - start_time
    if logger:
        logger.info(f"[{label}] Done: {len(prompts)} prompts, {elapsed:.1f}s")

    result = torch.stack(all_activations)
    if logger:
        logger.info(f"[{label}] Activations shape: {result.shape}")
    return result


def compute_refusal_direction(harmful_acts, harmless_acts, logger=None):
    mean_harmful = harmful_acts.mean(dim=0)
    mean_harmless = harmless_acts.mean(dim=0)

    refusal_dir = mean_harmful - mean_harmless
    norms = refusal_dir.norm(dim=-1, keepdim=True)
    refusal_dir_normalized = refusal_dir / norms

    if logger:
        logger.info("Per-layer refusal direction L2 norms:")
        for i in range(refusal_dir.shape[0]):
            bar = "#" * int(norms[i].item() * 20)
            logger.info(f"  Layer {i:2d}: {norms[i].item():.4f}  {bar}")

    return refusal_dir_normalized, norms.squeeze()


def main():
    args = parse_args()
    device = get_device(args.device)
    dtype = get_dtype(args.dtype)
    logger = setup_logging(args.log_file)

    logger.info("=" * 60)
    logger.info("Extract Refusal Direction")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Device: {device}, Dtype: {args.dtype}")

    harmful_prompts = load_prompts(args.harmful_data)
    harmless_prompts = load_prompts(args.harmless_data)
    logger.info(f"Loaded {len(harmful_prompts)} harmful, {len(harmless_prompts)} harmless prompts")

    logger.info("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype, device_map=device)
    model.eval()
    logger.info(f"Model loaded: {type(model).__name__}, {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params")

    num_layers = len(get_layers(model))
    logger.info(f"Number of layers: {num_layers}")

    harmful_acts = extract_activations(model, tokenizer, harmful_prompts, num_layers, device, "harmful", logger)
    harmless_acts = extract_activations(model, tokenizer, harmless_prompts, num_layers, device, "harmless", logger)

    refusal_dir, norms = compute_refusal_direction(harmful_acts, harmless_acts, logger)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save({
        "refusal_direction": refusal_dir,
        "layer_norms": norms,
        "harmful_mean": harmful_acts.mean(dim=0),
        "harmless_mean": harmless_acts.mean(dim=0),
        "num_harmful": len(harmful_prompts),
        "num_harmless": len(harmless_prompts),
        "model_path": args.model_path,
        "num_layers": num_layers,
        "hidden_size": refusal_dir.shape[1],
    }, args.output)
    logger.info(f"Saved to {args.output}")
    logger.info(f"Refusal direction shape: {refusal_dir.shape}")
    logger.info(f"Max norm layer: {norms.argmax().item()} ({norms.max().item():.4f})")
    logger.info(f"Min norm layer: {norms.argmin().item()} ({norms.min().item():.4f})")


if __name__ == "__main__":
    main()
