"""
plot_figures.py
生成论文级图表（英文版）

图表列表：
  fig1: alpha-ASR 权衡曲线
  fig2: 层选择策略对比柱状图
  fig3: 各层拒绝方向 L2 范数分布
  fig4: 干预前后拒绝率对比
  fig5: 四合一面板总结图

用法：
  python plot_figures.py --results_dir results --output_dir results/figures
"""

import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def parse_args():
    parser = argparse.ArgumentParser(description="Generate publication figures")
    parser.add_argument("--results_dir", default="results", help="Directory containing result JSONs")
    parser.add_argument("--output_dir", default="results/figures", help="Output directory for figures")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--formats", nargs="+", default=["pdf", "png"])
    return parser.parse_args()


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_fig(fig, path, formats, dpi):
    for fmt in formats:
        fig.savefig(f"{path}.{fmt}", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_alpha_tradeoff(results_dir, output_dir, formats, dpi):
    data = load_json(os.path.join(results_dir, "alpha_ablation_results.json"))
    results = data["results"]

    alphas = [r["alpha"] for r in results]
    asrs = [r["asr"] * 100 for r in results]
    h_ppls = [r["harmful_avg_ppl"] for r in results]
    hl_ppls = [r["harmless_avg_ppl"] for r in results]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    color1 = "#2196F3"
    color2 = "#FF5722"
    color3 = "#4CAF50"

    ax1.plot(alphas, asrs, "o-", color=color1, linewidth=2, markersize=8, label="ASR (%)")
    ax1.set_xlabel("Alpha (Intervention Strength)", fontsize=13)
    ax1.set_ylabel("Attack Success Rate (%)", color=color1, fontsize=13)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim([0, 105])

    ax2 = ax1.twinx()
    ax2.plot(alphas, h_ppls, "s--", color=color2, linewidth=1.5, markersize=6, label="Harmful PPL")
    ax2.plot(alphas, hl_ppls, "^--", color=color3, linewidth=1.5, markersize=6, label="Harmless PPL")
    ax2.set_ylabel("Perplexity", fontsize=13)
    ax2.tick_params(axis="y")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=10)

    ax1.axvline(x=0.4, color="gray", linestyle=":", alpha=0.5)
    ax1.annotate("Sweet spot\nalpha=0.4", xy=(0.4, asrs[2]), xytext=(0.8, 70),
                fontsize=10, ha="center", color="gray",
                arrowprops=dict(arrowstyle="->", color="gray"))

    ax1.grid(True, alpha=0.3)
    fig.suptitle("Alpha-ASR Tradeoff Curve", fontsize=14, y=1.02)
    save_fig(fig, os.path.join(output_dir, "fig1_alpha_tradeoff"), formats, dpi)


def plot_layer_selection(results_dir, output_dir, formats, dpi):
    data = load_json(os.path.join(results_dir, "layer_ablation_results.json"))
    strategies = data["strategies"]

    names = list(strategies.keys())
    asrs = [strategies[n]["ASR"] * 100 for n in names]
    hl_fr = [strategies[n]["harmless_refusal_rate"] * 100 for n in names]
    n_layers = [strategies[n]["n_layers"] for n in names]

    # Shorten labels
    short_names = [n.split("(")[0].strip() for n in names]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    x = np.arange(len(names))
    width = 0.35

    bars1 = ax1.bar(x - width/2, asrs, width, label="ASR (%)", color="#2196F3", alpha=0.8)
    bars2 = ax1.bar(x + width/2, hl_fr, width, label="Harmless False Refusal (%)", color="#FF9800", alpha=0.8)

    ax1.set_xlabel("Layer Intervention Strategy", fontsize=13)
    ax1.set_ylabel("Rate (%)", fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_names, rotation=15, ha="right", fontsize=10)
    ax1.legend(fontsize=11)
    ax1.set_ylim([0, 110])

    for bar, val in zip(bars1, asrs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{val:.1f}%",
                ha="center", va="bottom", fontsize=9, color="#1565C0")

    ax1.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Layer Selection Strategy Comparison", fontsize=14, y=1.02)
    save_fig(fig, os.path.join(output_dir, "fig2_layer_selection"), formats, dpi)


def plot_layer_norms(results_dir, output_dir, formats, dpi):
    refusal_path = os.path.join(results_dir, "refusal_direction.pt")
    if not os.path.exists(refusal_path):
        print(f"  Skipped fig3: {refusal_path} not found")
        return

    import torch
    data = torch.load(refusal_path, map_location="cpu")
    norms = data["layer_norms"].numpy()

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.RdYlBu_r(np.linspace(0.1, 0.9, len(norms)))

    ax.bar(range(len(norms)), norms, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Layer Index", fontsize=13)
    ax.set_ylabel("L2 Norm", fontsize=13)
    ax.set_xticks(range(0, len(norms), max(1, len(norms)//10)))
    ax.grid(True, axis="y", alpha=0.3)

    # Highlight middle layers
    n = len(norms)
    third = n // 3
    ax.axvspan(third - 0.5, 2*third - 0.5, alpha=0.1, color="red", label="Middle 1/3 (selected)")
    ax.legend(fontsize=11)

    fig.suptitle("Refusal Direction L2 Norm Distribution", fontsize=14, y=1.02)
    save_fig(fig, os.path.join(output_dir, "fig3_layer_norms"), formats, dpi)


def plot_intervention_comparison(results_dir, output_dir, formats, dpi):
    alpha_data = load_json(os.path.join(results_dir, "alpha_ablation_results.json"))
    alphas = [r["alpha"] for r in alpha_data["results"]]
    asrs = [r["asr"] * 100 for r in alpha_data["results"]]

    layer_data = load_json(os.path.join(results_dir, "layer_ablation_results.json"))
    baseline_refusal = layer_data["baseline"]["harmful_refused"]
    baseline_rate = baseline_refusal / layer_data["config"]["n_harmful"] * 100

    best_strategy = max(layer_data["strategies"].items(), key=lambda x: x[1]["ASR"])
    best_asr = best_strategy[1]["ASR"] * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Before/After bar chart
    ax = axes[0]
    categories = ["Before\n(Baseline)", "After\n(Best Strategy)"]
    refusal_rates = [baseline_rate, 100 - best_asr]
    colors = ["#f44336", "#4CAF50"]
    bars = ax.bar(categories, refusal_rates, color=colors, width=0.5, edgecolor="white")
    ax.set_ylabel("Refusal Rate (%)", fontsize=12)
    ax.set_title("Harmful Prompt Refusal Rate", fontsize=13)
    ax.set_ylim([0, 110])
    for bar, val in zip(bars, refusal_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{val:.1f}%",
               ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    # ASR vs alpha
    ax = axes[1]
    ax.plot(alphas, asrs, "o-", color="#2196F3", linewidth=2, markersize=8)
    ax.set_xlabel("Alpha", fontsize=12)
    ax.set_ylabel("ASR (%)", fontsize=12)
    ax.set_title("ASR vs Intervention Strength", fontsize=13)
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3)

    fig.suptitle("Intervention Effect Summary", fontsize=14, y=1.02)
    plt.tight_layout()
    save_fig(fig, os.path.join(output_dir, "fig4_intervention_comparison"), formats, dpi)


def plot_summary(results_dir, output_dir, formats, dpi):
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    alpha_data = load_json(os.path.join(results_dir, "alpha_ablation_results.json"))
    layer_data = load_json(os.path.join(results_dir, "layer_ablation_results.json"))

    # Panel 1: Alpha-ASR
    ax1 = fig.add_subplot(gs[0, 0])
    alphas = [r["alpha"] for r in alpha_data["results"]]
    asrs = [r["asr"] * 100 for r in alpha_data["results"]]
    ax1.plot(alphas, asrs, "o-", color="#2196F3", linewidth=2, markersize=6)
    ax1.set_xlabel("Alpha")
    ax1.set_ylabel("ASR (%)")
    ax1.set_title("(a) Alpha-ASR Tradeoff")
    ax1.grid(True, alpha=0.3)

    # Panel 2: Layer strategies
    ax2 = fig.add_subplot(gs[0, 1])
    strats = layer_data["strategies"]
    names = [n.split("(")[0].strip() for n in strats.keys()]
    s_asrs = [strats[n]["ASR"] * 100 for n in strats.keys()]
    ax2.barh(names, s_asrs, color="#4CAF50", alpha=0.8)
    ax2.set_xlabel("ASR (%)")
    ax2.set_title("(b) Layer Strategy Comparison")
    ax2.set_xlim([0, 110])

    # Panel 3: Layer norms
    ax3 = fig.add_subplot(gs[1, 0])
    norms_path = os.path.join(results_dir, "refusal_direction.pt")
    if os.path.exists(norms_path):
        import torch
        nd = torch.load(norms_path, map_location="cpu")
        norms = nd["layer_norms"].numpy()
        n = len(norms)
        third = n // 3
        colors = ["#E53935" if third <= i < 2*third else "#90A4AE" for i in range(n)]
        ax3.bar(range(n), norms, color=colors, width=0.8)
        ax3.set_xlabel("Layer Index")
        ax3.set_ylabel("L2 Norm")
        ax3.set_title("(c) Refusal Direction Norms by Layer")
    ax3.grid(True, axis="y", alpha=0.3)

    # Panel 4: Summary stats
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    best_strat = max(strats.items(), key=lambda x: x[1]["ASR"])
    sweet_alpha = max(alpha_data["results"], key=lambda x: x["asr"])

    stats_text = (
        f"Key Findings\n"
        f"{'='*30}\n"
        f"Optimal alpha: {sweet_alpha['alpha']}\n"
        f"  ASR: {sweet_alpha['asr']*100:.1f}%\n"
        f"  Harmful PPL: {sweet_alpha['harmful_avg_ppl']:.1f}\n\n"
        f"Best layer strategy:\n"
        f"  {best_strat[0]}\n"
        f"  ASR: {best_strat[1]['ASR']*100:.1f}%\n"
        f"  Layers: {best_strat[1]['n_layers']}\n\n"
        f"Harmless false refusal: ~0%\n"
        f"Model: Qwen2.5-3B-Instruct"
    )
    ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes, fontsize=11,
            verticalalignment="center", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5", alpha=0.8))
    ax4.set_title("(d) Summary")

    fig.suptitle("Refusal Direction Abliteration: Experimental Results", fontsize=15, y=1.01)
    save_fig(fig, os.path.join(output_dir, "fig5_summary"), formats, dpi)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Generating figures...")
    plot_alpha_tradeoff(args.results_dir, args.output_dir, args.formats, args.dpi)
    plot_layer_selection(args.results_dir, args.output_dir, args.formats, args.dpi)
    plot_layer_norms(args.results_dir, args.output_dir, args.formats, args.dpi)
    plot_intervention_comparison(args.results_dir, args.output_dir, args.formats, args.dpi)
    plot_summary(args.results_dir, args.output_dir, args.formats, args.dpi)
    print(f"\nAll figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
