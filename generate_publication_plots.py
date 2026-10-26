"""
generate_publication_plots.py
==============================
Publication-grade visual analytics for IEEE Transactions on Multimedia (TMM).

Generates 5 publication-ready 300 DPI figures:
  1. fig_accuracy_vs_params.png: Parameter Footprint (M) vs. Top-1 Accuracy (%) with Pareto Frontier.
  2. fig_flops_vs_latency.png: FLOPs (G) vs. Inference Latency (ms) Hardware Efficiency Landscape.
  3. fig_robustness_stress.png: Clean vs. Adverse Noise & Illumination Robustness Bar Chart.
  4. fig_modality_ablation.png: Modality Isolation & Synergistic Fusion Decomposition.
  5. fig_statistical_significance.png: 5-Seed Accuracy Distribution with 95% Confidence Intervals.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Configure IEEE Publication Typography & Aesthetics
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#E0E0E0'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.7

PALETTE = {
    "Baseline (CNN + Bi-LSTM)": "#4C72B0",
    "MobileNetV3-Large": "#55A868",
    "Comparative (ConvNeXt-Femto)": "#C44E52",
    "Comparative (MobileViT-S)": "#8172B2",
    "Pure Skeletal GCN": "#CCB974",
    "Proposed 1 (Enhanced CNN + Mamba)": "#64B5CD",
    "Proposed 2 (Enhanced GCN + CNN)": "#D55E00" # Distinct bold vermillion
}


def plot_accuracy_vs_params():
    df = pd.read_csv("benchmark_results.csv")
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.grid(True)

    for i, row in df.iterrows():
        name = row['Architecture']
        params = row['Params (M)']
        acc = row['Top-1 Acc (%)']
        lat = row['Latency (ms)']
        color = PALETTE.get(name, '#1f77b4')

        # Bubble size represents latency
        s = np.clip(lat * 1.8, 80, 700)
        ax.scatter(params, acc, s=s, color=color, alpha=0.85, edgecolors='black', linewidth=1.2, zorder=5)

        # Smart label offsets
        offset_y = 1.2 if "Proposed 2" in name or "ConvNeXt" in name else (-3.2 if "MobileViT" in name else 1.2)
        offset_x = -0.5 if "Proposed 2" in name else 0.35
        ha = 'right' if "Proposed 2" in name else 'left'

        ax.annotate(
            f"{name}\n({params:.2f}M, {acc:.1f}%)",
            xy=(params, acc),
            xytext=(params + offset_x, acc + offset_y),
            fontsize=8.5,
            weight='bold' if "Proposed" in name else 'normal',
            ha=ha,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, lw=1, alpha=0.85),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1", color=color, lw=0.8),
            zorder=6
        )

    # Dynamic Pareto Optimal curve calculation
    # Sort models by parameter footprint ascending
    sorted_df = df.sort_values(by='Params (M)').reset_index(drop=True)
    pareto_x = []
    pareto_y = []
    max_acc = -1.0
    for _, r in sorted_df.iterrows():
        if r['Top-1 Acc (%)'] > max_acc:
            pareto_x.append(r['Params (M)'])
            pareto_y.append(r['Top-1 Acc (%)'])
            max_acc = r['Top-1 Acc (%)']
    if len(pareto_x) > 1:
        ax.plot(pareto_x, pareto_y, 'r--', lw=1.5, alpha=0.7, label='Empirical Pareto Frontier', zorder=3)

    min_p, max_p = df['Params (M)'].min(), df['Params (M)'].max()
    min_a, max_a = df['Top-1 Acc (%)'].min(), df['Top-1 Acc (%)'].max()
    ax.set_xlim(max(-0.5, min_p - 0.8), max_p + 1.8)
    ax.set_ylim(max(0, min_a - 8), min(104, max_a + 6))

    ax.set_xlabel("Parameter Footprint (M) $\\rightarrow$ Lower is Better", weight='bold')
    ax.set_ylabel("Top-1 Test Accuracy (%) $\\rightarrow$ Higher is Better", weight='bold')
    ax.set_title("Recognition Accuracy vs. Parameter Footprint (Bubble Area $\\propto$ Latency)", weight='bold', pad=10)
    ax.legend(loc='lower right', frameon=True, framealpha=0.9, facecolor='white')

    plt.tight_layout()
    plt.savefig("fig_accuracy_vs_params.png")
    plt.close()
    print("[*] Generated fig_accuracy_vs_params.png")


def plot_flops_vs_latency():
    df = pd.read_csv("benchmark_results.csv")
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.grid(True)

    for _, row in df.iterrows():
        name = row['Architecture']
        flops = row['FLOPs (G)']
        lat = row['Latency (ms)']
        color = PALETTE.get(name, '#1f77b4')

        ax.scatter(flops, lat, s=180, color=color, edgecolors='black', linewidth=1.2, zorder=5)
        
        offset_y = 8 if "Proposed 1" in name or "MobileViT" in name else -16
        ax.annotate(
            name.replace(" (", "\n("),
            xy=(flops, lat),
            xytext=(flops + 0.12, lat + offset_y),
            fontsize=8.5,
            weight='bold' if "Proposed" in name else 'normal',
            bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="#999999", lw=0.6, alpha=0.8),
            zorder=6
        )

    # Highlight Ultra-Efficient Edge Deployment Zone
    ax.axvspan(-0.1, 0.5, ymin=0, ymax=0.45, color='#D4EDDA', alpha=0.4, label='Target Mobile Edge Budget (<0.5G FLOPs, <100ms)', zorder=1)

    ax.set_xlabel("Theoretical Computational Complexity - FLOPs (G)", weight='bold')
    ax.set_ylabel("Inference Latency (ms)", weight='bold')
    ax.set_title("Computational Efficiency Landscape: FLOPs vs. Inference Latency", weight='bold', pad=10)
    min_f, max_f = df['FLOPs (G)'].min(), df['FLOPs (G)'].max()
    max_lat = df['Latency (ms)'].max()
    ax.set_xlim(-0.2, max(4.0, max_f + 0.5))
    ax.set_ylim(0, max(250, max_lat * 1.25))
    ax.legend(loc='upper left', frameon=True, framealpha=0.9, facecolor='white')

    plt.tight_layout()
    plt.savefig("fig_flops_vs_latency.png")
    plt.close()
    print("[*] Generated fig_flops_vs_latency.png")


def plot_robustness_stress():
    df = pd.read_csv("stress_test_results.csv")
    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    x = np.arange(len(df))
    width = 0.36

    rects1 = ax.bar(x - width/2, df['Clean Top-1 Acc (%)'], width, label='Standard Benchmark (Clean Set)', color='#2B5C8F', edgecolor='black', linewidth=0.8)
    rects2 = ax.bar(x + width/2, df['Stress Top-1 Acc (%)'], width, label='Adverse Stress: Low Light ($\\gamma=0.3$) + Noise ($\\sigma=0.15$)', color='#C93B3B', edgecolor='black', linewidth=0.8)

    ax.set_ylabel('Top-1 Recognition Accuracy (%)', weight='bold')
    ax.set_title('Invariance & Robustness Under Adverse Illumination and Camera Sensor Noise', weight='bold', pad=12)
    ax.set_xticks(x)
    
    clean_labels = [
        "Baseline\n(CNN+LSTM)",
        "MobileNetV3\nLarge",
        "ConvNeXt\nFemto",
        "MobileViT-S\n(Transformer)",
        "Pure Skeletal\nGCN",
        "Proposed 1\n(CNN+Mamba)",
        "Proposed 2\n(GCN+CNN)"
    ]
    ax.set_xticklabels(clean_labels, fontsize=8.5, weight='bold')
    ax.set_ylim(0, 115)
    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.95)

    # Annotate values and degradation retention percentages
    for i in range(len(df)):
        c_val = df['Clean Top-1 Acc (%)'].iloc[i]
        s_val = df['Stress Top-1 Acc (%)'].iloc[i]
        retention = (s_val / c_val) * 100.0

        ax.annotate(f"{c_val:.1f}%", xy=(x[i] - width/2, c_val), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontsize=8, weight='bold')
        ax.annotate(f"{s_val:.1f}%\n({retention:.0f}%)", xy=(x[i] + width/2, s_val), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontsize=7.5, weight='bold', color='#8B0000')

    plt.tight_layout()
    plt.savefig("fig_robustness_stress.png")
    plt.close()
    print("[*] Generated fig_robustness_stress.png")


def plot_modality_ablation():
    df = pd.read_csv("modality_ablation.csv")
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    x = np.arange(len(df))
    width = 0.35

    rects1 = ax.bar(x - width/2, df['Clean Top-1 (%)'], width, label='Clean Accuracy (%)', color='#3B6978', edgecolor='black', linewidth=0.8)
    rects2 = ax.bar(x + width/2, df['Stress Top-1 (%)'], width, label='Stress Accuracy (%)', color='#E05A47', edgecolor='black', linewidth=0.8)

    ax.set_ylabel('Top-1 Accuracy (%)', weight='bold')
    ax.set_title('Modality Stream Decomposition: Pure Skeletal vs. Pure Visual vs. Synergistic Dual-Modal', weight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(df['Stream Configuration'].str.replace(" (", "\n("), fontsize=8.5, weight='bold')
    ax.set_ylim(0, 115)
    ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.95)

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontsize=8, weight='bold')
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontsize=8, weight='bold')

    plt.tight_layout()
    plt.savefig("fig_modality_ablation.png")
    plt.close()
    print("[*] Generated fig_modality_ablation.png")


def plot_statistical_significance():
    df = pd.read_csv("statistical_significance.csv")
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    y_pos = np.arange(len(df))
    colors = [PALETTE.get(arch, '#4C72B0') for arch in df['Architecture']]

    bars = ax.barh(y_pos, df['Mean Top-1 (%)'], xerr=df['95% CI (±%)'], color=colors,
                   edgecolor='black', linewidth=0.8, alpha=0.85, capsize=4, height=0.6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['Architecture'], fontsize=9, weight='bold')
    ax.invert_yaxis()
    ax.set_xlabel('Mean Top-1 Accuracy (%) with 95% Confidence Intervals', weight='bold')
    ax.set_title('Empirical Accuracy Distribution & Statistical Significance Across 5 Evaluation Seeds', weight='bold', pad=10)
    min_m = df['Mean Top-1 (%)'].min()
    max_m = (df['Mean Top-1 (%)'] + df['95% CI (±%)']).max()
    ax.set_xlim(max(0, min_m - 12), min(105, max_m + 8))

    for i, bar in enumerate(bars):
        w = bar.get_width()
        p_val_str = f"p={df['p-value vs Baseline'].iloc[i]:.3f}" if df['p-value vs Baseline'].iloc[i] != 1.0 else "Baseline"
        ax.annotate(f" {w:.2f}% (±{df['95% CI (±%)'].iloc[i]:.2f}) [{p_val_str}]",
                    xy=(w + df['95% CI (±%)'].iloc[i] + 0.3, y_pos[i]),
                    va='center', fontsize=8, weight='bold')

    plt.tight_layout()
    plt.savefig("fig_statistical_significance.png")
    plt.close()
    print("[*] Generated fig_statistical_significance.png")


if __name__ == '__main__':
    plot_accuracy_vs_params()
    plot_flops_vs_latency()
    plot_robustness_stress()
    plot_modality_ablation()
    plot_statistical_significance()
    print("[*] All 5 publication-ready charts successfully generated.")
