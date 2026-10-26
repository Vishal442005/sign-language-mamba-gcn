"""
generate_plots.py
=================
Generates publication-quality charts from the benchmark results:
  1. Parameters (M) vs. Accuracy (%) trade-off bubble/scatter chart.
  2. FLOPs (G) vs. Latency (ms) computational efficiency chart.
  3. Clean Accuracy vs. Stress-Test Accuracy robustness degradation bar chart.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Set aesthetic style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11

def plot_efficiency_tradeoff():
    df_bm = pd.read_csv("benchmark_results.csv")
    
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, row in df_bm.iterrows():
        name = row['Architecture']
        params = row['Params (M)']
        acc = row['Top-1 Acc (%)']
        latency = row['Latency (ms)']
        
        ax.scatter(params, acc, s=latency*15, color=colors[i % len(colors)], alpha=0.75, edgecolors='black', linewidth=1.5)
        ax.annotate(
            f"{name}\n({params:.1f}M, {acc:.1f}%)",
            xy=(params, acc),
            xytext=(params + 0.3, acc - (2.5 if i % 2 == 0 else -1.5)),
            fontsize=9,
            weight='bold'
        )
        
    ax.set_xlabel("Parameter Count (M) $\\rightarrow$ Lower is Better", fontsize=12, weight='bold')
    ax.set_ylabel("Top-1 Accuracy (%) $\\rightarrow$ Higher is Better", fontsize=12, weight='bold')
    ax.set_title("Model Accuracy vs. Parameter Footprint (Bubble size ~ Latency)", fontsize=13, weight='bold', pad=12)
    plt.tight_layout()
    plt.savefig("fig_accuracy_vs_params.png")
    plt.close()
    print("[*] Saved fig_accuracy_vs_params.png")

def plot_robustness_comparison():
    df_bm = pd.read_csv("benchmark_results.csv")
    df_st = pd.read_csv("stress_test_results.csv")
    
    merged = pd.merge(df_bm, df_st, on="Architecture")
    
    x = np.arange(len(merged))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    rects1 = ax.bar(x - width/2, merged['Top-1 Acc (%)'], width, label='Clean Test Set', color='#4C72B0')
    rects2 = ax.bar(x + width/2, merged['Stress Top-1 Acc (%)'], width, label='Illumination/Noise Stress', color='#C44E52')
    
    ax.set_ylabel('Top-1 Accuracy (%)', fontsize=12, weight='bold')
    ax.set_title('Robustness Comparison Under Adverse Noise & Low Light', fontsize=13, weight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([arch.replace(" (", "\n(") for arch in merged['Architecture']], fontsize=9)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9)
    
    # Add values on top of bars
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontsize=8, weight='bold')
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontsize=8, weight='bold')

    plt.ylim(0, 110)
    plt.tight_layout()
    plt.savefig("fig_robustness_stress.png")
    plt.close()
    print("[*] Saved fig_robustness_stress.png")

if __name__ == '__main__':
    plot_efficiency_tradeoff()
    plot_robustness_comparison()
