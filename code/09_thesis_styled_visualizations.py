"""
09_thesis_styled_visualizations.py - Clean Publication Figures & Tables (Without Figure/Table Numbering)
Generates clean figures and tables matching exact thesis styling without Figure/Table label prefixes.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Color Palette matching target sample images
COLOR_MISTRAL = "#2b82c9"   # Steel Blue
COLOR_GEMMA   = "#e54b4b"   # Vibrant Coral Red
COLOR_HEADER  = "#0f172a"   # Dark Navy for Table Headers
COLOR_GRID    = "#e2e8f0"   # Subtle Grid Gray
COLOR_BG      = "#ffffff"   # Card Background

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "visualisations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Full Dataset matching performance tables
PERFORMANCE_DATA = [
    {"model": "Mistral-7B", "approach": "Zero-Shot", "rouge1": 0.218, "rougel": 0.160, "bertscore": 0.666, "em": 0.12, "f1": 0.621},
    {"model": "Gemma-7B",   "approach": "Zero-Shot", "rouge1": 0.255, "rougel": 0.193, "bertscore": 0.664, "em": 0.14, "f1": 0.645},
    {"model": "Mistral-7B", "approach": "Few-Shot (3-Shot)", "rouge1": 0.297, "rougel": 0.241, "bertscore": 0.702, "em": 0.21, "f1": 0.689},
    {"model": "Gemma-7B",   "approach": "Few-Shot (3-Shot)", "rouge1": 0.331, "rougel": 0.272, "bertscore": 0.711, "em": 0.24, "f1": 0.712},
    {"model": "Mistral-7B", "approach": "Few-Shot (5-Shot)", "rouge1": 0.314, "rougel": 0.258, "bertscore": 0.714, "em": 0.25, "f1": 0.701},
    {"model": "Gemma-7B",   "approach": "Few-Shot (5-Shot)", "rouge1": 0.348, "rougel": 0.289, "bertscore": 0.723, "em": 0.27, "f1": 0.734},
    {"model": "Mistral-7B", "approach": "Chain-of-Thought",  "rouge1": 0.342, "rougel": 0.287, "bertscore": 0.731, "em": 0.29, "f1": 0.725},
    {"model": "Gemma-7B",   "approach": "Chain-of-Thought",  "rouge1": 0.371, "rougel": 0.312, "bertscore": 0.742, "em": 0.33, "f1": 0.758},
    {"model": "Mistral-7B", "approach": "Prompt Ensemble",   "rouge1": 0.358, "rougel": 0.301, "bertscore": 0.741, "em": 0.31, "f1": 0.742},
    {"model": "Gemma-7B",   "approach": "Prompt Ensemble",   "rouge1": 0.378, "rougel": 0.319, "bertscore": 0.748, "em": 0.34, "f1": 0.763},
]

def add_card_frame(fig, ax, title, subtitle):
    """Adds clean title & subtitle at the bottom in serif font."""
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.22, top=0.88, left=0.10, right=0.95)
    
    fig.text(0.5, 0.09, title, ha='center', va='center', fontsize=14, fontweight='bold', fontfamily='serif', color='#111111')
    fig.text(0.5, 0.04, subtitle, ha='center', va='center', fontsize=11, fontstyle='italic', fontfamily='serif', color='#444444')


def generate_figure_5_1():
    """F1-Score Comparison for Prompt Engineering Approaches"""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)

    categories = ["Zero-Shot", "Few-Shot (3-Shot)", "Few-Shot (5-Shot)", "CoT", "Ensemble"]
    mistral_f1 = [0.621, 0.689, 0.701, 0.725, 0.742]
    gemma_f1   = [0.645, 0.712, 0.734, 0.758, 0.763]

    x = np.arange(len(categories))
    width = 0.35

    rects1 = ax.bar(x - width/2, mistral_f1, width, label="Mistral-7B", color=COLOR_MISTRAL, edgecolor="#1d5b8c", linewidth=0.8)
    rects2 = ax.bar(x + width/2, gemma_f1, width, label="Gemma-7B", color=COLOR_GEMMA, edgecolor="#a82e2e", linewidth=0.8)

    for bar in rects1:
        height = bar.get_height()
        ax.annotate(f"{height:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    for bar in rects2:
        height = bar.get_height()
        ax.annotate(f"{height:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11, color="#333333")
    ax.set_ylim(0.50, 0.82)
    ax.yaxis.grid(True, linestyle='-', color=COLOR_GRID, alpha=0.8)
    ax.xaxis.grid(False)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False, fontsize=11)

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#cccccc")

    add_card_frame(fig, ax, 
                   "F1-Score Comparison for Prompt Engineering Approaches",
                   "Comparison of F1-Score for prompt engineering approaches across Mistral-7B and Gemma-7B")

    out_path = OUTPUT_DIR / "thesis_fig_5_1_f1_comparison.png"
    plt.savefig(out_path, dpi=300, facecolor=COLOR_BG)
    plt.close(fig)
    print(f"Saved: {out_path}")


def generate_figure_5_2():
    """Few-Shot Prompting Performance Improvement"""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)

    shots = ["Zero-Shot", "3-Shot", "5-Shot"]
    x = np.arange(len(shots))
    
    mistral_f1 = [0.621, 0.689, 0.701]
    gemma_f1   = [0.645, 0.712, 0.734]

    ax.plot(x, mistral_f1, 'o-', color=COLOR_MISTRAL, linewidth=2.5, markersize=8, label="Mistral-7B")
    ax.fill_between(x, 0.55, mistral_f1, color=COLOR_MISTRAL, alpha=0.15)

    ax.plot(x, gemma_f1, 'o-', color=COLOR_GEMMA, linewidth=2.5, markersize=8, label="Gemma-7B")
    ax.fill_between(x, mistral_f1, gemma_f1, color=COLOR_GEMMA, alpha=0.15)

    for i, txt in enumerate(mistral_f1):
        ax.annotate(f"{txt:.3f}", (x[i], mistral_f1[i]), xytext=(0, -15), textcoords="offset points", ha='center', fontsize=9, fontweight='bold')
    
    for i, txt in enumerate(gemma_f1):
        ax.annotate(f"{txt:.3f}", (x[i], gemma_f1[i]), xytext=(0, 8), textcoords="offset points", ha='center', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(shots, fontsize=11, color="#333333")
    ax.set_ylim(0.55, 0.76)
    ax.yaxis.grid(True, linestyle='-', color=COLOR_GRID, alpha=0.8)
    ax.xaxis.grid(False)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False, fontsize=11)

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#cccccc")

    add_card_frame(fig, ax,
                   "Few-Shot Prompting Performance Improvement",
                   "Few-Shot prompting performance improvement across shot counts")

    out_path = OUTPUT_DIR / "thesis_fig_5_2_few_shot_improvement.png"
    plt.savefig(out_path, dpi=300, facecolor=COLOR_BG)
    plt.close(fig)
    print(f"Saved: {out_path}")


def generate_figure_5_3():
    """Chain-of-Thought vs Standard Few-Shot Performance"""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)

    categories = ["Few-Shot (5-Shot)", "Chain-of-Thought"]
    mistral_val = [0.701, 0.725]
    gemma_val   = [0.734, 0.758]

    x = np.arange(len(categories))
    width = 0.35

    rects1 = ax.bar(x - width/2, mistral_val, width, label="Mistral-7B", color=COLOR_MISTRAL, edgecolor="#1d5b8c", linewidth=0.8)
    rects2 = ax.bar(x + width/2, gemma_val, width, label="Gemma-7B", color=COLOR_GEMMA, edgecolor="#a82e2e", linewidth=0.8)

    for bar in rects1:
        height = bar.get_height()
        ax.annotate(f"{height:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    for bar in rects2:
        height = bar.get_height()
        ax.annotate(f"{height:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11, color="#333333")
    ax.set_ylim(0.65, 0.79)
    ax.yaxis.grid(True, linestyle='-', color=COLOR_GRID, alpha=0.8)
    ax.xaxis.grid(False)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False, fontsize=11)

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#cccccc")

    add_card_frame(fig, ax,
                   "Chain-of-Thought vs Standard Few-Shot Performance",
                   "Chain-of-Thought prompting compared to standard few-shot prompting")

    out_path = OUTPUT_DIR / "thesis_fig_5_3_cot_vs_fewshot.png"
    plt.savefig(out_path, dpi=300, facecolor=COLOR_BG)
    plt.close(fig)
    print(f"Saved: {out_path}")


def generate_figure_5_4():
    """ROUGE Lexical Overlap Metric Comparison"""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)

    categories = ["Zero-Shot", "Few-Shot (3-Shot)", "Few-Shot (5-Shot)", "CoT", "Ensemble"]
    mistral_r1 = [0.218, 0.297, 0.314, 0.342, 0.358]
    gemma_r1   = [0.255, 0.331, 0.348, 0.371, 0.378]

    x = np.arange(len(categories))
    width = 0.35

    rects1 = ax.bar(x - width/2, mistral_r1, width, label="Mistral-7B (ROUGE-1)", color=COLOR_MISTRAL, edgecolor="#1d5b8c", linewidth=0.8)
    rects2 = ax.bar(x + width/2, gemma_r1, width, label="Gemma-7B (ROUGE-1)", color=COLOR_GEMMA, edgecolor="#a82e2e", linewidth=0.8)

    for bar in rects1:
        height = bar.get_height()
        ax.annotate(f"{height:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    for bar in rects2:
        height = bar.get_height()
        ax.annotate(f"{height:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11, color="#333333")
    ax.set_ylim(0.18, 0.42)
    ax.yaxis.grid(True, linestyle='-', color=COLOR_GRID, alpha=0.8)
    ax.xaxis.grid(False)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False, fontsize=11)

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#cccccc")

    add_card_frame(fig, ax,
                   "ROUGE-1 Lexical Overlap Comparison Across Strategies",
                   "ROUGE-1 lexical overlap scores comparing Mistral-7B and Gemma-7B")

    out_path = OUTPUT_DIR / "thesis_fig_5_4_rouge_comparison.png"
    plt.savefig(out_path, dpi=300, facecolor=COLOR_BG)
    plt.close(fig)
    print(f"Saved: {out_path}")


def generate_figure_5_5():
    """BERTScore Semantic Similarity Comparison"""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)

    categories = ["Zero-Shot", "Few-Shot (3-Shot)", "Few-Shot (5-Shot)", "CoT", "Ensemble"]
    mistral_bert = [0.666, 0.702, 0.714, 0.731, 0.741]
    gemma_bert   = [0.664, 0.711, 0.723, 0.742, 0.748]

    x = np.arange(len(categories))
    width = 0.35

    rects1 = ax.bar(x - width/2, mistral_bert, width, label="Mistral-7B", color=COLOR_MISTRAL, edgecolor="#1d5b8c", linewidth=0.8)
    rects2 = ax.bar(x + width/2, gemma_bert, width, label="Gemma-7B", color=COLOR_GEMMA, edgecolor="#a82e2e", linewidth=0.8)

    for bar in rects1:
        height = bar.get_height()
        ax.annotate(f"{height:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    for bar in rects2:
        height = bar.get_height()
        ax.annotate(f"{height:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11, color="#333333")
    ax.set_ylim(0.60, 0.78)
    ax.yaxis.grid(True, linestyle='-', color=COLOR_GRID, alpha=0.8)
    ax.xaxis.grid(False)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False, fontsize=11)

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#cccccc")

    add_card_frame(fig, ax,
                   "BERTScore Semantic Similarity Comparison Across Strategies",
                   "BERTScore semantic similarity metrics evaluating model response quality")

    out_path = OUTPUT_DIR / "thesis_fig_5_5_bertscore_comparison.png"
    plt.savefig(out_path, dpi=300, facecolor=COLOR_BG)
    plt.close(fig)
    print(f"Saved: {out_path}")


def generate_table_5_1_image():
    """Healthcare QA Performance Across Prompt Engineering Approaches"""
    fig, ax = plt.subplots(figsize=(11, 7), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    ax.axis("off")

    plt.text(0.5, 0.95, "Healthcare QA Performance Across Prompt Engineering Approaches",
             ha='center', va='center', fontsize=14, fontweight='bold', fontfamily='serif', color='#0f172a')
    plt.text(0.5, 0.91, "Comparison of Mistral-7B and Gemma-7B across six prompt engineering strategies",
             ha='center', va='center', fontsize=10, fontstyle='italic', fontfamily='serif', color='#475569')

    table_data = []
    headers = ["Model", "Approach", "ROUGE-1", "ROUGE-L", "BERTScore", "EM", "F1-Score"]
    
    for row in PERFORMANCE_DATA:
        table_data.append([
            row["model"],
            row["approach"],
            f"{row['rouge1']:.3f}",
            f"{row['rougel']:.3f}",
            f"{row['bertscore']:.3f}",
            f"{row['em']:.2f}",
            f"{row['f1']:.3f}"
        ])

    table = ax.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center', bbox=[0.05, 0.18, 0.90, 0.68])
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)

    for col_idx in range(len(headers)):
        cell = table[(0, col_idx)]
        cell.set_facecolor("#0f172a")
        cell.get_text().set_color("white")
        cell.get_text().set_weight("bold")

    def get_color(val, min_v=0.12, max_v=0.763):
        norm = (val - min_v) / (max_v - min_v + 1e-5)
        r = int(220 * (1 - norm) + 34 * norm)
        g = int(38 * (1 - norm) + 197 * norm)
        b = int(38 * (1 - norm) + 94 * norm)
        return f"#{r:02x}{g:02x}{b:02x}"

    for row_idx, row in enumerate(PERFORMANCE_DATA, start=1):
        m_cell = table[(row_idx, 0)]
        m_cell.get_text().set_weight("bold")
        m_cell.get_text().set_color(COLOR_MISTRAL if row["model"] == "Mistral-7B" else COLOR_GEMMA)
        m_cell.set_facecolor("#f1f5f9" if row_idx % 2 == 0 else "#ffffff")

        ap_cell = table[(row_idx, 1)]
        ap_cell.set_facecolor("#f1f5f9" if row_idx % 2 == 0 else "#ffffff")

        metrics = [row["rouge1"], row["rougel"], row["bertscore"], row["em"], row["f1"]]
        for col_offset, val in enumerate(metrics, start=2):
            cell = table[(row_idx, col_offset)]
            cell_color = get_color(val)
            cell.set_facecolor(cell_color)
            cell.get_text().set_color("white" if val < 0.35 or val > 0.65 else "black")
            cell.get_text().set_weight("bold")

    plt.text(0.5, 0.08, "Healthcare QA performance across prompt engineering approaches",
             ha='center', va='center', fontsize=9.5, fontweight='bold', fontfamily='serif', color='#334155')

    out_path = OUTPUT_DIR / "thesis_table_5_1_performance_summary.png"
    plt.savefig(out_path, dpi=300, facecolor="#ffffff", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def generate_table_5_2_image():
    """Performance of Zero-Shot Prompting Across Both Models"""
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    ax.axis("off")

    plt.text(0.5, 0.88, "Performance of Zero-Shot Prompting Across Both Models",
             ha='center', va='center', fontsize=14, fontweight='bold', fontfamily='serif', color='#0f172a')
    plt.text(0.5, 0.78, "Baseline performance comparison of Mistral-7B and Gemma-7B without in-context examples",
             ha='center', va='center', fontsize=10, fontstyle='italic', fontfamily='serif', color='#475569')

    headers = ["Model", "ROUGE-1", "ROUGE-L", "BERTScore", "EM", "F1-Score"]
    table_data = [
        ["Mistral-7B", "0.218", "0.160", "0.666", "0.12", "0.621"],
        ["Gemma-7B",   "0.255 (*)", "0.193 (*)", "0.664 (*)", "0.14 (*)", "0.645 (*)"]
    ]

    table = ax.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center', bbox=[0.05, 0.30, 0.90, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)

    for col_idx in range(len(headers)):
        cell = table[(0, col_idx)]
        cell.set_facecolor("#0f172a")
        cell.get_text().set_color("white")
        cell.get_text().set_weight("bold")

    for col_idx in range(len(headers)):
        cell = table[(1, col_idx)]
        cell.set_facecolor("#f8fafc")
        if col_idx == 0:
            cell.get_text().set_color(COLOR_MISTRAL)
            cell.get_text().set_weight("bold")

    for col_idx in range(len(headers)):
        cell = table[(2, col_idx)]
        cell.set_facecolor("#dcfce7")
        cell.get_text().set_color("#166534")
        cell.get_text().set_weight("bold")
        if col_idx == 0:
            cell.get_text().set_color(COLOR_GEMMA)

    plt.text(0.5, 0.15, "Performance of Zero-Shot Prompting Across Both Models",
             ha='center', va='center', fontsize=9.5, fontweight='bold', fontfamily='serif', color='#334155')
    plt.text(0.5, 0.07, "(*) Indicates best performance across both models for each metric.",
             ha='center', va='center', fontsize=8.5, fontstyle='italic', fontfamily='serif', color='#64748b')

    out_path = OUTPUT_DIR / "thesis_table_5_2_zero_shot_baseline.png"
    plt.savefig(out_path, dpi=300, facecolor="#ffffff", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def generate_all():
    print("="*60)
    print("RE-GENERATING VISUALIZATIONS & TABLES WITHOUT Figure/Table PREFIXES")
    print("="*60)
    generate_figure_5_1()
    generate_figure_5_2()
    generate_figure_5_3()
    generate_figure_5_4()
    generate_figure_5_5()
    generate_table_5_1_image()
    generate_table_5_2_image()
    print(f"\nAll clean figures & tables successfully re-generated in: {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_all()
