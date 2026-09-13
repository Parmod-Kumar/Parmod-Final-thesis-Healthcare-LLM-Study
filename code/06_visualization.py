"""
06_visualization.py - Baseline visualisation utilities
Self‑contained version – runs with or without existing results.

This script creates a small set of standard benchmark plots to compare model and
method performance. It loads the evaluation summary data, builds simple charts,
and saves the output PNG files to the visualisations directory.
"""

import os
import sys
import random
from pathlib import Path
import pandas as pd
import numpy as np

# Force a non-GUI backend and white figure backgrounds for reliable saved plots.
import matplotlib
matplotlib.use("Agg")

# Check required packages
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError as e:
    print("Missing required package. Please install: matplotlib, seaborn, pandas, numpy")
    print(f"Error: {e}")
    sys.exit(1)

# Set style
sns.set_style("whitegrid")
plt.rcParams.update({
    "font.family": "Times New Roman",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#111111",
    "xtick.color": "#111111",
    "ytick.color": "#111111",
    "text.color": "#111111",
})

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

class Config:
    """Project-wide paths for the evaluation and visualisation pipeline."""
    output_dir = str(PROJECT_ROOT / "results")

config = Config()

# ---------------------------------------------------------------------------
# Visualiser class
# ---------------------------------------------------------------------------
class Visualiser:
    """Generate baseline comparison charts from the study result summaries."""
    def __init__(self):
        self.output_dir = str(PROJECT_ROOT / "visualisations")
        os.makedirs(self.output_dir, exist_ok=True)
        self.data = None

    def _generate_sample_data(self):
        """Create dummy evaluation summary if none exists."""
        print("No evaluation summary found. Generating sample data for demonstration...")
        os.makedirs(config.output_dir, exist_ok=True)

        methods = ["prompt_engineering", "lora", "qlora"]
        models = ["distilgpt2", "phi3_mini"]

        rows = []
        for method in methods:
            for model in models:
                # Simulate random metrics
                f1 = round(random.uniform(0.5, 0.9), 3)
                exact_match = round(random.uniform(0.4, 0.85), 3)
                inference_time = round(random.uniform(0.1, 0.6), 3)
                rows.append({
                    "method": method,
                    "model": model,
                    "f1": f1,
                    "exact_match": exact_match,
                    "avg_inference_time": inference_time,
                    "num_samples": 10
                })
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(config.output_dir, "evaluation_summary.csv"), index=False)
        print(f"Generated sample data: {config.output_dir}/evaluation_summary.csv")

    def load_data(self):
        """Load evaluation summary data; generate sample if missing."""
        summary_path = os.path.join(config.output_dir, "evaluation_summary.csv")
        if os.path.exists(summary_path):
            self.data = pd.read_csv(summary_path)
            print(f"Loaded data from {summary_path}")
            return True
        else:
            self._generate_sample_data()
            self.data = pd.read_csv(summary_path)
            print(f"Loaded generated data from {summary_path}")
            return True

    def plot_f1_comparison(self):
        """Plot F1 score comparison across methods and models."""
        if self.data is None:
            return

        fig, ax = plt.subplots(figsize=(12, 6))
        methods = self.data["method"].unique()
        models = self.data["model"].unique()
        x = np.arange(len(models))
        width = 0.12
        colors = ["#42a5f5", "#43a047", "#fb8c00", "#7b1fa2", "#c62828", "#00838f"]

        for i, method in enumerate(methods):
            method_data = self.data[self.data["method"] == method]
            values = []
            for model in models:
                val = method_data[method_data["model"] == model]["f1"].values
                values.append(val[0] if len(val) > 0 else 0)
            ax.bar(x + i * width, values, width, label=method, color=colors[i % len(colors)])

        ax.set_xlabel("Model", fontsize=12, fontweight="bold")
        ax.set_ylabel("F1 Score", fontsize=12, fontweight="bold")
        ax.set_title("F1 Score Comparison by Method and Model", fontsize=14, fontweight="bold")
        ax.set_xticks(x + width * (len(methods) - 1) / 2)
        ax.set_xticklabels(models, rotation=15, ha="right")
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "f1_comparison.png"), dpi=300, facecolor="white")
        plt.close(fig)
        print("Saved: f1_comparison.png")

    def plot_inference_time(self):
        """Plot inference time comparison."""
        if self.data is None:
            return

        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        methods = self.data["method"].unique()
        models = self.data["model"].unique()
        x = np.arange(len(models))
        width = 0.12
        colors = ["#42a5f5", "#43a047", "#fb8c00", "#7b1fa2", "#c62828", "#00838f"]

        for i, method in enumerate(methods):
            method_data = self.data[self.data["method"] == method]
            values = []
            for model in models:
                val = method_data[method_data["model"] == model]["avg_inference_time"].values
                values.append(val[0] if len(val) > 0 else 0)
            ax.bar(x + i * width, values, width, label=method, color=colors[i % len(colors)])

        ax.set_xlabel("Model", fontsize=12, fontweight="bold")
        ax.set_ylabel("Inference Time (seconds)", fontsize=12, fontweight="bold")
        ax.set_title("Inference Time Comparison by Method and Model", fontsize=14, fontweight="bold")
        ax.set_xticks(x + width * (len(methods) - 1) / 2)
        ax.set_xticklabels(models, rotation=15, ha="right")
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "inference_time.png"), dpi=300, facecolor="white")
        plt.close(fig)
        print("Saved: inference_time.png")

    def plot_performance_summary(self):
        """Plot overall performance summary (heatmap)."""
        if self.data is None:
            return

        pivot = self.data.pivot_table(
            index="method",
            columns="model",
            values="f1",
            aggfunc="mean"
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn",
                   vmin=0.3, vmax=0.9, ax=ax, cbar_kws={"label": "F1 Score"})
        ax.set_title("F1 Score Heatmap: Method vs Model", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "performance_summary.png"), dpi=300, facecolor="white")
        plt.close(fig)
        print("Saved: performance_summary.png")

    def plot_error_analysis(self):
        """Plot error analysis: accuracy by method and F1 vs inference time."""
        if self.data is None:
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor("white")
        for ax in axes:
            ax.set_facecolor("white")

        # Accuracy by method
        method_accuracy = self.data.groupby("method")["exact_match"].mean().sort_values(ascending=False)
        colors = ["#42a5f5", "#43a047", "#fb8c00", "#7b1fa2", "#c62828", "#00838f"]
        axes[0].bar(method_accuracy.index, method_accuracy.values,
                   color=colors[:len(method_accuracy)])
        axes[0].set_xlabel("Method", fontsize=12, fontweight="bold")
        axes[0].set_ylabel("Accuracy", fontsize=12, fontweight="bold")
        axes[0].set_title("Accuracy by Method", fontsize=13, fontweight="bold")
        axes[0].set_ylim(0, 1)
        axes[0].tick_params(axis="x", rotation=15)

        # F1 vs Inference Time
        method_summary = self.data.groupby("method").agg({
            "f1": "mean",
            "avg_inference_time": "mean"
        }).reset_index()

        for i, row in method_summary.iterrows():
            axes[1].scatter(row["avg_inference_time"], row["f1"],
                          s=200, label=row["method"], color=colors[i % len(colors)])

        axes[1].set_xlabel("Inference Time (seconds)", fontsize=12, fontweight="bold")
        axes[1].set_ylabel("F1 Score", fontsize=12, fontweight="bold")
        axes[1].set_title("F1 vs Inference Time Trade-off", fontsize=13, fontweight="bold")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "error_analysis.png"), dpi=300, facecolor="white")
        plt.close(fig)
        print("Saved: error_analysis.png")

    def generate_all_plots(self):
        """Generate all visualisations."""
        print("\n" + "=" * 60)
        print("GENERATING VISUALISATIONS")
        print("=" * 60)

        if not self.load_data():
            print("No data to visualise.")
            return

        self.plot_f1_comparison()
        self.plot_inference_time()
        self.plot_performance_summary()
        self.plot_error_analysis()

        print(f"\nAll baseline visualisations saved to: {self.output_dir}/")

        # Call targeted prompting strategy visualisations
        try:
            import importlib.util
            from pathlib import Path

            module_path = Path(__file__).resolve().with_name("08_prompting_visualizations.py")
            spec = importlib.util.spec_from_file_location("prompting_visualizations", module_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load module from {module_path}")

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            prompt_vis = module.PromptingVisualizer()
            prompt_vis.generate_all()
        except Exception as e:
            print(f"Note: Could not run 08_prompting_visualizations module: {e}")

# -------------------- Main --------------------
def main():
    visualiser = Visualiser()
    visualiser.generate_all_plots()

if __name__ == "__main__":
    main()