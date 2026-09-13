"""
05_evaluation.py - Evaluation and Comparison
Self‑contained version – runs with or without existing results.

This module aggregates model outputs, computes standard evaluation metrics, and
summarises the performance differences between prompting and fine-tuning
approaches in a single comparison table.
"""

import os
import random
import sys

# Check required packages
try:
    import pandas as pd
    import numpy as np
    from sklearn.metrics import precision_score, recall_score, f1_score
except ImportError as e:
    print("Missing required package. Please install: pandas, numpy, scikit-learn")
    print(f"Error: {e}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42):
    """Set deterministic seeds for evaluation reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass  # PyTorch not installed – fine

def save_results(results, filename, output_dir="./results"):
    """Save results (dict or list of dicts) to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, filename), index=False)
    print(f"Results saved to {os.path.join(output_dir, filename)}")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
class Config:
    output_dir = "./results"

config = Config()

# ---------------------------------------------------------------------------
# Evaluator class
# ---------------------------------------------------------------------------
class Evaluator:
    def __init__(self):
        self.results = []

    def load_results(self, filepath: str) -> pd.DataFrame:
        """Load results from CSV file."""
        return pd.read_csv(filepath)

    def calculate_metrics(self, df: pd.DataFrame) -> dict:
        """Calculate evaluation metrics."""
        if df.empty:
            return {
                "exact_match": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "avg_inference_time": 0.0,
                "num_samples": 0,
            }

        predictions = df["prediction"].tolist()
        ground_truths = df["ground_truth"].tolist()

        # Exact match (case‑insensitive trimmed)
        exact_match = sum(
            1 for p, g in zip(predictions, ground_truths)
            if str(p).strip().lower() == str(g).strip().lower()
        ) / len(predictions) if predictions else 0.0

        # Weighted classification metrics
        try:
            p_str = [str(p).strip() for p in predictions]
            g_str = [str(g).strip() for g in ground_truths]
            precision = precision_score(g_str, p_str, average="weighted", zero_division=0)
            recall = recall_score(g_str, p_str, average="weighted", zero_division=0)
            f1 = f1_score(g_str, p_str, average="weighted", zero_division=0)
        except Exception:
            precision = 0.0
            recall = 0.0
            f1 = 0.0

        avg_inference_time = df["inference_time"].mean() if "inference_time" in df.columns else 0.0

        return {
            "exact_match": exact_match,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "avg_inference_time": avg_inference_time,
            "num_samples": len(df)
        }

    def _generate_sample_results(self):
        """Create dummy result files if none exist, to demonstrate evaluation."""
        print("No result files found. Generating sample results for demonstration...")
        os.makedirs(config.output_dir, exist_ok=True)

        sample_questions = [
            "Is aspirin effective for headache?",
            "Can antibiotics treat viral infections?",
            "Does smoking increase lung cancer risk?",
            "Is coffee harmful during pregnancy?",
            "Can stress cause ulcers?"
        ]
        sample_answers = ["yes", "no", "yes", "maybe", "yes"]

        # Simulate predictions with some errors
        predictions_variants = [
            ["yes", "no", "yes", "maybe", "yes"],   # perfect
            ["yes", "maybe", "yes", "maybe", "yes"], # one error
            ["no", "no", "yes", "maybe", "yes"],    # two errors
            ["yes", "no", "maybe", "maybe", "yes"], # one error
        ]

        methods = ["prompt_engineering", "lora", "qlora"]
        models = ["distilgpt2", "phi3_mini"]

        for method in methods:
            for model in models:
                # Choose a prediction variant (fallback to first if random fails)
                try:
                    preds = random.choice(predictions_variants)
                except IndexError:
                    preds = predictions_variants[0]

                data = []
                for q, gt, pred in zip(sample_questions, sample_answers, preds):
                    data.append({
                        "question": q,
                        "ground_truth": gt,
                        "prediction": pred,
                        "inference_time": random.uniform(0.1, 0.5),
                        "method": method,
                        "model": model
                    })
                df = pd.DataFrame(data)
                if method == "prompt_engineering":
                    fname = f"prompt_engineering_{model}.csv"
                else:
                    fname = f"{method}_results_{model}.csv"
                df.to_csv(os.path.join(config.output_dir, fname), index=False)
                print(f"  Generated: {fname}")

    def evaluate_all(self):
        """Evaluate all experiment results."""
        print("Evaluating all results...")
        results_dir = config.output_dir
        all_metrics = []

        if not os.path.exists(results_dir) or not os.listdir(results_dir):
            self._generate_sample_results()

        files = [f for f in os.listdir(results_dir) if f.endswith(".csv") and f != "evaluation_summary.csv"]
        if not files:
            print("No CSV files found after generation attempt.")
            return pd.DataFrame()

        for file in files:
            filepath = os.path.join(results_dir, file)
            df = self.load_results(filepath)

            required_cols = {"prediction", "ground_truth"}
            if not required_cols.issubset(df.columns):
                print(f"  Skipped (missing columns): {file}")
                continue

            filename = file.replace(".csv", "")
            if filename.startswith("prompt_engineering_"):
                method = "prompt_engineering"
                model = filename.replace("prompt_engineering_", "")
            elif filename.startswith("lora_results_"):
                method = "lora"
                model = filename.replace("lora_results_", "")
            elif filename.startswith("qlora_results_"):
                method = "qlora"
                model = filename.replace("qlora_results_", "")
            else:
                method = df["method"].iloc[0] if "method" in df.columns and not df.empty else "unknown"
                model = df["model"].iloc[0] if "model" in df.columns and not df.empty else "unknown"

            metrics = self.calculate_metrics(df)
            metrics["method"] = method
            metrics["model"] = model
            metrics["file"] = file
            all_metrics.append(metrics)
            print(f"  Evaluated: {file}")

        if not all_metrics:
            print("No valid result files found to evaluate.")
            return pd.DataFrame()

        summary_df = pd.DataFrame(all_metrics)
        summary_df.to_csv(os.path.join(results_dir, "evaluation_summary.csv"), index=False)
        print(f"Evaluation summary saved to {os.path.join(results_dir, 'evaluation_summary.csv')}")
        return summary_df

    def generate_comparison_table(self) -> pd.DataFrame:
        """Generate a cleaned comparison table of all approaches."""
        summary_path = os.path.join(config.output_dir, "evaluation_summary.csv")
        if not os.path.exists(summary_path):
            self.evaluate_all()

        df = pd.read_csv(summary_path)
        if df.empty:
            print("No data to generate comparison table.")
            return pd.DataFrame()

        comparison = df.pivot_table(
            index=["model"],
            columns=["method"],
            values=["f1", "exact_match", "avg_inference_time"],
            aggfunc="mean"
        )

        # Flatten the MultiIndex columns into a clean table with one column per metric-method pair.
        comparison = comparison.reset_index()
        flat_columns = []
        for col in comparison.columns:
            if isinstance(col, tuple):
                metric, method = col
                if method == "":
                    flat_columns.append("model")
                else:
                    flat_columns.append(f"{metric}_{method}")
            else:
                flat_columns.append(str(col))

        comparison.columns = flat_columns
        ordered = ["model"] + [c for c in comparison.columns if c != "model"]
        comparison = comparison[ordered]
        comparison.to_csv(os.path.join(config.output_dir, "comparison_table.csv"), index=False)
        print(f"Comparison table saved to {os.path.join(config.output_dir, 'comparison_table.csv')}")
        return comparison

    def print_summary(self):
        """Print a summary of results."""
        summary_path = os.path.join(config.output_dir, "evaluation_summary.csv")
        if not os.path.exists(summary_path):
            self.evaluate_all()

        df = pd.read_csv(summary_path)
        if df.empty:
            print("No summary data available.")
            return

        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)

        method_summary = df.groupby("method").agg({
            "f1": ["mean", "std"],
            "exact_match": ["mean", "std"],
            "avg_inference_time": ["mean", "std"]
        }).round(4)

        print("\nMethod Summary:")
        print(method_summary)

        best_row = df.loc[df["f1"].idxmax()]
        print(f"\nBest F1 Score: {best_row['method']} on {best_row['model']} - {best_row['f1']:.4f}")

        fastest_row = df.loc[df["avg_inference_time"].idxmin()]
        print(f"Fastest Inference: {fastest_row['method']} on {fastest_row['model']} - {fastest_row['avg_inference_time']:.4f}s")

# -------------------- Main --------------------
def main():
    set_seed(42)
    print("=" * 60)
    print("EVALUATION AND COMPARISON")
    print("=" * 60)

    evaluator = Evaluator()
    evaluator.evaluate_all()
    evaluator.generate_comparison_table()
    evaluator.print_summary()

    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()