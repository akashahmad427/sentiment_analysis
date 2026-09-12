"""
Evaluation Module
------------------
Computes metrics, generates comparison plots, and produces
the model comparison dashboard for ML vs DL models.
"""

import time
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

CLASSES = ["negative", "neutral", "positive"]
COLORS = {"negative": "#e74c3c", "neutral": "#95a5a6", "positive": "#2ecc71"}
ML_COLOR = "#3498db"
DL_COLOR = "#9b59b6"


class ModelEvaluator:
    """
    Unified evaluator for ML and DL sentiment models.
    Produces comparison metrics and visualization dashboard.
    """

    def __init__(self, output_dir: str = "evaluation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def compute_metrics(self, y_true, y_pred, model_name: str) -> dict:
        """Compute full evaluation metrics for a model."""
        prec, rec, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=[0, 1, 2], average=None
        )
        macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro"
        )
        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

        per_class = {
            cls: {
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, cls in enumerate(CLASSES)
        }

        return {
            "model": model_name,
            "accuracy": float(acc),
            "macro_precision": float(macro_prec),
            "macro_recall": float(macro_rec),
            "macro_f1": float(macro_f1),
            "per_class": per_class,
            "confusion_matrix": cm,
        }

    def plot_confusion_matrix(self, ax, cm: np.ndarray, title: str, color: str):
        """Plot a single confusion matrix on given axes."""
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        sns.heatmap(
            cm_norm, annot=cm, fmt="d", ax=ax,
            xticklabels=CLASSES, yticklabels=CLASSES,
            cmap=sns.light_palette(color, as_cmap=True),
            linewidths=0.5, linecolor="white",
        )
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual", fontsize=10)
        ax.tick_params(axis="x", rotation=30)
        ax.tick_params(axis="y", rotation=0)

    def plot_per_class_metrics(self, ax, ml_metrics: dict, dl_metrics: dict, metric: str):
        """Bar chart comparing ML vs DL for each class on a given metric."""
        x = np.arange(len(CLASSES))
        width = 0.35
        ml_vals = [ml_metrics["per_class"][c][metric] for c in CLASSES]
        dl_vals = [dl_metrics["per_class"][c][metric] for c in CLASSES]

        bars_ml = ax.bar(x - width / 2, ml_vals, width, label="ML Model", color=ML_COLOR, alpha=0.85)
        bars_dl = ax.bar(x + width / 2, dl_vals, width, label="DL Model", color=DL_COLOR, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([c.capitalize() for c in CLASSES])
        ax.set_ylim(0, 1.1)
        ax.set_title(f"{metric.capitalize()} per Class", fontweight="bold")
        ax.set_ylabel(metric.capitalize())
        ax.legend(fontsize=9)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)

        for bar in bars_ml:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)
        for bar in bars_dl:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)

    def plot_overall_comparison(self, ax, ml_metrics: dict, dl_metrics: dict,
                                 ml_extra: dict, dl_extra: dict):
        """Bar chart for overall model comparison (accuracy, macro F1, speed, size)."""
        metrics = ["accuracy", "macro_f1"]
        ml_vals = [ml_metrics[m] for m in metrics]
        dl_vals = [dl_metrics[m] for m in metrics]
        labels = ["Accuracy", "Macro F1"]

        x = np.arange(len(labels))
        width = 0.35
        ax.bar(x - width / 2, ml_vals, width, label="ML Model", color=ML_COLOR, alpha=0.85)
        ax.bar(x + width / 2, dl_vals, width, label="DL Model", color=DL_COLOR, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.1)
        ax.set_title("Overall Performance Comparison", fontweight="bold")
        ax.legend(fontsize=9)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)

        for i, (mv, dv) in enumerate(zip(ml_vals, dl_vals)):
            ax.text(i - width / 2, mv + 0.02, f"{mv:.3f}", ha="center", fontsize=9)
            ax.text(i + width / 2, dv + 0.02, f"{dv:.3f}", ha="center", fontsize=9)

    def plot_inference_speed(self, ax, ml_extra: dict, dl_extra: dict):
        """Bar chart for inference speed and training time comparison."""
        models = ["ML Model", "DL Model"]
        speeds = [ml_extra.get("avg_inference_ms", 0.5), dl_extra.get("avg_inference_ms", 15.0)]
        colors = [ML_COLOR, DL_COLOR]
        bars = ax.bar(models, speeds, color=colors, alpha=0.85, width=0.4)
        ax.set_title("Avg Inference Speed (ms/sample)", fontweight="bold")
        ax.set_ylabel("Milliseconds")
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        for bar, val in zip(bars, speeds):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f"{val:.2f}ms", ha="center", fontsize=10, fontweight="bold")

    def build_comparison_table(self, ml_metrics: dict, dl_metrics: dict,
                                ml_extra: dict, dl_extra: dict) -> pd.DataFrame:
        """Create a clean comparison DataFrame."""
        rows = []
        rows.append({
            "Metric": "Accuracy",
            "ML Model": f"{ml_metrics['accuracy']:.4f}",
            "DL Model": f"{dl_metrics['accuracy']:.4f}",
        })
        for key, label in [("macro_precision", "Macro Precision"),
                            ("macro_recall", "Macro Recall"),
                            ("macro_f1", "Macro F1")]:
            rows.append({"Metric": label, "ML Model": f"{ml_metrics[key]:.4f}",
                          "DL Model": f"{dl_metrics[key]:.4f}"})
        for cls in CLASSES:
            for m in ["precision", "recall", "f1"]:
                rows.append({
                    "Metric": f"{cls.capitalize()} {m.capitalize()}",
                    "ML Model": f"{ml_metrics['per_class'][cls][m]:.4f}",
                    "DL Model": f"{dl_metrics['per_class'][cls][m]:.4f}",
                })
        rows.append({
            "Metric": "Avg Inference (ms)",
            "ML Model": f"{ml_extra.get('avg_inference_ms', 0.5):.3f}",
            "DL Model": f"{dl_extra.get('avg_inference_ms', 15.0):.3f}",
        })
        rows.append({
            "Metric": "Training Time (s)",
            "ML Model": f"{ml_extra.get('training_time_s', 5.0):.2f}",
            "DL Model": f"{dl_extra.get('training_time_s', 120.0):.2f}",
        })
        return pd.DataFrame(rows)

    def generate_dashboard(
        self,
        y_test, ml_preds, dl_preds,
        ml_extra: dict = None, dl_extra: dict = None,
        save_path: str = None,
    ) -> dict:
        """
        Generate the full model comparison dashboard.
        Returns metrics dict and saves PNG.
        """
        ml_extra = ml_extra or {}
        dl_extra = dl_extra or {}

        y_arr = np.array(y_test)
        ml_metrics = self.compute_metrics(y_arr, ml_preds, "ML Model")
        dl_metrics = self.compute_metrics(y_arr, dl_preds, "DL Model")

        # --- Build figure ---
        fig = plt.figure(figsize=(20, 18))
        fig.patch.set_facecolor("#f8f9fa")
        fig.suptitle(
            "Sentiment Analysis: ML vs DL Model Comparison Dashboard",
            fontsize=18, fontweight="bold", y=0.98
        )

        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

        # Row 0: Confusion matrices + overall comparison
        ax_cm_ml = fig.add_subplot(gs[0, 0])
        ax_cm_dl = fig.add_subplot(gs[0, 1])
        ax_overall = fig.add_subplot(gs[0, 2])

        self.plot_confusion_matrix(ax_cm_ml, ml_metrics["confusion_matrix"], "ML: Confusion Matrix", ML_COLOR)
        self.plot_confusion_matrix(ax_cm_dl, dl_metrics["confusion_matrix"], "DL: Confusion Matrix", DL_COLOR)
        self.plot_overall_comparison(ax_overall, ml_metrics, dl_metrics, ml_extra, dl_extra)

        # Row 1: Per-class metrics
        ax_prec = fig.add_subplot(gs[1, 0])
        ax_rec = fig.add_subplot(gs[1, 1])
        ax_f1 = fig.add_subplot(gs[1, 2])

        self.plot_per_class_metrics(ax_prec, ml_metrics, dl_metrics, "precision")
        self.plot_per_class_metrics(ax_rec, ml_metrics, dl_metrics, "recall")
        self.plot_per_class_metrics(ax_f1, ml_metrics, dl_metrics, "f1")

        # Row 2: Speed + table
        ax_speed = fig.add_subplot(gs[2, 0])
        self.plot_inference_speed(ax_speed, ml_extra, dl_extra)

        ax_table = fig.add_subplot(gs[2, 1:])
        ax_table.axis("off")
        table_df = self.build_comparison_table(ml_metrics, dl_metrics, ml_extra, dl_extra)
        tbl = ax_table.table(
            cellText=table_df.values,
            colLabels=table_df.columns,
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.4)
        for (row, col), cell in tbl.get_celld().items():
            if row == 0:
                cell.set_facecolor("#2c3e50")
                cell.set_text_props(color="white", fontweight="bold")
            elif row % 2 == 0:
                cell.set_facecolor("#ecf0f1")
        ax_table.set_title("Full Metrics Comparison Table", fontweight="bold", fontsize=12, pad=10)

        if save_path is None:
            save_path = str(self.output_dir / "model_comparison_dashboard.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        logger.info(f"Dashboard saved to {save_path}")

        return {
            "ml_metrics": ml_metrics,
            "dl_metrics": dl_metrics,
            "table": table_df,
            "dashboard_path": save_path,
        }


if __name__ == "__main__":
    # Quick test with dummy data
    np.random.seed(42)
    y_test = np.random.choice([0, 1, 2], size=100)
    ml_preds = np.random.choice([0, 1, 2], size=100)
    dl_preds = np.random.choice([0, 1, 2], size=100)

    evaluator = ModelEvaluator(output_dir="evaluation_results")
    results = evaluator.generate_dashboard(
        y_test, ml_preds, dl_preds,
        ml_extra={"avg_inference_ms": 0.4, "training_time_s": 8.2},
        dl_extra={"avg_inference_ms": 14.7, "training_time_s": 132.5},
    )
    print("Dashboard generated!")
    print(results["table"].to_string(index=False))
