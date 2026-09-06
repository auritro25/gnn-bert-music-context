from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    multilabel_confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def tune_global_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.10, 0.91, 0.05):
        pred = (y_prob >= t).astype(int)
        score = f1_score(y_true, pred, average="macro", zero_division=0)
        if score > best_f1:
            best_f1, best_t = float(score), float(t)
    return best_t, best_f1


def _safe_roc_auc(y_true, y_prob, average):
    try:
        return float(roc_auc_score(y_true, y_prob, average=average))
    except ValueError:
        return float("nan")


def multilabel_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    pred = (y_prob >= threshold).astype(int)
    out = {
        "threshold": float(threshold),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, pred, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, pred, average="macro", zero_division=0)),
        "micro_precision": float(precision_score(y_true, pred, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y_true, pred, average="micro", zero_division=0)),
        "macro_auc_pr": float(average_precision_score(y_true, y_prob, average="macro")),
        "micro_auc_pr": float(average_precision_score(y_true, y_prob, average="micro")),
        "macro_roc_auc": _safe_roc_auc(y_true, y_prob, "macro"),
        "micro_roc_auc": _safe_roc_auc(y_true, y_prob, "micro"),
    }
    return out


def per_label_metrics(y_true, y_prob, labels, threshold):
    pred = (y_prob >= threshold).astype(int)
    rows = []
    for i, label in enumerate(labels):
        yt, yp, pp = y_true[:, i], y_prob[:, i], pred[:, i]
        try:
            ap = average_precision_score(yt, yp)
        except ValueError:
            ap = np.nan
        try:
            auc = roc_auc_score(yt, yp)
        except ValueError:
            auc = np.nan
        rows.append(
            {
                "label": label,
                "support": int(yt.sum()),
                "precision": precision_score(yt, pp, zero_division=0),
                "recall": recall_score(yt, pp, zero_division=0),
                "f1": f1_score(yt, pp, zero_division=0),
                "auc_pr": ap,
                "roc_auc": auc,
            }
        )
    return pd.DataFrame(rows)


def plot_history(history: pd.DataFrame, out_dir: str | Path):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))
    plt.plot(history["epoch"], history["train_loss"], label="Train loss")
    plt.plot(history["epoch"], history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "training_loss.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(history["epoch"], history["val_macro_f1"], label="Validation Macro-F1")
    plt.plot(history["epoch"], history["val_micro_f1"], label="Validation Micro-F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.title("Validation F1 curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "f1_curves.png", dpi=220)
    plt.close()


def plot_pr_curves(y_true, y_prob, labels, out_path: str | Path, max_labels: int = 10):
    supports = y_true.sum(axis=0)
    idxs = np.argsort(supports)[::-1][:max_labels]

    plt.figure(figsize=(8, 6))
    for i in idxs:
        if len(np.unique(y_true[:, i])) < 2:
            continue
        p, r, _ = precision_recall_curve(y_true[:, i], y_prob[:, i])
        plt.plot(r, p, label=labels[i][:30])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall curves (most-supported labels)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_roc_curves(y_true, y_prob, labels, out_path: str | Path, max_labels: int = 10):
    supports = y_true.sum(axis=0)
    idxs = np.argsort(supports)[::-1][:max_labels]

    plt.figure(figsize=(8, 6))
    for i in idxs:
        if len(np.unique(y_true[:, i])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
        plt.plot(fpr, tpr, label=labels[i][:30])
    plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curves (most-supported labels)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_aggregate_confusion(y_true, y_prob, threshold, out_path):
    pred = (y_prob >= threshold).astype(int)
    matrices = multilabel_confusion_matrix(y_true, pred)
    cm = matrices.sum(axis=0)

    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    plt.title("Aggregate multilabel confusion matrix")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_tsne(embeddings, y_true, labels, out_path):
    if embeddings is None or len(embeddings) < 5:
        return

    n = len(embeddings)
    perplexity = min(30, max(2, (n - 1) // 3))
    xy = TSNE(
        n_components=2,
        random_state=42,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
    ).fit_transform(embeddings)

    # For multi-label visualization, color by the first active target label.
    color_ids = []
    for row in y_true:
        active = np.flatnonzero(row > 0.5)
        color_ids.append(int(active[0]) if len(active) else -1)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(xy[:, 0], xy[:, 1], c=color_ids, s=18, alpha=0.8)
    plt.title("t-SNE of learned fused/model embeddings")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_test_artifacts(
    y_true,
    y_prob,
    embeddings,
    labels,
    threshold,
    out_dir,
    track_ids=None,
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    metrics = multilabel_metrics(y_true, y_prob, threshold)
    with open(out / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    per_label_metrics(y_true, y_prob, labels, threshold).to_csv(
        out / "per_label_metrics.csv", index=False
    )

    np.savez_compressed(
        out / "test_predictions.npz",
        y_true=y_true,
        y_prob=y_prob,
        embeddings=embeddings if embeddings is not None else np.array([]),
        track_ids=np.array(track_ids or []),
    )

    plot_pr_curves(y_true, y_prob, labels, out / "pr_curves.png")
    plot_roc_curves(y_true, y_prob, labels, out / "roc_curves.png")
    plot_aggregate_confusion(y_true, y_prob, threshold, out / "confusion_matrix.png")
    plot_tsne(embeddings, y_true, labels, out / "tsne.png")

    return metrics
