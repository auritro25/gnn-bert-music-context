from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def main():
    result_root = Path("results")
    rows = []

    for name in ["cnn", "bert", "gnn", "concat", "fusion"]:
        path = result_root / name / "test_metrics.json"
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            m = json.load(f)
        rows.append(
            {
                "model": name,
                "test_loss": m.get("test_loss"),
                "macro_f1": m.get("macro_f1"),
                "micro_f1": m.get("micro_f1"),
                "macro_auc_pr": m.get("macro_auc_pr"),
                "micro_auc_pr": m.get("micro_auc_pr"),
                "macro_roc_auc": m.get("macro_roc_auc"),
                "micro_roc_auc": m.get("micro_roc_auc"),
            }
        )

    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(result_root / "ablation_comparison.csv", index=False)
        print(df.to_string(index=False))
        print("\nSaved results/ablation_comparison.csv")
    else:
        print("No supervised result files found yet.")


if __name__ == "__main__":
    main()
