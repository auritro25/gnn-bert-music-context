from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from .cnn_model import CNNMelBaseline
from .datasets import MelDataset
from .evaluation import multilabel_metrics, plot_history, save_test_artifacts, tune_global_threshold
from .labels import load_label_vocab
from .utils import ensure_dir, load_config, resolve_device, save_json, set_seed


def run(model, loader, criterion, device, optimizer=None, use_amp=False, grad_clip=1.0):
    train = optimizer is not None
    model.train(train)
    losses, ys, probs, ids = [], [], [], []

    for batch in tqdm(loader, leave=False):
        mel = batch["mel"].to(device)
        y = batch["labels"].to(device)
        ids.extend(batch["track_id"])

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=(use_amp and device.type == "cuda")):
            logits = model(mel)
            loss = criterion(logits, y)

        if train:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        losses.append(loss.item())
        ys.append(y.detach().cpu().numpy())
        probs.append(torch.sigmoid(logits).detach().cpu().numpy())

    return {
        "loss": float(np.mean(losses)),
        "y_true": np.concatenate(ys),
        "y_prob": np.concatenate(probs),
        "track_ids": ids,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    device = resolve_device(cfg.get("device", "cuda:0"))
    labels = load_label_vocab(cfg["data"]["label_vocab"])

    train_ds, val_ds, test_ds = (MelDataset(cfg, s) for s in ["train", "val", "test"])
    bs, nw = int(cfg["train"]["batch_size"]), int(cfg["train"]["num_workers"])
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw)

    model = CNNMelBaseline(len(labels), float(cfg["model"]["dropout"])).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr_head"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    out_dir = ensure_dir(Path(cfg["train"]["output_dir"]) / "cnn")
    save_json(cfg, out_dir / "resolved_config.json")

    history, best_state, best_score, stale = [], None, -1.0, 0
    use_amp = bool(cfg["train"]["use_amp"])
    default_t = float(cfg["train"]["default_threshold"])

    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        tr = run(model, train_loader, criterion, device, optimizer, use_amp, float(cfg["train"]["grad_clip"]))
        va = run(model, val_loader, criterion, device, None, use_amp)
        vm = multilabel_metrics(va["y_true"], va["y_prob"], default_t)

        history.append({
            "epoch": epoch,
            "train_loss": tr["loss"],
            "val_loss": va["loss"],
            "val_macro_f1": vm["macro_f1"],
            "val_micro_f1": vm["micro_f1"],
            "val_auc_pr": vm["macro_auc_pr"],
        })
        print(f"Epoch {epoch:02d} | val_macro_f1={vm['macro_f1']:.4f}")

        if vm["macro_f1"] > best_score:
            best_score = vm["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())
            torch.save({"model_state": best_state, "num_labels": len(labels)}, out_dir / "best_model.pt")
            stale = 0
        else:
            stale += 1
            if stale >= int(cfg["train"]["patience"]):
                break

    hist = pd.DataFrame(history)
    hist.to_csv(out_dir / "history.csv", index=False)
    plot_history(hist, out_dir)
    model.load_state_dict(best_state)

    va = run(model, val_loader, criterion, device, None, use_amp)
    threshold, val_f1 = tune_global_threshold(va["y_true"], va["y_prob"])
    save_json({"threshold": threshold, "validation_macro_f1_at_threshold": val_f1}, out_dir / "val_threshold.json")

    te = run(model, test_loader, criterion, device, None, use_amp)
    metrics = save_test_artifacts(
        te["y_true"], te["y_prob"], None, labels, threshold, out_dir, te["track_ids"]
    )
    metrics["test_loss"] = te["loss"]
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
