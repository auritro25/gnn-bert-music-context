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

from .datasets import PairedGraphTextDataset, graph_text_collate
from .evaluation import (
    multilabel_metrics,
    plot_history,
    save_test_artifacts,
    tune_global_threshold,
)
from .factory import build_supervised_model
from .labels import load_label_vocab
from .utils import ensure_dir, load_config, resolve_device, save_json, set_seed


def move_batch(batch, device):
    out = dict(batch)
    out["graph"] = batch["graph"].to(device)
    out["input_ids"] = batch["input_ids"].to(device)
    out["attention_mask"] = batch["attention_mask"].to(device)
    out["labels"] = batch["labels"].to(device)
    return out


def collect_train_targets(dataset):
    ys = [dataset[i]["labels"].numpy() for i in range(len(dataset))]
    return np.stack(ys)


def make_pos_weight(y_train):
    pos = y_train.sum(axis=0)
    neg = len(y_train) - pos
    weights = neg / np.clip(pos, 1.0, None)
    # Prevent extremely rare labels from destabilizing BCE.
    weights = np.clip(weights, 1.0, 20.0)
    return torch.tensor(weights, dtype=torch.float32)


def optimizer_for(model, cfg):
    bert_lr = float(cfg["train"]["lr_bert"])
    head_lr = float(cfg["train"]["lr_head"])
    wd = float(cfg["train"]["weight_decay"])

    bert_params, other_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "text_encoder.transformer" in name:
            bert_params.append(p)
        else:
            other_params.append(p)

    groups = []
    if bert_params:
        groups.append({"params": bert_params, "lr": bert_lr})
    if other_params:
        groups.append({"params": other_params, "lr": head_lr})
    return AdamW(groups, weight_decay=wd)


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, use_amp=False, grad_clip=1.0):
    train = optimizer is not None
    model.train(train)

    losses = []
    all_y, all_p, all_e = [], [], []

    for raw_batch in tqdm(loader, leave=False):
        batch = move_batch(raw_batch, device)
        labels = batch["labels"]

        if train:
            optimizer.zero_grad(set_to_none=True)

        amp_enabled = use_amp and device.type == "cuda"
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            out = model(batch)
            loss = criterion(out["logits"], labels)

        if train:
            if scaler is not None and amp_enabled:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        losses.append(loss.item())
        all_y.append(labels.detach().cpu().numpy())
        all_p.append(torch.sigmoid(out["logits"]).detach().cpu().numpy())
        if out.get("embedding") is not None:
            all_e.append(out["embedding"].detach().cpu().numpy())

    return {
        "loss": float(np.mean(losses)),
        "y_true": np.concatenate(all_y, axis=0),
        "y_prob": np.concatenate(all_p, axis=0),
        "embedding": np.concatenate(all_e, axis=0) if all_e else None,
    }


def evaluate_with_ids(model, loader, criterion, device, use_amp=False):
    model.eval()
    losses, ys, probs, embs, ids = [], [], [], [], []

    with torch.no_grad():
        for raw_batch in tqdm(loader, leave=False):
            ids.extend(raw_batch["track_id"])
            batch = move_batch(raw_batch, device)
            with torch.autocast(device_type=device.type, enabled=(use_amp and device.type == "cuda")):
                out = model(batch)
                loss = criterion(out["logits"], batch["labels"])
            losses.append(loss.item())
            ys.append(batch["labels"].cpu().numpy())
            probs.append(torch.sigmoid(out["logits"]).cpu().numpy())
            if out.get("embedding") is not None:
                embs.append(out["embedding"].cpu().numpy())

    return {
        "loss": float(np.mean(losses)),
        "y_true": np.concatenate(ys),
        "y_prob": np.concatenate(probs),
        "embedding": np.concatenate(embs) if embs else None,
        "track_ids": ids,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", required=True, choices=["bert", "gnn", "concat", "fusion"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    device = resolve_device(cfg.get("device", "cuda:0"))
    print("Using device:", device)
    if torch.cuda.is_available():
        print("Visible CUDA devices:", torch.cuda.device_count(), "| This code uses only", device)

    labels = load_label_vocab(cfg["data"]["label_vocab"])
    n_labels = len(labels)

    train_ds = PairedGraphTextDataset(cfg, "train")
    val_ds = PairedGraphTextDataset(cfg, "val")
    test_ds = PairedGraphTextDataset(cfg, "test")

    bs = int(cfg["train"]["batch_size"])
    nw = int(cfg["train"]["num_workers"])

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, num_workers=nw,
        collate_fn=graph_text_collate, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False, num_workers=nw,
        collate_fn=graph_text_collate, pin_memory=(device.type == "cuda")
    )
    test_loader = DataLoader(
        test_ds, batch_size=bs, shuffle=False, num_workers=nw,
        collate_fn=graph_text_collate, pin_memory=(device.type == "cuda")
    )

    model = build_supervised_model(cfg, args.model, n_labels).to(device)

    pos_weight = None
    if bool(cfg["train"].get("use_pos_weight", True)):
        pos_weight = make_pos_weight(collect_train_targets(train_ds)).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optimizer_for(model, cfg)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1
    )

    use_amp = bool(cfg["train"].get("use_amp", True))
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    out_dir = ensure_dir(Path(cfg["train"]["output_dir"]) / args.model)
    save_json(cfg, out_dir / "resolved_config.json")
    save_json({"labels": labels}, out_dir / "labels.json")

    history = []
    best_score = -1.0
    best_state = None
    patience = 0
    max_patience = int(cfg["train"]["patience"])
    default_t = float(cfg["train"]["default_threshold"])

    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        tr = run_epoch(
            model, train_loader, criterion, device,
            optimizer=optimizer, scaler=scaler, use_amp=use_amp,
            grad_clip=float(cfg["train"]["grad_clip"])
        )
        va = run_epoch(
            model, val_loader, criterion, device,
            optimizer=None, scaler=None, use_amp=use_amp
        )

        val_metrics = multilabel_metrics(va["y_true"], va["y_prob"], default_t)
        score = val_metrics["macro_f1"]
        scheduler.step(score)

        row = {
            "epoch": epoch,
            "train_loss": tr["loss"],
            "val_loss": va["loss"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_micro_f1": val_metrics["micro_f1"],
            "val_auc_pr": val_metrics["macro_auc_pr"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d} | train_loss={tr['loss']:.4f} "
            f"| val_loss={va['loss']:.4f} | val_macro_f1={score:.4f}"
        )

        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "model_type": args.model,
                    "model_state": best_state,
                    "num_labels": n_labels,
                    "best_val_macro_f1": best_score,
                },
                out_dir / "best_model.pt",
            )
            patience = 0
        else:
            patience += 1
            if patience >= max_patience:
                print("Early stopping.")
                break

    history_df = pd.DataFrame(history)
    history_df.to_csv(out_dir / "history.csv", index=False)
    plot_history(history_df, out_dir)

    # Reload best validation-selected state.
    model.load_state_dict(best_state)

    # Validation-only threshold selection.
    val_best = evaluate_with_ids(model, val_loader, criterion, device, use_amp)
    threshold, threshold_macro_f1 = tune_global_threshold(val_best["y_true"], val_best["y_prob"])
    save_json(
        {
            "threshold": threshold,
            "validation_macro_f1_at_threshold": threshold_macro_f1,
            "selection_split": "val",
        },
        out_dir / "val_threshold.json",
    )

    # Final held-out test evaluation.
    test = evaluate_with_ids(model, test_loader, criterion, device, use_amp)
    metrics = save_test_artifacts(
        y_true=test["y_true"],
        y_prob=test["y_prob"],
        embeddings=test["embedding"],
        labels=labels,
        threshold=threshold,
        out_dir=out_dir,
        track_ids=test["track_ids"],
    )
    metrics["test_loss"] = test["loss"]
    with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Final test metrics:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
