from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from .contrastive import GraphTextContrastiveModel
from .datasets import PairedGraphTextDataset, graph_text_collate
from .factory import build_components
from .utils import ensure_dir, load_config, resolve_device, save_json, set_seed


def move_batch(batch, device):
    out = dict(batch)
    out["graph"] = batch["graph"].to(device)
    out["input_ids"] = batch["input_ids"].to(device)
    out["attention_mask"] = batch["attention_mask"].to(device)
    return out


def retrieval_metrics(graph_emb, text_emb):
    sim = text_emb @ graph_emb.T
    n = sim.shape[0]
    ks = [1, 5, 10]
    result = {}

    def recall(mat, prefix):
        order = np.argsort(mat, axis=1)[:, ::-1]
        truth = np.arange(n)
        for k in ks:
            k_eff = min(k, n)
            hit = np.any(order[:, :k_eff] == truth[:, None], axis=1).mean()
            result[f"{prefix}_R@{k}"] = float(hit)

    recall(sim, "text_to_music")
    recall(sim.T, "music_to_text")
    return result


def collect(model, loader, device):
    model.eval()
    gs, ts, ids = [], [], []
    with torch.no_grad():
        for raw in tqdm(loader, leave=False):
            ids.extend(raw["track_id"])
            batch = move_batch(raw, device)
            gs.append(model.encode_graph(batch["graph"]).cpu().numpy())
            ts.append(model.encode_text(batch["input_ids"], batch["attention_mask"]).cpu().numpy())
    return np.concatenate(gs), np.concatenate(ts), ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    device = resolve_device(cfg.get("device", "cuda:0"))

    train_ds = PairedGraphTextDataset(cfg, "train")
    val_ds = PairedGraphTextDataset(cfg, "val")
    test_ds = PairedGraphTextDataset(cfg, "test")

    bs = int(cfg["contrastive"]["batch_size"])
    nw = int(cfg["train"]["num_workers"])
    make_loader = lambda ds, shuffle: DataLoader(
        ds, batch_size=bs, shuffle=shuffle, num_workers=nw, collate_fn=graph_text_collate
    )
    train_loader = make_loader(train_ds, True)
    val_loader = make_loader(val_ds, False)
    test_loader = make_loader(test_ds, False)

    gnn, text = build_components(cfg)
    model = GraphTextContrastiveModel(
        gnn=gnn,
        text_encoder=text,
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        projection_dim=int(cfg["contrastive"]["projection_dim"]),
        temperature=float(cfg["contrastive"]["temperature"]),
    ).to(device)

    bert_params, other = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (bert_params if "text_encoder.transformer" in name else other).append(p)

    optimizer = AdamW(
        [
            {"params": bert_params, "lr": float(cfg["contrastive"]["bert_lr"])},
            {"params": other, "lr": float(cfg["contrastive"]["lr"])},
        ],
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    out_dir = ensure_dir(Path(cfg["train"]["output_dir"]) / "contrastive")
    save_json(cfg, out_dir / "resolved_config.json")

    best_r5, best_state, stale = -1.0, None, 0
    for epoch in range(1, int(cfg["contrastive"]["epochs"]) + 1):
        model.train()
        losses = []
        for raw in tqdm(train_loader, desc=f"Contrastive epoch {epoch}", leave=False):
            batch = move_batch(raw, device)
            optimizer.zero_grad(set_to_none=True)
            out = model(batch)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip"]))
            optimizer.step()
            losses.append(out["loss"].item())

        vg, vt, _ = collect(model, val_loader, device)
        vm = retrieval_metrics(vg, vt)
        r5 = vm["text_to_music_R@5"]
        print(f"Epoch {epoch:02d} | loss={np.mean(losses):.4f} | val text->music R@5={r5:.4f}")

        if r5 > best_r5:
            best_r5 = r5
            best_state = copy.deepcopy(model.state_dict())
            torch.save({"model_state": best_state, "best_val_R@5": best_r5}, out_dir / "best_model.pt")
            stale = 0
        else:
            stale += 1
            if stale >= int(cfg["contrastive"]["patience"]):
                break

    model.load_state_dict(best_state)
    tg, tt, ids = collect(model, test_loader, device)
    metrics = retrieval_metrics(tg, tt)
    with open(out_dir / "test_retrieval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    np.savez_compressed(out_dir / "test_embeddings.npz", graph=tg, text=tt, track_ids=np.array(ids))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
