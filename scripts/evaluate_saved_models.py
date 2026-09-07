import argparse
import json

import torch
import yaml

from torch.utils.data import DataLoader

from src.factory import build_supervised_model
from src.datasets import PairedGraphTextDataset, graph_text_collate
from src.evaluation import save_test_artifacts, tune_global_threshold


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["device"])

    with open(
        cfg["data"]["label_vocab"],
        encoding="utf-8"
    ) as f:
        labels = json.load(f)

    if isinstance(labels, dict):
        if "labels" in labels:
            labels = labels["labels"]
        elif "label_to_id" in labels:
            labels = list(labels["label_to_id"].keys())
        else:
            labels = list(labels.keys())

    num_labels = len(labels)

    print("Number of labels:", num_labels)

    dataset = PairedGraphTextDataset(
        cfg,
        split="test"
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        collate_fn=graph_text_collate,
        num_workers=0,
    )

    model = build_supervised_model(
        cfg,
        args.model,
        num_labels
    ).to(device)

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state"]
    )

    model.eval()

    y_true = []
    y_prob = []
    embeddings = []
    track_ids = []

    with torch.no_grad():

        for batch in loader:

            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device)

                elif hasattr(v, "to"):
                    batch[k] = v.to(device)

            out = model(batch)

            y_true.append(
                batch["labels"].cpu()
            )

            y_prob.append(
                torch.sigmoid(out["logits"]).cpu()
            )

            if "embedding" in out:
                embeddings.append(
                    out["embedding"].cpu()
                )

            if "track_ids" in batch:
                track_ids.extend(
                    batch["track_ids"]
                )

    y_true = torch.cat(y_true).numpy()
    y_prob = torch.cat(y_prob).numpy()

    emb = None

    if len(embeddings) > 0:
        emb = torch.cat(embeddings).numpy()

    threshold, _ = tune_global_threshold(
        y_true,
        y_prob
    )

    metrics = save_test_artifacts(
        y_true=y_true,
        y_prob=y_prob,
        embeddings=emb,
        labels=labels,
        threshold=threshold,
        out_dir=args.output,
        track_ids=track_ids,
    )

    print(json.dumps(metrics, indent=2))
    print("Saved evaluation:", args.output)


if __name__ == "__main__":
    main()