from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm


def safe_id(x):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(x))


def fixed_hash_split(track_id: str):
    # Stable 80/10/10 split based only on ID.
    h = int(hashlib.sha1(track_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if h < 80:
        return "train"
    if h < 90:
        return "val"
    return "test"


def parse_aspects(x):
    if isinstance(x, list):
        return [str(v).strip() for v in x if str(v).strip()]
    text = str(x)
    try:
        y = ast.literal_eval(text)
        if isinstance(y, list):
            return [str(v).strip() for v in y if str(v).strip()]
    except Exception:
        pass
    return [v.strip() for v in text.split(",") if v.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="google/MusicCaps")
    p.add_argument("--output", default="data/splits/musiccaps.csv")
    p.add_argument("--audio-root", default="data/raw/musiccaps_audio")
    p.add_argument("--tag-source", choices=["aspect_list", "audioset_positive_labels"], default="aspect_list")
    p.add_argument("--save-audio", action="store_true", help="Save decoded audio if dataset exposes an audio column.")
    args = p.parse_args()

    ds = load_dataset(args.dataset, split="train")
    audio_root = Path(args.audio_root)
    audio_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, row in enumerate(tqdm(ds, desc="Preparing metadata")):
        raw_id = row.get("ytid") or row.get("index") or row.get("fname") or f"row_{i:06d}"
        track_id = safe_id(raw_id)

        tags = parse_aspects(row.get(args.tag_source, ""))

        if args.save_audio:
            if "audio" not in row or row["audio"] is None:
                raise RuntimeError(
                    f"{args.dataset} does not expose decoded audio. "
                    "Use the metadata-only mode and provide audio separately."
                )
            audio = row["audio"]
            wav_path = audio_root / f"{track_id}.wav"
            sf.write(wav_path, np.asarray(audio["array"], dtype=np.float32), int(audio["sampling_rate"]))
            audio_path = wav_path.name
        else:
            audio_path = f"{track_id}.wav"

        rows.append(
            {
                "track_id": track_id,
                "audio_path": audio_path,
                "caption": str(row.get("caption") or row.get("caption_ground_truth") or ""),
                "tags": json.dumps(tags, ensure_ascii=False),
                "split": fixed_hash_split(track_id),
            }
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved {len(rows)} rows to {out}")
    print(pd.DataFrame(rows)["split"].value_counts())


if __name__ == "__main__":
    main()
