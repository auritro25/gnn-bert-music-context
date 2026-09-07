from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import pandas as pd


def safe_list(value):
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            return [value]
    return []


def clean_text(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    text = str(x).replace("<br>", " ").replace("<br/>", " ").replace("<br />", " ")
    return " ".join(text.split())


def subset_mask(series, subset):
    s = series.astype(str).str.lower()
    if subset == "small":
        return s.eq("small")
    if subset == "medium":
        return s.isin(["small", "medium"])
    raise ValueError("subset must be small or medium")


def audio_relative_path(track_id: int) -> str:
    return f"{track_id // 1000:03d}/{track_id:06d}.mp3"


def normalize_split(x):
    x = str(x).strip().lower()
    mapping = {"training": "train", "train": "train", "validation": "val", "valid": "val", "val": "val", "test": "test", "testing": "test"}
    if x not in mapping:
        raise ValueError(f"Unknown FMA split value: {x}")
    return mapping[x]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tracks-csv", required=True)
    p.add_argument("--subset", choices=["small", "medium"], required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--audio-root", default="data/raw/fma_medium")
    p.add_argument("--drop-missing-audio", action="store_true")
    args = p.parse_args()

    tracks = pd.read_csv(args.tracks_csv, index_col=0, header=[0, 1])
    mask = subset_mask(tracks[("set", "subset")], args.subset)
    df = tracks.loc[mask].copy()

    rows, missing_audio = [], []
    for tid, row in df.iterrows():
        tid = int(tid)
        split = normalize_split(row[("set", "split")])
        genre = clean_text(row.get(("track", "genre_top"), ""))
        tags = [clean_text(t) for t in safe_list(row.get(("track", "tags"), [])) if clean_text(t)]

        title = clean_text(row.get(("track", "title"), ""))
        artist_name = clean_text(row.get(("artist", "name"), ""))
        album_title = clean_text(row.get(("album", "title"), ""))
        artist_bio = clean_text(row.get(("artist", "bio"), ""))
        album_info = clean_text(row.get(("album", "information"), ""))

        parts = []
        if title: parts.append(f"Track title: {title}.")
        if artist_name: parts.append(f"Artist: {artist_name}.")
        if album_title: parts.append(f"Album: {album_title}.")
        if artist_bio: parts.append(f"Artist description: {artist_bio}")
        if album_info: parts.append(f"Album information: {album_info}")
        caption = " ".join(parts).strip() or "Music track with unavailable textual metadata."

        rel = audio_relative_path(tid)
        full = Path(args.audio_root) / rel
        if not full.exists():
            missing_audio.append((tid, str(full)))
            if args.drop_missing_audio:
                continue

        rows.append({
            "track_id": str(tid), "audio_path": rel, "caption": caption,
            "tags": json.dumps(tags, ensure_ascii=False), "genre": genre,
            "split": split, "artist_id": row.get(("artist", "id"), ""),
            "album_id": row.get(("album", "id"), ""),
            "fma_subset": str(row.get(("set", "subset"), "")), "source_dataset": "FMA",
        })

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows); out_df.to_csv(out, index=False)

    # Strict artist leakage audit.
    clean = out_df.dropna(subset=["artist_id"]).copy()
    clean = clean[clean["artist_id"].astype(str).str.strip() != ""]
    groups = {s: set(clean.loc[clean["split"] == s, "artist_id"].astype(str)) for s in ["train", "val", "test"]}
    overlaps = []
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        n = len(groups[a] & groups[b])
        if n: overlaps.append((a, b, n))
    if overlaps:
        raise RuntimeError("Artist leakage detected: " + ", ".join(f"{a}-{b}: {n}" for a,b,n in overlaps))

    print(f"Saved {len(out_df)} rows -> {out}")
    print(out_df["split"].value_counts())
    print("Artist leakage audit: PASS")
    if missing_audio:
        print(f"Audio existence audit: {len(missing_audio)} missing file(s). Example: {missing_audio[0]}")
        if not args.drop_missing_audio:
            print("Metadata was written; extract FMA-medium audio before preprocessing.")
    else:
        print("Audio existence audit: PASS")

if __name__ == "__main__":
    main()
