# Data layout

Place audio under:

```text
data/raw/musiccaps_audio/
```

and metadata at:

```text
data/splits/musiccaps.csv
```

Required CSV columns:

- `track_id`
- `audio_path`
- `caption`
- `tags`
- `split`

`split` must be exactly `train`, `val`, or `test`.

The preprocessing step writes:

```text
data/processed/graphs/<track_id>.pt
data/processed/mels/<track_id>.npy
data/processed/label_vocab.json
```
