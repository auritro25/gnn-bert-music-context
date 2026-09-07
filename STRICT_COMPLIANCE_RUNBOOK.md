# Strict Assignment-Compliance Runbook

This path resolves the remaining compliance issues while preserving the original MusicCaps experiments.

## Dataset plan

- **MusicCaps**: Task 1 BERT caption→tag proxy and Task 4 music↔text retrieval.
- **FMA-small**: exact Task 2 genre classification benchmark.
- **FMA-medium**: exact Task 3 GNN–BERT fusion benchmark.

FMA-small is nested inside FMA-medium, so one FMA-medium audio download can serve both Tasks 2 and 3.

## 0. Preserve the results you already have

```powershell
powershell -ExecutionPolicy Bypass -File scripts\archive_current_results.ps1
```

This copies the current MusicCaps outputs to `results/musiccaps_original/`.

## 1. Download FMA-medium to the project drive

The official FMA-medium audio archive is about 22 GiB, plus extraction space.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\download_fma_medium.ps1
```

Expected after extraction:

```text
data/raw/fma_metadata/tracks.csv
data/raw/fma_medium/000/*.mp3
...
```

After verifying extraction, recover space by deleting the ZIPs:

```powershell
Remove-Item data\raw\fma_medium.zip
Remove-Item data\raw\fma_metadata.zip
```

## 2. Build metadata from the official FMA split

```powershell
python scripts\prepare_fma_metadata.py `
  --tracks-csv data\raw\fma_metadata\tracks.csv `
  --subset medium `
  --audio-root data\raw\fma_medium `
  --output data\splits\fma_medium.csv

python scripts\prepare_fma_metadata.py `
  --tracks-csv data\raw\fma_metadata\tracks.csv `
  --subset small `
  --audio-root data\raw\fma_medium `
  --output data\splits\fma_small.csv
```

The preparation script uses FMA's provided train/validation/test split and fails if artist overlap is detected.

Audit explicitly:

```powershell
python scripts\audit_splits.py --csv data\splits\fma_medium.csv --group-col artist_id
python scripts\audit_splits.py --csv data\splits\fma_small.csv --group-col artist_id
```

## 3. Preprocess FMA-medium once

```powershell
$env:MPLBACKEND="Agg"
python -m src.preprocess --config configs\fma_medium_tags.yaml
```

This uses:
- 22,050 Hz resampling
- 30-second FMA clips
- 5-second nodes
- MFCC/chroma node features
- temporal + cosine-similarity graph edges
- 128-bin log-mel features for the CNN

FMA-small uses the same processed files, so do not preprocess it again.

## 4. Task 1 — leakage-controlled MusicCaps BERT

Keep the original result, and run the stronger masked-caption version separately:

```powershell
python -m src.train --config configs\musiccaps_masked.yaml --model bert
```

Exact target tag phrases are masked in the caption before BERT sees the text. Results go to `results/musiccaps_masked/bert/`, so the original run is not overwritten.

## 5. Task 2 — FMA-small genre classification

GNN:

```powershell
python -m src.train_genre --config configs\fma_small_genre.yaml --model gnn
```

CNN baseline:

```powershell
python -m src.train_genre_cnn --config configs\fma_small_genre.yaml
```

These use `CrossEntropyLoss` and report genre accuracy and Macro-F1.

## 6. Task 3 — FMA-medium GNN–BERT fusion

The BERT input is built from track title, artist, album, artist biography and album information. The target tag list is deliberately excluded from the text input, and exact target terms are additionally masked.

Train all required ablations:

```powershell
python -m src.train --config configs\fma_medium_tags.yaml --model bert
python -m src.train --config configs\fma_medium_tags.yaml --model gnn
python -m src.train --config configs\fma_medium_tags.yaml --model concat
python -m src.train --config configs\fma_medium_tags.yaml --model fusion
```

Generate the required three graph/text case studies:

```powershell
python -m src.case_studies `
  --config configs\fma_medium_tags.yaml `
  --checkpoint results\fma_medium_tags\fusion\best_model.pt `
  --num-cases 3
```

## 7. Task 4 — MusicCaps contrastive retrieval

```powershell
python -m src.train_contrastive --config configs\musiccaps_task4.yaml
```

Run zero-shot tag ranking from the contrastive graph embedding against BERT tag prompts:

```powershell
python -m src.zero_shot_tags `
  --config configs\musiccaps_task4.yaml `
  --checkpoint results\musiccaps_task4\contrastive\best_model.pt `
  --split test `
  --top-k 3 `
  --output results\musiccaps_task4\contrastive\zero_shot_tags.json
```

This reports threshold-free Macro/Micro AUC-PR plus fixed top-3 Macro/Micro F1, which can be compared with the supervised tagging results.

Generate ten caption queries with their top-3 retrieved clips:

```powershell
python -m src.retrieval_examples `
  --embeddings results\musiccaps_task4\contrastive\test_embeddings.npz `
  --metadata data\splits\musiccaps.csv `
  --output results\musiccaps_task4\contrastive\retrieval_examples.csv `
  --num-queries 10 `
  --top-k 3
```

## 8. Final tables and compliance check

```powershell
python scripts\compare_strict_compliance.py
python scripts\compliance_check.py
```

The comparison table is written to:

```text
results/strict_compliance_comparison.csv
```

## 9. What this fixes

- Uses more than one Table-1 source: FMA + MusicCaps.
- Uses **FMA-small** for the exact Task-2 genre experiment.
- Uses **FMA-medium** for the exact Task-3 fusion experiment.
- Uses FMA's official reproducible split rather than creating a random split.
- Audits artist leakage.
- Preserves the original MusicCaps result and adds a leakage-controlled caption experiment.
- Permanently uses a headless Matplotlib backend to avoid the Tkinter crash.
- Produces Task-4 retrieval metrics and 10 qualitative retrieval examples.
- Leaves the demo notebook in place.

## 10. Final report

Do not write final numerical conclusions for FMA until the new experiments have actually run. Once `scripts/compliance_check.py` reports all required artifacts as PASS, use the saved real metrics/plots for the 6–10 page final report.
