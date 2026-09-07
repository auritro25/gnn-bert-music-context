# GNN–BERT Music Context Understanding

Complete PyTorch/PyTorch-Geometric implementation for the CSE425/EEE474/CSE715 project:

- Task 1: BERT multi-label music-tag baseline
- Task 2: GraphSAGE/GAT on music segment graphs
- Baseline B2: CNN on log-mel spectrograms
- Task 3: GNN–BERT early-concatenation and cross-attention fusion
- Ablation-ready model selection: `bert`, `gnn`, `concat`, `fusion`
- Task 4: Optional contrastive GNN–BERT music↔text retrieval
- Macro-F1, Micro-F1, precision, recall, AUC-PR, ROC-AUC
- Validation-only threshold selection
- Training/validation curves
- ROC and PR curves
- Aggregate multilabel confusion matrix
- t-SNE of learned embeddings
- Example end-to-end inference notebook

The code is designed around **paired music + text + multi-label tags**, which matches MusicCaps-style experiments well. It also works with your own paired metadata CSV.

## 1. Important experimental rules

1. Build the label vocabulary from the **training split only**.
2. Do not use validation/test samples to choose tags, normalization parameters, thresholds, or checkpoints.
3. Choose the best checkpoint using validation Macro-F1 only.
4. Tune the classification threshold on validation only, then freeze it before test evaluation.
5. Keep the provided/fixed split file unchanged after it is created.
6. Do not use `DataParallel` or DDP. The code always uses one device (`cuda:0` when available).
7. For Kaggle with T4x2, launch with `CUDA_VISIBLE_DEVICES=0`.

## 2. Repository

```text
gnn-bert-music-context/
├── README.md
├── requirements.txt
├── config.yaml
├── data/
│   ├── README.md
│   ├── raw/
│   ├── processed/
│   │   ├── graphs/
│   │   └── mels/
│   └── splits/
├── notebooks/
│   ├── eda.ipynb
│   └── demo_context.ipynb
├── scripts/
│   └── prepare_musiccaps_metadata.py
├── src/
│   ├── __init__.py
│   ├── audio_features.py
│   ├── graph_builder.py
│   ├── labels.py
│   ├── datasets.py
│   ├── bert_encoder.py
│   ├── gnn_model.py
│   ├── fusion_model.py
│   ├── cnn_model.py
│   ├── contrastive.py
│   ├── factory.py
│   ├── evaluation.py
│   ├── preprocess.py
│   ├── train.py
│   ├── train_cnn.py
│   ├── train_contrastive.py
│   ├── inference.py
│   └── utils.py
└── results/
```

## 3. Installation

Create a fresh environment and install:

```bash
pip install -r requirements.txt
```

PyTorch Geometric can be installed directly with `pip install torch_geometric` on current PyG releases. If your platform has a special CUDA/PyTorch combination, follow the official PyG installation page.

## 4. Dataset format

The main pipeline expects `data/splits/musiccaps.csv` (or any CSV path set in `config.yaml`) containing:

```text
track_id,audio_path,caption,tags,split
abc001,abc001.wav,"A slow sad piano ballad","[""sad"", ""piano"", ""ballad""]",train
abc002,abc002.wav,"Fast energetic drums","[""energetic"", ""drums""]",val
abc003,abc003.wav,"Soft guitar instrumental","[""guitar"", ""instrumental""]",test
```

- `audio_path` may be absolute or relative to `data.audio_root`.
- `tags` can be a JSON/Python-list string, comma-separated string, or `|`-separated string.
- `split` must be `train`, `val`, or `test`.

### MusicCaps helper

If you have Internet access, the included script can create metadata from a Hugging Face MusicCaps source.

Metadata-only Google source:

```bash
python scripts/prepare_musiccaps_metadata.py \
  --dataset google/MusicCaps \
  --audio-root data/raw/musiccaps_audio \
  --output data/splits/musiccaps.csv \
  --tag-source aspect_list
```

If using a dataset mirror that exposes decoded audio in an `audio` column, add:

```bash
--save-audio
```

The script makes one deterministic split file. Do **not** regenerate it between experiments.

> MusicCaps' official Google release is metadata pointing to 10-second AudioSet/YouTube excerpts. Make sure your audio acquisition method complies with the dataset/source terms that apply to you.

## 5. Configure

Edit `config.yaml`.

Important defaults:

- sample rate: 22,050 Hz
- 10 s clip duration
- 2 s segments
- node feature = MFCC mean/std + chroma mean
- temporal + cosine-similarity graph edges
- `bert-base-uncased`
- GraphSAGE
- 15 epochs
- validation Macro-F1 checkpoint selection
- mixed precision on CUDA
- single GPU only

## 6. Preprocess

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.preprocess --config config.yaml
```

This will:

1. Load only the training labels to create `label_vocab.json`.
2. Load/resample/pad/crop audio.
3. Split each clip into fixed windows.
4. Extract segment MFCC/chroma node features.
5. Build temporal and similarity graph edges.
6. Save PyG `.pt` graphs.
7. Save fixed log-mel arrays for the CNN baseline.

## 7. Train required models

### Task 1 — BERT-only

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.train --config config.yaml --model bert
```

### Task 2 — GNN-only

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.train --config config.yaml --model gnn
```

### CNN baseline

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.train_cnn --config config.yaml
```

### Task 3 — Early concatenation ablation

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.train --config config.yaml --model concat
```

### Task 3 — Cross-attention GNN–BERT fusion

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.train --config config.yaml --model fusion
```

Each run saves its own folder under `results/<model>/`.

## 8. Task 4 — Contrastive music↔text retrieval

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.train_contrastive --config config.yaml
```

The contrastive model learns aligned graph and BERT embeddings with symmetric InfoNCE loss and reports:

- text → music R@1, R@5, R@10
- music → text R@1, R@5, R@10

## 9. Outputs

Each supervised run produces:

```text
results/<model>/
├── best_model.pt
├── history.csv
├── val_threshold.json
├── test_metrics.json
├── test_predictions.npz
├── per_label_metrics.csv
├── training_loss.png
├── f1_curves.png
├── pr_curves.png
├── roc_curves.png
├── confusion_matrix.png
└── tsne.png
```

## 10. Recommended final comparison table

| Model | Macro-F1 | Micro-F1 | Mean AUC-PR | ROC-AUC |
|---|---:|---:|---:|---:|
| Majority/random | | | | |
| CNN mel-spec | | | | |
| BERT-only | | | | |
| GNN-only | | | | |
| GNN+BERT concat | | | | |
| GNN+BERT cross-attention | | | | |

Do not copy illustrative values from the project brief. Replace them with your real experimental results.

## 11. Leakage note

The code intentionally:

- builds tag vocabulary using `train` only;
- computes positive class weights using `train` only;
- chooses best model on `val` only;
- tunes decision threshold using `val` only;
- evaluates `test` only after model/threshold selection;
- never fits a scaler on combined splits.

MusicCaps captions may literally mention target aspects. For a stronger experiment, enable:

```yaml
data:
  mask_label_terms_in_text: true
```

This replaces exact target tag phrases with `[MASK]` in captions, reducing obvious text-label leakage.

## 12. Demo

Open:

```text
notebooks/demo_context.ipynb
```

Set the checkpoint path to `results/fusion/best_model.pt`, then run the cells to show one end-to-end prediction.

## 13. Reproducibility

- Random seed is fixed in `config.yaml`.
- Split creation is deterministic.
- No multi-GPU wrappers are used.
- Config and label vocabulary are saved alongside outputs.
- Best model selection criterion is explicit.
## 14. Generate the required 3 case studies

After training the fusion model:

```bash
python -m src.case_studies \
  --config config.yaml \
  --checkpoint results/fusion/best_model.pt \
  --num-cases 3
```

This saves graph/text alignment heatmaps, graph visualizations, predictions and a JSON summary under:

```text
results/fusion/case_studies/
```

## 15. Automatic ablation table

After training the supervised models:

```bash
python scripts/compare_models.py
```

It writes:

```text
results/ablation_comparison.csv
```

## 16. Run everything

Once the metadata/audio are ready:

```bash
bash run_all.sh
```

Or use `notebooks/full_project_pipeline.ipynb`.

## Strict assignment-compliance extension

A stricter dataset/evaluation path is included in `STRICT_COMPLIANCE_RUNBOOK.md`. It preserves the original MusicCaps results and adds FMA-small Task 2, FMA-medium Task 3, official artist-safe split auditing, masked-caption analysis, Task 4 qualitative retrieval examples, and an automated compliance checker.


A submission-aligned report skeleton is available at `report/REPORT_TEMPLATE.md`.
