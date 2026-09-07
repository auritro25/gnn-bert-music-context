#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

python -m src.preprocess --config config.yaml
python -m src.train_cnn --config config.yaml
python -m src.train --config config.yaml --model bert
python -m src.train --config config.yaml --model gnn
python -m src.train --config config.yaml --model concat
python -m src.train --config config.yaml --model fusion
python -m src.case_studies --config config.yaml --checkpoint results/fusion/best_model.pt --num-cases 3
python -m src.train_contrastive --config config.yaml
python scripts/compare_models.py
