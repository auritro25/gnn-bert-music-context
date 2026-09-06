from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np


@dataclass
class AudioConfig:
    sample_rate: int = 22050
    duration_seconds: float = 10.0
    segment_seconds: float = 2.0
    n_mfcc: int = 20
    n_chroma: int = 12
    n_mels: int = 128
    hop_length: int = 512


def load_audio_fixed(path: str | Path, cfg: AudioConfig) -> np.ndarray:
    y, _ = librosa.load(path, sr=cfg.sample_rate, mono=True)
    target = int(cfg.sample_rate * cfg.duration_seconds)

    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    elif len(y) > target:
        y = y[:target]

    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    return y.astype(np.float32)


def split_audio(y: np.ndarray, cfg: AudioConfig) -> list[np.ndarray]:
    seg_len = max(1, int(cfg.sample_rate * cfg.segment_seconds))
    segments = []
    for start in range(0, len(y), seg_len):
        seg = y[start : start + seg_len]
        if len(seg) < seg_len:
            seg = np.pad(seg, (0, seg_len - len(seg)))
        segments.append(seg.astype(np.float32))
    return segments


def segment_node_feature(segment: np.ndarray, cfg: AudioConfig) -> np.ndarray:
    mfcc = librosa.feature.mfcc(
        y=segment,
        sr=cfg.sample_rate,
        n_mfcc=cfg.n_mfcc,
        hop_length=cfg.hop_length,
    )
    chroma = librosa.feature.chroma_stft(
        y=segment,
        sr=cfg.sample_rate,
        n_chroma=cfg.n_chroma,
        hop_length=cfg.hop_length,
    )

    # 20 MFCC means + 20 MFCC stds + 12 chroma means = 52 dimensions by default.
    feat = np.concatenate(
        [
            mfcc.mean(axis=1),
            mfcc.std(axis=1),
            chroma.mean(axis=1),
        ]
    )
    return np.nan_to_num(feat).astype(np.float32)


def extract_node_features(y: np.ndarray, cfg: AudioConfig) -> np.ndarray:
    feats = [segment_node_feature(seg, cfg) for seg in split_audio(y, cfg)]
    return np.stack(feats, axis=0).astype(np.float32)


def extract_logmel(y: np.ndarray, cfg: AudioConfig) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=cfg.sample_rate,
        n_mels=cfg.n_mels,
        hop_length=cfg.hop_length,
        power=2.0,
    )
    db = librosa.power_to_db(mel, ref=np.max)
    # Per-track normalization only: no dataset-wide statistics and no cross-split fitting.
    mean = db.mean()
    std = db.std() + 1e-6
    return ((db - mean) / std).astype(np.float32)
