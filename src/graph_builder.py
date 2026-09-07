from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data


@dataclass
class GraphConfig:
    add_temporal_edges: bool = True
    add_similarity_edges: bool = True
    similarity_threshold: float = 0.82
    max_similarity_neighbors: int = 4
    undirected: bool = True


def _add_edge(edges, edge_types, weights, i, j, edge_type, weight, undirected):
    edges.append((i, j))
    edge_types.append(edge_type)
    weights.append(weight)
    if undirected and i != j:
        edges.append((j, i))
        edge_types.append(edge_type)
        weights.append(weight)


def build_segment_graph(node_features: np.ndarray, cfg: GraphConfig) -> Data:
    x = torch.tensor(node_features, dtype=torch.float32)
    n = x.size(0)
    edges, edge_types, weights = [], [], []

    if cfg.add_temporal_edges and n > 1:
        for i in range(n - 1):
            _add_edge(edges, edge_types, weights, i, i + 1, 0, 1.0, cfg.undirected)

    if cfg.add_similarity_edges and n > 1:
        norm = F.normalize(x, p=2, dim=-1)
        sim = (norm @ norm.T).cpu().numpy()
        np.fill_diagonal(sim, -1.0)

        for i in range(n):
            candidates = np.argsort(sim[i])[::-1]
            added = 0
            for j in candidates:
                if j == i:
                    continue
                score = float(sim[i, j])
                if score < cfg.similarity_threshold:
                    break
                # Avoid duplicating immediate temporal pairs as similarity edges.
                if abs(int(i) - int(j)) == 1:
                    continue
                _add_edge(edges, edge_types, weights, int(i), int(j), 1, score, False)
                added += 1
                if added >= cfg.max_similarity_neighbors:
                    break

    if not edges:
        edges = [(i, i) for i in range(n)]
        edge_types = [2] * n
        weights = [1.0] * n

    # Remove exact duplicate directed edges while preserving the first edge type.
    seen = set()
    clean_edges, clean_types, clean_weights = [], [], []
    for e, t, w in zip(edges, edge_types, weights):
        if e not in seen:
            seen.add(e)
            clean_edges.append(e)
            clean_types.append(t)
            clean_weights.append(w)

    edge_index = torch.tensor(clean_edges, dtype=torch.long).T.contiguous()
    edge_type = torch.tensor(clean_types, dtype=torch.long)
    edge_weight = torch.tensor(clean_weights, dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_type=edge_type, edge_weight=edge_weight)
