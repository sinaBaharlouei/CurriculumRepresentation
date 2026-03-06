from __future__ import annotations
import numpy as np
import lightgbm as lgb
from dataclasses import dataclass
from typing import Iterable, Tuple, Dict, List, Sequence

N_ALL_FEATURES = 136
HISTORY_FEATURE_IDS = {134, 135, 136}  # your history-based features

@dataclass
class LetorData:
    X: np.ndarray          # shape [n_docs, n_features]
    y: np.ndarray          # shape [n_docs]
    group: np.ndarray      # shape [n_queries], sums to n_docs


def _compute_group_from_sorted_qids(qids_sorted: np.ndarray) -> np.ndarray:
    # qids must be contiguous by query
    group = []
    cnt = 1
    for i in range(1, len(qids_sorted)):
        if qids_sorted[i] == qids_sorted[i - 1]:
            cnt += 1
        else:
            group.append(cnt)
            cnt = 1
    group.append(cnt)
    return np.asarray(group, dtype=np.int32)

def load_letor(
        path: str,
        *,
        use_all_features: bool,
        n_all_features: int = N_ALL_FEATURES,
) -> LetorData:
    """
    Loads a LETOR-format file like:
      <label> qid:<qid> 1:<v> 2:<v> ... 136:<v> # comment

    If use_all_features=False, drops features {134,135,136} and returns 133-dim X.
    """
    # Decide output feature dimensionality and mapping
    if use_all_features:
        out_dim = n_all_features
        def out_index(fid: int) -> int:  # fid: 1..136
            return fid - 1
        def keep(fid: int) -> bool:
            return 1 <= fid <= n_all_features
    else:
        # keep 1..133 only
        out_dim = n_all_features - len(HISTORY_FEATURE_IDS)  # 133
        def out_index(fid: int) -> int:
            # fid in 1..133 -> 0..132
            return fid - 1
        def keep(fid: int) -> bool:
            return (1 <= fid <= n_all_features) and (fid not in HISTORY_FEATURE_IDS)

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    qid_list: List[int] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            line = line.split("#", 1)[0].strip()  # remove comments
            parts = line.split()
            y = int(parts[0])
            qid = int(parts[1].split(":")[1])

            x = np.zeros(out_dim, dtype=np.float32)
            for tok in parts[2:]:
                fid_s, val_s = tok.split(":")
                fid = int(fid_s)
                if keep(fid):
                    x[out_index(fid)] = float(val_s)

            y_list.append(y)
            qid_list.append(qid)
            X_list.append(x)

    X = np.vstack(X_list)
    y = np.asarray(y_list, dtype=np.int32)
    qids = np.asarray(qid_list, dtype=np.int32)

    # Ensure contiguous-by-qid ordering (safe even if file already grouped)
    order = np.argsort(qids, kind="mergesort")
    X, y, qids = X[order], y[order], qids[order]

    group = _compute_group_from_sorted_qids(qids)
    return LetorData(X=X, y=y, group=group)

def _dcg(rels: np.ndarray) -> float:
    # rels already sorted by predicted order
    gains = (2.0 ** rels - 1.0)
    discounts = 1.0 / np.log2(np.arange(2, rels.size + 2))
    return float(np.sum(gains * discounts))

def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    if y_true.size == 0:
        return 0.0
    k = min(k, y_true.size)

    order = np.argsort(-y_score)[:k]
    ideal = np.sort(y_true)[::-1][:k]

    dcg = _dcg(y_true[order])
    idcg = _dcg(ideal)
    return 0.0 if idcg == 0.0 else (dcg / idcg)

def mean_ndcg(
        y: np.ndarray,
        scores: np.ndarray,
        group: np.ndarray,
        ks: Sequence[int] = (1, 3, 5, 10),
) -> Dict[int, float]:
    out = {k: [] for k in ks}
    start = 0
    for g in group:
        end = start + int(g)
        y_q = y[start:end]
        s_q = scores[start:end]
        for k in ks:
            out[k].append(ndcg_at_k(y_q, s_q, k))
        start = end
    return {k: float(np.mean(vals)) for k, vals in out.items()}


def make_feature_names(n_features: int) -> list[str]:
    return [f"f{idx}" for idx in range(1, n_features + 1)]
