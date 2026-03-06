from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from dataclasses import dataclass
from typing import Iterable, Tuple, Dict, List, Sequence
from sklearn.metrics import roc_auc_score
from utilities import make_feature_names
import pandas as pd

N_ALL_FEATURES = 136
HISTORY_FEATURE_IDS = {134, 135, 136}  # your history-based features


def feature_importance_from_booster_safe(
        booster: lgb.Booster,
        importance_type: str = "gain",     # "gain" or "split"
        feature_names: list[str] | None = None,
) -> pd.DataFrame:
    imp = booster.feature_importance(importance_type=importance_type)

    # LightGBM's own feature names (length should match imp)
    booster_names = booster.feature_name()
    if booster_names is not None and len(booster_names) == len(imp):
        names = booster_names
    else:
        names = None

    # If user-provided names exist, use them only if lengths match
    if feature_names is not None:
        if len(feature_names) == len(imp):
            names = feature_names
        else:
            print(
                f"[warn] feature_names length ({len(feature_names)}) != "
                f"importance length ({len(imp)}). Using booster feature names."
            )

    # Final fallback: generate names with the correct length
    if names is None:
        names = make_feature_names(len(imp))

    df = pd.DataFrame({"feature": names, importance_type: imp})
    return df.sort_values(importance_type, ascending=False).reset_index(drop=True)


def feature_importance_from_ranker_safe(
        ranker: lgb.LGBMRanker,
        importance_type: str = "gain",
        feature_names: list[str] | None = None,
) -> pd.DataFrame:
    return feature_importance_from_booster_safe(
        ranker.booster_,
        importance_type=importance_type,
        feature_names=feature_names,
    )



def binarize_labels(y: np.ndarray) -> np.ndarray:
    # relevant if label is 3 or 4
    return (y >= 3).astype(np.int32)


def dcg_binary(rels: np.ndarray) -> float:
    discounts = 1.0 / np.log2(np.arange(2, rels.size + 2))
    return float(np.sum(rels * discounts))


def ndcg_at_k_binary(y_true_bin: np.ndarray, y_score: np.ndarray, k: int) -> float:
    if y_true_bin.size == 0:
        return 0.0
    k = min(k, y_true_bin.size)
    order = np.argsort(-y_score)[:k]
    ideal = np.sort(y_true_bin)[::-1][:k]
    dcg = dcg_binary(y_true_bin[order])
    idcg = dcg_binary(ideal)
    return 0.0 if idcg == 0.0 else (dcg / idcg)


def precision_at_k(y_true_bin: np.ndarray, y_score: np.ndarray, k: int) -> float:
    if y_true_bin.size == 0:
        return 0.0
    k = min(k, y_true_bin.size)
    order = np.argsort(-y_score)[:k]
    return float(np.mean(y_true_bin[order]))


def recall_at_k(y_true_bin: np.ndarray, y_score: np.ndarray, k: int) -> float:
    total_rel = int(np.sum(y_true_bin))
    if total_rel == 0:
        return 0.0
    k = min(k, y_true_bin.size)
    order = np.argsort(-y_score)[:k]
    return float(np.sum(y_true_bin[order]) / total_rel)


def average_precision_at_k(y_true_bin: np.ndarray, y_score: np.ndarray, k: int) -> float:
    total_rel = int(np.sum(y_true_bin))
    if total_rel == 0:
        return 0.0
    k = min(k, y_true_bin.size)
    order = np.argsort(-y_score)[:k]
    rels = y_true_bin[order]
    precisions = []
    hits = 0
    for i, r in enumerate(rels, start=1):
        if r == 1:
            hits += 1
            precisions.append(hits / i)
    return float(np.mean(precisions)) if precisions else 0.0

def mrr(y_true_bin: np.ndarray, y_score: np.ndarray) -> float:
    if y_true_bin.size == 0:
        return 0.0
    order = np.argsort(-y_score)
    rels = y_true_bin[order]
    idx = np.where(rels == 1)[0]
    return float(1.0 / (idx[0] + 1)) if idx.size > 0 else 0.0


def mean_query_auc(y: np.ndarray, scores: np.ndarray, group: np.ndarray) -> float:
    """
    Mean AUC across queries (binary labels). Skips queries where AUC undefined.
    """
    yb = binarize_labels(y)
    aucs = []
    start = 0
    for g in group:
        end = start + int(g)
        y_q = yb[start:end]
        s_q = scores[start:end]
        if np.min(y_q) != np.max(y_q):  # both classes exist
            aucs.append(roc_auc_score(y_q, s_q))
        start = end
    return float(np.mean(aucs)) if aucs else float("nan")


def evaluate_binary_ranking_with_auc(
        y: np.ndarray,
        scores: np.ndarray,
        group: np.ndarray,
        ks=(1, 3, 5, 10),
) -> dict:
    yb = binarize_labels(y)

    metrics = {f"ndcg@{k}": [] for k in ks}
    metrics.update({f"p@{k}": [] for k in ks})
    metrics.update({f"r@{k}": [] for k in ks})
    metrics.update({f"map@{k}": [] for k in ks})
    metrics["mrr"] = []
    aucs = []

    start = 0
    for g in group:
        end = start + int(g)
        y_q = yb[start:end]
        s_q = scores[start:end]

        for k in ks:
            metrics[f"ndcg@{k}"].append(ndcg_at_k_binary(y_q, s_q, k))
            metrics[f"p@{k}"].append(precision_at_k(y_q, s_q, k))
            metrics[f"r@{k}"].append(recall_at_k(y_q, s_q, k))
            metrics[f"map@{k}"].append(average_precision_at_k(y_q, s_q, k))

        metrics["mrr"].append(mrr(y_q, s_q))

        if np.min(y_q) != np.max(y_q):
            aucs.append(roc_auc_score(y_q, s_q))

        start = end

    results = {m: float(np.mean(v)) for m, v in metrics.items()}
    results["mean_query_auc"] = float(np.mean(aucs)) if aucs else float("nan")
    results["auc_queries_used"] = int(len(aucs))
    results["total_queries"] = int(len(group))
    return results


def train_lgbm_ranker(train: LetorData, val: LetorData) -> lgb.LGBMRanker:
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        train.X, train.y, group=train.group,
        eval_set=[(val.X, val.y)],
        eval_group=[val.group],
        eval_at=(1, 3, 5, 10),
    )
    return model


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


# Load all features (1..136)
train_a = load_letor("MSLR-WEB10K/Fold1/train.txt", use_all_features=True)   # 136 dims
val_a   = load_letor("MSLR-WEB10K/Fold1/vali.txt",  use_all_features=True)
test_a  = load_letor("MSLR-WEB10K/Fold1/test.txt",  use_all_features=True)

model_all = train_lgbm_ranker(train_a, val_a)

test_scores_a = model_all.predict(test_a.X)

binary_results_all = evaluate_binary_ranking_with_auc(
    y=test_a.y,
    scores=test_scores_a,
    group=test_a.group,
    ks=(1, 3, 5, 10),
)

for k, v in binary_results_all.items():
    if isinstance(v, float) and np.isfinite(v):
        print(f"{k:16s} {v:.5f}")
    else:
        print(f"{k:16s} {v}")

#### Now Cold Items
DWELL_COL_0IDX = 135  # feature 136

@dataclass
class LetorSplit:
    X: np.ndarray
    y: np.ndarray
    group: np.ndarray  # rebuilt per subset


def split_cold_warm(
        X: np.ndarray,
        y: np.ndarray,
        group: np.ndarray,
        dwell_col: int = DWELL_COL_0IDX,
):
    """
    Returns (cold_subset, warm_subset), each with rebuilt group arrays.
    Cold is defined as X[:, dwell_col] == 0.
    """
    cold_mask = (X[:, dwell_col] == 0.0)

    X_c, y_c, g_c = [], [], []
    X_w, y_w, g_w = [], [], []

    start = 0
    for g in group:
        end = start + int(g)

        m_q = cold_mask[start:end]
        X_q = X[start:end]
        y_q = y[start:end]

        # cold rows within this query
        if np.any(m_q):
            X_c.append(X_q[m_q])
            y_c.append(y_q[m_q])
            g_c.append(int(np.sum(m_q)))

        # warm rows within this query
        mw_q = ~m_q
        if np.any(mw_q):
            X_w.append(X_q[mw_q])
            y_w.append(y_q[mw_q])
            g_w.append(int(np.sum(mw_q)))

        start = end

    cold = LetorSplit(
        X=np.vstack(X_c) if X_c else np.empty((0, X.shape[1]), dtype=X.dtype),
        y=np.concatenate(y_c) if y_c else np.empty((0,), dtype=y.dtype),
        group=np.asarray(g_c, dtype=np.int32),
    )
    warm = LetorSplit(
        X=np.vstack(X_w) if X_w else np.empty((0, X.shape[1]), dtype=X.dtype),
        y=np.concatenate(y_w) if y_w else np.empty((0,), dtype=y.dtype),
        group=np.asarray(g_w, dtype=np.int32),
    )
    return cold, warm

def eval_on_cold_and_warm(
        model,                 # LightGBM Booster or LGBMRanker
        X: np.ndarray,
        y: np.ndarray,
        group: np.ndarray,
        *,
        ks=(1,3,5,10),
        dwell_col=DWELL_COL_0IDX,
):
    cold, warm = split_cold_warm(X, y, group, dwell_col=dwell_col)

    out = {}

    # Cold
    if cold.X.shape[0] > 0 and cold.group.size > 0:
        s_cold = model.predict(cold.X)
        out["cold"] = evaluate_binary_ranking_with_auc(cold.y, s_cold, cold.group, ks=ks)
        out["cold"]["n_rows"] = int(cold.X.shape[0])
        out["cold"]["n_queries"] = int(cold.group.size)
    else:
        out["cold"] = {"note": "no cold rows"}

    # Warm
    if warm.X.shape[0] > 0 and warm.group.size > 0:
        s_warm = model.predict(warm.X)
        out["warm"] = evaluate_binary_ranking_with_auc(warm.y, s_warm, warm.group, ks=ks)
        out["warm"]["n_rows"] = int(warm.X.shape[0])
        out["warm"]["n_queries"] = int(warm.group.size)
    else:
        out["warm"] = {"note": "no warm rows"}

    return out


def _compute_group_from_sorted_qids(qids_sorted: np.ndarray) -> np.ndarray:
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


def load_letor_136(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_list, y_list, qid_list = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            line = line.split("#", 1)[0].strip()
            parts = line.split()

            y_list.append(int(parts[0]))
            qid_list.append(int(parts[1].split(":")[1]))

            x = np.zeros(N_ALL_FEATURES, dtype=np.float32)
            for tok in parts[2:]:
                k, v = tok.split(":")
                fid = int(k)
                if 1 <= fid <= N_ALL_FEATURES:
                    x[fid - 1] = float(v)
            X_list.append(x)

    X = np.vstack(X_list)
    y = np.asarray(y_list, dtype=np.int32)
    qids = np.asarray(qid_list, dtype=np.int32)

    order = np.argsort(qids, kind="mergesort")
    X, y, qids = X[order], y[order], qids[order]
    group = _compute_group_from_sorted_qids(qids)
    return X, y, group


# Example: evaluate curriculum booster on cold/warm test rows
Xte, yte, gte = load_letor_136("MSLR-WEB10K/Fold1/test.txt")   # returns X, y, group
results = eval_on_cold_and_warm(model_all, Xte, yte, gte, ks=(1,3,5,10))

print("=== COLD ===")
for k, v in results["cold"].items():
    print(f"{k:16s} {v}" if not isinstance(v, float) else f"{k:16s} {v:.5f}")

print("\n=== WARM ===")
for k, v in results["warm"].items():
    print(f"{k:16s} {v}" if not isinstance(v, float) else f"{k:16s} {v:.5f}")

print("Feature Importance:")
fi_curr_gain = feature_importance_from_ranker_safe(
    model_all,
    feature_names=make_feature_names(136),
    importance_type="gain",
)

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 0)

print(fi_curr_gain)
