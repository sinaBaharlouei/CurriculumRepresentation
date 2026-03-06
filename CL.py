from __future__ import annotations
import numpy as np
import lightgbm as lgb
from dataclasses import dataclass
from typing import Sequence, Dict
from sklearn.metrics import roc_auc_score
from utilities import make_feature_names
import pandas as pd


N_ALL_FEATURES = 136
HISTORY_COLS_0IDX = [133, 134, 135]  # features 134-136 in 1-indexed -> 133-135 in 0-indexed


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



@dataclass
class LetorData:
    X: np.ndarray
    y: np.ndarray
    group: np.ndarray


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


def mask_history_features(X: np.ndarray) -> np.ndarray:
    Xm = X.copy()
    Xm[:, HISTORY_COLS_0IDX] = 0.0
    return Xm


# ---------- Binary ranking metrics (+ mean per-query AUC) ----------
def binarize_labels(y: np.ndarray) -> np.ndarray:
    return (y >= 3).astype(np.int32)  # 3/4 relevant


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


def evaluate_binary_ranking_with_auc(
        y: np.ndarray,
        scores: np.ndarray,
        group: np.ndarray,
        ks: Sequence[int] = (1, 3, 5, 10),
) -> Dict[str, float]:
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

        if np.min(y_q) != np.max(y_q):  # AUC defined only if both classes present
            aucs.append(roc_auc_score(y_q, s_q))

        start = end

    out = {m: float(np.mean(v)) for m, v in metrics.items()}
    out["mean_query_auc"] = float(np.mean(aucs)) if aucs else float("nan")
    out["auc_queries_used"] = float(len(aucs))
    out["total_queries"] = float(len(group))
    return out


# ---------- Curriculum training ----------
def train_curriculum_lambdamart(
        X_train: np.ndarray, y_train: np.ndarray, g_train: np.ndarray,
        X_val: np.ndarray,   y_val: np.ndarray,   g_val: np.ndarray,
        *,
        n_content_trees: int,
        total_trees: int,
        params: dict | None = None,
) -> lgb.Booster:
    assert 0 < n_content_trees <= total_trees
    m_all_trees = total_trees - n_content_trees

    default_params = dict(
        objective="lambdarank",
        metric="ndcg",
        ndcg_eval_at=[1, 3, 5, 10],
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        feature_pre_filter=False,   # important for init_model continuation
        seed=42,
        verbosity=-1,
    )
    if params:
        default_params.update(params)

    # Stage 1: content-only via masking history features to 0
    Xtr_c = mask_history_features(X_train)
    Xva_c = mask_history_features(X_val)

    dtrain_c = lgb.Dataset(Xtr_c, label=y_train, group=g_train, free_raw_data=False)
    dval_c   = lgb.Dataset(Xva_c, label=y_val,   group=g_val,   free_raw_data=False)

    booster = lgb.train(
        default_params,
        dtrain_c,
        num_boost_round=n_content_trees,
        valid_sets=[dval_c],
        valid_names=["val"],
    )

    # Stage 2: continue training on all features for remaining trees
    if m_all_trees > 0:
        dtrain_all = lgb.Dataset(X_train, label=y_train, group=g_train, free_raw_data=False)
        dval_all   = lgb.Dataset(X_val,   label=y_val,   group=g_val,   free_raw_data=False)

        booster = lgb.train(
            default_params,
            dtrain_all,
            num_boost_round=m_all_trees,
            valid_sets=[dval_all],
            valid_names=["val"],
            init_model=booster,
            keep_training_booster=True,
        )

    return booster

# ---------- Example run on Fold1 ----------
# Load 136-dim data
Xtr, ytr, gtr = load_letor_136("MSLR-WEB10K/Fold1/train.txt")
Xva, yva, gva = load_letor_136("MSLR-WEB10K/Fold1/vali.txt")
Xte, yte, gte = load_letor_136("MSLR-WEB10K/Fold1/test.txt")

# Match your prior setting, e.g. total_trees=2000; choose n (content stage)
total_trees = 2000
n_content = 1000  # example; m will be 1200

booster_curr = train_curriculum_lambdamart(
    Xtr, ytr, gtr,
    Xva, yva, gva,
    n_content_trees=n_content,
    total_trees=total_trees,
)

scores_test = booster_curr.predict(Xte)  # uses all built trees by default
results = evaluate_binary_ranking_with_auc(yte, scores_test, gte, ks=(1,3,5,10))

for k, v in results.items():
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
        ks=(1, 3, 5, 10),
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


# Example: evaluate curriculum booster on cold/warm test rows
results = eval_on_cold_and_warm(booster_curr, Xte, yte, gte, ks=(1,3,5,10))

print("=== COLD ===")
for k, v in results["cold"].items():
    print(f"{k:16s} {v}" if not isinstance(v, float) else f"{k:16s} {v:.5f}")

print("\n=== WARM ===")
for k, v in results["warm"].items():
    print(f"{k:16s} {v}" if not isinstance(v, float) else f"{k:16s} {v:.5f}")


print("Feature Importance:")
fi_curr_gain = feature_importance_from_booster_safe(
    booster_curr,
    feature_names=make_feature_names(136),
    importance_type="gain",
)

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 0)

print(fi_curr_gain)