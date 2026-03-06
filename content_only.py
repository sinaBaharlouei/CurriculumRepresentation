"""
Content-only LambdaMART (LightGBM) on MSLR-WEB10K:
- 5 folds (Fold1..Fold5)
- 5 repeats per fold (different seeds)
- trains ONLY on content (history features f134,f135,f136 are masked to 0 for train/val)
- evaluates on test with masked inputs too
- cold split is defined using ORIGINAL Xte (f136 == 0), but the model is fed masked features
- writes CSVs:
    content_only_per_fold_repeat.csv
    content_only_mean.csv
    content_only_std.csv
"""

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from dataclasses import dataclass
from typing import Sequence, Dict

# --------------------------
# CONFIG
# --------------------------
BASE_DIR = "MSLR-WEB10K"         # change if needed
TOTAL_TREES = 2000
FOLDS = [1, 2, 3, 4, 5]
REPEATS = 5
KS = (1, 3, 5, 10)
OUT_PREFIX = "NEW_content_only_5fold_5repeat"

# f134,f135,f136 are columns 133,134,135 (0-indexed) in 136-dim matrix
HIST_COLS_0IDX = [133, 134, 135]
DWELL_COL_0IDX = 135  # f136


N_ALL_FEATURES = 136
HISTORY_FEATURE_IDS = {134, 135, 136}  # your history-based features


@dataclass
class LetorData:
    X: np.ndarray          # shape [n_docs, n_features]
    y: np.ndarray          # shape [n_docs]
    group: np.ndarray      # shape [n_queries], sums to n_docs



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

# --------------------------
# REQUIRED: you already have these in your codebase
# --------------------------
# def load_letor_136(path): -> (X, y, group)
# def evaluate_binary_ranking_with_auc(y, scores, group, ks=(1,3,5,10)): -> dict with map@k, mean_query_auc, auc_queries_used, etc.


# --------------------------
# Helpers
# --------------------------
def mask_history(X: np.ndarray) -> np.ndarray:
    X2 = X.copy()
    X2[:, HIST_COLS_0IDX] = 0.0
    return X2

def pooled_auc(y: np.ndarray, scores: np.ndarray) -> float:
    yb = (y >= 3).astype(np.int32)
    if yb.size == 0 or np.min(yb) == np.max(yb):
        return float("nan")
    return float(roc_auc_score(yb, scores))

def _dcg(rels: np.ndarray) -> float:
    gains = (2.0 ** rels - 1.0)
    discounts = 1.0 / np.log2(np.arange(2, rels.size + 2))
    return float(np.sum(gains * discounts))

def graded_ndcg_at_k(y: np.ndarray, scores: np.ndarray, group: np.ndarray, ks=(1,3,5,10)) -> dict:
    out = {f"ndcg@{k}": 0.0 for k in ks}
    start = 0
    n_q = 0
    for g in group:
        g = int(g)
        end = start + g
        yq = y[start:end]
        sq = scores[start:end]

        order = np.argsort(-sq, kind="mergesort")
        y_sorted = yq[order]
        ideal = np.sort(yq)[::-1]

        for k in ks:
            kk = min(k, g)
            dcg_k = _dcg(y_sorted[:kk])
            idcg_k = _dcg(ideal[:kk])
            ndcg = (dcg_k / idcg_k) if idcg_k > 0 else 0.0
            out[f"ndcg@{k}"] += ndcg

        n_q += 1
        start = end

    for k in ks:
        out[f"ndcg@{k}"] = out[f"ndcg@{k}"] / n_q if n_q > 0 else float("nan")
    return out

def with_seed(params: dict, seed: int) -> dict:
    p = dict(params)
    p["seed"] = seed
    p["bagging_seed"] = seed
    p["feature_fraction_seed"] = seed
    p["data_random_seed"] = seed
    return p

def train_content_only_booster(
        Xtr, ytr, gtr,
        Xva, yva, gva,
        *,
        total_trees: int,
        params: dict,
) -> lgb.Booster:
    Xtr_m = mask_history(Xtr)
    Xva_m = mask_history(Xva)

    dtrain = lgb.Dataset(Xtr_m, label=ytr, group=gtr, free_raw_data=False)
    dval   = lgb.Dataset(Xva_m, label=yva, group=gva, free_raw_data=False)

    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=total_trees,
        valid_sets=[dval],
        valid_names=["val"],
    )
    return booster

def split_cold_warm_by_original_Xte(Xte: np.ndarray, yte: np.ndarray, gte: np.ndarray, dwell_col: int = DWELL_COL_0IDX):
    """
    Cold rows are those with f136==0. Keeps query structure by rebuilding group arrays.
    Returns: (Xc,yc,gc), (Xw,yw,gw)
    """
    X_c, y_c, g_c = [], [], []
    X_w, y_w, g_w = [], [], []

    start = 0
    for g in gte:
        g = int(g)
        end = start + g
        Xq = Xte[start:end]
        yq = yte[start:end]

        cold_mask = (Xq[:, dwell_col] == 0.0)
        warm_mask = ~cold_mask

        if np.any(cold_mask):
            X_c.append(Xq[cold_mask])
            y_c.append(yq[cold_mask])
            g_c.append(int(np.sum(cold_mask)))

        if np.any(warm_mask):
            X_w.append(Xq[warm_mask])
            y_w.append(yq[warm_mask])
            g_w.append(int(np.sum(warm_mask)))

        start = end

    def stack(partsX, partsy, partsg):
        if not partsX:
            return np.empty((0, Xte.shape[1])), np.empty((0,)), np.asarray([], dtype=np.int32)
        return np.vstack(partsX), np.concatenate(partsy), np.asarray(partsg, dtype=np.int32)

    Xc, yc, gc = stack(X_c, y_c, g_c)
    Xw, yw, gw = stack(X_w, y_w, g_w)
    return (Xc, yc, gc), (Xw, yw, gw)


# --------------------------
# Main run
# --------------------------
def run_content_only_5x5_to_csv(
        base_dir: str = BASE_DIR,
        out_prefix: str = OUT_PREFIX,
):
    base_params = dict(
        objective="lambdarank",
        metric="ndcg",
        ndcg_eval_at=list(KS),
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=20,
        subsample=1,
        colsample_bytree=1,
        feature_pre_filter=False,
        verbosity=-1,
    )

    rows = []

    for fold in FOLDS:
        train_path = os.path.join(base_dir, f"Fold{fold}", "train.txt")
        val_path   = os.path.join(base_dir, f"Fold{fold}", "vali.txt")
        test_path  = os.path.join(base_dir, f"Fold{fold}", "test.txt")

        Xtr, ytr, gtr = load_letor_136(train_path)
        Xva, yva, gva = load_letor_136(val_path)
        Xte, yte, gte = load_letor_136(test_path)

        (Xc, yc, gc), (Xw, yw, gw) = split_cold_warm_by_original_Xte(Xte, yte, gte)

        for r in range(REPEATS):
            seed = 10_000 * fold + r
            params = with_seed(base_params, seed)

            booster = train_content_only_booster(
                Xtr, ytr, gtr,
                Xva, yva, gva,
                total_trees=TOTAL_TREES,
                params=params,
            )

            # Predict using MASKED inputs (content-only)
            scores_all = booster.predict(mask_history(Xte))
            scores_cold = booster.predict(mask_history(Xc)) if Xc.shape[0] > 0 else np.asarray([])
            scores_warm = booster.predict(mask_history(Xw)) if Xw.shape[0] > 0 else np.asarray([])

            # Overall: graded NDCG
            overall_ndcg = graded_ndcg_at_k(yte, scores_all, gte, ks=KS)

            # Overall: binary MAP/AUC
            overall_bin = evaluate_binary_ranking_with_auc(yte, scores_all, gte, ks=KS)
            overall_pooled = pooled_auc(yte, scores_all)

            # Cold: graded NDCG
            cold_ndcg = graded_ndcg_at_k(yc, scores_cold, gc, ks=KS) if yc.size > 0 else {f"ndcg@{k}": float("nan") for k in KS}
            cold_bin = evaluate_binary_ranking_with_auc(yc, scores_cold, gc, ks=KS) if yc.size > 0 else {}
            cold_pooled = pooled_auc(yc, scores_cold) if yc.size > 0 else float("nan")

            # Warm (optional)
            warm_ndcg = graded_ndcg_at_k(yw, scores_warm, gw, ks=KS) if yw.size > 0 else {f"ndcg@{k}": float("nan") for k in KS}

            row = {
                "fold": fold,
                "repeat": r,
                "seed": seed,

                # history gains should be ~0 in content-only baseline
                "f134_gain": 0.0,
                "f135_gain": 0.0,
                "f136_gain": 0.0,

                "overall_pooled_auc": overall_pooled,
                "cold_pooled_auc": cold_pooled,

                "overall_auc": float(overall_bin.get("mean_query_auc", np.nan)),
                "cold_auc": float(cold_bin.get("mean_query_auc", np.nan)) if yc.size > 0 else float("nan"),
                "overall_auc_queries_used": float(overall_bin.get("auc_queries_used", np.nan)),
                "cold_auc_queries_used": float(cold_bin.get("auc_queries_used", np.nan)) if yc.size > 0 else float("nan"),
            }

            for k in KS:
                row[f"overall_ndcg@{k}"] = overall_ndcg[f"ndcg@{k}"]
                row[f"cold_ndcg@{k}"]    = cold_ndcg[f"ndcg@{k}"]
                row[f"warm_ndcg@{k}"]    = warm_ndcg[f"ndcg@{k}"]

                row[f"overall_map@{k}"] = float(overall_bin.get(f"map@{k}", np.nan))
                row[f"cold_map@{k}"]    = float(cold_bin.get(f"map@{k}", np.nan)) if yc.size > 0 else float("nan")

            rows.append(row)

            print(
                f"Fold {fold} rep {r} | "
                f"overall ndcg@10={row['overall_ndcg@10']:.5f} map@10={row['overall_map@10']:.5f} "
                f"mqAUC={row['overall_auc']:.5f} pooledAUC={row['overall_pooled_auc']:.5f} | "
                f"cold ndcg@10={row['cold_ndcg@10']:.5f} map@10={row['cold_map@10']:.5f} "
                f"mqAUC={row['cold_auc']:.5f} pooledAUC={row['cold_pooled_auc']:.5f}"
            )

    df = pd.DataFrame(rows).sort_values(["fold", "repeat"]).reset_index(drop=True)

    metric_cols = (
            ["f134_gain", "f135_gain", "f136_gain"]
            + [f"cold_ndcg@{k}" for k in KS]
            + [f"cold_map@{k}" for k in KS]
            + ["cold_auc", "cold_pooled_auc", "cold_auc_queries_used"]
            + [f"overall_ndcg@{k}" for k in KS]
            + [f"overall_map@{k}" for k in KS]
            + ["overall_auc", "overall_pooled_auc", "overall_auc_queries_used"]
            + [f"warm_ndcg@{k}" for k in KS]
    )

    mean_df = df[metric_cols].mean(numeric_only=True).to_frame("mean").reset_index().rename(columns={"index": "metric"})
    std_df  = df[metric_cols].std(numeric_only=True, ddof=1).to_frame("std").reset_index().rename(columns={"index": "metric"})

    per_path = f"{out_prefix}_per_fold_repeat.csv"
    mean_path = f"{out_prefix}_mean.csv"
    std_path = f"{out_prefix}_std.csv"

    df.to_csv(per_path, index=False)
    mean_df.to_csv(mean_path, index=False)
    std_df.to_csv(std_path, index=False)

    print("\nSaved CSVs:")
    print(" -", per_path)
    print(" -", mean_path)
    print(" -", std_path)

    return df, mean_df, std_df


if __name__ == "__main__":
    run_content_only_5x5_to_csv()
