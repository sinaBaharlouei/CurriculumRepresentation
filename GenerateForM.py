import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from utilities import make_feature_names
from dataclasses import dataclass
from typing import Sequence, Dict
from sklearn.metrics import roc_auc_score

N_ALL_FEATURES = 136
DWELL_COL_0IDX = 135  # feature 136
HISTORY_COLS_0IDX = [133, 134, 135]  # features 134-136 in 1-indexed -> 133-135 in 0-indexed
TOTAL_TREES = 2000
M_VALUES = list(range(0, 2000, 100))  # 0..1900 step 100 (exclude 2000 as requested)
FOLDS = [1, 2, 3, 4, 5]
KS = (1, 3, 5, 10)
REPEATS = 5
HIST_FEATURES = ["f134", "f135", "f136"]


def get_hist_gains(booster: lgb.Booster) -> dict:
    """
    Returns gain importance for f134/f135/f136 from the full model.
    Requires feature_importance_from_booster_safe + make_feature_names.
    """
    fi = feature_importance_from_booster_safe(
        booster,
        importance_type="gain",
        feature_names=make_feature_names(booster.num_feature()),
    )
    fi_map = dict(zip(fi["feature"], fi["gain"]))

    return {
        "f134_gain": float(fi_map.get("f134", 0.0)),
        "f135_gain": float(fi_map.get("f135", 0.0)),
        "f136_gain": float(fi_map.get("f136", 0.0)),
    }


def _dcg(rels: np.ndarray) -> float:
    # DCG with gains 2^rel - 1
    gains = (2.0 ** rels - 1.0)
    discounts = 1.0 / np.log2(np.arange(2, rels.size + 2))
    return float(np.sum(gains * discounts))


def graded_ndcg_at_k(y: np.ndarray, scores: np.ndarray, group: np.ndarray, ks=(1,3,5,10)) -> dict:
    """
    Graded NDCG@k using original labels (0..4), per query then averaged.
    """
    out = {f"ndcg@{k}": 0.0 for k in ks}
    start = 0
    n_q = 0
    for g in group:
        g = int(g)
        end = start + g
        yq = y[start:end]
        sq = scores[start:end]

        # sort by predicted score desc
        order = np.argsort(-sq, kind="mergesort")
        y_sorted = yq[order]

        # ideal by label desc
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


base_params = dict(
    objective="lambdarank",
    metric="ndcg",
    ndcg_eval_at=list(KS),
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=20,
    subsample=1.0,
    colsample_bytree=1.0,
    feature_pre_filter=False,
    verbosity=-1,
)

@dataclass
class LetorSplit:
    X: np.ndarray
    y: np.ndarray
    group: np.ndarray  # rebuilt per subset


def with_seed(params: dict, seed: int) -> dict:
    p = dict(params)
    p["seed"] = seed
    p["bagging_seed"] = seed
    p["feature_fraction_seed"] = seed
    p["data_random_seed"] = seed
    return p


def pick_metrics_multi_k(
        *,
        graded_ndcg_dict: dict,
        binary_eval_dict: dict,
        ks=(1,3,5,10),
) -> dict:
    out = {}
    for k in ks:
        out[f"ndcg@{k}"] = float(graded_ndcg_dict.get(f"ndcg@{k}", np.nan))
        out[f"map@{k}"]  = float(binary_eval_dict.get(f"map@{k}", np.nan))
    out["auc"] = float(binary_eval_dict.get("mean_query_auc", np.nan))  # mean per-query AUC (binary)
    out["auc_queries_used"] = float(binary_eval_dict.get("auc_queries_used", np.nan))
    return out



def mask_history_features(X: np.ndarray) -> np.ndarray:
    Xm = X.copy()
    Xm[:, HISTORY_COLS_0IDX] = 0.0
    return Xm


# ---------- Binary ranking metrics (+ mean per-query AUC) ----------
def binarize_labels(y: np.ndarray) -> np.ndarray:
    return (y >= 3).astype(np.int32)  # 3/4 relevant


def pooled_auc(y: np.ndarray, scores: np.ndarray) -> float:
    """
    Global AUC across all rows (binary labels), ignoring query boundaries.
    """
    yb = (y >= 3).astype(np.int32)  # relevant if 3/4
    if yb.size == 0 or np.min(yb) == np.max(yb):
        return float("nan")  # undefined if only one class
    return float(roc_auc_score(yb, scores))


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


def train_all_features_booster(
        X_train, y_train, g_train,
        X_val,   y_val,   g_val,
        *,
        total_trees: int = TOTAL_TREES,
        params: dict | None = None,
) -> lgb.Booster:
    """Train a standard all-features LambdaMART booster for total_trees rounds."""
    default_params = dict(
        objective="lambdarank",
        metric="ndcg",
        ndcg_eval_at=[1, 3, 5, 10],
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        feature_pre_filter=False,
        seed=42,
        verbosity=-1,
    )
    if params:
        default_params.update(params)

    dtrain = lgb.Dataset(X_train, label=y_train, group=g_train, free_raw_data=False)
    dval   = lgb.Dataset(X_val,   label=y_val,   group=g_val,   free_raw_data=False)

    booster = lgb.train(
        default_params,
        dtrain,
        num_boost_round=total_trees,
        valid_sets=[dval],
        valid_names=["val"],
    )
    return booster


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


def get_f134_gain(booster: lgb.Booster) -> float:
    """Return gain importance for f134 from full model."""
    fi = feature_importance_from_booster_safe(
        booster,
        importance_type="gain",
        feature_names=make_feature_names(booster.num_feature()),
    )
    row = fi.loc[fi["feature"] == "f134"]
    if row.empty:
        return 0.0
    # column name is "gain"
    return float(row["gain"].iloc[0])


def pick_metrics(result_dict: dict) -> dict:
    """Extract the key metrics we care about."""
    return {
        "ndcg@10": float(result_dict.get("ndcg@10", np.nan)),
        "map@10": float(result_dict.get("map@10", np.nan)),
        "auc": float(result_dict.get("mean_query_auc", np.nan)),
        "auc_queries_used": float(result_dict.get("auc_queries_used", np.nan)),
    }


def run_sweep_all_folds(
        base_dir: str = "MSLR-WEB10K",
        out_prefix: str = "curriculum_latest_5folds",
):
    rows = []

    for fold in FOLDS:
        train_path = os.path.join(base_dir, f"Fold{fold}", "train.txt")
        val_path   = os.path.join(base_dir, f"Fold{fold}", "vali.txt")
        test_path  = os.path.join(base_dir, f"Fold{fold}", "test.txt")

        print(f"\n===== Fold {fold} =====")
        print("Loading:", train_path)

        Xtr, ytr, gtr = load_letor_136(train_path)
        Xva, yva, gva = load_letor_136(val_path)
        Xte, yte, gte = load_letor_136(test_path)
        cold_subset, warm_subset = split_cold_warm(Xte, yte, gte)  # LetorSplit objects

        for m in M_VALUES:
            print(f"\n--- Fold {fold} | m={m} ---")
            for r in range(REPEATS):
                seed = 1000*fold + 10*m + r  # any deterministic scheme is fine
                params = with_seed(base_params, seed)
                # Train booster
                if m == 0:
                    booster = train_all_features_booster(
                        Xtr, ytr, gtr, Xva, yva, gva, total_trees=TOTAL_TREES, params=params
                    )
                else:
                    booster = train_curriculum_lambdamart(
                        Xtr, ytr, gtr,
                        Xva, yva, gva,
                        n_content_trees=m,
                        total_trees=TOTAL_TREES,
                        params=params
                    )

                # f134, f135, f136 gains from full model importance
                hist_gains = get_hist_gains(booster)

                # overall
                scores_all = booster.predict(Xte)
                # graded ndcg on original labels
                overall_ndcg = graded_ndcg_at_k(yte, scores_all, gte, ks=KS)

                # binary MAP/AUC (thresholding handled inside your function)
                overall_bin = evaluate_binary_ranking_with_auc(yte, scores_all, gte, ks=KS)

                overall_pooled = pooled_auc(yte, scores_all)

                overall_k = pick_metrics_multi_k(
                    graded_ndcg_dict=overall_ndcg,
                    binary_eval_dict=overall_bin,
                    ks=KS
                )

                # cold
                scores_cold = booster.predict(cold_subset.X)

                cold_ndcg = graded_ndcg_at_k(cold_subset.y, scores_cold, cold_subset.group, ks=KS)
                cold_bin  = evaluate_binary_ranking_with_auc(cold_subset.y, scores_cold, cold_subset.group, ks=KS)

                cold_pooled = pooled_auc(cold_subset.y, scores_cold)

                cold_k = pick_metrics_multi_k(
                    graded_ndcg_dict=cold_ndcg,
                    binary_eval_dict=cold_bin,
                    ks=KS
                )

                row = {
                    "fold": fold,
                    "m": m,
                    "repeat": r,
                    "seed": seed,

                    **hist_gains,

                    "overall_pooled_auc": overall_pooled,
                    "cold_pooled_auc": cold_pooled,

                    "overall_auc": overall_k["auc"],
                    "overall_auc_queries_used": overall_k["auc_queries_used"],

                    "cold_auc": cold_k["auc"],
                    "cold_auc_queries_used": cold_k["auc_queries_used"],
                }

                # attach ndcg/map at all ks + mean_query_auc
                for k in KS:
                    row[f"overall_ndcg@{k}"] = overall_k[f"ndcg@{k}"]
                    row[f"overall_map@{k}"]  = overall_k[f"map@{k}"]
                    row[f"cold_ndcg@{k}"]    = cold_k[f"ndcg@{k}"]
                    row[f"cold_map@{k}"]     = cold_k[f"map@{k}"]

                rows.append(row)

                print(
                    f"m={m:4d} rep={r} fold={fold} | "
                    f"gains(f134={row['f134_gain']:.1f}, f135={row['f135_gain']:.1f}, f136={row['f136_gain']:.1f}) | "
                    f"cold(ndcg10={row['cold_ndcg@10']:.5f}, map10={row['cold_map@10']:.5f}, "
                    f"mqAUC={row['cold_auc']:.5f}, pooledAUC={row['cold_pooled_auc']:.5f}) | "
                    f"overall(ndcg10={row['overall_ndcg@10']:.5f}, map10={row['overall_map@10']:.5f}, "
                    f"mqAUC={row['overall_auc']:.5f}, pooledAUC={row['overall_pooled_auc']:.5f})"
                )

    metric_cols = [
        "f134_gain", "f135_gain", "f136_gain",

        *[f"cold_ndcg@{k}" for k in KS],
        *[f"cold_map@{k}"  for k in KS],
        "cold_auc", "cold_auc_queries_used", "cold_pooled_auc",

        *[f"overall_ndcg@{k}" for k in KS],
        *[f"overall_map@{k}"  for k in KS],
        "overall_auc", "overall_auc_queries_used", "overall_pooled_auc",
    ]

    df = pd.DataFrame(rows).sort_values(["m", "fold", "repeat"]).reset_index(drop=True)

    # 1) average over folds, separately for each (m, repeat)
    mean_over_folds_by_repeat = df.groupby(["m", "repeat"], as_index=False)[metric_cols].mean()
    std_over_folds_by_repeat  = df.groupby(["m", "repeat"], as_index=False)[metric_cols].std(ddof=1)

    # 2) now summarize over repeats (this is the "final" curve you plot)
    mean_df = mean_over_folds_by_repeat.groupby("m", as_index=False)[metric_cols].mean()
    std_df  = mean_over_folds_by_repeat.groupby("m", as_index=False)[metric_cols].std(ddof=1)

    print("\n===== Per fold+repeat results (head) =====")
    print(df.head(10))

    print("\n===== Mean over folds, per repeat (head) =====")
    print(mean_over_folds_by_repeat.head(10))

    print("\n===== Final mean over repeats (per m) =====")
    print(mean_df)

    print("\n===== Final std over repeats (per m) =====")
    print(std_df)

    # Write CSVs
    df.to_csv(f"{out_prefix}_per_fold_per_repeat.csv", index=False)
    mean_over_folds_by_repeat.to_csv(f"{out_prefix}_mean_over_folds_by_repeat.csv", index=False)
    std_over_folds_by_repeat.to_csv(f"{out_prefix}_std_over_folds_by_repeat.csv", index=False)
    mean_df.to_csv(f"{out_prefix}_mean_over_folds_and_repeats.csv", index=False)
    std_df.to_csv(f"{out_prefix}_std_over_folds_and_repeats.csv", index=False)

    # # Write CSVs
    # per_fold_path = f"{out_prefix}_per_fold.csv"
    # mean_path     = f"{out_prefix}_mean_over_folds.csv"
    # std_path      = f"{out_prefix}_std_over_folds.csv"
    #
    # df.to_csv(per_fold_path, index=False)
    # mean_df.to_csv(mean_path, index=False)
    # std_df.to_csv(std_path, index=False)
    #
    # print("\nSaved CSVs:")
    # print(" -", per_fold_path)
    # print(" -", mean_path)
    # print(" -", std_path)

    return df, mean_df, std_df


if __name__ == "__main__":
    # Change base_dir if needed (e.g. "MSLR-WEB30K")
    run_sweep_all_folds(
        base_dir="MSLR-WEB10K",
        out_prefix="curriculum_sweep_5folds",
    )
