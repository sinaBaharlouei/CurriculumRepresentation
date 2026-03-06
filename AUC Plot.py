"""
Cold vs Overall plots across M for:
  - NDCG@{1,3,5,10}
  - MAP@{1,3,5,10}
  - AUC (mean per-query AUC) + pooled AUC (optional)

No error bars. Mean over folds×repeats per M.
Exports PDF+PNG to OUT_DIR.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG: set your file paths
# -----------------------------
content_only_csv = "NEW_content_only_5fold_5repeat_per_fold_repeat.csv"  # m=2000 file
rest_csv = "new_all.csv"             # m=0..1900 file

OUT_DIR = "figs_kdd"
os.makedirs(OUT_DIR, exist_ok=True)

KS = (1, 3, 5, 10)

# -----------------------------
# Styling
# -----------------------------
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "lines.linewidth": 2.0,
    "lines.markersize": 5,
})

# -----------------------------
# Helpers
# -----------------------------
def load_and_merge(content_path: str, rest_path: str) -> pd.DataFrame:
    df_rest = pd.read_csv(rest_path)

    df_content = pd.read_csv(content_path).copy()
    df_content["m"] = 2000  # inject content-only point

    all_cols = sorted(set(df_rest.columns).union(df_content.columns))
    for c in all_cols:
        if c not in df_rest.columns:
            df_rest[c] = np.nan
        if c not in df_content.columns:
            df_content[c] = np.nan

    df_all = pd.concat([df_rest[all_cols], df_content[all_cols]], ignore_index=True)
    df_all["m"] = df_all["m"].astype(int)
    return df_all

def mean_by_m(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df.groupby("m", as_index=False)[cols].mean(numeric_only=True).sort_values("m")

def set_m_ticks(ax, ms: np.ndarray, label_every: int = 200):
    ms = np.asarray(ms, dtype=int)
    show = ms[ms % label_every == 0]
    if 0 in ms and 0 not in show:
        show = np.r_[show, 0]
    if 2000 in ms and 2000 not in show:
        show = np.r_[show, 2000]
    show = np.unique(show)
    ax.set_xticks(show)
    ax.set_xticklabels([str(int(x)) for x in show])

def savefig(fig, base_name: str):
    pdf_path = os.path.join(OUT_DIR, f"{base_name}.pdf")
    png_path = os.path.join(OUT_DIR, f"{base_name}.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    print("Saved:", pdf_path)
    print("Saved:", png_path)

def plot_two_lines(mean_df: pd.DataFrame, y1: str, y2: str, title: str, ylabel: str, fname: str, label_every: int = 200):
    ms = mean_df["m"].to_numpy(dtype=int)

    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ax.plot(ms, mean_df[y1].to_numpy(), marker="o", label=y1.replace("_", " "))
    ax.plot(ms, mean_df[y2].to_numpy(), marker="o", label=y2.replace("_", " "))

    ax.set_title(title)
    ax.set_xlabel("M (content-only trees before history)")
    ax.set_ylabel(ylabel)
    set_m_ticks(ax, ms, label_every=label_every)
    ax.legend(frameon=True)
    fig.tight_layout()
    savefig(fig, fname)
    plt.show()

# -----------------------------
# Main
# -----------------------------
df_all = load_and_merge(content_only_csv, rest_csv)

# NDCG plots
ndcg_cols = []
for k in KS:
    ndcg_cols += [f"cold_ndcg@{k}", f"overall_ndcg@{k}"]

# MAP plots
map_cols = []
for k in KS:
    map_cols += [f"cold_map@{k}", f"overall_map@{k}"]

# AUC plots (mean-query AUC) + pooled AUC if present
auc_cols = []
for c in ["cold_auc", "overall_auc", "cold_pooled_auc", "overall_pooled_auc"]:
    if c in df_all.columns:
        auc_cols.append(c)

needed = ["m"] + ndcg_cols + map_cols + auc_cols
missing = [c for c in needed if c not in df_all.columns]
if missing:
    raise ValueError(f"Missing columns in your merged data: {missing}")

mean_df = mean_by_m(df_all, needed)

# --- NDCG ---
for k in KS:
    plot_two_lines(
        mean_df,
        y1=f"cold_ndcg@{k}",
        y2=f"overall_ndcg@{k}",
        title=f"Cold vs Overall NDCG@{k} across curriculum depth M",
        ylabel=f"NDCG@{k} (mean over folds×repeats)",
        fname=f"ndcg{k}_cold_vs_overall",
        label_every=200,  # set 100 to label every M
    )

# --- MAP ---
for k in KS:
    plot_two_lines(
        mean_df,
        y1=f"cold_map@{k}",
        y2=f"overall_map@{k}",
        title=f"Cold vs Overall MAP@{k} across curriculum depth M",
        ylabel=f"MAP@{k} (mean over folds×repeats)",
        fname=f"map{k}_cold_vs_overall",
        label_every=200,
    )

# --- AUC (mean per-query AUC) ---
plot_two_lines(
    mean_df,
    y1="cold_auc",
    y2="overall_auc",
    title="Cold vs Overall AUC (mean per-query) across curriculum depth M",
    ylabel="AUC (mean over folds×repeats)",
    fname="auc_mean_query_cold_vs_overall",
    label_every=200,
)

# --- AUC (pooled) if available ---
if "cold_pooled_auc" in mean_df.columns and "overall_pooled_auc" in mean_df.columns:
    plot_two_lines(
        mean_df,
        y1="cold_pooled_auc",
        y2="overall_pooled_auc",
        title="Cold vs Overall AUC across curriculum depth M",
        ylabel="AUC (mean over folds×repeats)",
        fname="auc_pooled_cold_vs_overall",
        label_every=200,
    )
