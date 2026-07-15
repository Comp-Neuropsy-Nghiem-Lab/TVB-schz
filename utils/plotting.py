"""
utils/plotting.py

Reusable plots for visualization.ipynb: classification score violins,
J_i group differences, shuffle comparisons, graph-shuffle diagnostics and
summary tables. Statistics follow the same convention throughout the
project: paired Wilcoxon signed-rank test + Benjamini-Hochberg FDR
correction, with results shown as significance stars.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

CHANCE_LEVEL = 0.5
BENCHMARK = 0.98


def p_to_stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def paired_wilcoxon_stars(groups_a, groups_b, fdr_method="fdr_bh"):
    """
    Paired Wilcoxon signed-rank test per aligned pair of score lists,
    FDR-corrected across all pairs. groups_a/groups_b are same-length
    sequences of score arrays (one array per x-axis category).
    Returns (p_values_corrected, stars) both length len(groups_a).
    """
    p_values = [wilcoxon(a, b).pvalue for a, b in zip(groups_a, groups_b)]
    _, p_corrected, _, _ = multipletests(p_values, method=fdr_method)
    stars = [p_to_stars(p) for p in p_corrected]
    return p_corrected, stars


def _annotate_stars(ax, stars, y):
    for i, s in enumerate(stars):
        if s:
            ax.text(i, y, s, ha="center", va="bottom", fontsize=14)


def plot_score_violin(
    df,
    x,
    hue,
    hue_order,
    score_col="score",
    title=None,
    chance=CHANCE_LEVEL,
    benchmark=BENCHMARK,
    ax=None,
):
    """
    Split violin of test scores, grouped by `x` and split by `hue`
    (exactly two hue levels, given by `hue_order`). Adds chance/benchmark
    reference lines and significance stars from a paired Wilcoxon test
    between the two hue levels within each x category.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    sns.violinplot(
        data=df, x=x, y=score_col, hue=hue, hue_order=hue_order,
        split=True, gap=0.1, inner="quart", ax=ax,
    )
    ax.set_ylim(0, 1.0)
    ax.axhline(chance, color="red", linestyle="--", linewidth=1)
    ax.axhline(benchmark, color="black", linestyle="--", linewidth=1)
    if title:
        ax.set_title(title)

    categories = df[x].drop_duplicates().tolist()
    a_scores = [df[(df[x] == c) & (df[hue] == hue_order[0])][score_col].to_numpy() for c in categories]
    b_scores = [df[(df[x] == c) & (df[hue] == hue_order[1])][score_col].to_numpy() for c in categories]
    valid = [len(a) == len(b) and len(a) > 0 for a, b in zip(a_scores, b_scores)]
    if all(valid):
        _, stars = paired_wilcoxon_stars(a_scores, b_scores)
        y_max = df[score_col].max()
        _annotate_stars(ax, stars, y_max + 0.03)

    ax.scatter([], [], color="black", marker="$*$", s=100, label="p < 0.05")
    ax.scatter([], [], color="black", marker="$**$", s=100, label="p < 0.01")
    ax.scatter([], [], color="black", marker="$***$", s=100, label="p < 0.001")
    ax.legend(fontsize=8)
    return ax


def plot_feature_vs_feature(
    df, feature_a, feature_b, x="weight_metric", classifier=None, exp="normal", ax=None
):
    """Violin comparing two feature sets (e.g. 'J_i' vs 'FC_SC') across weight metrics."""
    subset = df[df["feature"].isin([feature_a, feature_b]) & (df["exp"] == exp)]
    if classifier is not None:
        subset = subset[subset["classifier"] == classifier]
    title = f"{classifier or ''} classifier: {feature_a} vs {feature_b} ({exp})".strip()
    return plot_score_violin(subset, x=x, hue="feature", hue_order=[feature_a, feature_b], title=title, ax=ax)


def summary_table(df, index=("classifier", "exp", "feature", "weight_metric")):
    """
    Mean +/- std accuracy per group, plus the best weight_metric per
    (classifier, exp, feature). Returns a plain DataFrame — call
    .style.highlight_max(...) yourself for notebook display.
    """
    index = list(index)
    grouped = df.groupby(index)["score"]
    table = grouped.agg(mean="mean", std="std", median="median", min="min", max="max").reset_index()
    table["mean ± std"] = table.apply(lambda r: f"{r['mean']:.3f} ± {r['std']:.3f}", axis=1)
    return table


def save_table(table, path, **to_csv_kwargs):
    table.to_csv(path, index=False, **to_csv_kwargs)
    return path


def plot_J_i_difference(J_i_ctrl, J_i_schz, region_labels=None, title=None, ax=None):
    """
    Mean(schz) - mean(ctrl) J_i per region, with SEM error bars and
    per-region significance stars (Wilcoxon rank-sum is not paired here
    since group sizes may differ, so Mann-Whitney U is used instead).
    """
    from scipy.stats import mannwhitneyu

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))

    n_regions = J_i_ctrl.shape[1]
    diff = J_i_schz.mean(axis=0) - J_i_ctrl.mean(axis=0)
    sem = np.sqrt(J_i_ctrl.var(axis=0) / len(J_i_ctrl) + J_i_schz.var(axis=0) / len(J_i_schz))

    p_values = [
        mannwhitneyu(J_i_ctrl[:, r], J_i_schz[:, r], alternative="two-sided").pvalue
        for r in range(n_regions)
    ]
    _, p_corrected, _, _ = multipletests(p_values, method="fdr_bh")

    x = np.arange(n_regions)
    ax.bar(x, diff, yerr=sem, color=np.where(diff >= 0, "tab:red", "tab:blue"), alpha=0.8)
    ax.axhline(0, color="black", linewidth=1)

    y_max = np.abs(diff).max() + np.abs(sem).max()
    for r in range(n_regions):
        stars = p_to_stars(p_corrected[r])
        if stars:
            ax.text(r, y_max * 1.05, stars, ha="center", va="bottom", fontsize=8, rotation=90)

    ax.set_xlabel("Region index" if region_labels is None else "Region")
    ax.set_ylabel("J_i(schz) - J_i(ctrl)")
    if region_labels is not None:
        ax.set_xticks(x)
        ax.set_xticklabels(region_labels, rotation=90, fontsize=6)
    if title:
        ax.set_title(title)
    return ax, p_corrected


def plot_shuffle_comparison(
    df, feature, weight_metric, classifier, baseline_exp="normal",
    other_exps=("shuffle_region", "shuffle_subj"), ax=None,
):
    """
    Test-score violin for `baseline_exp` vs each of `other_exps`, split by
    exp, for one (feature, weight_metric, classifier) combination.
    """
    subset = df[
        (df["feature"] == feature)
        & (df["weight_metric"] == weight_metric)
        & (df["classifier"] == classifier)
        & (df["exp"].isin([baseline_exp, *other_exps]))
    ]
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    sns.violinplot(data=subset, x="exp", y="score", order=[baseline_exp, *other_exps], inner="quart", ax=ax)
    ax.set_ylim(0, 1.0)
    ax.axhline(CHANCE_LEVEL, color="red", linestyle="--", linewidth=1)
    ax.axhline(BENCHMARK, color="black", linestyle="--", linewidth=1)
    ax.set_title(f"{classifier} classifier: {feature} ({weight_metric})")

    baseline_scores = subset[subset["exp"] == baseline_exp].sort_values("split")["score"].to_numpy()
    y_max = subset["score"].max()
    for i, exp in enumerate(other_exps, start=1):
        exp_scores = subset[subset["exp"] == exp].sort_values("split")["score"].to_numpy()
        if len(exp_scores) == len(baseline_scores) and len(exp_scores) > 0:
            _, p = wilcoxon(baseline_scores, exp_scores)
            stars = p_to_stars(p)
            if stars:
                ax.text(i, y_max + 0.03, stars, ha="center", va="bottom", fontsize=14)
    return ax


def plot_graph_shuffle_examples(graphs, weight_type, sparsity, n=10):
    """Grid of example Erdos-Renyi surrogate graphs for one (weight_type, sparsity)."""
    n = min(n, len(graphs))
    n_cols = 5
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for i in range(n):
        axes[i].imshow(graphs[i], cmap="viridis", aspect="equal")
        axes[i].set_title(f"graph {i}", fontsize=10)
        axes[i].axis("off")
    for i in range(n, len(axes)):
        axes[i].axis("off")

    fig.suptitle(f"Erdos-Renyi surrogates — {weight_type}, sparsity={sparsity}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_graph_shuffle_fit_quality(df, weight_type=None, ax=None):
    """Boxplot of final FC correlation across graph_idx, one box per sparsity."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    sparsities = sorted(df["sparsity"].unique())
    data = [df[df["sparsity"] == s]["fc_corr"].to_numpy() for s in sparsities]

    ax.boxplot(data, labels=[str(s) for s in sparsities])
    ax.set_xlabel("Sparsity")
    ax.set_ylabel("Final FC correlation")
    title = "Graph-shuffle fit quality vs sparsity"
    if weight_type:
        title += f" ({weight_type})"
    ax.set_title(title)
    ax.grid(alpha=0.3)
    return ax
