import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from loguru import logger
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import statsmodels.stats.multitest as smm
from sklearn.metrics import (
    confusion_matrix, accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)

from src.constants import LABELS


def compute_metrics(
    res_df: pd.DataFrame,
    adata_pred,
    adata_gt,
    save_path: Path,
    label_names: list[str] = LABELS,
    alpha_corr: float = 0.05,
) -> None:
    """Compute and save gene correlation and classification metrics.

    Genes present in only one AnnData are excluded with a warning.
    """
    save_path.mkdir(exist_ok=True, parents=True)

    # --- Input validation ---
    # Slides: all predicted slides must be present in GT
    missing_slides = set(res_df.index) - set(adata_gt.obs_names)
    assert len(missing_slides) == 0, (
        f"{len(missing_slides)} predicted slide(s) not found in adata_gt.obs_names: "
        f"{sorted(missing_slides)}"
    )

    # GT structure
    assert "target" in adata_gt.obs.columns, (
        "adata_gt.obs must contain a 'target' column with consensus subtype labels"
    )
    invalid_labels = set(adata_gt.obs["target"]) - set(label_names)
    if len(invalid_labels) > 0:
        logger.warning(
            f"Excluding {len(invalid_labels)} unknown label(s) from GT: {sorted(invalid_labels)}. "
            f"Supported labels: {label_names}"
        )
        valid_mask = adata_gt.obs["target"].isin(label_names)
        adata_gt = adata_gt[valid_mask]
        res_df = res_df[res_df.index.isin(adata_gt.obs_names)]
        adata_pred = adata_pred[adata_pred.obs_names.isin(adata_gt.obs_names)]

    # Gene intersection — sorted list for deterministic, aligned indexing
    common_genes = sorted(adata_pred.var_names.intersection(adata_gt.var_names))
    n_pred, n_gt = len(adata_pred.var_names), len(adata_gt.var_names)

    assert len(common_genes) > 0, (
        f"No common genes between predicted ({n_pred} genes) and GT ({n_gt} genes) AnnData. "
        "Check that both use Ensembl gene IDs as var_names."
    )
    overlap_ratio = len(common_genes) / n_pred
    assert overlap_ratio >= 0.5, (
        f"Gene overlap too low: {len(common_genes)}/{n_pred} predicted genes found in GT "
        f"({overlap_ratio:.1%}). Expected ≥ 50%. "
        "Check that adata_gt uses the same Ensembl gene IDs."
    )

    if len(common_genes) < max(n_pred, n_gt):
        logger.warning(
            f"Gene mismatch: {n_pred} predicted, {n_gt} GT → keeping {len(common_genes)} common genes"
        )
    else:
        logger.info(f"Gene sets match: {len(common_genes)} genes")


    # Gene correlation
    logger.info("Running gene correlation evaluation...")
    pred_obs = adata_pred.obs_names
    gene_names = adata_pred[:, common_genes].var["hgnc_symbol"].tolist()
    corr_df = gene_expression_metrics(
        target = adata_gt[pred_obs, :][:, common_genes].X,
        pred = adata_pred[:, common_genes].X,
        gene_names = gene_names,
        ensemble_gene_ids = common_genes,
        alpha_corr = alpha_corr,
    )
    corr_df.to_csv(save_path / "gene_corr.csv", index=False)
    plot_corr(corr_df, save_path=save_path / "gene_corr.png")

    # Classification
    logger.info("Running classification metrics...")
    if "target" not in res_df.columns:
        res_df = res_df.copy()
        res_df["target"] = adata_gt.obs.loc[res_df.index, "target"].values
    classification_metrics(res_df, save_path, label_names=label_names)


def gene_expression_metrics(
    target: np.ndarray,
    pred: np.ndarray,
    gene_names: list[str],
    ensemble_gene_ids: list[str],
    alpha_corr: float = 0.05,
) -> pd.DataFrame:
    """Compute per-gene Pearson correlation between predicted and ground truth expression.

    Constant genes are excluded. Returns a DataFrame with correlations,
    p-values, and significance flags under Holm-Sidak and Benjamini-Hochberg corrections.
    """
    non_constant = np.var(target, axis=0) > 1e-6
    n_excluded = (~non_constant).sum()
    if n_excluded > 0:
        logger.warning(f"Excluding {n_excluded} constant genes from correlation evaluation")

    logger.info(f"Computing Pearson correlation on {non_constant.sum()} genes ({target.shape[0]} samples)")
    r, p = pearsonr(target[:, non_constant], pred[:, non_constant])

    signif_hs, p_hs, _, _ = smm.multipletests(p, alpha=alpha_corr, method="holm-sidak")
    signif_bh, p_bh, _, _ = smm.multipletests(p, alpha=alpha_corr, method="fdr_bh")
    logger.info(f"Significant genes — Benjamini-Hochberg: {signif_bh.sum()}")

    return pd.DataFrame({
        "ensembl_gene_id": np.array(ensemble_gene_ids)[non_constant],
        "hgnc_symbol": np.array(gene_names)[non_constant],
        "corr": r,
        "p_value": p,
        "signif": p < alpha_corr,
        "p_value_HolmSidak": p_hs,
        "signif_HolmSidak": signif_hs,
        "p_value_BenjaminiHochberg": p_bh,
        "signif_BenjaminiHochberg": signif_bh,
    })


def classification_metrics(
    res_df: pd.DataFrame,
    save_path: Path,
    label_names: list[str] = LABELS,
) -> None:
    """Compute classification metrics and save them alongside a confusion matrix.

    Computes accuracy, balanced accuracy, F1, precision, recall, and (if probability
    columns are present) weighted OVR ROC-AUC. Results are written to metrics.csv.

    Args:
        res_df: DataFrame with 'target' and 'pred' columns, and optional '<label>_proba' columns.
        save_path: Directory in which to write metrics.csv and the confusion matrix PNG.
        label_names: Ordered list of class labels.
    """
    labels, preds = res_df["target"], res_df["pred"]
    conf_matrix = confusion_matrix(labels, preds, labels=label_names)
    plot_conf_matrix(conf_matrix, label_names, save_path=save_path / "test_conf_matrix.png")

    acc = accuracy_score(labels, preds)
    ba = balanced_accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    precision = precision_score(labels, preds, average="weighted", zero_division=0)
    recall = recall_score(labels, preds, average="weighted", zero_division=0)

    test_metrics = {
        "accuracy": acc,
        "ba": ba,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }
    logger.info(f"Accuracy: {acc:.3f} | Balanced accuracy: {ba:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f}")

    # ROC AUC
    columns = [label + "_proba" for label in sorted(label_names)]
    if all(col in res_df.columns for col in columns):
        y_proba = np.array(res_df[columns])
        roc_auc = roc_auc_score(labels, y_proba, average="weighted", multi_class="ovr")
        roc_auc_per_class = roc_auc_score(labels, y_proba, average=None, multi_class="ovr")

        test_metrics["roc_auc"] =  roc_auc
        for i, label in enumerate(sorted(label_names)):
            test_metrics[f"roc_auc_{label}"] = roc_auc_per_class[i]

        logger.info(
            f"ROC AUC (weighted OVR): {roc_auc:.3f} | "
            + " | ".join(
                f"ROC AUC {label}: {roc_auc_per_class[i]:.3f}"
                for i, label in enumerate(sorted(label_names))
            )
        )

    # Save metrics
    metrics_df = pd.DataFrame([test_metrics])
    metrics_df.to_csv(save_path / "metrics.csv", index=False)
    logger.info(f"Metrics saved to {save_path / 'metrics.csv'}")


def plot_conf_matrix(
    conf_matrix: np.ndarray,
    labels: list[str],
    save_path: Path | None = None,
) -> None:
    """Plot raw counts and row-normalized (recall) confusion matrices side by side."""
    heatmap_kwargs = dict(cmap="Blues", cbar=False, xticklabels=labels, yticklabels=labels)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Raw counts
    sns.heatmap(conf_matrix, annot=True, fmt=".0f", ax=axes[0], **heatmap_kwargs)
    axes[0].set_xlabel("Predictions")
    axes[0].set_ylabel("GT Labels")
    axes[0].set_title("Confusion Matrix (counts)")

    # Row-normalized: recall per class
    norm = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)
    sns.heatmap(norm, annot=True, fmt=".2f", ax=axes[1], **heatmap_kwargs)
    axes[1].set_xlabel("Predictions")
    axes[1].set_ylabel("GT Labels")
    axes[1].set_title("Confusion Matrix (recall)")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path)
        logger.info(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()


def plot_corr(
    corr_df: pd.DataFrame,
    method: str | None = "BenjaminiHochberg",
    save_path: Path | None = None,
) -> None:
    """Plot violin plot of per-gene Pearson correlations with significance threshold.

    Args:
        corr_df: Output of gene_correlation_eval, must contain 'corr' and 'signif_<method>' columns.
        method: Multiple testing correction method used to determine significance threshold line.
    """
    if method is not None:
        col_name = "signif_" + method.replace(" ", "").replace("-", "")
        title = f"{method} corrected"
    else:
        col_name = "signif"
        title = "Uncorrected"

    n_signif = corr_df[col_name].sum()
    # Threshold: lowest correlation among significant genes (conservative boundary)
    signif_threshold = corr_df.loc[corr_df[col_name], "corr"].min()

    fig, ax = plt.subplots(figsize=(6, 5))

    # Violin plot with significance threshold
    sns.violinplot(data=corr_df, y="corr", ax=ax, inner="quartile", color="lightblue")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1)
    ax.axhline(y=signif_threshold, color="red", linestyle="--", linewidth=1.5, label=f"{method} (p < 0.05)")
    ax.text(0.35, signif_threshold + 0.02, f"{n_signif} genes (p < 0.05)", color="red", fontsize=10, ha="center")
    ax.text(0.35, signif_threshold - 0.04, f"{len(corr_df) - n_signif} genes (p ≥ 0.05)", color="red", fontsize=10, ha="center")
    ax.set_ylabel("Pearson Correlation")
    ax.set_title(f"Correlation distribution ({title})")
    ax.legend()

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path)
        logger.info(f"Correlation plot saved to {save_path}")
    else:
        plt.show()
