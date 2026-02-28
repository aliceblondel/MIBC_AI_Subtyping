import torch
import numpy as np
import pandas as pd
import lightning as L
import torch.nn.functional as F
import torch.nn as nn
from loguru import logger
from scipy.stats import pearsonr
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, balanced_accuracy_score, roc_auc_score
)

from src.model.mil import LinearBatchNorm

class LitModel(L.LightningModule):
    """PyTorch Lightning wrapper for BulkMIL training (gene expression + subtype classification)."""

    def __init__(
            self,
            model: nn.Module,
            ensembl_gene_ids: list[str],
            label_names: list[str] = ["Ba.Sq", "LumU", "Stroma.rich", "LumP", "LumNS"],
            lr: float = 1e-4,
            weight_decay: float = 1e-3,
            gene_exp_loss_factor: float = 0.2,
            classif_loss_factor: float = 1,
        ):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=['model'])

        self.lr = lr
        self.weight_decay = weight_decay
        self.gene_exp_loss_factor = gene_exp_loss_factor
        self.classif_loss_factor = classif_loss_factor
        self.ensembl_gene_ids = ensembl_gene_ids
        self.label_names = label_names
        self.test_target_expr = []
        self.test_pred_expr = []
        self.test_labels = []
        self.test_preds = []
        self.test_probas = []
        self.test_ids = []

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(x)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Compute combined MSE (gene expression) + cross-entropy (subtype) loss."""
        inputs, targets, labels = batch
        gene_exp, preds = self.model(inputs)

        exp_loss = F.mse_loss(gene_exp, targets)
        if self.classif_loss_factor > 0:
            classif_loss = F.cross_entropy(preds, labels)
        else:
            classif_loss = 0
        loss = self.gene_exp_loss_factor * exp_loss + self.classif_loss_factor * classif_loss

        self.log("train_exp_loss", exp_loss, prog_bar=True)
        self.log("train_classif_loss", classif_loss, prog_bar=True)
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        inputs, targets, labels = batch
        gene_exp, preds = self.model(inputs)

        exp_loss = F.mse_loss(gene_exp, targets)
        if self.classif_loss_factor > 0:
            classif_loss = F.cross_entropy(preds, labels)
        else:
            classif_loss = 0
        loss = self.gene_exp_loss_factor * exp_loss + self.classif_loss_factor * classif_loss

        self.log("val_exp_loss", exp_loss, prog_bar=True)
        self.log("val_classif_loss", classif_loss, prog_bar=True)
        self.log("val_loss", loss, prog_bar=True)

        return loss

    def test_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        inputs, xy, targets, labels, ids = batch
        gene_exp, preds = self.model(inputs)
        proba = F.softmax(preds, dim=1)

        if not self.classif_loss_factor > 0:
            logger.debug("Using consensus classifier ...")
            from src.consensus_class import pred_consensus_class

            df_expression = pd.DataFrame(
                gene_exp.T.detach().cpu().numpy(), index=self.ensembl_gene_ids,
            )
            preds = pred_consensus_class(df_expression).values

        self.test_ids.extend(ids)
        if self.classif_loss_factor > 0:
            self.test_probas.append(np.array(proba.cpu()))
            self.test_preds.extend(np.array(preds.cpu().argmax(dim=1)))
            self.test_labels.extend(np.array(labels.cpu()))
        else:
            self.test_preds.extend(np.array(preds))
            self.test_labels.extend(np.array(labels))

        self.test_target_expr.append(targets.cpu().detach().numpy())
        self.test_pred_expr.append(gene_exp.cpu().detach().numpy())

        exp_loss = F.mse_loss(gene_exp, targets)
        if self.classif_loss_factor > 0:
            classif_loss = F.cross_entropy(preds, labels)
        else:
            classif_loss = 0
        loss = self.gene_exp_loss_factor * exp_loss + self.classif_loss_factor * classif_loss

        self.log("test_exp_loss", exp_loss, prog_bar=True)
        if classif_loss > 0:
            self.log("test_classif_loss", classif_loss, prog_bar=True)
        self.log("test_loss", loss, prog_bar=True)

        return loss

    def on_test_epoch_end(self) -> None:

        target_expr = np.concatenate(self.test_target_expr, axis=0)
        pred_expr = np.concatenate(self.test_pred_expr, axis=0)
        corr = pearsonr(target_expr, pred_expr)[0]
        corr_mean = np.array([c for c in corr if not np.isnan(c)]).mean()
        self.log("final_test_corr", corr_mean, prog_bar=True)

        preds = np.array(self.test_preds)
        labels = np.array(self.test_labels)
        self.test_outputs = {
            "ids": self.test_ids,
            "label": labels,
            "pred": preds,
            "target_expr": target_expr,
            "pred_expr": pred_expr,
        }
        acc = accuracy_score(labels, preds)
        ba = balanced_accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="weighted")
        precision = precision_score(labels, preds, average="weighted", zero_division=0)
        recall = recall_score(labels, preds, average="weighted", zero_division=0)

        self.metrics = {
            "final_test_accuracy": acc,
            "final_test_ba": ba,
            "final_test_f1": f1,
            "final_test_precision": precision,
            "final_test_recall": recall,
        }

        if self.classif_loss_factor > 0:
            y_proba = np.concatenate(self.test_probas, axis=0)
            self.test_outputs["proba"] = y_proba
            roc_auc = roc_auc_score(labels, y_proba, average="weighted", multi_class="ovr")
            roc_auc_per_class = roc_auc_score(labels, y_proba, average=None, multi_class="ovr")
            self.metrics["final_test_roc_auc"] = roc_auc
            for i, label in enumerate(sorted(self.label_names)):
                self.metrics[f"final_test_roc_auc_{label}"] = roc_auc_per_class[i]
        self.log_dict(self.metrics, prog_bar=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return optimizer


class TileClassifier(L.LightningModule):
    """
    Tile-level classifier for MIBC / NMIBC / Non-Tumor prediction.
    """
    def __init__(
            self,
            feature_depth: int,
            num_classes: int,
            dropout: float = 0.1,
            classifier_hidden_dims: list[int] = [128],
            lr: float = 1e-4,
            weight_decay: float = 1e-3,
            class_weights: torch.Tensor | None = None,
            tests_on_zone: bool = False,
        ):
        super(TileClassifier, self).__init__()

        self.dropout = dropout
        self.feature_depth = feature_depth
        self.classifier_hidden_dims = classifier_hidden_dims
        self.num_classes = num_classes
        self.classifier = self._build_classifier()

        self.lr = lr
        self.weight_decay = weight_decay
        self.class_weights = class_weights
        self.tests_on_zone = tests_on_zone

    def _build_classifier(self) -> nn.ModuleDict:
        classifier = []
        prev_dim = int(self.feature_depth)
        for hidden_dim in self.classifier_hidden_dims:
            classifier.append(
                LinearBatchNorm(prev_dim, hidden_dim, self.dropout, True))
            prev_dim = hidden_dim
        classifier.append(nn.Linear(self.classifier_hidden_dims[-1], self.num_classes))
        return nn.ModuleDict({"mlp": nn.Sequential(*classifier)})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier["mlp"](x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        proba = F.softmax(logits, dim=1)
        preds = torch.argmax(proba, dim=1)
        return preds

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        inputs, labels, _ = batch
        logits = self.forward(inputs)
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(labels.device)
        loss = F.cross_entropy(logits, labels, weight=self.class_weights)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        inputs, labels, _ = batch
        logits = self.forward(inputs)
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(labels.device)
        loss = F.cross_entropy(logits, labels, weight=self.class_weights)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch: tuple, batch_idx: int) -> torch.Tensor | None:
        if self.tests_on_zone:
            self.zone_test_step(batch, batch_idx)
        else:
            inputs, labels, ids = batch
            logits = self.forward(inputs)
            proba = F.softmax(logits, dim=1)

            self.test_ids.extend(ids)
            self.test_probas.append(np.array(proba.cpu()))
            self.test_preds.extend(np.array(logits.cpu().argmax(dim=1)))
            self.test_labels.extend(np.array(labels.cpu()))

            if self.class_weights is not None:
                self.class_weights = self.class_weights.to(labels.device)
            loss = F.cross_entropy(logits, labels, weight=self.class_weights)
            self.log("test_loss", loss, prog_bar=True)

            return loss

    def on_test_epoch_start(self) -> None:
        self.test_labels = []
        self.test_preds = []
        self.test_probas = []
        self.test_ids = []

    def on_test_epoch_end(self) -> None:

        preds = np.array(self.test_preds)
        labels = np.array(self.test_labels)
        y_proba = np.concatenate(self.test_probas, axis=0)

        self.test_outputs = {
            "ids": self.test_ids,
            "label": labels,
            "pred": preds,
            "proba": y_proba,
        }
        acc = accuracy_score(labels, preds)
        ba = balanced_accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="weighted")
        precision = precision_score(labels, preds, average="weighted", zero_division=0)
        recall = recall_score(labels, preds, average="weighted", zero_division=0)
        roc_auc = roc_auc_score(labels, y_proba, average="weighted", multi_class="ovr")
        self.metrics = {
            "final_test_accuracy": acc,
            "final_test_ba": ba,
            "final_test_f1": f1,
            "final_test_precision": precision,
            "final_test_recall": recall,
            "final_test_roc_auc": roc_auc,
        }
        self.log_dict(self.metrics, prog_bar=True)

    def zone_test_step(self, batch: tuple, batch_idx: int) -> None:
        inputs, _, ids, labels = batch

        proba_list = []
        for x in inputs:
            logits = self.classifier(x)
            proba = F.softmax(logits, dim=1)
            proba = proba.mean(axis=0)
            proba_list.append(proba)
        proba = torch.cat(proba_list, dim=0)

        self.test_ids.extend(ids)
        self.test_probas.append([np.array(proba.cpu())])
        self.test_preds.extend([np.array(proba.cpu().argmax())])
        self.test_labels.extend(np.array(labels.cpu()))

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return optimizer
