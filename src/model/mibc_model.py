import torch
import pandas as pd
from pathlib import Path
from loguru import logger
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from src.constants import LABELS
from src.model.mil import BulkMIL
from src.model.model import LitModel, TileClassifier
from src.config import get_model_args, get_mibc_model_args


MIBC_ID = 0
NMIBC_ID = 1
NT_ID = 2

def get_device() -> str:
    """Return the best available device string ('cuda', 'mps', or 'cpu')."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

class MIBCModel():
    """Ensemble inference model for MIBC molecular subtyping from H&E tile embeddings.

    Combines a 10-fold MIBC-Subtyping ensemble (gene expression + subtype prediction)
    with a 10-fold MIBC-Detect ensemble (MIBC / NMIBC / Non-Tumor tile classification).
    Models are downloaded automatically from the HuggingFace Hub.
    """

    def __init__(self, use_learnt_classifier: bool = True):
        """Download and initialise all subtyping and detection models.

        Args:
            use_learnt_classifier: If True, use the trained MLP classifier head.
                If False, fall back to the consensusMIBC R-based classifier.
        """
        super().__init__()

        self.repo_id = "aliceblondel/mibc-ai-subtyping"
        self.use_learnt_classifier = use_learnt_classifier
        self.label_names = sorted(LABELS)

        self.device = get_device()
        logger.info(f"Using device: {self.device}")

        self.models = self.set_models()
        self.models = [model.eval().to(self.device) for model in self.models]
        logger.info(f"✅ Load MIBC-Subtyping model")
       
        self.mibc_models = self.set_mibc_models()
        self.mibc_models = [model.eval().to(self.device) for model in self.mibc_models]
        logger.info(f"✅ Load MIBC-Detect model")

        if self.use_learnt_classifier:
            logger.debug("Using learnt classifier MIBC-Clf")
        else:
            logger.debug("Using consensus classifier")

    def download_models(self) -> tuple[list[str], str]:

        model_paths = [
            hf_hub_download(
                repo_id=self.repo_id, 
                filename=f"MIBCSubtyping_checkpoints/test_{k}.ckpt"
            )
            for k in range(10)
        ]
        config_path = hf_hub_download(
            repo_id=self.repo_id, filename="MIBCSubtyping_checkpoints/config.yaml"
        )
    
        return model_paths, config_path

    def download_mibc_models(self) -> tuple[list[str], str]:
        model_paths = [
            hf_hub_download(
                repo_id=self.repo_id, 
                filename=f"MIBCDetect_checkpoints/test_{k}.ckpt"
            )
            for k in range(10)
        ]
        config_path = hf_hub_download(
            repo_id=self.repo_id, filename="MIBCDetect_checkpoints/config.yaml"
        )
        return model_paths, config_path

    def set_models(self) -> list:
        """Download and instantiate the MIBC-Subtyping ensemble (10 LitModel checkpoints)."""
        model_paths, config_path = self.download_models()
        args = get_model_args(config_path=config_path)
        self.ensembl_gene_ids = args.ensembl_gene_ids
        self.hgnc_symbols = args.hgnc_symbols
        self.num_genes = args.num_genes
        self.num_classes = args.num_classes
        
        models = []
        for path in model_paths:
            # Get model
            model = BulkMIL(
                feature_depth = args.feature_depth,
                num_genes = args.num_genes,
                num_classes = args.num_classes, 
                # Predictors
                predictor_hidden_dims = args.predictor_hidden_dims,
                classifier_hidden_dims = args.classifier_hidden_dims,
                dropout = args.dropout, 
                num_heads = args.num_heads, 
                atn_dim = args.atn_dim,
                classif_on_gene_exp=args.classif_on_gene_exp,
                # Aggregator
                pooling_fct = args.pooling_fct,
                instance_based = args.instance_based,
                # Encoder
                encoder_type = args.encoder_type,
                nb_tiles = args.nb_tiles,
                encoder_num_heads=args.encoder_num_heads,
                encoder_num_layers=args.encoder_num_layers, 
                encoder_ff_dim=args.encoder_ff_dim, 
                encoder_dropout=args.encoder_dropout,
            )

            pl_model = LitModel.load_from_checkpoint(
                path, model=model, ensembl_gene_ids=self.ensembl_gene_ids,
            )
            models.append(pl_model)

        return models

    def set_mibc_models(self) -> list:
        """Download and instantiate the MIBC-Detect ensemble (10 TileClassifier checkpoints)."""
        mibc_model_paths, config_path = self.download_mibc_models()
        mibc_args = get_mibc_model_args(config_path=config_path)

        models = []
        for path in mibc_model_paths:
            pl_model = TileClassifier.load_from_checkpoint(
                path, 
                feature_depth=mibc_args.feature_depth, 
                num_classes=mibc_args.num_classes,
                classifier_hidden_dims = mibc_args.classifier_hidden_dims, 
            )
            models.append(pl_model)

        return models
    
    def predict_nmibc_mibc_nt(self, he_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run tile-level MIBC / NMIBC / Non-Tumor classification with the detection ensemble.

        Args:
            he_emb: Tile embeddings of shape (1, N, F).

        Returns:
            Tuple of (per-tile predicted class ids, per-tile softmax probabilities).
        """
        model_mibc_probas = []
        with torch.no_grad():
            for mibc_model in self.mibc_models:
                model_mibc_proba = F.softmax(mibc_model(he_emb[0]), dim=-1)
                model_mibc_probas.append(model_mibc_proba)
        mibc_probas = torch.stack(model_mibc_probas, dim=0).mean(axis=0)
        mibc_probas = F.softmax(mibc_probas, dim=-1)
        mibc_preds = torch.argmax(mibc_probas, axis=-1).cpu()
        return mibc_preds, mibc_probas
    
    def predict_molecular_subtypes(self, he_emb: torch.Tensor, return_proba: bool = False) -> tuple:
        """Predict slide-level molecular subtype and gene expression with the subtyping ensemble.

        Args:
            he_emb: Tile embeddings of shape (1, N, F).
            return_proba: If True, also return the class probability vector.

        Returns:
            (gene_exp, pred) or (gene_exp, pred, proba) depending on return_proba.
        """
        assert not (return_proba and not self.use_learnt_classifier), \
            "return_proba=True is not supported with consensus classifier"

        model_gene_exps, model_classifs = [], []
        for i, model in enumerate(self.models):
            with torch.no_grad():
                model_gene_exp, model_classif_proba = model.model(he_emb)
                model_classif_proba = F.softmax(model_classif_proba, dim=-1)
                model_gene_exps.append(model_gene_exp[0])
                model_classifs.append(model_classif_proba[0])
        gene_exp = torch.stack(model_gene_exps, dim=0).mean(axis=0).cpu()
        classif_proba = torch.stack(model_classifs, dim=0).mean(axis=0).cpu()

        if self.use_learnt_classifier:
            pred = torch.argmax(classif_proba, axis=-1).cpu()
            pred = self.label_names[pred]
            if return_proba:
                return gene_exp, pred, classif_proba
        else:
            from src.consensus_class import pred_consensus_class
            df_expression = pd.DataFrame(gene_exp.T, index=self.ensembl_gene_ids)
            pred = pred_consensus_class(df_expression).values[0]

        return gene_exp, pred
    
    def slide_predict(self, he_emb: torch.Tensor, nmibc_threshold: float = 0.99, nt_threshold: float = 0.99, return_proba: bool = False) -> tuple:
        """Predict slide-level subtype, filtering non-MIBC tiles before subtyping.

        If more than nmibc_threshold of tiles are classified as NMIBC (or NT),
        returns 'NMIBC' / 'Non-Tumor' directly. Otherwise runs predict_molecular_subtypes
        on the MIBC tiles only.

        Args:
            he_emb: Tile embeddings of shape (1, N, F).
            nmibc_threshold: Fraction of NMIBC tiles above which the slide is called NMIBC.
            nt_threshold: Fraction of NT tiles above which the slide is called Non-Tumor.
            return_proba: If True, also return class probabilities.
        """
        n_tiles = he_emb.shape[1]
        mibc_preds, _ = self.predict_nmibc_mibc_nt(he_emb)
        if (mibc_preds == NMIBC_ID).sum() > nmibc_threshold * n_tiles:
            gene_exp = torch.zeros(self.num_genes, device=he_emb.device)
            if return_proba:
                proba = torch.ones(self.num_classes, device=he_emb.device)
                proba = proba / proba.sum() 
                return gene_exp, "NMIBC", proba
            return gene_exp, "NMIBC"
        
        if (mibc_preds == NT_ID).sum() > nt_threshold * n_tiles:
            gene_exp = torch.zeros(self.num_genes, device=he_emb.device)
            if return_proba:
                proba = torch.ones(self.num_classes, device=he_emb.device)
                proba = proba / proba.sum() 
                return gene_exp, "Non-Tumor", proba
            return gene_exp, "Non-Tumor"
        
        mibc_mask = mibc_preds == MIBC_ID
        filtered_he_emb = he_emb[:,mibc_mask,:]
        
        return self.predict_molecular_subtypes(
                    filtered_he_emb, return_proba=return_proba)
    
    def tile_molecular_subtypes(self, he_emb: torch.Tensor, return_proba: bool = False) -> tuple:
        """Predict per-tile molecular subtype and gene expression.

        Args:
            he_emb: Tile embeddings of shape (1, N, F).
            return_proba: If True, also return per-tile class probabilities.

        Returns:
            (gene_exp, pred_series) or (gene_exp, pred_series, y_proba).
        """
        assert not (return_proba and not self.use_learnt_classifier), \
            "return_proba=True is not supported with consensus classifier"

        # Forward gene expression - per tile
        model_gene_exps, model_classifs = [], []
        for model in self.models:
            with torch.no_grad():
                model_gene_exp, model_classif = model.model.forward_per_tile(he_emb)
                model_gene_exps.append(model_gene_exp[0]) 
                model_classifs.append(model_classif[0])           
        gene_exp = torch.stack(model_gene_exps, dim=0).mean(axis=0).cpu()
        y_proba = torch.stack(model_classifs, dim=0).mean(axis=0).cpu()
        y_proba = F.softmax(y_proba, dim=1)

        if self.use_learnt_classifier:
            pred = torch.argmax(y_proba, axis=-1).cpu()
            pred = pd.Series([self.label_names[p] for p in pred.tolist()])
        else:
            from src.consensus_class import pred_consensus_class
            df_expression = pd.DataFrame(
                gene_exp.T, index=self.ensembl_gene_ids, columns=range(gene_exp.shape[0]))
            pred = pred_consensus_class(df_expression)
            y_proba = None
        if return_proba:
            return gene_exp, pred, y_proba
        else:
            return gene_exp, pred

    def tile_predict(self, he_emb: torch.Tensor, return_proba: bool = False) -> tuple:
        """Per-tile prediction: subtype + gene expression, with NMIBC/NT tiles zeroed out.

        Args:
            he_emb: Tile embeddings of shape (1, N, F).
            return_proba: If True, also return per-tile class probabilities.
        """
        mibc_preds, _ = self.predict_nmibc_mibc_nt(he_emb)
        if return_proba:
            gene_exp, pred, y_proba = self.tile_molecular_subtypes(he_emb, return_proba=True)
        else:
            gene_exp, pred = self.tile_molecular_subtypes(he_emb)
            
        # Remove NT / NMIB
        not_mibc = mibc_preds != MIBC_ID
        gene_exp[not_mibc] = 0
        pred[mibc_preds.detach().cpu().numpy() == NMIBC_ID] = "NMIBC"
        pred[mibc_preds.detach().cpu().numpy() == NT_ID] = "Non-Tumor"

        if return_proba:
            return gene_exp, pred, y_proba
        return gene_exp, pred
    