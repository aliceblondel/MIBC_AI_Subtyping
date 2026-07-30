import torch
import pandas as pd
from pathlib import Path
from loguru import logger
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
    
    def _ensemble_mean(self, models: list, fn) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Run fn(model) for every ensemble member and average the outputs.

        fn may return a single tensor or a tuple of tensors; either way, the per-model
        outputs are stacked along a new leading dim and averaged over it.
        """
        with torch.no_grad():
            outputs = [fn(model) for model in models]
        if isinstance(outputs[0], tuple):
            return tuple(
                torch.stack([o[i] for o in outputs], dim=0).mean(dim=0)
                for i in range(len(outputs[0]))
            )
        return torch.stack(outputs, dim=0).mean(dim=0)

    def _get_pred(self, gene_exp: torch.Tensor, proba: torch.Tensor | None = None, tile_level: bool = False):
        """Predict subtype label(s) from proba (learnt classifier) or gene_exp (consensus classifier)."""
        if self.use_learnt_classifier:
            pred_idx = torch.argmax(proba, axis=-1).cpu()
            if tile_level:
                return pd.Series([self.label_names[p] for p in pred_idx.tolist()])
            return self.label_names[pred_idx]

        from src.consensus_class import pred_consensus_class
        logger.debug("Applying consensus classifier ...")
        kw = {"columns": range(gene_exp.shape[0])} if tile_level else {}
        df = pd.DataFrame(gene_exp.T, index=self.ensembl_gene_ids, **kw)
        result = pred_consensus_class(df)
        return result if tile_level else result.values[0]

    def predict_nmibc_mibc_nt(self, he_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run tile-level MIBC / NMIBC / Non-Tumor classification with the detection ensemble.

        Args:
            he_emb: Tile embeddings of shape (1, N, F).

        Returns:
            Tuple of (per-tile predicted class ids, per-tile softmax probabilities).
        """
        mibc_probas = self._ensemble_mean(
            self.mibc_models,
            lambda mibc_model: mibc_model.predict(he_emb[0]),
        )
        mibc_preds = torch.argmax(mibc_probas, axis=-1).cpu()
        return mibc_preds, mibc_probas

    def predict_molecular_subtypes(
        self, he_emb: torch.Tensor, use_tiles: bool = False,
    ) -> tuple:
        """Predict slide-level molecular subtype and gene expression with the subtyping ensemble.

        Args:
            he_emb: Tile embeddings of shape (1, N, F).
            use_tiles: If True, patient-level prediction from per-tile classification instead
                of the attention-pooled slide representation: each tile votes its argmax class
                and the slide probability is the per-class fraction of tiles ("% of tiles",
                no attention).

        Returns:
            (gene_exp, pred, proba) — proba is None when using the consensus classifier.
        """
        def fn(model):
            if use_tiles:
                # Checkpoints are trained with attmil pooling; force plain averaging of the
                # per-tile one-hot votes so the "% of tiles" prediction bypasses attention.
                pooling = model.model.pooling_function
                trained_pooling_fct, pooling.pooling = pooling.pooling, "mean"
                gene_exp, classif_proba = model.predict_tiles(he_emb, one_hot=True)
                pooling.pooling = trained_pooling_fct
            else:
                gene_exp, classif_proba = model.predict(he_emb)
            return gene_exp[0], classif_proba[0]

        gene_exp, classif_proba = self._ensemble_mean(self.models, fn)
        gene_exp, classif_proba = gene_exp.cpu(), classif_proba.cpu()

        pred = self._get_pred(gene_exp, classif_proba)
        proba = classif_proba if self.use_learnt_classifier else None
        return gene_exp, pred, proba

    def slide_predict(
        self, he_emb: torch.Tensor, nmibc_threshold: float = 0.9, nt_threshold: float = 0.9,
        use_tiles: bool = False,
    ) -> tuple:
        """Predict slide-level subtype, filtering non-MIBC tiles before subtyping.

        If more than nmibc_threshold of tiles are classified as NMIBC (or NT),
        returns 'NMIBC' / 'Non-Tumor' directly. Otherwise runs predict_molecular_subtypes
        on the MIBC tiles only.

        Args:
            he_emb: Tile embeddings of shape (1, N, F).
            nmibc_threshold: Fraction of NMIBC tiles above which the slide is called NMIBC.
            nt_threshold: Fraction of NT tiles above which the slide is called Non-Tumor.
            use_tiles: If True, patient-level prediction from per-tile classification instead
                of the attention-pooled slide representation ("% of tiles", no attention).

        Returns:
            (gene_exp, pred, proba) — proba is None when using the consensus classifier.
        """
        n_tiles = he_emb.shape[1]
        mibc_preds, _ = self.predict_nmibc_mibc_nt(he_emb)
        for pred_id, threshold, label in [
            (NMIBC_ID, nmibc_threshold, "NMIBC"),
            (NT_ID,    nt_threshold,    "Non-Tumor"),
        ]:
            if (mibc_preds == pred_id).sum() > threshold * n_tiles:
                gene_exp = torch.zeros(self.num_genes, device=he_emb.device)
                proba = torch.ones(self.num_classes, device=he_emb.device) / self.num_classes \
                    if self.use_learnt_classifier else None
                return gene_exp, label, proba

        mibc_mask = mibc_preds == MIBC_ID
        filtered_he_emb = he_emb[:,mibc_mask,:]

        return self.predict_molecular_subtypes(filtered_he_emb, use_tiles=use_tiles)
    
    def tile_molecular_subtypes(self, he_emb: torch.Tensor) -> tuple:
        """Predict per-tile molecular subtype and gene expression.

        Args:
            he_emb: Tile embeddings of shape (1, N, F).

        Returns:
            (gene_exp, pred_series, y_proba) — y_proba is None when using the consensus classifier.
        """
        def fn(model):
            gene_exp, classif_proba = model.predict_per_tile(he_emb)
            return gene_exp[0], classif_proba[0]

        gene_exp, y_proba = self._ensemble_mean(self.models, fn)
        gene_exp, y_proba = gene_exp.cpu(), y_proba.cpu()

        pred = self._get_pred(gene_exp, y_proba, tile_level=True)
        if not self.use_learnt_classifier:
            y_proba = None
        return gene_exp, pred, y_proba

    def tile_predict(self, he_emb: torch.Tensor) -> tuple:
        """Per-tile prediction: subtype + gene expression, with NMIBC/NT tiles zeroed out.

        Args:
            he_emb: Tile embeddings of shape (1, N, F).

        Returns:
            (gene_exp, pred_series, y_proba) — y_proba is None when using the consensus classifier.
        """
        mibc_preds, _ = self.predict_nmibc_mibc_nt(he_emb)
        gene_exp, pred, y_proba = self.tile_molecular_subtypes(he_emb)

        # Remove NT / NMIB
        not_mibc = mibc_preds != MIBC_ID
        gene_exp[not_mibc] = 0
        pred[mibc_preds.detach().cpu().numpy() == NMIBC_ID] = "NMIBC"
        pred[mibc_preds.detach().cpu().numpy() == NT_ID] = "Non-Tumor"

        return gene_exp, pred, y_proba
    