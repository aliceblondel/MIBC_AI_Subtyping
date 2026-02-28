import cv2
import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from argparse import Namespace
from torch.utils.data import DataLoader

from src.model import MIBCModel
from src.he_dataset import HEDataset
from src.config import get_maps_args, save_args
from src.wsi_visualization import display_heatmap, display_map


def _resolve_gene_indices(display_gene_exp: list[str] | None, model: MIBCModel) -> list[int] | None:
    """
    Resolve which gene indices to display from the display_gene_exp config value.

    Accepted values (from YAML):
      - None                    → no individual gene maps
      - ['none'] / ['0']        → no individual gene maps
      - ['all'] / ['1']         → all genes
      - ['TP53', 'FGFR3', ...]  → specific genes (HGNC symbols or Ensembl IDs)

    Genes are validated against model.hgnc_symbols and model.ensembl_gene_ids.
    Unrecognised genes are skipped with a warning.
    Returns a list of integer indices, or None if nothing to display.
    """
    if display_gene_exp is None:
        return None
    if len(display_gene_exp) == 1 and display_gene_exp[0].lower() in ('0', 'none', 'false'):
        return None
    ensembl_ids = list(model.ensembl_gene_ids)
    hgnc_syms = list(model.hgnc_symbols)
    if len(display_gene_exp) == 1 and display_gene_exp[0].lower() in ('all', '1', 'true'):
        return list(range(len(ensembl_ids)))
    indices = []
    for gene in display_gene_exp:
        if gene in ensembl_ids:
            indices.append(ensembl_ids.index(gene))
        elif gene in hgnc_syms:
            indices.append(hgnc_syms.index(gene))
        else:
            logger.warning(f"Gene '{gene}' not found in model genes (HGNC or Ensembl), skipping.")
    return indices if indices else None


def save_maps(args_maps: Namespace) -> None:
    """Generate and save tile-level classification and gene-expression maps for each slide.

    For each slide in the CSV, runs per-tile inference with MIBCModel, saves:
    - predicted subtype CSV and expression AnnData per slide,
    - a summed gene-expression heatmap overlaid on the WSI thumbnail,
    - individual gene heatmaps (if display_gene_exp is set),
    - a classification colour map overlaid on the WSI thumbnail.

    Args:
        args_maps: Parsed namespace from get_maps_args().
    """
    # Save paths
    save_path = Path(args_maps.save_path)
    save_path.mkdir(exist_ok=True, parents=True)

    savemaps_path = Path(args_maps.save_path) / "maps"
    savemaps_path.mkdir(exist_ok=True, parents=True)

    # Save Path
    savepred_path = Path(savemaps_path) / "predictions"
    savepred_path.mkdir(exist_ok=True, parents=True)

    # Save Path
    savefigure_path = Path(savemaps_path) / "figures"
    savefigure_path.mkdir(exist_ok=True, parents=True)

    # Get Data
    slide_df = pd.read_csv(args_maps.slide_csv)
    dataset = HEDataset(
        slide_df, 
        patient_id_col=args_maps.patient_id_col,
        slide_id_col=args_maps.slide_id_col,
        aggregate_by_patient=False,
        emb_folder=args_maps.slide_emb_folder, 
        localized=args_maps.localized,
        include_info=True,
    )
    logger.info(f"Dataset loaded: {len(dataset)} slides")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Get Model
    model = MIBCModel(use_learnt_classifier=args_maps.use_learnt_classifier)

    # Resolve gene indices once (before the loop)
    gene_indices = _resolve_gene_indices(args_maps.display_gene_exp, model)

    # Predict Slide by Slide
    logger.info("Running inference...")
    for he_emb, xy, slide_id, info in tqdm(dataloader):
        
        slide_id = slide_id[0]
        he_emb, xy = he_emb.to(model.device), xy[0].cpu().numpy()

        # Get thumbnail
        thumbnail_path = Path(args_maps.slide_emb_folder) / "thumbnails" / f"{slide_id}.png"
        thumbnail = cv2.imread(str(thumbnail_path))
        thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)

        if args_maps.use_learnt_classifier:
            gene_exp, classif_pred, y_proba = model.tile_predict(he_emb, return_proba=True)
        else:
            gene_exp, classif_pred = model.tile_predict(he_emb)
        gene_exp = np.array(gene_exp)

        # Save Predictions
        df_results = pd.DataFrame({"pred": classif_pred}, columns=["pred"])
        if args_maps.use_learnt_classifier:
            proba_columns = [label + "_proba" for label in sorted(model.label_names)]
            df_results[proba_columns] = np.array(y_proba)
        df_results.to_csv(savepred_path / f"{slide_id}_predicted_subtype.csv")

        adata_pred = sc.AnnData(
            X=gene_exp,
            obs=pd.DataFrame(index=range(gene_exp.shape[0])),
            var=pd.DataFrame(
                {"hgnc_symbol": model.hgnc_symbols},
                index=model.ensembl_gene_ids,
            ),
        )
        adata_pred.obs = df_results
        saveadata_path = savepred_path / f"{slide_id}_predicted_expression.h5ad"
        adata_pred.write(saveadata_path)
        logger.info(f"Predicted expression saved to {saveadata_path}")

        # Save Maps
        tile_size = info["tile_size_mag0"]
        display_heatmap(
            thumbnail, info, xy, gene_exp.sum(axis=1),
            gene_name="ALL", tile_size=tile_size,
            path=savefigure_path / f"{slide_id}_gene_expression.png"
        )
        if gene_indices is not None:
            slidefigure_path = savefigure_path / f"{slide_id}_gene_expression"
            slidefigure_path.mkdir(exist_ok=True)
            for gene_idx in gene_indices:
                gene_name = adata_pred.var.iloc[gene_idx]["hgnc_symbol"]
                display_heatmap(
                    thumbnail, info, xy, gene_exp[:, gene_idx],
                    gene_name=gene_name, tile_size=tile_size,
                    path=slidefigure_path / f"{gene_name}.png"
                )

        # Classification map
        display_map(
            thumbnail, info, xy, list(classif_pred), tile_size=tile_size,
            path=savefigure_path / f"{slide_id}_classification.png"
        )



if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True,
                        help='Path to maps YAML config file')
    config_path = parser.parse_args().config_path

    args_maps = get_maps_args(config_path=config_path)
    Path(args_maps.save_path).mkdir(exist_ok=True, parents=True)
    save_args(args_maps, Path(args_maps.save_path) / "maps_config.yaml")
    save_maps(args_maps)
