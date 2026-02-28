import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from torch.utils.data import DataLoader


from argparse import Namespace

from src.he_dataset import HEDataset
from src.model import MIBCModel
from src.config import get_predict_args, save_args
from src.metrics import compute_metrics

def predict(predict_args: Namespace) -> None:
    """Run slide-level (or patient-level) inference and optionally compute evaluation metrics.

    Loads H&E tile embeddings, runs the MIBCModel ensemble, saves predicted subtypes
    and predicted gene expression to disk. If ground-truth AnnData is provided and
    compute_metrics is enabled, also computes classification and gene-correlation metrics.

    Args:
        predict_args: Parsed namespace from get_predict_args().
    """
    assert not (predict_args.compute_metrics and predict_args.adata_gt_path is None), (
        "compute_metrics is set to True but adata_gt_path is not provided. "
        "Please set adata_gt_path in the config."
    )

    # Save paths
    save_path = Path(predict_args.save_path)
    save_path.mkdir(exist_ok=True, parents=True)

    savepred_path = Path(predict_args.save_path) / "predictions"
    savepred_path.mkdir(exist_ok=True, parents=True)

    # Get Data
    df = pd.read_csv(predict_args.csv_path)
    dataset = HEDataset(
        df=df,
        patient_id_col=predict_args.patient_id_col,
        slide_id_col=predict_args.slide_id_col,
        aggregate_by_patient=predict_args.aggregate_by_patient,
        emb_folder=predict_args.emb_folder,
    )
    if predict_args.aggregate_by_patient:
        logger.info(f"Dataset loaded: {len(dataset)} patients")
    else:
        logger.info(f"Dataset loaded: {len(dataset)} slides")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Get Model
    model = MIBCModel(use_learnt_classifier=predict_args.use_learnt_classifier)

    # Predict Patient by Patient
    logger.info("Running inference...")
    ids, all_preds, all_proba, all_pred_exp = [], [], [], []
    for he_emb, _, group_id, _ in tqdm(dataloader):

        he_emb = he_emb.to(model.device)
        if predict_args.use_mibc_detect:
            predict_fct = model.slide_predict
        else:
            predict_fct = model.predict_molecular_subtypes
        
        if predict_args.use_learnt_classifier:
            gene_exp, pred, proba = predict_fct(he_emb, return_proba=True)
            all_proba.append(proba.cpu())
        else:
            gene_exp, pred = predict_fct(he_emb, return_proba=False)

        ids.append(group_id[0])
        all_preds.append(pred)
        all_pred_exp.append(gene_exp.cpu())

    # Save results
    df_results = pd.DataFrame({"pred": all_preds}, index=ids)
    if predict_args.use_learnt_classifier:
        proba_columns = [label + "_proba" for label in sorted(model.label_names)]
        df_results[proba_columns] = np.array(all_proba)
    df_results.to_csv(savepred_path / "predicted_subtype.csv")
    logger.info(f"Predictions saved to {savepred_path / 'predicted_subtype.csv'}")

    adata_pred = sc.AnnData(
        X=np.array(all_pred_exp),
        obs=pd.DataFrame(index=ids),
        var=pd.DataFrame(
            {"hgnc_symbol": model.hgnc_symbols},
            index=model.ensembl_gene_ids,
        ),
    )
    adata_pred.obs = df_results
    adata_pred.write(savepred_path / "predicted_expression.h5ad")
    logger.info(f"Predicted expression saved to {savepred_path / 'predicted_expression.h5ad'}")

    if (predict_args.adata_gt_path is not None) and (predict_args.compute_metrics):
        adata_gt = sc.read_h5ad(predict_args.adata_gt_path)
        res_df = pd.read_csv(savepred_path / "predicted_subtype.csv", index_col = 0)
        adata_pred = sc.read_h5ad(savepred_path / "predicted_expression.h5ad")

        # Validate that GT slides cover all predicted slides
        predicted_ids = set(res_df.index)
        gt_ids = set(adata_gt.obs_names)
        missing = predicted_ids - gt_ids
        assert len(missing) == 0, (
            f"{len(missing)} predicted id(s) not found in adata_gt: {sorted(missing)}. "
            f"Make sure adata_gt.obs_names match the '{predict_args.patient_id_col}' column values."
        )

        save_metric_path = Path(savepred_path.parent / "metrics")
        compute_metrics(
            res_df,
            adata_pred = adata_pred,
            adata_gt = adata_gt,
            save_path = save_metric_path,
        )


if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True,
                        help='Path to evaluation YAML config file')
    config_path = parser.parse_args().config_path

    predict_args = get_predict_args(config_path=config_path)
    Path(predict_args.save_path).mkdir(exist_ok=True, parents=True)
    save_args(predict_args, Path(predict_args.save_path) / "predict_config.yaml")
    predict(predict_args)
