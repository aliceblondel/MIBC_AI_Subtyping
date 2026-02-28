import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri, numpy2ri

import numpy as np
import pandas as pd
from tqdm import tqdm

# Consensus Classification
pandas2ri.activate()
numpy2ri.activate()
devtools = importr('devtools')
consensusMIBC = importr('consensusMIBC')

def get_consensus_class(x: pd.Series, minCor: float = 0.2, gene_id: str = "entrezgene") -> pd.DataFrame:
    """Call the R consensusMIBC::getConsensusClass function for a single sample.

    Args:
        x: Gene expression series indexed by gene identifiers.
        minCor: Minimum correlation threshold for class assignment.
        gene_id: Gene identifier type used as index (e.g. 'entrezgene', 'ensembl_gene_id').

    Returns:
        DataFrame with one row containing the consensus class and per-class scores.
        Returns a row of NaN values if the R call fails.
    """
    columns = ["consensusClass", "cor_pval", "separationLevel", "LumP", "LumNS", "LumU", "Stroma-rich", "Ba/Sq", "NE-like"]
    none_df = pd.DataFrame([[np.nan] * len(columns)], columns=columns)
    try:
        ro.globalenv['x'] = x
        result = ro.r('suppressWarnings(getConsensusClass(x, minCor = {}, gene_id = "{}"))'.format(minCor, gene_id))
        if not isinstance(result, pd.DataFrame):
            result = pandas2ri.rpy2py(result)
    except Exception:
        result = none_df
    return result

def pred_consensus_class(df_expression: pd.DataFrame, gene_id: str = "ensembl_gene_id") -> pd.Series:
    """Predict consensus MIBC subtypes for all columns of a gene expression DataFrame.

    Args:
        df_expression: Genes × samples expression DataFrame (genes as index, samples as columns).
        gene_id: Gene identifier type used as index.

    Returns:
        Series of predicted subtype labels aligned with the DataFrame columns.
        Labels are remapped to the project's canonical names (e.g. 'Ba/Sq' → 'Ba.Sq').
    """
    results = []
    for k in range(df_expression.shape[1]):
        x = df_expression.iloc[:, k]
        result = get_consensus_class(x=x, minCor=0.2, gene_id=gene_id)
        results.append(result)
    classif_df = pd.concat(results, axis=0)
    pred = classif_df["consensusClass"].map({
        "Ba/Sq": "Ba.Sq", 
        "Stroma-rich": "Stroma.rich",
        "LumU": "LumU",
        "LumP": "LumP",
        "LumNS": "LumNS",
        "NE-like": "NE.like",
    })
    return pred
