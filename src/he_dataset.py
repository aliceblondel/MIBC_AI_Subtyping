import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import scipy.sparse as sp

import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors


class HEDataset(Dataset):
    def __init__(self,
            df: pd.DataFrame,
            emb_folder: str | Path,
            patient_id_col: str = "patient_id",
            slide_id_col: str = "slide_id",
            aggregate_by_patient: bool = True,
            localized: bool = False,
            include_info: bool = False,
            tile_size: int = None,
        ):
        """Dataset wrapping pre-computed H&E tile embeddings.

        Args:
            df: Metadata DataFrame containing at least patient_id_col and slide_id_col columns.
            emb_folder: Root folder with 'tiles/' and 'xy/' sub-directories of .npy files.
            patient_id_col: Column name for patient identifiers.
            slide_id_col: Column name for slide identifiers.
            aggregate_by_patient: If True, concatenate all slides for a patient into one item.
            localized: If True, apply LocalizedEmbeddingLayer spatial aggregation.
            include_info: If True, load slide width/height metadata into the info dict.
            tile_size: Tile size (pixels) required when localized=True.
        """
        assert patient_id_col in df.columns, f"patient_id_col '{patient_id_col}' not found in df"
        assert slide_id_col in df.columns, f"slide_id_col '{slide_id_col}' not found in df"

        self.df = df
        self.patient_id_col = patient_id_col
        self.slide_id_col = slide_id_col
        self.aggregate_by_patient = aggregate_by_patient
        self.emb_folder = Path(emb_folder)
        self.info, self.ids = self._build_he_emb(include_info=include_info)

        self.localized = localized
        self.tile_size = tile_size

    def _build_he_emb(self, include_info: bool = False) -> tuple[dict, list]:
        """Index available embeddings and build the patient→slides mapping.

        Returns:
            Tuple of (info dict keyed by id, ordered list of ids).
        """
        info = {}
        self._patient_slides = {}

        if self.aggregate_by_patient:
            all_patient_ids = self.df[self.patient_id_col].unique()
            ids = []
            for patient_id in tqdm(all_patient_ids):
                patient_df = self.df[self.df[self.patient_id_col] == patient_id]
                slide_ids = []
                for slide_id in patient_df[self.slide_id_col].values:
                    if not (self.emb_folder / "tiles" / f"{slide_id}.npy").exists():
                        raise FileNotFoundError(f"Missing embedding file for slide {slide_id}")
                    slide_ids.append(slide_id)
                if slide_ids:
                    self._patient_slides[patient_id] = slide_ids
                    info[patient_id] = {}
                    ids.append(patient_id)
        else:
            ids = []
            for slide_id in tqdm(self.df[self.slide_id_col].values):
                if not (self.emb_folder / "tiles" / f"{slide_id}.npy").exists():
                    raise FileNotFoundError(f"Missing embedding file for slide {slide_id}")
                self._patient_slides[slide_id] = [slide_id]
                info[slide_id] = {}
                if include_info:
                    slide_df = self.df[self.df[self.slide_id_col] == slide_id]
                    info[slide_id]["width"] = slide_df["width"].values[0]
                    info[slide_id]["height"] = slide_df["height"].values[0]
                    info[slide_id]["tile_size_mag0"] = slide_df["tile_size_mag0"].values[0]
                ids.append(slide_id)

        return info, ids

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> tuple:

        id = self.ids[idx]
        slide_ids = self._patient_slides[id]

        slides_emb = [np.load(self.emb_folder / "tiles" / f"{sid}.npy") for sid in slide_ids]
        slides_xy  = [np.load(self.emb_folder / "xy"    / f"{sid}.npy") for sid in slide_ids]
        he_emb_np = np.concatenate(slides_emb, axis=0)
        xy_np     = np.concatenate(slides_xy,  axis=0)

        xy = torch.from_numpy(xy_np).float()
        he_emb = torch.from_numpy(he_emb_np).float()
        info = self.info[id]

        if self.localized:
            tile_size = info["tile_size_mag0"] if self.tile_size is None else self.tile_size
            self.localized_layer = LocalizedEmbeddingLayer(
                tile_size=tile_size,  sigma=tile_size//2, grid_step=2)
            he_emb = self.localized_layer(he_emb, xy_np)

        return he_emb, xy, id, info
    

class LocalizedEmbeddingLayer(nn.Module):
    """
    PyTorch layer that aggregates embeddings based on spatial neighborhoods.
    Given a set of coordinates, it constructs a sparse weight matrix W using
    a Gaussian kernel over neighbors within a radius, then computes:
        H_localized = W @ H
    """
    def __init__(self, tile_size: int = 448, sigma: float = 200, grid_step: int = 2):
        """
        Args:
            tile_size (int): spatial step between tiles/points
            sigma (float): standard deviation of Gaussian kernel
            grid_step (int): multiplier for tile_size to define neighbor radius
        """
        super().__init__()
        self.tile_size = tile_size
        self.sigma = sigma
        self.grid_step = grid_step

    def create_weight_matrix(self, xy: np.ndarray) -> torch.Tensor:
        """
        Build the sparse weight matrix W based on coordinates.

        Args:
            xy (np.ndarray): (N,2) coordinates of all points
        """
        N = xy.shape[0]
        # Compute radius to include immediate neighbors on the grid
        radius = np.ceil(np.sqrt(2 * ((self.grid_step * self.tile_size) ** 2)))

        # Find neighbors within radius
        nbrs = NearestNeighbors(radius=radius, algorithm='auto').fit(xy)
        distances, indices = nbrs.radius_neighbors(xy)

        row_list, col_list, data_list = [], [], []
        for i, (inds, dists) in enumerate(zip(indices, distances)):
            if len(inds) == 0:
                continue
            # Gaussian kernel weights
            w = np.exp(- (dists ** 2) / (2 * self.sigma ** 2))
            w /= w.sum()  # normalize weights per row
            row_list.extend([i] * len(inds))
            col_list.extend(inds)
            data_list.extend(w)

        # Build sparse COO matrix
        W_scipy = sp.coo_matrix((data_list, (row_list, col_list)), shape=(N, N))
        indices_torch = torch.tensor([W_scipy.row, W_scipy.col], dtype=torch.long)
        values_torch = torch.tensor(W_scipy.data, dtype=torch.float32)
        W = torch.sparse_coo_tensor(indices_torch, values_torch, (N, N))

        # Move to GPU if available
        if torch.cuda.is_available():
            W = W.cuda()
        return W

    def forward(self, H: torch.Tensor, xy: np.ndarray = None) -> torch.Tensor:
        """
        Forward pass: compute localized embeddings H_localized = W @ H

        Args:
            H (torch.Tensor): (N,d) embeddings/features
            xy (np.ndarray, optional): (N,2) coordinates; required if W is not yet built

        Returns:
            torch.Tensor: localized embeddings (N,d)
        """
        assert xy is not None, "Coordinates xy must be provided to build W on first forward."
        W = self.create_weight_matrix(xy)

        # Ensure W is on the same device as H
        if W.device != H.device:
            W = W.to(H.device)

        # Sparse matrix multiplication
        H_localized = torch.sparse.mm(W, H)
        return H_localized 