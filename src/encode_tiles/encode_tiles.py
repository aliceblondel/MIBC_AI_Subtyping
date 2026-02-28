import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from torch.utils.data import DataLoader

from src.encode_tiles.model import Hoptimus1Model
from src.encode_tiles.dataset import st_collate_fn, WSITileDataset


def get_device() -> str:
    """Return the best available device string ('cuda', 'mps', or 'cpu')."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def encode_slide(
        model,
        slide_dict: dict,
        segmentation_csv_path: str | Path | None = None,
        save_df_path: str | Path | None = None,
        save_masked_thumbnail_folder: str | Path | None = None,
        save_thumbnail_folder: str | Path | None = None,
        save_tile_image_folder: str | Path | None = None,
        final_tile_size: int = 224,
        magnification_tile: float = 10,
        max_tiles_per_slide: int | None = None,
        mask_tolerance: float = 0.9,
        hes2he: bool = False,
        num_workers: int = 0,
        batch_size: int = 128,
    ) -> tuple:
    """Encode all tiles of one slide (or one zone) into H-Optimus embeddings.

    Returns:
        Tuple of (embeddings, xys, tile_df, slide_metadata), or all-None
        if no tiles were found for the slide.
    """
    # Get Data
    data = WSITileDataset(
        slide_dict,
        final_tile_size=final_tile_size,
        magnification_tile=magnification_tile,
        max_tiles_per_slide=max_tiles_per_slide,
        mask_tolerance=mask_tolerance,
        segmentation_csv_path=segmentation_csv_path,
        save_df_path=save_df_path,
        save_masked_thumbnail_folder=save_masked_thumbnail_folder,
        save_thumbnail_folder=save_thumbnail_folder,
        save_tile_image_folder=save_tile_image_folder,
        transform=None,
        hes2he=hes2he,
        random_state=0,
    )
    if len(data) == 0:
        logger.warning("No tiles found for this slide, skipping.")
        return None, None, None, None
    batch_size = min(batch_size, len(data))
    logger.info(f"DataLoader: {len(data)} tiles, batch_size={batch_size}, num_workers={num_workers}")
    dataloader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=num_workers > 0 and torch.cuda.is_available(),
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        collate_fn=st_collate_fn,
    )

    # Encode slide
    embeddings = []
    xys = []
    n_batches = len(dataloader)
    for i, batch in enumerate(tqdm(dataloader, desc="Encoding batches")):
        with torch.no_grad():
            im, xy = batch
            im = im.to(model.device)
            xys.append(xy)
            emb = model(im)
            embeddings.append(emb)
    embeddings = np.concatenate(embeddings)
    xys = torch.concatenate(xys)
    logger.info(f"Embeddings shape: {embeddings.shape}")

    tile_df = data.tile_df
    metadata = data.slide_metadata

    del dataloader, data
    del batch

    return embeddings, xys, tile_df, metadata


def read_df(csv_path: str | Path) -> pd.DataFrame:
    """Read a CSV or Excel file into a DataFrame."""
    ext = Path(csv_path).suffix
    if ext.lower() == '.csv':
        return pd.read_csv(csv_path)
    elif ext.lower() == '.xlsx':
        return pd.read_excel(csv_path)
    else:
        raise ValueError("Unsupported file extension: {}".format(ext))


def find_slide_path(slides_folder: str | Path, slide_id: str) -> Path:
    """Find the unique slide file matching slide_id (any extension) in slides_folder."""
    matches = list(Path(slides_folder).glob(f"{slide_id}.*"))
    if len(matches) == 1:
        return matches[0]
    elif len(matches) == 0:
        raise FileNotFoundError(f"No slide found for '{slide_id}' in {slides_folder}")
    else:
        raise ValueError(f"Multiple files found for '{slide_id}': {matches}")


def get_slide_dict(df: pd.DataFrame, id: str, encode_by: str, slide_folder: str | Path, annotation_folder: str | Path | None, slide_id_col: str = "slide_id") -> dict:
    """Build the slide_dict expected by WSITileDataset for a single encode_by unit.

    Supports 'slide_id' (one slide, no annotation) and 'zone_id' (one zone with GeoJSON annotation).
    """
    if encode_by == "zone_id":
        zone_df = df[df[encode_by] == id]
        slide_dict = zone_df[[slide_id_col, "id"]].groupby(slide_id_col)['id'].apply(list).to_dict()
        slide_dict = {
            slide_id: {
                "slide_path": find_slide_path(slide_folder, slide_id),
                "annotation_path": annotation_folder / (slide_id + ".geojson"),
                "contour_ids": contour_ids,
            } for slide_id, contour_ids in slide_dict.items()
        }
        return slide_dict
    elif encode_by == "slide_id":
        return {id:{
                "slide_path": find_slide_path(slide_folder, id),
                "annotation_path": None,
                "contour_ids": None,
        }}
    else:
        raise NotImplementedError


def encode_dataset(
        emb_folder: Path,
        slide_folder: str | Path,
        csv_path: str | Path,
        segmentation_folder: str | Path | None = None,
        final_tile_size: int = 224,
        magnification_tile: float = 10,
        save_images: bool = False,
        mask_tolerance: float = 0.9,
        hes2he: bool = False,
        encode_by: str = "slide_id",
        slide_id_col: str = "slide_id",
        patient_id_col: str = "patient_id",
        annotation_folder: str | Path | None = None,
        num_workers: int = 0,
        batch_size: int = 128,
    ) -> None:
    """Encode all slides (or zones) in a dataset and save tile embeddings to disk.

    Iterates over unique IDs in the ground-truth CSV, encodes each with
    H-Optimus-1, and saves tile embeddings (.npy), xy coordinates (.npy),
    tile metadata CSV, slide thumbnails, and a global info CSV.
    """
    # Paths
    tiles_folder = emb_folder / 'tiles'
    xy_folder = emb_folder / 'xy'

    images_folder = emb_folder / 'images'
    csv_folder = emb_folder / 'csv'
    thumbnail_mask_folder = emb_folder / 'thumbnail_masks'
    thumbnail_folder = emb_folder / 'thumbnails'

    for folder in [emb_folder, emb_folder, tiles_folder, xy_folder, images_folder, csv_folder, thumbnail_folder]:
        folder.mkdir(parents=True, exist_ok=True)

    # Model
    device = get_device()
    logger.info(f"Using device: {device}")
    model = Hoptimus1Model(device = device)
    logger.info(f"✅ Load H-Optimus-1 model")
    
    # Encodes Slides
    df = read_df(csv_path)

    dfs = []
    info_df = pd.DataFrame(columns=["slide_id", "patient_id", "tile_size_mag0", "mag_level0", "n_tiles", "width", "height"])
    ids = df[encode_by].unique()
    logger.info(f"Encoding {len(ids)} {encode_by}(s) → {emb_folder}")
    with torch.no_grad():
        for id in ids:
            slide_dict = get_slide_dict(df, id, encode_by,
                    slide_folder=slide_folder, annotation_folder=annotation_folder, slide_id_col=slide_id_col)
            logger.info(f"Encoding {encode_by}: {id}")

            segmentation_csv_path = None if segmentation_folder is None else Path(segmentation_folder) / f"{id}.csv"
            tile_embeddings, tile_xys, tile_df, metadata = encode_slide(
                model,
                slide_dict=slide_dict,
                segmentation_csv_path=segmentation_csv_path,
                save_df_path=csv_folder / f"{id}.csv",
                save_masked_thumbnail_folder=thumbnail_mask_folder / id,
                save_thumbnail_folder=thumbnail_folder,
                save_tile_image_folder=images_folder / id if save_images else None,
                final_tile_size=final_tile_size,
                magnification_tile=magnification_tile,
                max_tiles_per_slide=None,
                mask_tolerance=mask_tolerance,
                hes2he=hes2he,
                num_workers=num_workers,
                batch_size=batch_size,
            )
            tile_emb_path = tiles_folder / f'{id}.npy'
            tile_xy_path = xy_folder / f'{id}.npy'

            if tile_embeddings is not None and tile_embeddings.size > 0:
                np.save(tile_emb_path, tile_embeddings)
                np.save(tile_xy_path, tile_xys)
                tile_df[encode_by] = id
                dfs.append(tile_df)

                # Get metadata from slide_metadata (one entry per slide in slide_dict)
                for slide_id, slide_metadata in metadata.items():
                    slide_row = df[df[slide_id_col] == slide_id]
                    patient_id = slide_row[patient_id_col].values[0] if patient_id_col in df.columns and len(slide_row) > 0 else None
                    info_df = pd.concat([info_df, pd.DataFrame([{
                            "slide_id": slide_id,
                            "patient_id": patient_id,
                            "tile_size_mag0": slide_metadata.get("tile_size_mag0"),
                            "mag_level0": slide_metadata.get("mag_level0"),
                            "n_tiles": tile_embeddings.shape[0],
                            "width": slide_metadata.get("width"),
                            "height": slide_metadata.get("height"),
                        }])], ignore_index=True)
                logger.info(f"Saved {tile_embeddings.shape[0]} tile embeddings for {encode_by} {id}")
            else:
                logger.warning(f"Empty embeddings for {encode_by} {id}, skipping")


    tiles_df = pd.concat(dfs, ignore_index=True)
    tiles_df.to_csv(emb_folder / "tiles_xy.csv", index=False)
    info_df = info_df.drop_duplicates(subset=["slide_id"], keep="last")
    info_df.to_csv(emb_folder / "slide_info.csv", index=False)
    logger.info(f"Encoding done: {len(info_df)} slides, {int(info_df['n_tiles'].sum())} total tiles")


if __name__ == "__main__":

    import argparse
    from src.config import get_encode_args, save_args

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)
    config_path = parser.parse_args().config_path

    args = get_encode_args(config_path=config_path)

    Path(args.emb_folder).mkdir(exist_ok=True, parents=True)
    save_args(args, Path(args.emb_folder) / "encode_config.yaml")

    encode_dataset(
        emb_folder=Path(args.emb_folder),
        slide_folder=Path(args.slide_folder),
        csv_path=args.csv_path,
        annotation_folder=Path(args.annotation_folder) if args.annotation_folder else None,
        segmentation_folder=args.segmentation_folder,
        final_tile_size=args.final_tile_size,
        magnification_tile=args.magnification_tile,
        save_images=args.save_images,
        mask_tolerance=args.mask_tolerance,
        hes2he=args.hes2he,
        encode_by=args.encode_by,
        slide_id_col=args.slide_id_col,
        patient_id_col=args.patient_id_col,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )
