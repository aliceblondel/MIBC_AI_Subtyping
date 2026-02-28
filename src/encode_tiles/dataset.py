"""
Inspired by  https://github.com/trislaz/Democratizing_WSI.
"""
import json
import cv2
import itertools
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import openslide
from PIL import Image
from PIL import ImageDraw
from loguru import logger
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import closing, opening, square
from torch.utils.data import Dataset
from torchvision import transforms

from src.encode_tiles.augmentation import HES2HEAugmentation

DS_PER_LEVEL = {'.svs': 4, '.ndpi':2, '.tiff':4, '.tif':2}
HOPTIMUS_MEAN = [0.707223, 0.578729, 0.703617]
HOPTIMUS_STD = [0.211883, 0.230117, 0.177517]

class WSITileDataset(Dataset):
    """PyTorch Dataset that tiles whole-slide images and returns raw image patches.

    Supports automatic tissue masking (Otsu), optional zone annotations (GeoJSON),
    pre-computed segmentation CSVs, and HES→HE stain augmentation.
    """

    ds_per_level = DS_PER_LEVEL

    def __init__(
            self,
            slide_dict,
            magnification_tile=10,
            final_tile_size=224,
            resize = None,
            max_tiles_per_slide=None,
            save_tile_image_folder = None,
            save_df_path = None,
            save_masked_thumbnail_folder = None,
            save_thumbnail_folder = None,
            mask_tolerance = 0.9,
            zone_mask_tolerance=0.9,
            mag_level0=None,
            transform=None,
            hes2he=False,
            random_state = 0,
            segmentation_csv_path = None,
        ):

        np.random.seed(random_state)
        if transform is None:
            transform = self.get_default_transform(hes2he)
            logger.info(f"Transform: {'HES→HE augmentation enabled' if hes2he else 'no stain augmentation'}")
        self.transform = transform
        self.final_tile_size = final_tile_size
        self.max_tiles_per_slide = max_tiles_per_slide
        self.magnification_tile = magnification_tile
        self.resize = resize
        self.mask_level = -1
        self.mask_tolerance = mask_tolerance
        self.zone_mask_tolerance = zone_mask_tolerance
        self.mag_level0 = mag_level0
        
        # Load slides
        self.slides = {}
        for slide_id, slide_info in slide_dict.items():
            logger.info(f"Opening slide {slide_id}: {slide_info['slide_path']}")
            slide = openslide.open_slide(slide_info["slide_path"])
            logger.info(f"  Dimensions (W x H): {slide.dimensions[0]} x {slide.dimensions[1]}, levels: {slide.level_count}")
            self.slides[slide_id] = slide
        
        # Save folders
        self.save_tile_image_folder = save_tile_image_folder
        if save_tile_image_folder is not None:
            self.save_tile_image_folder = Path(save_tile_image_folder)
            self.save_tile_image_folder.mkdir(exist_ok=True, parents=True)

        self.save_masked_thumbnail_folder = save_masked_thumbnail_folder
        if save_masked_thumbnail_folder is not None:
            self.save_masked_thumbnail_folder = Path(save_masked_thumbnail_folder)
            self.save_masked_thumbnail_folder.mkdir(exist_ok=True, parents=True)

        self.save_thumbnail_folder = save_thumbnail_folder
        if save_thumbnail_folder is not None:
            self.save_thumbnail_folder = Path(save_thumbnail_folder)
            self.save_thumbnail_folder.mkdir(exist_ok=True, parents=True)


        # Tile df
        if segmentation_csv_path is not None:
            logger.info(f"Loading segmentation from CSV: {segmentation_csv_path}")
            self.tile_df = pd.read_csv(segmentation_csv_path)
            self.slide_metadata = {}
            logger.info(f"Loaded {len(self.tile_df)} tiles from CSV")
        else:
            logger.info("Computing segmentation ...")
            self.tile_df, self.slide_metadata = self._prepare_tile_coords(slide_dict)
            logger.info(f"Segmentation done: {len(self.tile_df)} tiles total")
            if save_df_path is not None:
                self.tile_df.to_csv(save_df_path, index=False)
                logger.info(f"Saved tile coords to {save_df_path}")

        if self.save_thumbnail_folder is not None:
            for slide_id, slide_df in self.tile_df.groupby("slide_id"):
                slide = self.slides[slide_id]
                thumbnail = self._get_thumbnail(slide)
                save_thumbnail = self.save_thumbnail_folder / f'{slide_id}.png'
                thumbnail.save(save_thumbnail)
                logger.debug(f"Saved thumbnail: {save_thumbnail}")

        
        if self.save_masked_thumbnail_folder is not None:
            for slide_id, slide_df in self.tile_df.groupby("slide_id"):
                slide = self.slides[slide_id]
                thumbnail = self._get_thumbnail(slide)

                level_tile = int(slide_df["level"].iloc[0])
                read_size = int(slide_df["read_size"].iloc[0])
                actual_ds = slide.level_downsamples[level_tile]
                size_at_0 = round(read_size * actual_ds)
                self._make_masked_thumbnail(slide_df, slide, thumbnail, size_at_0)
                
                save_masked_thumbnail = self.save_masked_thumbnail_folder / f'{slide_id}.png'
                thumbnail.save(save_masked_thumbnail)
         
    def __len__(self) -> int:
        return len(self.tile_df)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:

        tile, x, y = self._load_img(index)
        for t in self.transform.transforms[:-1]:
            tile = t(tile)

        if self.save_tile_image_folder is not None:
            save_path = self.save_tile_image_folder / f'{index}.png'
            if isinstance(tile, torch.Tensor):
                img = tile.permute(1, 2, 0).cpu().numpy()
                img = (img * 255.).astype('uint8')
            else:
                img = tile.copy()
            img = Image.fromarray(img)
            img.save(save_path)

        tile = self.transform.transforms[-1](tile)

        return tile, torch.Tensor(np.array([x, y]))
    
    def _load_img(self, index: int) -> tuple[np.ndarray, int, int]:

        row = self.tile_df.iloc[index]
        slide_id, level, x, y = row[["slide_id", "level", "x", "y"]]
        read_size = int(row["read_size"])
        slide = self.slides[slide_id]

        # Clamp to slide bounds at the chosen level (avoids read_region errors on border tiles)
        actual_ds = slide.level_downsamples[level]
        lw, lh = slide.level_dimensions[level]
        x_at_level = int(x / actual_ds)
        y_at_level = int(y / actual_ds)
        w = min(read_size, lw - x_at_level)
        h = min(read_size, lh - y_at_level)

        img = slide.read_region(location=(int(x), int(y)), level=level, size=(w, h))
        img = np.array(img.convert("RGB"))

        # Resize to final_tile_size when read_size differs (different native magnification)
        # or when the tile was clamped at a border
        if img.shape[0] != self.final_tile_size or img.shape[1] != self.final_tile_size:
            img = cv2.resize(img, dsize=(self.final_tile_size, self.final_tile_size), interpolation=cv2.INTER_CUBIC)

        if self.resize is not None:
            img = cv2.resize(img, dsize=self.resize, interpolation=cv2.INTER_CUBIC)

        return img, x, y


    def _prepare_tile_coords(self, slide_dict: dict) -> tuple[pd.DataFrame, dict]:
        """Compute valid tile coordinates for all slides and return as a DataFrame + slide metadata.

        Returns:
            Tuple of (tile_df, slide_metadata) where slide_metadata is a dict with slide dimensions and tile info.
        """
        tile_coords = []
        slide_metadata = {}

        for slide_id, slide_info in slide_dict.items():
            slide = self.slides[slide_id]
            width, height = slide.dimensions

            contour_tile_coords, size_at_0 = self._prepare_slide_tile_coords(
                slide_id = slide_id,
                slide_info = slide_info,
            )
            slide_metadata[slide_id] = {
                "width": width,
                "height": height,
                "tile_size_mag0": size_at_0,
                "mag_level0": self.mag_level0,
            }
            tile_coords.extend(contour_tile_coords)

        logger.info(f"Total tiles before sampling: {len(tile_coords)}")
        if isinstance(self.max_tiles_per_slide, int):
            indices = np.random.choice(
                len(tile_coords),
                min(self.max_tiles_per_slide, len(tile_coords)),
                replace=False,
            )
            tile_coords = [tile_coords[i] for i in indices]
            logger.info(f"Sampled {len(tile_coords)} tiles (max_tiles={self.max_tiles_per_slide})")
        else:
            np.random.shuffle(tile_coords)

        tile_df = pd.DataFrame(tile_coords)
        tile_df["tile_id"] = tile_df.index
        tile_df["tile_size"] = self.final_tile_size

        return tile_df[["tile_id", "slide_id", "level", "read_size", "x", "y"]], slide_metadata

    def _prepare_slide_tile_coords(self, slide_id: str, slide_info: dict) -> tuple[list[dict], int]:
        """Compute valid tile coordinates for a single slide. Returns (tile_coords, size_at_0)."""
        slide = self.slides[slide_id]
        thumbnail = self._get_thumbnail(slide)
        level_tile, read_size, size_at_0 = self._get_level_info(slide_id, slide_info["slide_path"])

        contour_coordinates = None
        if "annotation_path" in slide_info and "contour_ids" in slide_info:
            contour_coordinates = self._get_contour_coord(
                annotation_path = slide_info["annotation_path"],
                contour_ids = slide_info["contour_ids"],
            )

        dico = self._get_clean_grid(slide, thumbnail, level_tile, size_at_0, contour_coordinates)
        contour_tile_coords = [
            {"slide_id": slide_id, "level": level_tile, "read_size": read_size, "x": coord[1], "y": coord[0]}
            for coord in dico['tile_coords']
        ]
        logger.info(f"  Slide {slide_id}: {len(contour_tile_coords)} valid tiles at level {level_tile}")

        return contour_tile_coords, size_at_0

    def _get_level_info(self, slide_id: str, slide_path: str | Path) -> tuple[int, int, int]:
        """Return (level, read_size, size_at_0) for tile extraction at the target magnification.

        Uses the actual OpenSlide pyramid downsamples rather than assuming a fixed
        ds_per_level per file format.  This correctly handles any combination of
        native vs. target magnification (e.g. 20x from a 40x, 10x or 20x slide).

        Returns
        -------
        level      : pyramid level to call read_region on
        read_size  : width/height in pixels to request from read_region at that level
        size_at_0  : physical footprint in level-0 pixels (used for grid spacing)

        Examples
        --------
        40x native, 20x target → target_ds=2, best_level=0 (actual_ds=1)
            read_size = 224*2/1 = 448  →  resize 448→224  (= real 20x)
        40x native, 10x target → target_ds=4, best_level=1 (actual_ds=4 for SVS)
            read_size = 224*4/4 = 224  →  no resize needed
        10x native, 20x target → target_ds=0.5, best_level=0 (actual_ds=1)
            read_size = 224*0.5/1 = 112 → resize 112→224 (upsampling — warning logged)
        """
        self.ext = Path(slide_path).suffix
        self.mag_level0 = self._get_magnification(slide_path)
        slide = self.slides[slide_id]

        target_ds = self.mag_level0 / self.magnification_tile  # e.g. 40/20 = 2.0

        if target_ds < 1.0:
            logger.warning(
                f"  Slide {slide_id}: requested {self.magnification_tile}x > native "
                f"{self.mag_level0:.1f}x — upsampling will occur."
            )

        # Highest pyramid level whose actual downsample is ≤ target_ds
        # (i.e. at least as detailed as what we want)
        best_level = slide.get_best_level_for_downsample(target_ds)
        actual_ds = slide.level_downsamples[best_level]

        # How many pixels to read at this level to cover the same physical area
        # as final_tile_size pixels at the target magnification
        read_size = round(self.final_tile_size * target_ds / actual_ds)

        # Physical extent in level-0 pixels → used for non-overlapping grid spacing
        size_at_0 = round(self.final_tile_size * target_ds)

        logger.info(
            f"  Slide {slide_id}: native={self.mag_level0:.1f}x  target={self.magnification_tile}x  "
            f"target_ds={target_ds:.2f}  level={best_level}  actual_ds={actual_ds:.2f}  "
            f"read_size={read_size}px → resize to {self.final_tile_size}px"
        )
        return best_level, read_size, size_at_0
    
    def _get_magnification(self, slide_path: str | Path) -> float:
        """Read the scanning magnification from slide metadata (mpp or user override)."""
        slide_object = openslide.OpenSlide(slide_path)
        if self.mag_level0 is not None:
            mag = self.mag_level0
            logger.info(f"  Magnification: {mag}x (user-specified)")
        elif 'openslide.mpp-x' in slide_object.properties:
            mpp_slide = float(slide_object.properties['openslide.mpp-x'])
            mag = 1/mpp_slide * 10
            logger.info(f"  Magnification: {mag:.2f}x (from openslide.mpp-x={mpp_slide})")
        elif 'aperio.MPP' in slide_object.properties:
            mpp_slide = float(slide_object.properties['aperio.MPP'])
            mag = 1/mpp_slide * 10
            logger.info(f"  Magnification: {mag:.2f}x (from aperio.MPP={mpp_slide})")
        else:
            raise ValueError("Please specify mag_level0 value.")
        return mag
    
    def _get_thumbnail(self, slide: "openslide.OpenSlide") -> "Image.Image":
        """Return a downsampled thumbnail at the lowest resolution pyramid level."""
        thumbnail = slide.get_thumbnail(slide.level_dimensions[self.mask_level])
        return thumbnail
 
    def _make_masked_thumbnail(self, tile_df: pd.DataFrame, slide: "openslide.OpenSlide", thumbnail: "Image.Image", size_at_0: int) -> None:
        """Draw red tile outlines on the thumbnail in place."""
        draw = ImageDraw.Draw(thumbnail)
        ds = slide.level_downsamples[self.mask_level]
        for _, tile_row in tile_df.iterrows():
            x, y = tile_row["x"], tile_row["y"]
            scaled_x, scaled_y = x // ds, y // ds
            scaled_w, scaled_h = size_at_0 // ds, size_at_0 // ds
            draw.rectangle(
                (scaled_x, scaled_y, scaled_x + scaled_w, scaled_y + scaled_h),
                outline='red',
            )
            # draw.text((scaled_x, scaled_y), str(index), fill='red')

    def _get_clean_grid(self, slide: "openslide.OpenSlide", thumbnail: "Image.Image", level_tile: int, size_at_0: int, zone_coordinates: list | None = None) -> dict:
        """Build a tissue-filtered tile grid, optionally restricted to annotated zones."""
        slide_height, slide_width = slide.dimensions[1], slide.dimensions[0]
        mask_ds = int(slide.level_downsamples[self.mask_level])
        mask = self._make_auto_mask(thumbnail)
        if zone_coordinates is not None:
            zone_mask = self._make_zone_mask(zone_coordinates, mask.shape, mask_ds)
        else:
            zone_mask = None
        grid = self._grid_blob((0, 0), (slide_height, slide_width), (size_at_0, size_at_0))
        n_grid_total = len(grid)
        grid = [
            (x[0], x[1])
            for x in grid
            if self._check_coordinates(
                x[0], x[1],
                (size_at_0, size_at_0),
                mask,
                mask_ds,
                zone_mask=zone_mask,
            )
        ]
        logger.info(f"  Grid: {n_grid_total} candidates → {len(grid)} kept after tissue masking (tile_size_at_0={size_at_0}px)")
        dico = {'tile_coords': grid, 'mask': mask, 'size_at_0': size_at_0}
        return dico 

    def _make_auto_mask(self, thumbnail: "Image.Image", mean: float = 0.50, std: float = 0.30) -> np.ndarray:
        """
        Create a binary mask from a downsampled version of a WSI. 
        Uses the Otsu algorithm and a morphological opening.
        """
        im = np.array(thumbnail)[:, :, :3]
        im_gray = rgb2gray(im)
        im_gray = self._adjust_mean_std(im_gray, mean, std)
        size = im_gray.shape
        im_gray = im_gray.flatten()
        pixels_int = im_gray[np.logical_and(im_gray > 0.02, im_gray < 0.98)]
        t = threshold_otsu(pixels_int)
        mask = opening(closing((im_gray < t).reshape(size), square(2)), square(2))
        return mask

    def _adjust_mean_std(self, image: np.ndarray, mean: float, std: float) -> np.ndarray:
        """Linearly rescale image to have the given mean and standard deviation."""
        mean_origin = np.mean(image)
        std_origin = np.std(image)
        adjusted_image = std * ((image - mean_origin) / std_origin) + mean        
        return adjusted_image
    
    def _make_zone_mask(self, zone_coordinates: list, mask_shape: tuple, mask_downsample: int) -> np.ndarray:
        """Rasterise polygon contours into a boolean zone mask at thumbnail resolution."""
        mask = np.zeros(mask_shape)
        for coord in zone_coordinates:
            ds_coord = coord//mask_downsample 
            cv2.drawContours(mask, [ds_coord.astype(np.int32)], 0, (1), -1)

        return mask.astype(bool)


    def _grid_blob(self, point_start: tuple, point_end: tuple, patch_size: tuple, stride_ratio: float = 1.0) -> list[tuple]:
        """
        Forms a uniform grid starting from the top left point point_start
        and finishes at point point_end of size patch_size for the given slide.
        Args:
            point_start : Tuple like object of integers of size 2.
            point_end : Tuple like object of integers of size 2. (x,y) = (row, col) = (height, width)
            patch_size : Tuple like object of integers of size 2.
            stride_ratio: Float specifying the proportional stride of the patch. 
                If stride_ratio is set to 1.0, it creates a tiling grid with no overlap between the patches.

        Returns:
            List of coordinates of grid.
        """
        patch_size_0 = patch_size
        size_x, size_y = patch_size_0
        size_x, size_y = int(size_x*stride_ratio), int(size_y*stride_ratio)
        list_col = range(point_start[1], point_end[1], size_x)
        list_row = range(point_start[0], point_end[0], size_y)
        return list(itertools.product(list_row, list_col))

    def _check_coordinates(self, row: int, col: int, patch_size: tuple, mask: np.ndarray, mask_downsample: int, zone_mask: np.ndarray | None = None) -> bool:
        """
        Checks if the patch at coordinates x, y in res 0 is valid.
        Args:
            row : Integer. row coordinate of the patch.
            col : Integer. col coordinate of the patch.
            patch_size : Tuple of integers. Size of the patch.
            mask : Numpy array. Mask of the slide.
            mask_downsample : Integer. Resolution of the mask.
        Returns:
            Boolean. True if the patch is valid, False otherwise.
        """
        col_0, row_0 = col, row
        col_1, row_1 = col + patch_size[0], row + patch_size[1]
        # Convert coordinates to mask_downsample resolution
        col_0, row_0 = col_0 // mask_downsample, row_0 // mask_downsample
        col_1, row_1 = col_1 // mask_downsample, row_1 // mask_downsample
        if col_0 < 0 or row_0 < 0 or row_1 > mask.shape[0] or col_1 > mask.shape[1]:
            return False
        
        # Check if it in a selected zone
        if zone_mask is not None:
            zone_mask_path = zone_mask[row_0:row_1, col_0:col_1]
            zone_condition = (zone_mask_path.sum() > self.zone_mask_tolerance * np.ones(zone_mask_path.shape).sum())
            if not zone_condition:
                return False
            # return True
        
        # Check if it is part of tissue
        mask_patch = mask[row_0:row_1, col_0:col_1]
        mask_condition = (mask_patch.sum() > self.mask_tolerance * np.ones(mask_patch.shape).sum())
        if not mask_condition:
            return False
            
        return True


    def _get_contour_coord(self, annotation_path: str | Path | None, contour_ids: list | None) -> list[np.ndarray] | None:
        """Load polygon coordinates for the requested contour IDs from a GeoJSON annotation file."""
        if contour_ids is None or annotation_path is None:
            return None
        
        with open(annotation_path) as f:
            dict_annotations = json.load(f)

        contour_coordinates= []
        for dict_contour in dict_annotations["features"]:
            coord = dict_contour['geometry']['coordinates']
            assert dict_contour['geometry']['type']=="Polygon", logger.info("Wrong annotation type", dict_contour['geometry']['type'])
            assert len(coord) == 1, logger.info("Multi Coordinates")
            if dict_contour['id'] in contour_ids:
                contour_coordinates.append(np.array(coord[0]))
        
        assert len(contour_ids) == len(contour_coordinates), logger.info(
            f"The number of coords found ({len(contour_coordinates)}) do not match the number of coords  ({len(contour_ids)})"
        )

        return contour_coordinates
        
    @staticmethod
    def get_default_transform(hes2he: bool = True) -> "transforms.Compose":
        """Return the default preprocessing pipeline (ToTensor + optional HES→HE + H-Optimus normalisation)."""
        if hes2he:
            return transforms.Compose([
                transforms.ToTensor(),
                HES2HEAugmentation(),
                transforms.Normalize(mean=HOPTIMUS_MEAN, std=HOPTIMUS_STD),
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=HOPTIMUS_MEAN, std=HOPTIMUS_STD),
            ])


def st_collate_fn(samples: list) -> tuple[torch.Tensor, torch.Tensor]:
    imgs = torch.concat([sample[0].unsqueeze(0) for sample in samples])
    ids = torch.concat([sample[1].unsqueeze(0) for sample in samples])
    return imgs, ids
