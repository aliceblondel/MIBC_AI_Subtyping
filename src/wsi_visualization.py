from pathlib import Path

import numpy as np
from loguru import logger
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from src.constants import COLORS

def display_map(
    thumbnail: np.ndarray,
    info: dict,
    coords: np.ndarray,
    classif: list[str],
    path: Path | None = None,
    tile_size: int = 1792,
    colors: dict[str, str] = COLORS,
    title: str = "",
) -> None:
    """Overlay a per-tile classification colour map on a WSI thumbnail.

    Saves both a plain colour mask (map.png) and a side-by-side figure
    (thumbnail + colour map). If path is None the figure is shown interactively.

    Args:
        thumbnail: RGB thumbnail of the whole slide image.
        info: Slide metadata dict containing at least 'height' (pixels at level 0).
        coords: (N, 2) array of tile coordinates (row, col) at level 0.
        classif: List of N predicted subtype labels, one per tile.
        path: Optional output path for the figure; the mask is saved alongside it.
        tile_size: Tile size in pixels at level 0.
        colors: Mapping from subtype label to hex/named colour.
        title: Figure suptitle.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,9), constrained_layout=True)
    fig.suptitle(title, fontsize=16, fontweight="bold") 

    im = np.array(thumbnail)
    ax1.imshow(im)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title("Whole Slide Image", fontsize=14, fontweight="bold") 

    alpha = info["height"] / im.shape[0]
    mask = np.zeros((im.shape[0], im.shape[1], 3), dtype=np.float32)
    for c, coord in zip(classif, coords):
        rgb = mcolors.to_rgb(colors.get(c, "white"))
        x, y = int(coord[1]), int(coord[0])
        x_min, x_max = int(x / alpha), int((x + tile_size) / alpha)
        y_min, y_max = int(y / alpha), int((y + tile_size) / alpha)
        mask[x_min: x_max, y_min: y_max, :] = rgb 

    if path is not None:
        mask_uint8 = (mask * 255).astype(np.uint8)
        imageio.imwrite(path.parent / f"{path.stem}_map.png", mask_uint8)

    ax2.imshow(mask)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title(f"Classification", fontsize=14, fontweight="bold") 

    if path is not None:
        plt.savefig(path, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def display_gene_heatmap(
    ax: Axes,
    im: np.ndarray,
    info: dict,
    coords: np.ndarray,
    tile_scores: np.ndarray,
    tile_size: int = 1792,
    gene_name: str | None = None,
    set_min_to_white: bool = False,
) -> object:
    """Draw a gene-expression heatmap on an existing matplotlib axis.

    Scores are clipped to [p10, p99] before mapping to the inferno colormap.

    Args:
        ax: Matplotlib axis to draw on.
        im: RGB thumbnail array used to determine the canvas dimensions.
        info: Slide metadata dict containing at least 'height' (pixels at level 0).
        coords: (N, 2) array of tile coordinates (row, col) at level 0.
        tile_scores: (N,) array of per-tile scalar scores to visualise.
        tile_size: Tile size in pixels at level 0.
        gene_name: Gene label used as axis title.
        set_min_to_white: If True, zero values are rendered white instead of dark.

    Returns:
        The AxesImage object produced by imshow.
    """
    scores = np.clip(tile_scores, np.percentile(tile_scores, 10), np.percentile(tile_scores, 99))

    alpha = info["height"] / im.shape[0]
    mask = np.zeros_like(im[:, :, 0]).astype(float)
    
    for s, coord in zip(scores, coords):
        x, y = int(coord[1]), int(coord[0])
        x_min, x_max = int(x / alpha), int((x + tile_size) / alpha)
        y_min, y_max = int(y / alpha), int((y + tile_size) / alpha)
        mask[x_min:x_max, y_min:y_max] = s

    if set_min_to_white:
        cmap = plt.cm.inferno
        cmap.set_under(color="white")
        ims = ax.imshow(mask, cmap=cmap)
        ims.set_clim(np.min(scores), np.max(scores))
    else:
        ims = ax.imshow(mask, cmap="inferno")

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Gene expression: {gene_name}", fontsize=14, fontweight="bold")

    cax = inset_axes(ax, width="5%", height="50%", loc="right", borderpad=-5)
    cbar = plt.colorbar(ims, cax=cax)
    cbar.ax.tick_params(labelsize=12)

    if mask.sum() > 0:
        ims.set_clim(np.min(mask[mask > 0]), np.max(mask[mask > 0]))
    else:
        logger.debug("/!\ Only 0 values to plot")

    return ims

def display_heatmap(
    thumbnail: np.ndarray,
    info: dict,
    coords: np.ndarray,
    tile_scores: np.ndarray,
    gene_name: str | None = None,
    path: Path | None = None,
    tile_size: int = 1792,
    set_min_to_white: bool = False,
) -> None:
    """Save (or display) a side-by-side WSI thumbnail + gene-expression heatmap figure.

    Args:
        thumbnail: RGB thumbnail of the whole slide image.
        info: Slide metadata dict containing at least 'height' (pixels at level 0).
        coords: (N, 2) array of tile coordinates (row, col) at level 0.
        tile_scores: (N,) array of per-tile scalar scores to visualise.
        gene_name: Gene label shown as axis title on the heatmap panel.
        path: Optional output path; if None the figure is shown interactively.
        tile_size: Tile size in pixels at level 0.
        set_min_to_white: Passed to display_gene_heatmap.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10), constrained_layout=True)

    im = np.array(thumbnail)
    ax1.imshow(im)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title("Whole Slide Image", fontsize=14, fontweight="bold") 

    display_gene_heatmap(ax2, im, info, coords, tile_scores, tile_size=tile_size, gene_name=gene_name, set_min_to_white=set_min_to_white)

    if path is not None:
        plt.savefig(path, bbox_inches='tight')
    else:
        plt.show()
    plt.close()
