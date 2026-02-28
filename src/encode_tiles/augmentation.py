import numpy as np
import torch
from scipy import linalg
import skimage.color as skimage_color

HEX2RGB = np.array([
    [0.650, 0.704, 0.286], # H
    [0.216, 0.801, 0.558], # E
    [0.0, 0.0, 0.0]
])
HEX2RGB[2, :] = np.cross(HEX2RGB[0, :], HEX2RGB[1, :])
RGB2HEX = linalg.inv(HEX2RGB)

class HES2HEAugmentation:
    """ 
    Deconvolve input HES tile into H, E, and S. 
    Then remove S channel.
    Reapply scaled stains to produced HE tiles. 
    """

    def __init__(self,):
        pass
       

    def __call__(self, hes_img: np.ndarray | torch.Tensor, **kwargs) -> np.ndarray | torch.Tensor:
        """Convert an HES image to HE by zeroing the saffron (S) channel.

        Handles both uint8 and float inputs and numpy arrays or torch tensors.

        Args:
            hes_img: HES image as (H, W, 3) numpy array or (3, H, W) torch tensor.

        Returns:
            HE image in the same format and dtype as the input.
        """
        # Rescale if needed
        if hes_img.dtype == np.uint8:
            hes_img = hes_img / 255.
            rescale = True
        else:
            rescale = False

        if isinstance(hes_img, torch.Tensor):
            hes_img = hes_img.permute(1, 2, 0).cpu().numpy()
            tensor = True
        else:
            tensor = False
        separated_image = skimage_color.separate_stains(hes_img, RGB2HEX)
        separated_image[:, :, 2]=0
        he_img = skimage_color.combine_stains(separated_image, HEX2RGB)

        if rescale:
            he_img = (255 * he_img).astype(np.uint8)
        if tensor:
            he_img = torch.from_numpy(he_img).permute(2, 0, 1).float()

        return he_img
