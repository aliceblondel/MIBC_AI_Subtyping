import timm
import torch
import numpy as np
import torch.nn as nn


class Hoptimus0Model(nn.Module):
    """Wrapper around H-Optimus-0 (bioptimus/H-optimus-0) for tile encoding."""

    def __init__(self, device: str = "cpu", timm_kwargs: dict = {'init_values': 1e-5, 'dynamic_img_size': False}):
        super(Hoptimus0Model, self).__init__()
        self.device = device
        model = self._load(timm_kwargs)
        model.to(device)
        model.eval()
        self.model = model
    
    def _load(self, timm_kwargs: dict) -> nn.Module:
        return timm.create_model("hf-hub:bioptimus/H-optimus-0", pretrained=True, **timm_kwargs)

    def forward(self, x: torch.Tensor) -> np.ndarray:
        autocast_dtype = torch.float16 if self.device in ('cuda', 'mps') else torch.bfloat16
        with torch.inference_mode():
            with torch.autocast(device_type=self.device, dtype=autocast_dtype):
                embeddings = self.model(x)
        embeddings = embeddings.squeeze().detach().cpu().numpy()
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1,-1)
        return embeddings


class Hoptimus1Model(Hoptimus0Model):
    def _load(self, timm_kwargs: dict) -> nn.Module:
        return timm.create_model("hf-hub:bioptimus/H-optimus-1", pretrained=True, **timm_kwargs)
    