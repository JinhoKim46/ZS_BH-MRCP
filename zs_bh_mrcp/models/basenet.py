"""
Copyright (c) Jinho Kim (jinho.kim@fau.de).

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import Tuple

import torch
import torch.nn as nn


class _BaseNet(nn.Module):
    """
    Base class for all networks.
    """
    @staticmethod
    def complex_to_chan_dim(x: torch.Tensor) -> torch.Tensor:
        # when x = (b, c, h, w, 2) => (b, 2*c, h, w)
        return x.movedim(-1, 1).flatten(1, 2)  # [b, 2*c, h, w]


    @staticmethod
    def chan_complex_to_last_dim(x: torch.Tensor) -> torch.Tensor:
        target_shape = (x.shape[0], 2, x.shape[1] // 2, *x.shape[2:])  # [b, 2, c, h, w]
        x = x.reshape(target_shape)  # [b, 2, c, h, w]
        x = x.movedim(1, -1)  # [b, c, h, w, 2]
        return x.contiguous()
    
    @staticmethod
    def norm(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = x.flatten(1, -2).mean(dim=1).view((-1,) + (1,) * (x.ndim - 2) + (x.shape[-1],))
        std = x.flatten(1, -2).std(dim=1).view((-1,) + (1,) * (x.ndim - 2) + (x.shape[-1],))
        return (x - mean) / std, mean, std
    
    @staticmethod
    def unnorm(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return x * std + mean
    
    
if __name__ == "__main__":
    #%% Import
    import torch

    from zs_bh_mrcp.models.basenet import _BaseNet

    #%% Test complex_to_chan_dim and chan_complex_to_last_dim for image (b, c, h, w, 2)
    x = torch.randn(1, 1, 10, 10, 2)
    
    y = _BaseNet.complex_to_chan_dim(x)
    x_recon = _BaseNet.chan_complex_to_last_dim(y)
    assert torch.allclose(x, x_recon)
    
    #%% Test complex_to_chan_dim and chan_complex_to_last_dim for kspace (b, c, coil, h, w, 2)
    x = torch.randn(1, 1, 4, 10, 10, 2)
    
    y = _BaseNet.complex_to_chan_dim(x)
    x_recon = _BaseNet.chan_complex_to_last_dim(y, coil=4)
    assert torch.allclose(x, x_recon)
    
# %%
