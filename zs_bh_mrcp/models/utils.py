"""
Copyright (c) Jinho Kim (jinho.kim@fau.de).

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import Optional

import torch

from .. import complex_conj, complex_mul, fft2c, ifft2c


def sens_expand(x: torch.Tensor, sens_maps: torch.Tensor, mask: Optional[torch.Tensor]=None) -> torch.Tensor:
    x = complex_mul(x, sens_maps)
    
    x = fft2c(x)
    
    if mask is not None:
        x = x * mask
        
    return x


def sens_reduce(
    x: torch.Tensor, sens_maps: torch.Tensor, dim=1, keepdim=True, mask: Optional[torch.Tensor]=None
) -> torch.Tensor:
    
    if mask is not None:
        x = x * mask
        
    x = ifft2c(x)
    
    x = complex_mul(x, complex_conj(sens_maps)).sum(dim=dim, keepdim=keepdim)   
    return x


