"""
Copyright (c) Jinho Kim (jinho.kim@fau.de).

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..data.data_consistency import conjgrad
from .resnet import ResNet
from .utils import sens_expand, sens_reduce


class BackboneNet(nn.Module):
    def __init__(
        self,
        num_stages: int = 10,
        num_resblocks: int = 15,
        chans: int = 64,
        cgdc_iter: int = 10,
    ):
        """
        Args:
            num_stages:      Number of stages (i.e., layers) for unrolled network.
            num_resblocks:   Number of residual blocks in each stage.
            chans:           Number of channels in the network.
            cgdc_iter:       Number of conjugate gradient iterations.
        """

        super().__init__()
        self.regularizer = ResNet(in_ch=2, chans=chans, num_of_resblocks=num_resblocks)

        self.dc = conjgrad

        self.num_stages = num_stages
        self.cgdc_iter = cgdc_iter
        self.lam = nn.Parameter(torch.tensor([0.05]))

    def forward(
        self,
        kdata_tau_OwoG: torch.Tensor,
        mask_tau_OwoG: torch.Tensor,
        sens_maps: torch.Tensor,
        convex_hull: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        input_x = sens_reduce(kdata_tau_OwoG, sens_maps)
        x = input_x
        for _ in range(self.num_stages):
            x = self.regularizer(x.float())

            rhs = input_x + self.lam * x
            x = conjgrad(rhs, sens_maps, mask_tau_OwoG, self.lam, self.cgdc_iter)
            
            x, kspace_pred = self.pf_threshold(x, sens_maps, convex_hull)
            
        x = x.squeeze(1)  # shape from (batch, coil, h, w, 2) to (batch, h, w, 2)
        return x, kspace_pred
    
    def pf_threshold(self, x, sens_maps, convex_hull):
        k_pf_thresh = sens_expand(x, sens_maps, mask=convex_hull)
        x_pf_thresh = sens_reduce(k_pf_thresh, sens_maps, mask=convex_hull)
        return x_pf_thresh, k_pf_thresh
