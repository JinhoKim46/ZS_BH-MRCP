"""
Copyright (c) Jinho Kim (jinho.kim@fau.de).

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""


import torch
import torch.nn as nn

from ..data.data_consistency import conjgrad
from .resnet import ResNet
from .utils import sens_expand, sens_reduce


class ZSNet(nn.Module):

    def __init__(
        self,
        num_stages: int = 12,
        num_resblocks: int = 15,
        chans: int = 64,
        cgdc_iter: int = 10,
        backbone: str = "backbone",
    ):
        """
        Args:
            num_stages:       Number of stages in the network. This is set to zero for shallow mode.
            num_resblocks:    Number of residual blocks.
            chans:            Number of channels in the network.
            cgdc_iter:        Number of conjugate gradient iterations.
            backbone:         Backbone model to use (e.g., "backbone").
        """ 
        super().__init__()
        self.regularizer = ResNet(in_ch=2, chans=chans, num_of_resblocks=num_resblocks)

        self.num_stages = num_stages
        self.cgdc_iter = cgdc_iter
        
        # If deep mode, set starting_img to the input_x
        if self.num_stages == 0:
            ckpt = torch.load(f"pretrained_models/{backbone}.ckpt")
            old_state_dict = ckpt["state_dict"]

            self.register_buffer("lam", old_state_dict['net.lam'])
        else:
            self.lam = nn.Parameter(torch.tensor([0.05]))


    def forward(self, starting_img: torch.Tensor, kdata_tau_OwoG:torch.Tensor, mask_tau_OwoG:torch.Tensor, sens_maps:torch.Tensor, convex_hull:torch.Tensor = None) -> torch.Tensor:
        input_x = sens_reduce(kdata_tau_OwoG, sens_maps)  # R

        # If deep mode, set starting_img to the input_x
        if self.num_stages > 0:
            starting_img = input_x
            
        # First stage in the deep mode. 
        # Last stage in the shallow mode.
        x = input_x + self.lam * self.regularizer(starting_img) # R
        x = conjgrad(x, sens_maps, mask_tau_OwoG, self.lam, self.cgdc_iter) # DC
        x, kspace_pred = self.pf_threshold(x, sens_maps, convex_hull)
        
        # Run from the second stage for deep mode. 
        for _ in range(1, self.num_stages):
            x = input_x + self.lam * self.regularizer(x)
            x = conjgrad(x, sens_maps, mask_tau_OwoG, self.lam, self.cgdc_iter)
            
            x, kspace_pred = self.pf_threshold(x, sens_maps, convex_hull)
            
        x = x.squeeze(1)  # shape from (batch, coil, h, w, 2) to (batch, h, w, 2)
        return x, kspace_pred
    
    def pf_threshold(self, x, sens_maps, convex_hull):
        k_pf_thresh = sens_expand(x, sens_maps, mask=convex_hull)
        x_pf_thresh = sens_reduce(k_pf_thresh, sens_maps, mask=convex_hull)
        return x_pf_thresh, k_pf_thresh
