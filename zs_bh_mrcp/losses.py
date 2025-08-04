"""
Copyright (c) Jinho Kim (jinho.kim@fau.de).

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import torch
import torch.nn as nn


class MixL1L2Loss(nn.Module):

    def __init__(self, eps=1e-10):
        super().__init__()
        self.eps = eps
        self.scaler = 0.5

    def forward(self, pred: torch.Tensor, targ: torch.Tensor):
        return 0.5 * (
            torch.linalg.vector_norm(pred - targ) / torch.linalg.vector_norm(targ)
        ) + 0.5 * (
            torch.linalg.vector_norm(pred - targ, ord=1)
            / torch.linalg.vector_norm(targ, ord=1)
        )
