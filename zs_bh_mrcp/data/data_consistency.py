"""
===============================================================================
Copyright (c) Jinho Kim (jinho.kim@fau.de).

Modifications and additional features by Jinho Kim are licensed under the MIT license, 
as detailed in the accompanying LICENSE file.
===============================================================================
"""

import torch

from .. import complex_conj, complex_mul
from ..models.utils import sens_expand, sens_reduce


def zdot_reduce_sum(input_x: torch.Tensor, input_y: torch.Tensor):
    # take only real part
    dims = tuple(range(len(input_x.shape[:-1])))
    inner = complex_mul(complex_conj(input_x), input_y).sum(dims)
    return inner[0]


def EhE_Op(
    img: torch.Tensor,
    sens_maps: torch.Tensor,
    mask: torch.Tensor,
    mu: torch.Tensor,
    coil_axis=0,
):
    """
    Performs (E^h*E + mu*I) x

    Parameters
    img: input image, (1, nrow, ncol, 2)
    sens_maps: coil sensitivity maps, (ncoil, nrow, ncol, 2)
    mask: undersampling mask, (1, nrow, ncol, 1)
    mu: penalty parameter
    """
    kspace = sens_expand(img, sens_maps) * mask
    image = sens_reduce(kspace, sens_maps, dim=coil_axis)

    return image + mu * img


def core_conjgrad(rhs, sens_maps, mask, mu, cg_iter):
    """
    Parameters
    ----------
    input_data : contains tuple of  reg output rhs = E^h*y + mu*z , sens_maps and mask
    rhs = (1, nrow, ncol, 2)
    sens_maps : coil sensitivity maps (ncoil, nrow, ncol, 2)
    mask : (1, nrow, ncol, 1)
    mu : penalty parameter
    Returns
    --------
    data consistency output, (1, nrow, ncol, 2)
    """
    # mu = mu.type(torch.complex64)
    x = torch.zeros_like(rhs)
    r, p = rhs, rhs
    rsnot = zdot_reduce_sum(r, r)
    rsold, rsnew = rsnot, rsnot

    for _ in range(cg_iter):
        Ap = EhE_Op(img=p, sens_maps=sens_maps, mask=mask, mu=mu)
        pAp = zdot_reduce_sum(p, Ap)
        alpha = rsold / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = zdot_reduce_sum(r, r)
        beta = rsnew / rsold
        rsold = rsnew
        p = beta * p + r

    return x


def conjgrad(rhs, sens_maps, mask, mu, cg_iter):
    """
    DC block employs conjugate gradient for data consistency,
    """
    batched_core = torch.vmap(core_conjgrad, in_dims=(0, 0, 0, None, None))
    return batched_core(rhs, sens_maps, mask, mu, cg_iter)

