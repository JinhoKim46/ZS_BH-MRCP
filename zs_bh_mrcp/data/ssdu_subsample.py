"""
Copyright (c) Jinho Kim (jinho.kim@fau.de).

Modifications and additional features by Jinho Kim are licensed under the MIT license, 
as detailed in the accompanying LICENSE file.
"""

import contextlib
from typing import Optional, Tuple, Union

import numpy as np


@contextlib.contextmanager
def temp_seed(rng: np.random.RandomState, seed: Optional[Union[int, Tuple[int, ...]]]):
    """A context manager for temporarily adjusting the random seed."""
    if seed is None:
        try:
            yield
        finally:
            pass
    else:
        state = rng.get_state()
        rng.seed(seed)
        try:
            yield
        finally:
            rng.set_state(state)


class SSDUMaskFunc:
    """
    A parent class for SSDU sampling masks.
    """

    def __init__(
        self,
        center_block: Tuple[int, int] = (5, 5),
        std_scale: int = 4,
        seed: Optional[int] = None,
    ):
        self.center_block = center_block
        self.std_scale = std_scale
        self.rng = np.random.RandomState(seed)

    def __call__(
        self,
        input_data: np.ndarray,
        mask_omega: np.ndarray,
        rho: float = 0.4,
        seed: Optional[Union[int, Tuple[int, ...]]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample and return a k-space mask.

        Args:
            shape: Shape of k-space.
            offset: Offset from 0 to begin mask (for equispaced masks). If no
                offset is given, then one is selected randomly.
            seed: Seed for random number generator for reproducibility.

        Returns:
            A 2-tuple containing 1) the k-space mask and 2) the number of
            center frequency lines.
        """
        assert input_data.ndim == 3, "input_data should have the shape of 3"

        with temp_seed(self.rng, seed):
            mask_theta, mask_lambda = self.sample_mask(input_data, mask_omega, rho)

        trn_mask = mask_theta
        loss_mask = mask_lambda
        return trn_mask, loss_mask

    def sample_mask(self, input_data: np.ndarray, mask_omega: np.ndarray, rho: float) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class SSDUUniformMask(SSDUMaskFunc):
    """Generate uniform distributed SSDU sampling mask."""

    def sample_mask(
        self, input_data: np.ndarray, mask_omega: np.ndarray, rho: float = 0.4
    ) -> Tuple[np.ndarray, np.ndarray]:

        nrow, ncol = input_data.shape[1], input_data.shape[2]
        center_kx = int(find_center_ind(input_data, axes=(0, 2)))
        center_ky = int(find_center_ind(input_data, axes=(0, 1)))

        center_block_x, center_block_y = self.center_block
        center_block_x_half = center_block_x // 2
        center_block_y_half = center_block_y // 2

        temp_mask = np.copy(mask_omega)
        temp_mask[
            0,
            center_kx - center_block_x_half : center_kx - center_block_x_half + center_block_x,
            center_ky - center_block_y_half : center_ky - center_block_y_half + center_block_y,
            0,
        ] = 0

        pr = np.ndarray.flatten(temp_mask)
        ind = self.rng.choice(
            np.arange(nrow * ncol),
            size=int(np.count_nonzero(pr) * rho),
            replace=False,
            p=pr / np.sum(pr),
        )

        [ind_x, ind_y] = index_flatten2nd(ind, (nrow, ncol))

        loss_mask = np.zeros_like(mask_omega)
        loss_mask[0, ind_x, ind_y, 0] = 1

        trn_mask = mask_omega - loss_mask

        return trn_mask, loss_mask


def norm(tensor, axes=(0, 1, 2), keepdims=True):
    """
    Parameters
    ----------
    tensor : It can be in image space or k-space.
    axes :  The default is (0, 1, 2).
    keepdims : The default is True.

    Returns
    -------
    tensor : applies l2-norm .

    """
    for axis in axes:
        tensor = np.linalg.norm(tensor, axis=axis, keepdims=True)

    if not keepdims:
        return tensor.squeeze()

    return tensor


def find_center_ind(kspace, axes=(1, 2, 3)):
    """
    Parameters
    ----------
    kspace : nrow x ncol x ncoil.
    axes :  The default is (1, 2, 3).

    Returns
    -------
    the center of the k-space

    """

    center_locs = norm(kspace, axes=axes).squeeze()

    return np.argsort(center_locs)[-1:]


def index_flatten2nd(ind, shape):
    """
    Parameters
    ----------
    ind : 1D vector containing chosen locations.
    shape : shape of the matrix/tensor for mapping ind.

    Returns
    -------
    list of >=2D indices containing non-zero locations

    """

    array = np.zeros(np.prod(shape))
    array[ind] = 1
    ind_nd = np.nonzero(np.reshape(array, shape))

    return [list(ind_nd_ii) for ind_nd_ii in ind_nd]