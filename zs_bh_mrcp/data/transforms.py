"""
Copyright (c) Facebook, Inc. and its affiliates.
Copyright (c) Jinho Kim (jinho.kim@fau.de).

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from pathlib import Path
from typing import NamedTuple, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import torch
from skimage import morphology

from ..models.backbonenet import BackboneNet
from .ssdu_subsample import SSDUMaskFunc


def to_tensor(data: np.ndarray) -> torch.Tensor:
    """
    Convert numpy array to PyTorch tensor.

    For complex arrays, the real and imaginary parts are stacked along the last
    dimension.

    Args:
        data: Input numpy array.

    Returns:
        PyTorch version of data.
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)

    # To compute with FloateTensor, the data should be float32
    return torch.from_numpy(data).to(torch.float32)


def get_us_direction(kspace):
    assert kspace.ndim == 3, "kspace should be 3D."
    _, _, ky = kspace.shape
    us_direction = "LR" if kspace[0, :, ky // 2].all() else "UD"

    return us_direction


def apply_ssdu_mask(
    data: np.ndarray,
    mask_func: SSDUMaskFunc,
    rho: float = 0.4,
    seed: Optional[float | Sequence[float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Subsample the Omega mask into the Theta mask and Lambda mask.

    Args:
        data:           The input k-space data. This is already undersampled data with
                        the Omega mask. The shape should be (Coil, rows, cols)
        mask_func:      A function that takes undersampled kdata and Omega mask and returns
                        the Theta mask for training and Lambda mask for loss.
        attrs:          A dictionary containing the attributes of the data. This is used
                        to determine the padding of the k-space data.
        seed:           Seed for the random number generator.

    Returns:
        tuple containing:
            mask_tau_OwoG:       SSDU Theta mask for training
            mask_lambda_gamma:    SSDU Lambda mask for loss
    """
    us_direction = get_us_direction(data)

    if us_direction == "UD":
        data = data.transpose(0, 2, 1)

    mask_omega = data[0].astype(bool).astype(int)[None, ..., None]  # [1, rows, cols, 1]

    mask_tau_OwoG, mask_lambda_gamma = mask_func(data, mask_omega, rho=rho, seed=seed)

    if us_direction == "UD":
        mask_tau_OwoG = mask_tau_OwoG.transpose(0, 2, 1, 3)
        mask_lambda_gamma = mask_lambda_gamma.transpose(0, 2, 1, 3)

    return mask_tau_OwoG.astype(np.int32), mask_lambda_gamma.astype(np.int32)


def _center_crop(data: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """
    Apply a center crop to the input real image or batch of real images.

    Args:
        data: The input tensor to be center cropped. It should
            have at least 2 dimensions and the cropping is applied along the
            last two dimensions.
        shape: The output shape. The shape should be smaller
            than the corresponding dimensions of data.

    Returns:
        The center cropped image.
    """
    if not (0 < shape[0] <= data.shape[-2] and 0 < shape[1] <= data.shape[-1]):
        raise ValueError("Invalid shapes.")

    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to]


def center_crop(images, crop_size=None):
    """
    Apply a center crop on the larger image to the target_size.

    The minimum is taken over dim=-1 and dim=-2. If x is smaller than y at
    dim=-1 and y is smaller than x at dim=-2, then the returned dimension will
    be a mixture of the two.

    Args:
        images: list of recon_images

    Returns:
        output: list of cropped images
    """
    output = [_center_crop(img, crop_size) for img in images]
    return output


def get_convex_hull(mask:np.ndarray):
    """
    Get the convex hull of the mask.
    Args:
        mask: The input mask. [1, rows, cols, 1]

    Returns:
        The convex hull of the mask.
    """
    orig_shape = mask.shape
    mask = np.squeeze(mask)
    convex_hull = morphology.convex_hull_image(mask)
    
    # to the same shape as the original mask
    convex_hull = convex_hull.reshape(orig_shape)
    return convex_hull


def get_val_masks_in_preproc(fname: str, kspace: np.ndarray, ssdu_mask_func: SSDUMaskFunc, rho_gamma: float):
    '''
    Get the SSDU validation masks in the preprocessed steps.
    
    Args:
        fname:                  File name.
        kspace:                 Acquired k-space data (omega mask).
        ssdu_mask_func:         SSDU mask function to use for subsampling.
        rho_gamma:              Rho_gamma value for validation data.
        
    Returns:
        mask_OwoG_dict:         Dictionary of the Omega \ Gamma mask (subset mask for train/loss).
        mask_gamma_dict:        Dictionary of the Gamma mask (val mask).
    '''
    mask_OwoG_dict, mask_gamma_dict = {}, {}
    for dataslice in range(kspace.shape[2]):
        hdf_key = f"{fname}_{dataslice:03d}"
        seed = list(map(ord, hdf_key))
        mask_OwoG, mask_gamma = apply_ssdu_mask(kspace[..., dataslice, :], ssdu_mask_func, rho=rho_gamma, seed=seed)
        mask_OwoG_dict[hdf_key] = mask_OwoG
        mask_gamma_dict[hdf_key] = mask_gamma

    return mask_OwoG_dict, mask_gamma_dict


def get_train_masks_in_preproc(
    fname: str, kspace: np.ndarray, zs_k: int, ssdu_mask_func: SSDUMaskFunc, rho_gamma: float, rho_lambda: float
):
    '''
    Get the SSDU training masks in the preprocessed steps.
    
    Args:   
        fname:                      File name.
        kspace:                     Acquired k-space data (omega mask).
        zs_k:                       k-set of the ZS mask.
        ssdu_mask_func:             SSDU mask function to use for subsampling.
        rho_gamma:                  Rho_gamma value for validation data.
        rho_lambda:                 Rho_lambda value for training data.
    
    Returns:
        mask_tau_dict:              Dictionary of the Theta mask (train mask).
        mask_lambda_dict:           Dictionary of the Lambda mask (loss mask).
    '''
    mask_tau_dict, mask_lambda_dict = {}, {}
    for dataslice in range(kspace.shape[2]):
        hdf_key_val = f"{fname}_{dataslice:03d}"
        seed = list(map(ord, hdf_key_val))
        mask_OwoG, _ = apply_ssdu_mask(kspace[..., dataslice, :], ssdu_mask_func, rho=rho_gamma, seed=seed)

        kdata_OwoG = kspace[..., dataslice, :] * mask_OwoG.squeeze(-1)

        for zs_k_idx in range(zs_k):
            hdf_key = f"{hdf_key_val}_{zs_k_idx:03d}"
            seed = list(map(ord, hdf_key))
            mask_tau, mask_lambda_gamma = apply_ssdu_mask(kdata_OwoG, ssdu_mask_func, rho=rho_lambda, seed=seed)
            mask_tau_dict[hdf_key] = mask_tau
            mask_lambda_dict[hdf_key] = mask_lambda_gamma

    return mask_tau_dict, mask_lambda_dict


def load_prepocessed_zs(
    hdf_key: str,
    file: Path,
):
    """
    Load the preprocessed zero-shot data from the HDF5 file.
    """
    with h5py.File(file, "r") as hf:
        group = hf[hdf_key]
        recon = np.array(group["recon"])
        mask_tau_OwoG = np.array(group["mask_tau_OwoG"])  # train: theta, val: Omega \gamma
        mask_lambda_gamma = np.array(group["mask_lambda_gamma"])  # train: lambda, val: Omega

    return recon, mask_tau_OwoG, mask_lambda_gamma


def backbone_inference(
    kdata_tau_OwoG: torch.Tensor,
    sens_maps: torch.Tensor,
    mask_tau_OwoG: torch.Tensor,
    convex_hull: torch.Tensor,
    model: BackboneNet,
    device: int,
) -> torch.Tensor:
    """
    Initialize the reconstruction by running the model on the input k-space data.
    
    Args:
        kdata_tau_OwoG:         Input k-space data with shape (batch, coil, h, w, 2).
        sens_maps:              Sensitivity maps with shape (batch, coil, h, w, 2).
        mask_tau_OwoG:          Input mask with shape (batch, 1, h, w, 1).
        convex_hull:            Convex hull of the mask with shape (batch, 1, h, w, 1).
        model:                  Backbone model for reconstruction.
        device:                 Device to run the model on.
    """
    kdata_tau_OwoG = kdata_tau_OwoG.to(device)  # Add batch dimension
    sens_maps = sens_maps.to(device)  # Add batch dimension
    mask_tau_OwoG = mask_tau_OwoG.to(device)  # Add batch dimension
    convex_hull = convex_hull.to(device)  # Add batch dimension

    with torch.no_grad():
        recon, _ = model(
            kdata_tau_OwoG=kdata_tau_OwoG,
            mask_tau_OwoG=mask_tau_OwoG,
            sens_maps=sens_maps,
            convex_hull=convex_hull,
        )

    return recon.detach()


def get_zs_shallow_tensors(preprocessed_set:dict, kdata_omega: torch.Tensor) -> dict:
    
    starting_img = preprocessed_set.get("recon", None)
    mask_tau_OwoG = torch.from_numpy(preprocessed_set.get("mask_tau_OwoG", None))
    mask_lambda_gamma = torch.from_numpy(preprocessed_set.get("mask_lambda_gamma", None))
    
    kdata_tau_OwoG = kdata_omega * mask_tau_OwoG 
    kdata_lambda_gamma = kdata_omega * mask_lambda_gamma 

    shallow_tensor_dict = {
        "starting_img": starting_img,
        "kdata_tau_OwoG": kdata_tau_OwoG,
        "mask_tau_OwoG": mask_tau_OwoG,
        "kdata_lambda_gamma": kdata_lambda_gamma,
        "mask_lambda_gamma": mask_lambda_gamma,
    }
    return shallow_tensor_dict


def get_zs_deep_tensors(kdata_omega: np.ndarray, mask_omega: np.ndarray, zs_val_seed: list, stage: str, **kwargs):
    ssdu_mask_func = kwargs.get("ssdu_mask_func", None)
    if ssdu_mask_func is None:
        raise ValueError("ssdu_mask_func is required for deep mode.")
    rho_lambda = kwargs.get("rho_lambda", 0.4)
    rho_gamma = kwargs.get("rho_gamma", 0.2)

    # omega_wo_gamma_mask: Omega \ Gamma
    # gamma_mask: Gamma
    mask_OwoG, mask_gamma = apply_ssdu_mask(kdata_omega, ssdu_mask_func, rho=rho_gamma, seed=zs_val_seed)

    if stage == "train":
        """
        input: theta
        target: lambda k
        """
        zs_train_seed = kwargs["zs_train_seed"]
        # zs_train_seed = None
        kdata_OwoG = kdata_omega * mask_OwoG.squeeze(-1)

        mask_tau_OwoG, mask_lambda_gamma = apply_ssdu_mask(kdata_OwoG, ssdu_mask_func, rho=rho_lambda, seed=zs_train_seed)

    elif stage == "val":
        """
        input: omega without gamma
        target: lambda gamma
        """
        mask_tau_OwoG = mask_OwoG.copy()
        mask_lambda_gamma = mask_gamma.copy()

    else:
        # use the equi-mask with 1 padding for knee data
        if mask_omega.ndim == 3:
            mask_omega = mask_omega[..., None]
        mask_tau_OwoG = mask_omega
        mask_lambda_gamma = mask_omega  # Will not be used

    """
    mask_tau_OwoG: 
        - for training: Theta mask
        - for validation: Omega without Gamma mask
        - for testing: Gamma mask
    mask_lambda_gamma: 
        - for training: Lambda mask
        - for validation & testing: NOT BE USED
    """
    mask_tau_OwoG = torch.from_numpy(mask_tau_OwoG)
    mask_lambda_gamma = torch.from_numpy(mask_lambda_gamma)

    kdata_tau_OwoG = to_tensor(kdata_omega) * mask_tau_OwoG  # (coil, row, col, 2)
    kdata_lambda_gamma = to_tensor(kdata_omega) * mask_lambda_gamma  # (coil, row, col, 2)

    # This is dummy data
    starting_img = torch.zeros_like(mask_tau_OwoG)  # (1, row, col, 1), Wouldn't be used in the deep mode.

    deep_tensor_dict = {
        "starting_img": starting_img,
        "kdata_tau_OwoG": kdata_tau_OwoG,
        "mask_tau_OwoG": mask_tau_OwoG,
        "kdata_lambda_gamma": kdata_lambda_gamma,
        "mask_lambda_gamma": mask_lambda_gamma,
    }
    return deep_tensor_dict


class BatchSample(NamedTuple):
    """
    A sample of the batch data for DL reconstruction module.

    Args:
        starting_img:               Starting reconstruction image for the shallow mode. This is zero-tensor in the deep mode.
        kdata_tau_OwoG:         k-space data being input to the zero-shot learning model
                                        - in training: theta mask applied.
                                        - in validation: Omega \ Gamma mask applied.
        mask_tau_OwoG:          Zero-shot learning mask used for kdata_tau_OwoG
                                        - in training: theta mask.
                                        - in validation: Omega \ Gamma mask.
        kdata_lambda_gamma:               k-space data applied with lambda mask for training and gamma mask for validation. During testing, this is not used.
        mask_lambda_gamma:                Zero-shot learning mask used for kdata_lambda_gamma. During testing, this is not used.
        kdata_omega:                Acquired k-space data (omega mask).
        ref_cs:                     Compressed sensing reconstruction to compare with zero-shot learning models.
        sens_maps:                  Sensitivity maps.
        convex_hull:                Convex hull of the mask_omega.
        fname:                      File name.
        readout_index:              The readout index.
        crop_size:                  Crop size.
    """

    starting_img: torch.Tensor
    kdata_tau_OwoG: torch.Tensor
    mask_tau_OwoG: torch.Tensor
    kdata_lambda_gamma: torch.Tensor
    mask_lambda_gamma: torch.Tensor
    kdata_omega: torch.Tensor
    ref_cs: torch.Tensor
    sens_maps: torch.Tensor
    convex_hull: torch.Tensor
    fname: str
    readout_index: list
    crop_size: Optional[Tuple[int, int]]


class DataTransform:
    """
    Data Transformer for training models.
    """

    def __init__(
        self,
        use_seed: bool = True,
        stage: str = "train",
        zs_mode: str = "shallow",
    ) -> None:
        """
        Args:
            use_seed:       Whether to use a seed for random number generation.
            stage:          The stage of the data transformation, either "train", "val", or "test".
            zs_mode:        Zero-shot mode, either "shallow" or "deep".
        """
        self.use_seed = use_seed
        self.stage = stage
        self.zs_mode = zs_mode

    def __call__(
        self,
        kdata_omega: np.ndarray,
        ref_cs: np.ndarray,
        sens_maps: Union[torch.Tensor, bool],
        mask_omega: np.ndarray,
        readout_index: int,
        fname: str,
        attrs: dict,
        zs_k_idx: int,
        preprocessed_set: Optional[dict] = None,
        **kwargs,
    ) -> BatchSample:
        """
        Args:
            kdata_omega:        Acquired k-space data (omega mask)
            ref_cs:             Compressed sensing reconstruction to compare with zero-shot learning models.
            sens_maps:          Sensitivity map
            mask_omega:         Omega mask applied to kdata_omega.
            readout_index:      Index number of the readout.
            fname:              File name.
            attrs:              Attributes of the data.
            zs_k_idx:           Index of the zero-shot learning k-space data.
            preprocessed_set:   Preprocessed set containing the precomputed arrays for the shallow mode.

        Returns:
            A VarNetSample with the following fields:
                - starting_img: torch.Tensor
                - kdata_tau_OwoG: torch.Tensor
                - mask_tau_OwoG: torch.Tensor
                - kdata_lambda_gamma: torch.Tensor
                - mask_lambda_gamma: torch.Tensor
                - kdata_omega: torch.Tensor
                - ref_cs: torch.Tensor
                - sens_maps: torch.Tensor
                - fname: str
                - readout_index: list
                - crop_size: Optional[Tuple[int, int]]
        """
        crop_size = attrs.get("recon_crop_size", ())

        crop_size = (crop_size[0], crop_size[2])

        ref_cs = torch.from_numpy(ref_cs)  # gt_img
        sens_maps_torch = to_tensor(sens_maps)

        #### NETWORK SETTING ####
        if self.zs_mode == "shallow":
            '''
                In the shallow mode, the preprocessed data is used.
                For training, the Tau and Lambda are used.
                For validation, the Gamma is used. 
                For testing, the Omega mask is used.
                
            '''
            kdata_omega = to_tensor(kdata_omega)  # Coil, row, col, 2
            shallow_tensor_dict = get_zs_shallow_tensors(preprocessed_set, kdata_omega)

            starting_img = shallow_tensor_dict["starting_img"]
            kdata_tau_OwoG = shallow_tensor_dict["kdata_tau_OwoG"]
            mask_tau_OwoG = shallow_tensor_dict["mask_tau_OwoG"]
            kdata_lambda_gamma = shallow_tensor_dict["kdata_lambda_gamma"]
            mask_lambda_gamma = shallow_tensor_dict["mask_lambda_gamma"]

        elif self.zs_mode == "deep":
            """
                In the deep mode, only tau and lambda are used for training and validation.
                For testing, the Omega mask is used.
            """
            zs_val_seed = list(map(ord, f"{fname}_{readout_index}"))
            if self.stage == "train":
                kwargs["zs_train_seed"] = list(map(ord, f"{fname}_{readout_index}_{zs_k_idx}"))

            deep_tensor_dict = get_zs_deep_tensors(kdata_omega, mask_omega, zs_val_seed=zs_val_seed, stage=self.stage, **kwargs)

            starting_img = deep_tensor_dict["starting_img"] # This is zero-tensor (dummy) in the deep mode.
            kdata_tau_OwoG = deep_tensor_dict["kdata_tau_OwoG"] 
            mask_tau_OwoG = deep_tensor_dict["mask_tau_OwoG"] # In the deep mode, only tau and lambda are used. 
            kdata_lambda_gamma = deep_tensor_dict["kdata_lambda_gamma"] # In the deep mode, only tau and lambda are used.     
            mask_lambda_gamma = deep_tensor_dict["mask_lambda_gamma"] # In the deep mode, only tau and lambda are used. 
            
            # Since apply_ssdu_mask takes np.ndarray, we need to convert kdata_omega to tensor here.
            kdata_omega = to_tensor(kdata_omega)
        else:
            raise ValueError(f"Invalid zs_mode: {self.zs_mode}. Choose 'shallow' or 'deep'.")

        convex_hull = get_convex_hull(mask_omega)

        sample = BatchSample(
            starting_img=starting_img,
            kdata_tau_OwoG=kdata_tau_OwoG,
            mask_tau_OwoG=mask_tau_OwoG,
            kdata_lambda_gamma=kdata_lambda_gamma,
            mask_lambda_gamma=mask_lambda_gamma,
            kdata_omega=kdata_omega,
            ref_cs=ref_cs,
            sens_maps=sens_maps_torch,
            convex_hull=convex_hull,
            fname=fname,
            readout_index=readout_index,
            crop_size=crop_size,
        )

        return sample
