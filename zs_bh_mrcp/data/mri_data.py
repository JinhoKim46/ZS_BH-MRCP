"""
Copyright (c) Facebook, Inc. and its affiliates.
Copyright (c) Jinho Kim (jinho.kim@fau.de).

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import h5py
import pandas as pd
import torch
from tqdm import tqdm

from ..data.transforms import (backbone_inference, get_convex_hull,
                               get_train_masks_in_preproc,
                               get_val_masks_in_preproc, to_tensor)
from ..models.backbonenet import BackboneNet
from .ssdu_subsample import SSDUMaskFunc


def get_hdf_key(stage, fname, dataslice, zs_k_idx=None):
    """
    Generate a unique key for the HDF5 dataset based on the stage, filename, slice index, and ZS k-index.
    """
    if stage == "train":
        return f"{fname}_{dataslice:03d}_{zs_k_idx:03d}"
    else:
        return f"{fname}_{dataslice:03d}"

class BaseDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(
        self,
        root: Union[str, Path, os.PathLike],
        transform: Callable = None,
        data_partition: str = "train",
        training_data_fname: str = None,
        zs_k: int = 10,
    ):
        """
        Args:
            root:                       Path to the dataset.
            transform:                  A transform object for the dataset.
            data_partition:             A string that specifies the data partition. It can be 'train', 'val', or 'test'.
            training_data_fname:        A string that specifies the data to use.
            zs_k:                       k-set of the ZS mask
        """
        self.transform = transform
        self.examples: List[Tuple[Path, int, int]] = []
        self.data_partition = data_partition
        self.zs_k = zs_k
        
        if not isinstance(training_data_fname, str):            
            raise ValueError("training_data_fname must be provided in String format for ZS training.")

        if not isinstance(root, Path):
            root = Path(root)
        training_fpath = root / f"{training_data_fname}.h5"

        # Get examples
        num_slices = self._retrieve_shape(training_fpath)
        self.examples += [
            (training_fpath, slice_idx, i) for slice_idx in range(num_slices) for i in range(zs_k if data_partition == "train" else 1)
            ]


    def _retrieve_shape(self, fname):
        with h5py.File(fname, "r") as hf:
            cha, lin, col, par = hf["kdata"].shape

        return col

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i: int):
        raise NotImplementedError


class MRCPDataset(BaseDataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(
        self,
        is_prototype: bool = False,
        ssdu_mask_func: Optional[SSDUMaskFunc] = None,
        rho_lambda: float = 0.4,
        rho_gamma: float = 0.2,
        device: int = 0,
        backbone: str = "backbone",
        zs_mode: str = "shallow",
        **kwargs,
    ):
        """
        Args:
            is_prototype:       If True, the dataset will be limited to a small number of examples for quick testing.
            ssdu_mask_func:     An instance of SSDUMaskFunc to generate SSDU masks.
            rho_lambda:         Sampling ratio of |LAMBDA|/|OMEGA\GAMMA| for training and loss.
            rho_gamma:          Sampling ratio of |GAMMA|/|OMEGA| for validation.
            device:             Device ID for PyTorch operations.
            backbone:           Name of the backbone model to use for reconstruction.
            zs_mode:            Zero-shot mode, either "shallow" or "deep".
        """
        super().__init__(**kwargs)
        self.ssdu_mask_func = ssdu_mask_func
        self.device = device
        self.rho_lambda = rho_lambda
        self.rho_gamma = rho_gamma
        self.backbone = backbone
        self.zs_mode = zs_mode
        self.preprocessed_set = dict() 
        
        if self.zs_mode == "shallow":    
            self._load_trained_model()
            self._preprocessing()
        
            
        if is_prototype:
            self.examples = self.examples[10::20]

    def _load_trained_model(self):
        '''
        Load the pre-trained backbone model for zero-shot reconstruction.
        This model is used to initialize the reconstruction process.
        It loads the model from a checkpoint file and prepares it for inference.
        The model is set to evaluation mode to ensure that it does not perform any training-specific operations
        such as dropout or batch normalization updates.
        This is crucial for ensuring that the model behaves consistently during inference.
        
        Please modified the pretrained model path and the backbone parameters
        '''
        # backbone model
        BACKBONE_NUM_STAGES = 12
        BACKBONE_NUM_RESBLOCKS = 8
        BACKBONE_CHANS = 64
        BACKBONE_CGDC_ITER = 10
        
        self.pretrained_model = BackboneNet(
            num_stages=BACKBONE_NUM_STAGES, 
            num_resblocks=BACKBONE_NUM_RESBLOCKS, 
            chans=BACKBONE_CHANS, 
            cgdc_iter=BACKBONE_CGDC_ITER
            ).to(self.device)

        ckpt = torch.load(f"pretrained_models/{self.backbone}.ckpt", map_location=torch.device(f"cuda:{self.device}"))
        old_state_dict = ckpt["state_dict"]

        # Remove the "net." prefix
        new_state_dict = {}
        for k, v in old_state_dict.items():
            # If the prefix is "net.", remove it
            if k.startswith("net."):
                new_k = k[len("net.") :]  # e.g. "net.lam" -> "lam"
            else:
                new_k = k
            new_state_dict[new_k] = v

        self.pretrained_model.load_state_dict(new_state_dict)
        self.pretrained_model.eval()

    def _preprocessing(self):
        """
        Preprocess the dataset by applying the pre-trained model to each example.
        This method initializes the reconstruction process for each example in the dataset.
        It iterates over the examples, applies the pre-trained model to reconstruct the images, and
        stores the results in the `preprocessed_set` dictionary.
        
        If there is a memory issue, you can reduce the BATCH_SIZE.

        kdata_tau_OwoG:                 The k-space data masked with the Tau for training and Omega \ Gamma (OwoG) for validation.
        mask_tau_OwoG:                  The mask applied to the k-space data. Tau for training and Omega \ Gamma (OwoG) for validation.
        kdata_lambda_gamma:             The k-space data masked with the Lambda for training and Gamma for validation.
        mask_lambda_gamma:              The mask applied to the k-space data. Lambda for training and Gamma for validation.
        """
        BATCH_SIZE = 500 # <==== Reducing this value can help with memory issues

        with h5py.File(self.examples[0][0], "r") as hf:
            kspace = hf["kdata"][:] # shape: (ncoil, phase_encoding, readout, slice_encoding)
            sens_maps = hf["sm_espirit"][:]  # shape: (ncoil, phase_encoding, readout, slice_encoding)

        mask_omega = kspace[0, :, 0].astype(bool)[None, ..., None]
        convex_hull = get_convex_hull(mask_omega)
        convex_hull_torch = torch.from_numpy(convex_hull)

        fname = self.examples[0][0].stem
        if self.data_partition == "train":
            '''
            mask_tau_OwoG_dict: Tau mask
            mask_lambda_gamma_dict: Lambda mask
            '''
            mask_tau_OwoG_dict, mask_lambda_gamma_dict = get_train_masks_in_preproc(
                fname, kspace, self.zs_k, self.ssdu_mask_func, self.rho_gamma, self.rho_lambda
            )
        elif self.data_partition == "val":
            """
            mask_tau_OwoG_dict: Omega \ Gamma mask
            mask_lambda_gamma_dict: Gamma mask
            """
            mask_tau_OwoG_dict, mask_lambda_gamma_dict = get_val_masks_in_preproc(fname, kspace, self.ssdu_mask_func, self.rho_gamma)
        else:
            pass

        total_examples = len(self.examples)
        index = 0
        pbar = tqdm(total=total_examples, unit="example")

        with pbar:
            while index < total_examples:
                batch_size = min(BATCH_SIZE, total_examples - index)
                batch_examples = self.examples[index : index + batch_size]
                index += batch_size

                batch_kdata_tau_OwoG = []
                batch_sens_maps = []
                batch_keys = []
                batch_mask_tau_OwoG = []
                batch_mask_lambda_gamma = []

                for _, dataslice, zs_k_idx in batch_examples:
                    hdf_key = get_hdf_key(self.data_partition, fname, dataslice, zs_k_idx)

                    pbar.set_description(f"[{self.data_partition}] Processing {hdf_key}")
                    pbar.update(1)

                    if self.data_partition in ("train", "val"):
                        m1 = mask_tau_OwoG_dict[hdf_key]
                        m2 = mask_lambda_gamma_dict[hdf_key]
                    else:
                        m1 = m2 = mask_omega # For testing, use the Omega mask.

                    m1_torch = torch.from_numpy(m1)
                    input_kspace = to_tensor(kspace[..., dataslice, :]) * m1_torch
                    input_sens_maps = to_tensor(sens_maps[..., dataslice, :])

                    batch_kdata_tau_OwoG.append(input_kspace)
                    batch_sens_maps.append(input_sens_maps)
                    batch_mask_tau_OwoG.append(m1)
                    batch_mask_lambda_gamma.append(m2)
                    batch_keys.append(hdf_key)

                if not batch_kdata_tau_OwoG:
                    continue  # all already processed

                batch_kdata_tau_OwoG = torch.stack(batch_kdata_tau_OwoG) # shape: (batch_size, ncoil, phase_encoding, slice_encoding, 2)
                batch_sens_maps = torch.stack(batch_sens_maps) # shape: (batch_size, ncoil, phase_encoding, slice_encoding, 2)
                batch_mask_tau_OwoG_torch = torch.stack([torch.from_numpy(m) for m in batch_mask_tau_OwoG]) # shape: (batch_size, 1, phase_encoding, slice_encoding, 1)
                
                recon_batch = backbone_inference(
                    kdata_tau_OwoG=batch_kdata_tau_OwoG,
                    sens_maps=batch_sens_maps,
                    mask_tau_OwoG=batch_mask_tau_OwoG_torch,
                    convex_hull=convex_hull_torch.expand(batch_kdata_tau_OwoG.shape[0], *convex_hull_torch.shape),
                    model=self.pretrained_model,
                    device=self.device,
                )

                recon_batch = recon_batch.detach().cpu().numpy()
            
                # Update the preprocessed data to the preprocessed_set dictionary
                self.preprocessed_set.update(
                        {
                            key: {
                                "recon": recon[None],  # adds batch dim, shape: (1, readout, phase_encoding, slice_encoding, 2)
                                "mask_tau_OwoG": m1, # shape: (1, phase_encoding, slice_encoding, 1)
                                "mask_lambda_gamma": m2, # shape: (1, phase_encoding, slice_encoding, 1)
                            }
                            for key, recon, m1, m2 in zip(batch_keys, recon_batch, batch_mask_tau_OwoG, batch_mask_lambda_gamma)
                        }
                    )
                
    def __getitem__(self, i: int):
        fname, readout_index, zs_k_idx = self.examples[i]

        kwargs = {"zs_mode": self.zs_mode}
        
        with h5py.File(fname, "r") as hf:
            # For ZS, iterate over RO
            kdata_omega = hf["kdata"][..., readout_index, :]  # shape: (ncoil, phase_encoding, (readout), slice_encoding)
            ref_cs = hf["cs"][..., readout_index, :]  # shape: (phase_encoding, (readout), slice_encoding)
            sens_maps = hf["sm_espirit"][..., readout_index, :]  # shape: (ncoil, phase_encoding, (readout), slice_encoding)

            mask_omega = kdata_omega[0].astype(bool)[None, ..., None]  # shape: (1, phase_encoding, slice_encoding, 1)
            attrs = dict(hf.attrs)
            
            if self.zs_mode == "deep":
                kwargs['ssdu_mask_func'] = self.ssdu_mask_func
                kwargs['rho_lambda'] = self.rho_lambda
                kwargs['rho_gamma'] = self.rho_gamma
                
        # For shallow mode, hdf_key exists in preprocessed_set. 
        # For deep mode, it is not used. => None
        hdf_key = get_hdf_key(self.data_partition, fname.stem, readout_index, zs_k_idx)
        preprocessed_set = self.preprocessed_set.get(hdf_key, None)

        sample = self.transform(
            kdata_omega, ref_cs, sens_maps, mask_omega, readout_index, fname.name, attrs, zs_k_idx, preprocessed_set=preprocessed_set, **kwargs
        )
        return sample
