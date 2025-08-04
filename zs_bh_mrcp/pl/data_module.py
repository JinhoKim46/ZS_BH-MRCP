"""
Copyright (c) Facebook, Inc. and its affiliates.
Copyright (c) Jinho Kim (jinho.kim@fau.de).

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from argparse import ArgumentParser
from pathlib import Path
from typing import Callable, Optional

import pytorch_lightning as pl
import torch

from ..data import MRCPDataset
from ..data.batch_sampler import ClusteredBatchSampler
from ..data.ssdu_subsample import SSDUMaskFunc


class DataModule(pl.LightningDataModule):
    """
    Data module class for fastMRI data sets.

    This class handles configurations for training on fastMRI data. It is set
    up to process configurations independently of training modules.

    Note that subsampling mask and transform configurations are expected to be
    done by the main client training scripts and passed into this data module.
    """

    def __init__(
        self,
        data_path: Path = Path(),
        train_transform: Optional[Callable] = None,
        val_transform: Optional[Callable] = None,
        test_transform: Optional[Callable] = None,
        batch_size: int = 1,
        num_workers: int = 4,
        distributed_sampler: bool = False,
        training_data_fname: str = None,
        zs_k: int = 10,
        is_prototype: bool = False,
        ssdu_mask_func: Optional[SSDUMaskFunc] = None,
        rho_lambda: float = 0.4,
        rho_gamma: float = 0.2,
        device: Optional[int] = None,
        backbone: str = "backbone",
        zs_mode: str = "shallow",
        **kwargs,  # additional arguments for flexibility
    ):
        """
        Args:
            data_path:                  Path to root data directory. For example, if knee/path
                                        is the root directory with subdirectories multicoil_train and
                                        multicoil_val, you would input knee/path for data_path.
            train_transform:            A transform object for the training split.
            val_transform:              A transform object for the validation split.
            test_transform:             A transform object for the test split.
            batch_size:                 Batch size.
            num_workers:                Number of workers for PyTorch dataloader.
            distributed_sampler:        Whether to use a distributed sampler.
            training_data_fname:        List of data files to use for training.
            zs_k:                       k-set of the ZS mask.            
            is_prototype:               Whether to use prototype data.
            ssdu_mask_func:             SSDU mask function to use for subsampling.
            rho_lambda:                 Rho value for training data.
            rho_gamma:                  Rho value for validation data.
            device:                     Device to use for training. If None, will use the first available device.
            backbone:                   Backbone model to use for training.
            zs_mode:                    Zero-shot mode, either "shallow" or "deep".
        """
        super().__init__()

        self.data_path = data_path
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed_sampler = distributed_sampler
        self.training_data_fname = training_data_fname
        self.zs_k = zs_k
        self.is_prototype = is_prototype
        self.ssdu_mask_func = ssdu_mask_func
        self.rho_lambda = rho_lambda
        self.rho_gamma = rho_gamma
        self.device = device
        self.backbone = backbone       
        self.zs_mode = zs_mode

    def _create_data_loader(
        self,
        data_transform,
        data_partition: str,
    ) -> torch.utils.data.DataLoader:

        if data_partition == "train":
            is_train = True
            
        else:
            is_train = False

        dataset = MRCPDataset(
            root=self.data_path,
            transform=data_transform,
            data_partition=data_partition,
            training_data_fname=self.training_data_fname,
            zs_k=self.zs_k,
            is_prototype=self.is_prototype,
            ssdu_mask_func=self.ssdu_mask_func,
            rho_lambda=self.rho_lambda,
            rho_gamma=self.rho_gamma,
            device=self.device,
            backbone=self.backbone,
            zs_mode=self.zs_mode
        )

        sampler = None
        if self.batch_size > 1:
            # ensure that batches contain only samples of the same size
            sampler = ClusteredBatchSampler(
                dataset,
                batch_size=self.batch_size,
                shuffle=is_train,
                distributed=self.distributed_sampler,
            )
        elif self.distributed_sampler:
            sampler = torch.utils.data.DistributedSampler(dataset)

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            num_workers=self.num_workers,
            batch_size=(self.batch_size if not isinstance(sampler, ClusteredBatchSampler) else 1),
            sampler=sampler if not isinstance(sampler, ClusteredBatchSampler) else None,
            batch_sampler=(sampler if isinstance(sampler, ClusteredBatchSampler) else None),
            pin_memory=True,
            shuffle=is_train if sampler is None else False,
            persistent_workers=True if self.num_workers > 0 else False,
        )

        return dataloader

    def train_dataloader(self):
        return self._create_data_loader(self.train_transform, data_partition="train")

    def val_dataloader(self):
        return self._create_data_loader(self.val_transform, data_partition="val")

    def test_dataloader(self):
        return self._create_data_loader(self.test_transform, data_partition="test")

    @staticmethod
    def add_data_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # data loader arguments
        parser.add_argument(
            "--batch_size", default=1, type=int, help="Data loader batch size"
        )
        parser.add_argument(
            "--num_workers",
            default=4,
            type=int,
            help="Number of workers to use in data loader",
        )
        parser.add_argument(
            "--is_prototype",
            default=False,
            type=bool,
            help="Whether to use prototype data or not",
        )

        return parser
