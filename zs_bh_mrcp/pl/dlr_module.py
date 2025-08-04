"""
Copyright (c) Facebook, Inc. and its affiliates.
Copyright (c) Jinho Kim (jinho.kim@fau.de).

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""


from typing import Optional, TypedDict

import pytorch_lightning as pl
import torch
import torchmetrics
import yaml

from ..data.transforms import center_crop
from ..losses import MixL1L2Loss
from ..models import ZSNet


class ZSParams(TypedDict):
    num_stages: int
    num_resblocks: int
    chans: int
    cgdc_iter: int
    backbone: str
    
    
class DLRModule(pl.LightningModule):
    """
    Training module of zero-shot self-supervised learning for breath-hold MRCP reconstruction.
    """

    def __init__(
        self,
        lr: float = 0.0003,
        lr_scheduler: Optional[str] = None,
        zs_mode: str = "shallow",
        zs_params: ZSParams = {},
        **kwargs,
    ):
        """
        Args:
            lr:                     Learning rate.
            lr_scheduler:           Learning rate scheduler type.
            zs_params:              Parameters for the ZSNet 
            zs_mode:                Zero-shot mode, either "shallow" or "deep".
            model.            
                - num_stages:       Number of stages in the network. This is set to zero for shallow mode.
                - num_resblocks:    Number of residual blocks.
                - chans:            Number of channels in the network.
                - cgdc_iter:        Number of conjugate gradient iterations.
                - backbone:         Backbone model to use (e.g., "backbone", "dcnet").    
            **kwargs:               Additional keyword arguments for the LightningModule.
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.lr = lr
        self.lr_scheduler = lr_scheduler

        if zs_mode == "shallow":
            zs_params["num_stages"] = 0
        elif zs_mode == "deep":
            if zs_params["num_stages"] <= 0:
                raise ValueError("num_stages must be greater than 0 for deep mode.")
            
        self.net = ZSNet(**zs_params)

        self.loss = MixL1L2Loss()

        # evaluation metrics
        self.val_loss = torchmetrics.MeanMetric()
        
    def forward(self, starting_img, kdata_tau_OwoG, mask_tau_OwoG, sens_maps, convex_hull):
        """
        Return:
            output_recon: Reconstructed image (..., 2) real vaued 2 chans.
            output_k: Predicted k-space (...) complex valued

        """
        output_recon, output_k = self.net(
            starting_img, kdata_tau_OwoG, mask_tau_OwoG, sens_maps, convex_hull
        )
        return torch.view_as_complex(output_recon), output_k

    def common_forward(self, batch):
        output_recon, output_k = self(
            starting_img=batch.starting_img,
            kdata_tau_OwoG=batch.kdata_tau_OwoG,
            mask_tau_OwoG=batch.mask_tau_OwoG,
            sens_maps=batch.sens_maps,
            convex_hull=batch.convex_hull,
        )

        output_k_lambda_gamma = output_k * batch.mask_lambda_gamma
        out_loss = torch.view_as_complex(output_k_lambda_gamma)
        target_loss = torch.view_as_complex(batch.kdata_lambda_gamma)

        loss = self.loss(out_loss, target_loss)

        return (
            output_recon,
            loss,
        )

    def training_step(self, batch, _):
        _, loss = self.common_forward(batch)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        output_recon, loss = self.common_forward(batch)
        
        # Crop oversampling
        ref_cs, output_recon = center_crop(
            [batch.ref_cs, output_recon], crop_size=batch.crop_size
        )

        return {
            "fname": batch.fname,
            "readout_index": batch.readout_index,
            "ref_cs": ref_cs,
            "output": output_recon,
            "val_loss": loss,
        }

    def test_step(self, batch, batch_idx):
        output_recon, _ = self(
            starting_img=batch.starting_img,
            kdata_tau_OwoG=batch.kdata_tau_OwoG,
            mask_tau_OwoG=batch.mask_tau_OwoG,
            sens_maps=batch.sens_maps,
            convex_hull=batch.convex_hull,
        )
        
        # Crop oversampling
        ref_cs, output_recon = center_crop(
            [batch.ref_cs, output_recon], crop_size=batch.crop_size
        )

        return {
            "fname": batch.fname,
            "readout_index": batch.readout_index,
            "output": output_recon,
            "ref_cs": ref_cs,
        }
            
    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        self.val_loss.update(outputs["val_loss"])

    def on_validation_epoch_end(self):
        # logging
        self.log("val_loss", self.val_loss, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = self._get_scheduler(optim)

        return ([optim], [scheduler]) if scheduler is not None else optim

    def _get_scheduler(self, optimizer):
        config_yaml = yaml.safe_load(open("configs/lr_scheduler.yaml", "r"))

        if self.lr_scheduler is None:
            scheduler = None
        else:
            scheduler_params = config_yaml[self.lr_scheduler]
            T_max = self.trainer.max_epochs

            if self.lr_scheduler == "StepLR":
                step_size = T_max // scheduler_params["decay_step"] if scheduler_params["decay_step"] < T_max else T_max
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, gamma=scheduler_params["gamma"], step_size=step_size
                )
            elif self.lr_scheduler == "CosineAnnealingLR":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, **scheduler_params)
            elif self.lr_scheduler == "CosineAnnealingWarmRestarts":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **scheduler_params)
            else:
                raise ValueError(f"Invalid lr_scheduler: {self.lr_scheduler}")

        return scheduler
