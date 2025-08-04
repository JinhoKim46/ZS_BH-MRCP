"""
Copyright (c) Facebook, Inc. and its affiliates.
Copyright (c) Marc Vornehm <marc.vornehm@fau.de>.
Copyright (c) Jinho Kim <jinho.kim@fau.de>.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import sys
from typing import Dict, Optional

import torch
from jsonargparse import Namespace
from pytorch_lightning.cli import LightningArgumentParser, LightningCLI

from ..data import SSDUUniformMask
from . import DataModule, DLRModule, TQDMProgressBarWithoutVersion
from .utils import (add_main_arguments, print_training_info, read_path_yaml,
                    save_file_dump, set_name)


class DLReconCLI(LightningCLI):
    """
    Customized LightningCLI for breath-hold MRCP DLRecon with zero-shot learning
    """

    def __init__(
        self,
        name: Optional[str] = None,
        **kwargs,
    ):
        self.name = name

        parser_kwargs = kwargs.pop("parser_kwargs", {})
        save_config_kwargs = kwargs.pop("save_config_kwargs", {})
        save_config_kwargs.update({"overwrite": True})

        # check if a config file has been passed via command line
        for i, arg in enumerate(sys.argv):
            if arg in ["-h", "--help", "--print_config"]:
                break 
            if arg in ["-c", "--config"]:
                if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("-"):
                    break
        else:
            raise RuntimeError("No config file given")

        super().__init__(
            DLRModule,
            DataModule,
            save_config_kwargs=save_config_kwargs,
            parser_kwargs=parser_kwargs,
            trainer_defaults={"callbacks": [TQDMProgressBarWithoutVersion()]},
            **kwargs,
        )

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        add_main_arguments(parser)

        group = parser.add_argument_group("Data transform and undersampling options:")
        group.add_argument(
            "--transform.ssdu_mask_center_block",
            type=int,
            default=5,
            help="The size of the SSDU center block.",
        )
        group.add_argument(
            "--transform.ssdu_mask_rho_lambda",
            type=float,
            default=0.4,
            help="A ratio of available points between the lambda mask (loss mask) and the Omega\Gamma mask (original undersampling mask) (||Mask_lambda||/||Mask_omega\Mask_gamma||)",
        )
        group.add_argument(
            "--transform.ssdu_mask_rho_gamma",
            type=float,
            default=0.2,
            help="A ratio of available points between the gamma mask (val mask) and the Omega mask (original undersampling mask) (||Mask_gamma||/||Mask_omega||)",
        )
        group.add_argument(
            "--transform.ssdu_mask_std_scale",
            type=int,
            default=4,
            help="A standard deviation scale for the SSDU Gaussian mask.",
        )

        group = parser.add_argument_group("Callback shortcut options:")
        group.add_argument(
            "--callback.val_log_images",
            type=int,
            default=16,
            help="Number of images to log during validation",
        )
        group.add_argument(
            "--callback.val_log_interval",
            type=int,
            default=10,
            help="Interval for logging validation images",
        )

        parser.add_argument(
            "--float32_matmul_precision",
            type=str,
            default="highest",
            choices=["highest", "high", "medium"],
            help="Precision of float32 matrix multiplications",
        )

    def before_instantiate_classes(self) -> None:
        subcommand = self.config["subcommand"]
        if subcommand == "predict":
            raise NotImplementedError(
                "Prediction is not supported, please use `test` subcommand for inference"
            )

        c = self.config[subcommand]
        if c.trainer.callbacks is None:
            c.trainer.callbacks = []  # initialize with empty list

        c.name = set_name(c) if self.name is None else self.name

        # set default paths based on directory config
        c.data_path = (
            read_path_yaml("data_path") if c.data_path is None else c.data_path
        )
        c.log_path = read_path_yaml("log_path") if c.log_path is None else c.log_path

        c.data.data_path = c.data_path
        c.trainer.default_root_dir = c.log_path / c.name

        # configure checkpointing in checkpoint_dir
        checkpoint_dir = c.trainer.default_root_dir / "checkpoints"
        if not checkpoint_dir.exists():
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for callback in c.trainer.callbacks:
            if callback["class_path"] == "pytorch_lightning.callbacks.ModelCheckpoint":
                callback["init_args"]["dirpath"] = checkpoint_dir

        # set default checkpoint if one exists in our checkpoint directory
        if checkpoint_dir.exists():
            ckpt_list = sorted(checkpoint_dir.glob("*.ckpt"), key=os.path.getmtime)
            if ckpt_list:
                c.ckpt_path = str(ckpt_list[-1])
        if self.subcommand in ["test", "predict"] and c.ckpt_path is None:
            raise RuntimeError("No checkpoint available")

        # logger
        c.trainer.logger = Namespace(
            {
                "class_path": "pytorch_lightning.loggers.tensorboard.TensorBoardLogger",
                "init_args": {
                    "save_dir": c.trainer.default_root_dir,
                    "version": (
                        "train" if self.subcommand in ["fit", "validate"] else "test"
                    ),
                },
            }
        )

        # logging callback
        c.trainer.callbacks.append(
            Namespace(
                {
                    "class_path": "zs_bh_mrcp.pl.ValImageLogger",
                    "init_args": {
                        "num_log_images": c.callback.val_log_images,
                        "logging_interval": c.callback.val_log_interval,
                    },
                }
            )
        )

        # save reconstructions callback
        c.trainer.callbacks.append(
            Namespace(
                {
                    "class_path": "zs_bh_mrcp.pl.TestImageLogger",
                    "init_args": {
                    },
                }
            )
        )

        # Validate zs_mode
        if c.model.zs_mode not in ["shallow", "deep"]:
            raise ValueError(f"Invalid zs_mode: {c.model.zs_mode}. Choose 'shallow' or 'deep'.")

        if c.data.train_transform is None and self.subcommand == "fit":
            c.data.train_transform = Namespace(
                {
                    "class_path": "zs_bh_mrcp.data.transforms.DataTransform",
                    "init_args": {
                        "use_seed": False,
                        "stage": "train",
                        "zs_mode": c.model.zs_mode,
                    },
                }
            )

        if c.data.val_transform is None and self.subcommand in ["fit", "validate"]:
            c.data.val_transform = Namespace(
                {
                    "class_path": "zs_bh_mrcp.data.transforms.DataTransform",
                    "init_args": {
                        "use_seed": True,
                        "stage": "val",
                        "zs_mode": c.model.zs_mode,
                    },
                }
            )
        if c.data.test_transform is None and self.subcommand == "test":
            c.data.test_transform = Namespace(
                {
                    "class_path": "zs_bh_mrcp.data.transforms.DataTransform",
                    "init_args": {
                        "use_seed": True,
                        "stage": "test",
                        "zs_mode": c.model.zs_mode,
                    },
                }
            )

        if c.data.is_prototype:
            print("Running in prototype mode")
            print("Setting num_workers to 0")
            c.data.num_workers = 0
            c.trainer.check_val_every_n_epoch = 1
            c.trainer.log_every_n_steps = 1

        c.data.device = int(c.trainer.devices[0]) # only support a single device. 
        c.data.rho_lambda = c.transform.ssdu_mask_rho_lambda
        c.data.rho_gamma = c.transform.ssdu_mask_rho_gamma
        c.data.zs_mode = c.model.zs_mode   

        block_size = c.transform.ssdu_mask_center_block
        c.data.ssdu_mask_func = SSDUUniformMask(center_block=(block_size, block_size), std_scale=c.transform.ssdu_mask_std_scale)

        # float32 matrix multiplication precision
        torch.set_float32_matmul_precision(c.float32_matmul_precision)

        if not c.data.is_prototype:
            save_file_dump(c, self.subcommand)
        print_training_info(c)
