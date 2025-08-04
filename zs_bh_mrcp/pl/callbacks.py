"""
Copyright (c) Facebook, Inc. and its affiliates.
Copyright (c) Marc Vornehm <marc.vornehm@fau.de>.
Copyright (c) Jinho Kim (jinho.kim@fau.de).

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import logging
import shutil
import tempfile
from collections import defaultdict
from glob import glob
from pathlib import Path

import h5py  # Added import
import numpy as np
import pytorch_lightning as pl

from .utils import (collect_h5_contents, log_images, normalize,
                    save_outputs_temp, save_recon_single)

log = logging.getLogger(__name__)

class ValImageLogger(pl.Callback):

    def __init__(
        self,
        num_log_images: int = 16,
        logging_interval: int = 10,
        log_always_after: float = 0.8,
    ):
        """
        Args:
            num_log_images: Number of validation images to log. Defaults to 16.
            logging_interval: After how many epochs to log validation images. Defaults to 10.
            log_always_after: After what percentage of trainer.max_epochs to log images in every epoch.
        """
        super().__init__()

        self.num_log_images = num_log_images
        self.val_log_indices = None
        self.logging_interval = logging_interval
        self.log_always_after = log_always_after

        self.output_imgs = defaultdict(dict)
        self.ref_cs_imgs = defaultdict(dict)



    @property
    def state_key(self) -> str:
        return self._generate_state_key(
            num_log_images=self.num_log_images,
            val_log_indices=self.val_log_indices,
            logging_interval=self.logging_interval,
            log_always_after=self.log_always_after,
        )

    def setup(self, trainer, pl_module, stage):
        tmp_dir = Path(trainer.default_root_dir) / "temp/val"
        if not tmp_dir.exists():
            tmp_dir.mkdir(parents=True, exist_ok=True)

        self.temp_dir = tempfile.mkdtemp(suffix=f"_{trainer.global_rank}", dir=tmp_dir)
        self.h5_path = Path(self.temp_dir) / "val_outputs.h5"
        self.h5_file_handle = None  # Initialize file handle
        print(f"[RANK {trainer.global_rank}] HDF5 file path is: {self.h5_path}")

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Open HDF5 file at the start of validation epoch, truncating if it's not the first time."""
        if not trainer.sanity_checking:
            try:
                # If the HDF5 path exists (i.e., from a previous validation epoch in this run),
                # open it in 'w' mode to truncate it, then close it.
                # This ensures a clean slate for the current validation epoch's data.
                if self.h5_path.exists():
                    with h5py.File(self.h5_path, "w") as f_truncate:
                        pass  # Open in write mode to truncate, then close.
                    log.info(
                        f"[RANK {trainer.global_rank}] HDF5 file {self.h5_path} truncated for new validation epoch."
                    )

                # Now open in append mode for the current epoch's writes.
                self.h5_file_handle = h5py.File(self.h5_path, "a")
                log.info(f"[RANK {trainer.global_rank}] HDF5 file {self.h5_path} opened in append mode for validation.")
            except Exception as e:
                log.error(f"[RANK {trainer.global_rank}] Error preparing HDF5 file {self.h5_path} for validation: {e}")
                self.h5_file_handle = None

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if not isinstance(outputs, dict):
            raise RuntimeError("Expected outputs to be a dict")

        if trainer.sanity_checking:
            return

        # For memory efficiency, save result data in the temporary h5 file.
        if self.h5_file_handle is not None:
            try:
                save_outputs_temp(outputs, self.h5_file_handle)
            except Exception as e:
                log.error(
                    f"[RANK {trainer.global_rank}] Error saving outputs to HDF5 file {self.h5_path} during validation: {e}"
                )
        else:
            log.warning(
                f"[RANK {trainer.global_rank}] HDF5 file handle is None, skipping save_outputs_temp for validation."
            )

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if trainer.sanity_checking:
            return

        # Close the HDF5 file handle on all ranks
        if hasattr(self, "h5_file_handle") and self.h5_file_handle is not None:
            try:
                self.h5_file_handle.close()
                print(f"[RANK {trainer.global_rank}] HDF5 file {self.h5_path} closed at validation epoch end.")
            except Exception as e:
                log.error(
                    f"[RANK {trainer.global_rank}] Error closing HDF5 file {self.h5_path} at validation epoch end: {e}"
                )
            self.h5_file_handle = None

        # Ensure all ranks have finished writing and closed their files before rank 0 gathers
        if trainer.strategy:
            trainer.strategy.barrier()

        if trainer.global_rank != 0:
            return

        val_h5_files = glob(trainer.default_root_dir + "/**/val_outputs.h5", recursive=True)

        # Reopen the HDF5 file in read-only mode to ensure all data is flushed and accessible
        collected_data = collect_h5_contents(val_h5_files)
        print(f"[RANK {trainer.global_rank}] Collected data from {len(collected_data)} validation HDF5 files.")
        for fname, group in collected_data.items():
            output_img = np.stack(
                [group[key][...] for key in sorted(group.keys()) if key.startswith("output_img")]
            )
            ref_cs_img = np.stack(
                [group[key][...] for key in sorted(group.keys()) if key.startswith("ref_cs_img")]
            )

            output_img = np.transpose(output_img, (2, 1, 0))
            ref_cs_img = np.transpose(ref_cs_img, (2, 1, 0))
            log.debug(
                f"[RANK {trainer.global_rank}] Processed {fname} with shape {output_img.shape} and ref_cs shape {ref_cs_img.shape}"
            )
            
            # Process results for this subject
            mip_recon = np.max(output_img, axis=0)
            mip_recon = normalize(mip_recon)

            mip_ref_cs = np.max(ref_cs_img, axis=0)
            mip_ref_cs = normalize(mip_ref_cs)

            mips = {
                f"MIP_val/{fname}/Recon": mip_recon,
                f"MIP_val/{fname}/CS": mip_ref_cs,
            }
            log_images(trainer, mips)
                    

    def teardown(self, trainer, pl_module, stage):
        # Safely close HDF5 file if it's still open
        if hasattr(self, "h5_file_handle") and self.h5_file_handle is not None:
            try:
                self.h5_file_handle.close()
                print(f"[RANK {trainer.global_rank}] HDF5 file {self.h5_path} closed in teardown (ValImageLogger).")
            except Exception as e:
                log.error(
                    f"[RANK {trainer.global_rank}] Error closing HDF5 file {self.h5_path} in teardown (ValImageLogger): {e}"
                )
            self.h5_file_handle = None

        # Remove the temporary directory
        if hasattr(self, "temp_dir") and Path(self.temp_dir).exists():
            try:
                shutil.rmtree(self.temp_dir)
                print(
                    f"[RANK {trainer.global_rank}] Temporary directory {self.temp_dir} (ValImageLogger) has been removed."
                )
            except Exception as e:
                log.error(
                    f"[RANK {trainer.global_rank}] Error removing temporary directory {self.temp_dir} (ValImageLogger): {e}"
                )

class TestImageLogger(pl.Callback):

    def __init__(self):
        '''
        Args:
            num_log_images: Number of test images to log. Defaults to 10.
            test_log_indices: Indices of the test images to log. If None, will be set during the first batch.
            save_img: Whether to save the images to disk.
        '''
        super().__init__()
        self.num_log_images = 10
        self.test_log_indices = None
        self.save_img = True

    @property
    def state_key(self) -> str:
        return self._generate_state_key(
            test_log_indices=self.test_log_indices, save_img=self.save_img
        )

    def setup(self, trainer, pl_module, stage):
        tmp_dir = Path(trainer.default_root_dir) / "temp/test"
        if not tmp_dir.exists():
            tmp_dir.mkdir(parents=True, exist_ok=True)

        self.temp_dir = tempfile.mkdtemp(suffix=f"_{trainer.global_rank}", dir=tmp_dir)
        self.h5_path = Path(self.temp_dir) / "test_outputs.h5"
        self.h5_file_handle = None  # Initialize file handle
        print(f"[RANK {trainer.global_rank}] HDF5 file path is: {self.h5_path}")

    def on_test_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Open HDF5 file at the start of test epoch, truncating if it exists."""
        # Not running sanity checking during test
        try:
            # If the HDF5 path exists (i.e., from a previous test epoch in this run, though unlikely),
            # open it in 'w' mode to truncate it, then close it.
            if self.h5_path.exists():
                with h5py.File(self.h5_path, "w") as f_truncate:
                    pass  # Open in write mode to truncate, then close.
                log.info(f"[RANK {trainer.global_rank}] HDF5 file {self.h5_path} truncated for new test epoch.")

            # Now open in append mode for the current epoch's writes.
            self.h5_file_handle = h5py.File(self.h5_path, "a")
            log.info(f"[RANK {trainer.global_rank}] HDF5 file {self.h5_path} opened in append mode for test.")
        except Exception as e:
            log.error(f"[RANK {trainer.global_rank}] Error preparing HDF5 file {self.h5_path} for test: {e}")
            self.h5_file_handle = None

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if not isinstance(outputs, dict):
            raise RuntimeError("Expected outputs to be a dict")

        # For memory efficiency, save result data in the temporary h5 file.
        if self.h5_file_handle is not None:
            try:
                save_outputs_temp(outputs, self.h5_file_handle)
            except Exception as e:
                log.error(
                    f"[RANK {trainer.global_rank}] Error saving outputs to HDF5 file {self.h5_path} during test: {e}"
                )
        else:
            log.warning(f"[RANK {trainer.global_rank}] HDF5 file handle is None, skipping save_outputs_temp for test.")

    def on_test_epoch_end(self, trainer, pl_module):
        # Close the HDF5 file handle on all ranks
        if hasattr(self, "h5_file_handle") and self.h5_file_handle is not None:
            try:
                self.h5_file_handle.close()
                print(f"[RANK {trainer.global_rank}] HDF5 file {self.h5_path} closed at test epoch end.")
            except Exception as e:
                log.error(f"[RANK {trainer.global_rank}] Error closing HDF5 file {self.h5_path} at test epoch end: {e}")
            self.h5_file_handle = None

        # Ensure all ranks have finished writing and closed their files before rank 0 gathers
        if trainer.strategy:
            trainer.strategy.barrier()

        if trainer.global_rank != 0:
            return

        ckpt_path = Path(trainer.ckpt_path).stem
        save_path = Path(trainer.default_root_dir) / f"results_{ckpt_path}"

        if trainer.global_rank != 0:
            return

        test_h5_files = glob(trainer.default_root_dir + "/**/test_outputs.h5", recursive=True)

        # Reopen the HDF5 file in read-only mode to ensure all data is flushed and accessible
        collected_data = collect_h5_contents(test_h5_files)
        for fname, group in collected_data.items():
            output_img = np.stack(
                [group[key][...] for key in sorted(group.keys()) if key.startswith("output_img")]
            )
            ref_cs_img = np.stack(
                [group[key][...] for key in sorted(group.keys()) if key.startswith("ref_cs_img")]
            )

            output_img = np.transpose(output_img, (2, 1, 0))
            ref_cs_img = np.transpose(ref_cs_img, (2, 1, 0))

            # Process results for this subject
            mip_recon = np.max(output_img, axis=0)
            mip_recon = normalize(mip_recon)

            mip_ref_cs = np.max(ref_cs_img, axis=0)
            mip_ref_cs = normalize(mip_ref_cs)

            mips = {
                f"MIP_test/{fname}/recon": mip_recon,
                f"MIP_test/{fname}/CS": mip_ref_cs,
            }
            log_images(trainer, mips)

            # Save reconstructions and any additional outputs for this subject.
            # For memory efficiency, consider processing and saving each subject individually.
            save_recon_single(fname, output_img, save_path)
            
    def teardown(self, trainer, pl_module, stage):
        # Safely close HDF5 file if it's still open
        if hasattr(self, "h5_file_handle") and self.h5_file_handle is not None:
            try:
                self.h5_file_handle.close()
                print(f"[RANK {trainer.global_rank}] HDF5 file {self.h5_path} closed in teardown (TestImageLogger).")
            except Exception as e:
                log.error(
                    f"[RANK {trainer.global_rank}] Error closing HDF5 file {self.h5_path} in teardown (TestImageLogger): {e}"
                )
            self.h5_file_handle = None

        # Remove the temporary directory
        if hasattr(self, "temp_dir") and Path(self.temp_dir).exists():
            try:
                shutil.rmtree(self.temp_dir)
                print(
                    f"[RANK {trainer.global_rank}] Temporary directory {self.temp_dir} (TestImageLogger) has been removed."
                )
            except Exception as e:
                log.error(
                    f"[RANK {trainer.global_rank}] Error removing temporary directory {self.temp_dir} (TestImageLogger): {e}"
                )