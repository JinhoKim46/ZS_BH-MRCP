"""
Copyright (c) Facebook, Inc. and its affiliates.
Copyright (c) Marc Vornehm <marc.vornehm@fau.de>.
Copyright (c) Jinho Kim <jinho.kim@fau.de>.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import datetime
import os
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from shutil import copy2, copytree
from typing import Dict, Union

import h5py
import numpy as np
import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar
from pytorch_lightning.cli import LightningArgumentParser

from .. import complex_abs

SCH = {"StepLR": "STEP", "CosineAnnealingLR": "CoAn", "CosineAnnealingWarmRestarts": "CAWR"}

class TQDMProgressBarWithoutVersion(TQDMProgressBar):
    """
    Progress bar that does not display a version number
    """

    def get_metrics(self, trainer, pl_module):
        metrics = super().get_metrics(trainer, pl_module)
        if "v_num" in metrics:
            del metrics["v_num"]
        return metrics


def log_image(trainer: pl.Trainer, name, image):
    if image.ndim == 2:
        image = image[None]  # [channel, y, x]
    trainer.logger.experiment.add_image(name, image, global_step=trainer.global_step)


def log_images(trainer: pl.Trainer, images: dict, one_time=False):
    if one_time and trainer.current_epoch > 1:
        return

    for name, image in images.items():
        # Save two log images for the ref cs (due to the progress bar in tensorboard.)
        if trainer.current_epoch > 1 and name in ["CS", "acquired_k"]:
            continue

        log_image(trainer, name, image)


def normalize(img):
    img = np.abs(img)
    return img / img.max()

def save_recon(reconstructions: Dict[str, np.ndarray], out_dir: Path):
    """
    Save reconstruction images.

    This function saves reconstructed images to the out_dir

    Args:
        reconstructions: A dictionary mapping input filenames to corresponding
            reconstructions.
        out_dir: Path to the output directory where the reconstructions should
            be saved.
    """
    npys_path = out_dir / "npys"

    npys_path.mkdir(exist_ok=True, parents=True)
    for fname, recons in reconstructions.items():
        file_dir = npys_path / fname
        file_dir.mkdir(exist_ok=True, parents=True)
        npy_path = file_dir / f"{fname}.npy"
        with open(npy_path, "wb") as f:
            np.save(f, recons)


def save_recon_single(fname: str, recon: np.ndarray, out_dir: Path):
    """
    Save reconstruction images.

    This function saves reconstructed images to the out_dir

    Args:
        reconstructions: A dictionary mapping input filenames to corresponding
            reconstructions.
        out_dir: Path to the output directory where the reconstructions should
            be saved.
    """
    file_dir = out_dir / "npys" / fname
    file_dir.mkdir(exist_ok=True, parents=True)
    npy_path = file_dir / f"{fname}.npy"
    with open(npy_path, "wb") as f:
        np.save(f, recon)

def collect_h5_contents(h5_files):
    collected_data = defaultdict(dict)

    for h5_file_path_str in h5_files:
        h5_file_path = Path(h5_file_path_str)
        if not h5_file_path.exists():
            print(f"Warning: HDF5 file {h5_file_path} not found. Skipping.")  # Consider using logging
            continue
        try:
            with h5py.File(h5_file_path, "r") as hf:
                for fname_group_name in hf.keys():
                    # Ensure that data for the same fname from different ranks are merged correctly.
                    # If fname_group_name is already in collected_data, new slice data will be added to its dict.
                    # If it's a new fname_group_name, a new entry in collected_data will be created.
                    group = hf[fname_group_name]
                    for key in group.keys():
                        # It's possible for different ranks to try to write the same key if not careful,
                        # but the current structure (output_img_XXX) should be unique per slice.
                        collected_data[fname_group_name][key] = group[key][...]
        except Exception as e:
            print(f"Error reading HDF5 file {h5_file_path}: {e}")  # Consider using logging
            # Decide if you want to continue or re-raise
            continue  # Skip problematic files

    # clear_h5_contents is problematic in a multi-epoch scenario if files are reused.
    # It should only be called if these specific h5_files are truly temporary and won't be needed again
    # in their current state by any process. Given the new epoch-based truncation, this might be okay
    # if called *after* all ranks are done with their files for the epoch and rank 0 has processed them.
    # However, it's safer to let the teardown method handle the deletion of the entire temp directory.
    # clear_h5_contents(h5_files)
    return collected_data

def clear_h5_contents(h5_files):
    print("Clearing h5 contents...")
    # Delete all top-level items in the file
    for h5_file_path_str in h5_files:
        h5_file_path = Path(h5_file_path_str)
        if not h5_file_path.exists():
            print(f"Warning: HDF5 file {h5_file_path} not found during clear. Skipping.")  # Consider using logging
            continue
        try:
            # Opening in 'w' mode truncates the file.
            with h5py.File(h5_file_path, "w") as hf:
                pass  # File is now empty
            print(f"Cleared HDF5 file: {h5_file_path}")
        except Exception as e:
            print(f"Error clearing HDF5 file {h5_file_path}: {e}")  # Consider using logging
            continue

def save_outputs_temp(outputs: dict, h5_file_handle_or_path: Union[Path, h5py.File]):
    """
    Save output dictionary to an HDF5 file.

    Args:
        outputs: Dictionary containing data to save.
                 Expected keys: "fname", "readout_index", "output_img", "target_img".
        h5_file_handle_or_path: Either a Path object to the HDF5 file (will be opened in append mode)
                                or an already open h5py.File object (file handle).
    """
    # For memory efficiency, save result data in the temporary h5 file.
    if isinstance(h5_file_handle_or_path, Path):
        # If a path is given, open the file in append mode
        h5_path = h5_file_handle_or_path
        try:
            with h5py.File(h5_path, "a") as h5_file:
                _save_to_h5_handle(outputs, h5_file)
        except Exception as e:
            # It's good to log the specific path that failed
            print(f"Error writing to HDF5 file {h5_path}: {e}")  # Consider using logging
            # Optionally re-raise or handle as appropriate
            raise
    elif isinstance(h5_file_handle_or_path, h5py.File):
        # If a file handle is given, use it directly
        h5_file = h5_file_handle_or_path
        try:
            _save_to_h5_handle(outputs, h5_file)
        except Exception as e:
            # It's good to log the file name if available from the handle
            print(f"Error writing to HDF5 file {h5_file.filename}: {e}")  # Consider using logging
            # Optionally re-raise or handle as appropriate
            raise
    else:
        raise TypeError("h5_file_handle_or_path must be a Path object or an h5py.File object.")


def _save_to_h5_handle(outputs: dict, h5_file: h5py.File):
    """Helper function to save outputs using an open h5py.File handle."""
    for i, (fname, readout_index) in enumerate(zip(outputs["fname"], outputs["readout_index"])):
        base_name = fname[:-3]
        slice_index = readout_index.item()  # convert to string for naming

        # Create a group for each file if it doesn't exist
        if base_name not in h5_file:
            grp = h5_file.create_group(base_name)
        else:
            grp = h5_file[base_name]

        # Save each array as a dataset; naming datasets by their slice number
        dataset_name_output = f"output_img_{slice_index:0>3}"
        dataset_name_ref_cs_img = f"ref_cs_img_{slice_index:0>3}"

        for dataset_name in [dataset_name_output, dataset_name_ref_cs_img]:
            if dataset_name in grp:
                del grp[dataset_name]  # Delete existing dataset

        grp.create_dataset(dataset_name_output, data=np.abs(outputs["output"][i].cpu().numpy()))
        grp.create_dataset(dataset_name_ref_cs_img, data=np.abs(outputs["ref_cs"][i].cpu().numpy()))



def read_path_yaml(target: str):
    yaml_path = Path.cwd() / "configs/paths.yaml"  # Relative path based on main.py

    if yaml_path.exists():
        with open(yaml_path, "r") as f:
            yaml_dict = yaml.safe_load(f)

        target_path = yaml_dict.get(target, None)
        
        if target_path is not None:
            target_path = Path(target_path)
        
    else:
        raise FileNotFoundError(f"Path YAML file not found at {yaml_path}. Please ensure it exists.")

    return target_path

def set_name(c):
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{date_str}_zs_{c.model.zs_mode}_bh_mrcp"

    return run_name


def save_file_dump(c, subcommand):
    save_path = c.trainer.default_root_dir / "script_dump"
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)
        

    version = "train" if subcommand in ["fit", "validate"] else "test"

    save_path = save_path / version

    dirs = ["configs", "zs_bh_mrcp"]
    for dir in dirs:
        try:
            copytree(dir, os.path.join(save_path, dir))
        except:
            pass
    files = ["main.py"]
    for file in files:
        try:
            copy2(file, os.path.join(save_path, file))
        except:
            print(f"{file} does not exist.")


def add_main_arguments(parser: Union[ArgumentParser, LightningArgumentParser]):
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Experiment name. If --optuna is given, this is the study name and the experiment "
        "names will be composed of the study name and a consecutive number.",
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        default=None,
        help="The path to the training data directory. By default, it is set in the paths.yaml file. Set it manually by passing --data_path parameter at run.",
    )
    parser.add_argument(
        "--log_path",
        type=Path,
        default=None,
        help="The path to the log directory. By default, it is set in the paths.yaml file. Set it manually by passing --log_path parameter at run.",
    )



def device_check(devices: Union[str, int]):
    if devices == "auto":
        return True
    elif devices == "-1":
        return True
    elif isinstance(devices, str):
        devices = [i for i in devices.split(",") if i != ""]
        return True if len(devices) > 1 else False
    else:  # isinstance(devices, int)
        return True if int > 1 else False


def print_training_info(c):
    Model_PRINT = f"""    - NUM_STAGES:           {c.model.zs_params["num_stages"]}
    - NUM_RESBLOCKS:        {c.model.zs_params["num_resblocks"]}
    - CHANS:                {c.model.zs_params["chans"]}
    - CGDC_ITER:            {c.model.zs_params["cgdc_iter"]}
    - BACKBONE:             {c.model.zs_params["backbone"]}"""

    print(
        f"""
================
HYPERPARAMETERS
================
- NAME:                     {c.name}
- Zero-shot mode:           {c.model.zs_mode}
- Trainer
    - MAX_EPOCHS:           {c.trainer.max_epochs}
- Model
{Model_PRINT}
    - LR:                   {c.model.lr}
    - SCHEDULER:            {c.model.lr_scheduler}
- Data
    - IS_PROTOTYPE:         {c.data.is_prototype}
- Path
    - DATA_PATH:            {c.data_path}
    - LOG_PATH:             {c.log_path}
"""
    )
