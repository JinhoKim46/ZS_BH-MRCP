"""
Copyright (c) Jinho Kim <jinho.kim@fau.de>.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import multiprocessing
import os

from zs_bh_mrcp.pl import cli
from zs_bh_mrcp.pl import utils as pl_utils

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
multiprocessing.set_start_method("spawn", force=True)

def run_cli():
    # base argument parser
    parser = argparse.ArgumentParser(add_help=False)
    pl_utils.add_main_arguments(parser)

    args = parser.parse_known_args()[0]

    _ = cli.DLReconCLI(name=args.name)

if __name__ == "__main__":
    run_cli()
