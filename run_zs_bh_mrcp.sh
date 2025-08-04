#!/bin/bash

# This script is used to run the zero-shot learning model for MRCP reconstruction.
# It sets up the environment, defines hyperparameters, and runs the training and testing stages.
# Make sure to adjust the paths and parameters according to your setup.
# Default parameters are defined in configs/config.yaml.

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# @                  DEFINE HYPERPARAMETERS                 @
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
##@ Trainer
STAGES=(fit test) # By giving fit and test, it runs both stages.
MAX_EPOCHS=3
DEVICE=0, # Recommend to use a single GPU.

##@ Model
LR=0.0003
LR_SCHEDULER=CosineAnnealingLR  
ZS_MODE=shallow # [deep, shallow]
ZS_NUM_STAGES=12 # For the shallow mode, this is set to zero automatically.
ZS_NUM_RESBLOCKS=8
ZS_CHANS=64
ZS_CGDC_ITER=4
ZS_BACKBONE=backbone # For shallow zero-shot learning

##@ Data
IS_PROTOTYPE=False
NUM_WORKERS=16
TRAINING_DATA_FNAME=data01 # Training data

##@ Transform
SSDU_MASK_RHO_LAMBDA=0.4
SSDU_MASK_RHO_GAMMA=0.2

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# @                  SETTING RUNNING NAME                    @
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
note="" # <====== Additional note about this run.
# Run
arr=(
    "$(date +"%Y%m%d_%H%M%S")"
    "${ZS_MODE}_ZS"
    "${DATA_SELECTION}"
    "${note}"
)

# Filter out empty strings
filtered=()
for item in "${arr[@]}"; do
  [[ -n "$item" ]] && filtered+=("$item")
done

NAME=$(printf "%s_" "${filtered[@]}")
NAME=${NAME%_}  # Trim trailing underscore

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# @           SETTING RUNNING ENVIRONMENT VARIABLES          @
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
LOG_DIR=./logs # Define the log directory. By default, ./logs defined in paths.yaml
DATA_DIR=./sample_data # Define the data directory. By default, ./data defined in paths.yaml

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# @         LOAD MODULES AND AVTIVATE CONDA ENV             @
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Initialize conda for the current shell
eval "$(conda shell.bash hook)"

# Activate the Conda environment
conda activate zs_bh_mrcp

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# @                    RUN PYTHON SCRIPTS                    @
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#
echo "================"
echo "RUNNING SCRIPT"
echo "================"
current_dir=$(pwd)
for stage in $STAGES
do
    echo "  - Running stage: $stage"
    python "$current_dir/main.py" $stage \
    --config configs/config.yaml \
    --name $NAME \
    --trainer.max_epochs $MAX_EPOCHS \
    --trainer.devices $DEVICE \
    --model.lr $LR \
    --model.lr_scheduler $LR_SCHEDULER \
    --model.zs_mode $ZS_MODE \
    --model.zs_params.num_stages $ZS_NUM_STAGES \
    --model.zs_params.num_resblocks $ZS_NUM_RESBLOCKS \
    --model.zs_params.chans $ZS_CHANS \
    --model.zs_params.cgdc_iter $ZS_CGDC_ITER \
    --model.zs_params.backbone $ZS_BACKBONE \
    --data.is_prototype $IS_PROTOTYPE \
    --data.num_workers $NUM_WORKERS \
    --data.training_data_fname $TRAINING_DATA_FNAME \
    --transform.ssdu_mask_rho_lambda $SSDU_MASK_RHO_LAMBDA \
    --transform.ssdu_mask_rho_gamma $SSDU_MASK_RHO_GAMMA \
    --data_path $DATA_DIR \
    --log_path $LOG_DIR
done