# Zero-shot self-supervised learning for breath-hold magnetic resonance cholangiopancreatography (MRCP) reconstruction
This repository is for zero-shot self-supervised learning reconstruction to reduce breath-hold times in magnetic resonance cholangiopancreatography (MRCP). 


## Installation
1. Clone the repository then navigate to the `ZS_BH-MRCP` root directory.
```sh
git clone git@github.com:JinhoKim46/ZS_BH-MRCP.git
cd ZS_BH-MRCP
```
2. Create a new conda environment
```sh
conda create -n zs_bh_mrcp python=3.10.14
```
3. Activate the conda environment
```sh
conda activate zs_bh_mrcp
```
4. Install pytorch
```sh
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
```
5. Install a requirement packages
```sh
pip install -r requirements.txt 
```

## Usage
### Data Preparation
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16731625.svg)](https://doi.org/10.5281/zenodo.16731625)

- A single breath-hold MRCP data is stored in the `HDF5` format containing the following structures:
  - Datasets:
    - `kdata`: Acquired k-space data decoubled along the readout direction using Fourier transform (nCoil $\times$ Phase Encoding $\times$ Readout (decoupled) $\times$ Partition Encoding)
    - `sm_espirit`: ESPIRiT-based sensitivity maps (nCoil $\times$ Phase Encoding $\times$ Readout (decoupled) $\times$ Partition Encoding)
    - `cs`: L1-wavelet compressed sensing reconstruction (Oversampled-Cropped Phase Encoding $\times$ Readout $\times$ Partition Encoding)
  - Attribute
    - `recon_crop_size`: The size of the reconstructed image cropping over-sampled in the phase encoding direction.
- The `sample_data` directory contains sample breath-hold MRCP data zero-shot learning. You can find the data [here](https://doi.org/10.5281/zenodo.16731625). Additional information, i.e., header, is ignored in the sample data. 

### Run
#### Train
-  Define the training configurations in the `configs/config.yaml` file.
- Run `main.py` by
```sh
python main.py fit --config configs/config.yaml
```
- You can define the run name by adding the `--name` argument at run. Unless you define the run name, the run name is set to `%Y%m%d_%H%M%S__zs_{zs_mode}_bh_mrcp`. 
  ```sh
  python main.py fit --config configs/config.yaml --name test_run
  ```
-  You can overwrite the configurations in the `configs/config.yaml` file by adding arguments at run. 
  ```sh
  python main.py fit --config configs/config.yaml --model.zs_mode shallow
  ```
- Log files containing `checkpoints/`, `lightning_logs/`, and `script_dump/` are stored in `log_path/run_name`. `log_path` is defined in the `configs/paths.yaml` file.
- You can resume the training by giving `run_name` with `fit` command. `*.ckpt` file should be placed in `log_path/run_name/checkpoints/` to resume the model.
  ```sh
  python main.py fit --config configs/config.yaml --name run_name
  ```
#### Test
- Run `main.py` with `run_name` by
```sh
python main.py test --config configs/config.yaml --name run_name
```
- `*.ckpt` file should be placed in `run_name/checkpoints/` to test the model.
- The output files are saved in `log_path/run_name/npys/FILENAME`.
#### Easy run
We offer an easy run script, `run_zs_bh_mrcp.sh` to run the training and testing. You can define the configurations directly in the script. You can run the script by 
```sh
source run_zs_bh_mrcp.sh
```

# Cite 
This work has been submitted for publication and is currently under revision. Until the peer-reviewed version is available, please cite the preprint on arXiv.

Kim, J., Nickel, M. D., & Knoll, F. (2025). Zero‑shot self‑supervised learning of single breath‑hold magnetic resonance cholangiopancreatography (MRCP) reconstruction. arXiv. https://doi.org/10.48550/arXiv.2508.09200

