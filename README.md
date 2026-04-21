# LandFormer

Research-grade surrogate modeling pipeline for VIC-style hydrological simulation. This repository trains Swin Transformer models on gridded meteorological forcing, static land-surface parameters, and optional lag features to predict evapotranspiration, soil moisture, and runoff.

Author: Siddik  
Contact: siddikbarbhuiya@gmail.com

## Overview

The default workflow in this repository is built around a multi-task Swin Transformer that learns from spatially gridded hydro-meteorological data. The training pipeline is designed for practical experimentation on large datasets and includes spatially disjoint grid splits, hydrology-specific loss functions, physics-aware penalties, mixed precision training on CUDA, checkpointing, TensorBoard logging, and automatic grid analysis exports.

This codebase is aimed at surrogate modeling for hydrological applications where spatial generalization matters. Instead of relying only on random train/validation/test splits, the trainer can sample different grid cells for each split so model performance is measured on unseen locations.

## What This Repository Provides

- Swin Transformer backbones for single-task and multi-task hydrological regression.
- Multi-output prediction for `evap`, `sm`, and `Q`.
- Grid-aware train/validation/test splitting for spatial generalization studies.
- Hydrology-native loss and evaluation support, including NSE and KGE.
- Optional physics-informed penalties for non-negative hydrological outputs.
- CUDA-aware AMP training, gradient clipping, learning-rate scheduling, and memory tuning.
- Automatic logging, checkpoint saving, configuration backup, normalization-stat export, and grid reports.

## Model and Data Flow

```text
Forcing sequences (7 variables, 270 time steps)
        +
Static VIC parameters (13 variables)
        +
Optional lag features (default: precipitation lags)
        |
        v
Patch embedding -> Swin Transformer backbone -> shared feature representation
        |
        +--> single-task head
        |
        +--> multi-task regression heads for evap, sm, Q
```

The default configuration uses `multi_task_swin`, where a shared Swin backbone is followed by separate regression heads for each target variable.

## Repository Layout

```text
surro_final_swin/
├── main.py
├── config_swin.yaml
├── QUICKSTART.md
├── requirements.txt
├── src/
│   ├── core/
│   │   ├── config.py
│   │   └── validator.py
│   ├── data/
│   │   ├── dataset.py
│   │   ├── grid_splitter.py
│   │   ├── memory_efficient_loader.py
│   │   ├── multi_input_dataset.py
│   │   └── utils.py
│   ├── models/
│   │   ├── losses.py
│   │   ├── physics_constraints.py
│   │   └── swin_transformer.py
│   ├── training/
│   │   └── swin_trainer.py
│   └── utils/
│       ├── hydro_metrics.py
│       └── memory.py
└── README.md
```

## Data Contract

### Inputs

The default configuration expects three data groups:

1. Forcing data in Zarr format.
2. Static parameter data in NetCDF format.
3. Target data in separate Zarr stores for each output variable.

### Default forcing variables

- `prcp`
- `tmax`
- `tmin`
- `vp`
- `swdown`
- `lwdown`
- `sp`

### Default parameter variables

- `infilt`
- `Dsmax`
- `Ds`
- `Ws`
- `d1`
- `d2`
- `d3`
- `Ksat1`
- `Ksat2`
- `Ksat3`
- `expt1`
- `expt2`
- `expt3`

### Default targets

- `evap` for evapotranspiration
- `sm` for soil moisture
- `Q` for runoff or discharge

### Sequence and feature settings

- Sequence length: `270`
- Default lag features: precipitation lags at `1, 2, 3, 7, 30, 90, 180`
- Input feature groups: forcing + parameters + optional lag features
- Normalization methods supported by the dataset: `zscore` and `minmax`

Training-set normalization statistics are saved automatically to `train_data_scaler.bin` inside the experiment output directory and reused by validation and test datasets.

## Installation

Create a Python environment, install a PyTorch build appropriate for your system, and then install the repository requirements.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you are training on GPU, make sure the installed PyTorch build matches your CUDA runtime.

## Quick Start

### 1. Update the configuration

Edit [config_swin.yaml](config_swin.yaml) and set the paths for your local datasets.

```yaml
model:
  model_type: 'multi_task_swin'
  sequence_length: 270
  patch_size: 10
  window_size: 12

data:
  paths:
    forcing: /path/to/forcing.zarr
    parameters: /path/to/parameters.nc
    target_files:
      evap: /path/to/evap.zarr
      sm: /path/to/sm.zarr
      Q: /path/to/Q.zarr

training:
  epochs: 40
  batch_size: 100
  loss_function: NSE

hardware:
  device: cuda:0
```

At minimum, review these sections before your first run:

- `data.paths`
- `data.grid_sampling`
- `model`
- `training`
- `output`
- `hardware`

### 2. Validate the configuration

```bash
python main.py --config config_swin.yaml --validate
```

### 3. Inspect the runtime environment

```bash
python main.py --info
```

### 4. Start training

```bash
python main.py --config config_swin.yaml
```

### 5. Resume training if needed

```bash
python main.py --config config_swin.yaml --resume
python main.py --config config_swin.yaml --checkpoint output/swin_surrogate/models/model_epoch_10.pth
```

When `--resume` is used without `--checkpoint`, the main script first looks for `best_model.pth` and then falls back to the latest `model_epoch_*.pth` checkpoint.

## Command Reference

```bash
python main.py --config config_swin.yaml
python main.py --config config_swin.yaml --resume
python main.py --config config_swin.yaml --checkpoint path/to/model.pth
python main.py --config config_swin.yaml --validate
python main.py --info
```

## Configuration Guide

### Model

Key model parameters live under `model` in [config_swin.yaml](config_swin.yaml):

- `model_type`: `swin` or `multi_task_swin`
- `embed_dim`: embedding width for the Swin backbone
- `patch_size`: temporal patch size for 1D patch embedding
- `depths`: number of blocks per Swin stage
- `num_heads`: attention heads per stage
- `window_size`: local attention window size
- `dropout` and `drop_path_rate`: regularization controls
- `sequence_length`: input time-window length
- `enable_lags` and `lag_config`: lag feature generation

### Data

Important data controls:

- `data.spatial`: geographic subsetting bounds
- `data.temporal`: date ranges for train, valid, and test
- `data.file_indices`: scenario index ranges per split
- `data.grid_sampling`: exact grid counts or percentage-driven sampling
- `data.variables`: forcing, parameter, and target variable lists
- `data.processing.normalization`: `zscore` or `minmax`

### Training

Important training controls:

- `epochs`, `batch_size`, `batch_size_valid`, `learning_rate`
- `loss_function`: `NSE`, `KGE`, `MSE`, or `MAE`
- `patience`: early stopping threshold
- `epoch_save`: periodic checkpoint frequency
- `multi_output.loss_weights`: target weighting for multi-task training
- `physical_constraints`: optional non-negative output penalty
- `scheduler`: cosine scheduling with warmup support
- `optimizations.use_amp`: mixed precision training on CUDA
- `optimizations.gradient_clipping`: gradient clipping threshold
- `memory`: DataLoader worker and transfer settings

### Output and logging

Important output controls:

- `output.base_dir`
- `output.experiment_name`
- `output.create_plots`
- `logging.tensorboard`
- `logging.level`
- `logging.log_file`

## Training Workflow

The runtime flow in [main.py](main.py) and [src/training/swin_trainer.py](src/training/swin_trainer.py) is:

1. Load YAML configuration and validate it.
2. Set reproducible seeds from `evaluation.grid_split_seed`.
3. Create grid-aware train, validation, and optional test splits.
4. Build PyTorch datasets and DataLoaders.
5. Initialize the selected Swin model, loss function, optimizer, and scheduler.
6. Train and validate for each epoch while tracking hydrological metrics.
7. Save the best model and periodic checkpoints.
8. Write training history and grid-location analysis outputs.
9. Run final test evaluation when a test split is configured.

## Metrics and Constraints

### Supported losses and metrics

The repository includes hydrology-aware losses and metrics in [src/models/losses.py](src/models/losses.py) and [src/utils/hydro_metrics.py](src/utils/hydro_metrics.py):

- Nash-Sutcliffe Efficiency (NSE)
- Kling-Gupta Efficiency (KGE)
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Pearson correlation
- Percent bias
- RMSE

### Physics-informed constraint behavior

The current physical-constraint implementation in [src/models/physics_constraints.py](src/models/physics_constraints.py) is centered on non-negative output enforcement through a mass-conservation-style penalty. In practice, this means the default documented physical constraint is a penalty on negative predicted values for hydrological outputs.

## Output Structure

By default, experiments are written to:

```text
output/swin_surrogate/
```

A typical run creates files like these:

```text
output/swin_surrogate/
├── config_backup.yaml
├── grid_splits.pkl
├── grid_locations_train.csv
├── grid_locations_train.nc
├── grid_locations_train.pkl
├── grid_locations_valid.csv
├── grid_locations_valid.nc
├── grid_locations_valid.pkl
├── grid_locations_test.csv         # when test data is configured
├── grid_locations_test.nc          # when test data is configured
├── grid_locations_test.pkl         # when test data is configured
├── grid_analysis_report.txt        # when grid analysis succeeds
├── grid_summary_table.csv          # when grid analysis succeeds
├── train_data_scaler.bin
├── training.log
├── training_history.json
├── models/
│   ├── best_model.pth
│   └── model_epoch_*.pth
├── tensorboard/
└── figures/
    ├── selected_grids_train.png
    ├── selected_grids_valid.png
    ├── selected_grids_test.png
    └── additional spatial analysis plots
```

The trainer also attempts to create consolidated grid-analysis figures and summary tables after training completes.

To inspect TensorBoard logs:

```bash
tensorboard --logdir output/swin_surrogate/tensorboard
```

## Practical Notes

- AMP is enabled only when `optimizations.use_amp: true` and CUDA is available.
- The default config uses exact grid counts with non-overlapping train, validation, and test selections.
- Validation and test normalization reuse the scaler generated during training.
- Early stopping is controlled by `training.patience`.
- Checkpoint frequency is controlled by `training.epoch_save`.

## Recommended First Edits

Before running a new experiment, update:

- dataset paths in [config_swin.yaml](config_swin.yaml)
- device selection in [config_swin.yaml](config_swin.yaml)
- grid counts in [config_swin.yaml](config_swin.yaml)
- output experiment name in [config_swin.yaml](config_swin.yaml)
- loss function and learning rate in [config_swin.yaml](config_swin.yaml)


