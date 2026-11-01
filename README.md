# LandFormer

Code for **LandFormer: A Swin Transformer Surrogate for Multi-Output Land Surface Modelling at Continental Scale**
(Siddik Barbhuiya and Vivek Gupta, Indian Institute of Technology Mandi).

LandFormer emulates the Variable Infiltration Capacity model (VIC 4.2.d) over the Indian subcontinent at 0.0625°
(about 6 km). For each grid cell it predicts daily evapotranspiration (ET), total runoff (surface runoff plus baseflow,
Q) and the volumetric water content of the top VIC soil layer (SM) from a 270-day window of ERA5 meteorological
forcing and 13 VIC soil parameters (`infilt`, `Dsmax`, `Ds`, `Ws`, `d1`–`d3`, `Ksat1`–`Ksat3`, `expt1`–`expt3`).

The model combines a hierarchical temporal Swin Transformer encoder, an optional spatial graph attention module
that connects neighbouring cells of the same day and parameter set, and dual-path regression heads for the three
outputs. It is trained with a multi-task loss based on the Kling–Gupta efficiency. The autoregressive configurations
additionally receive the outputs of the previous day as inputs.

## Contents

```
main.py                       training
run_inference.py              evaluation on the test configuration (per-cell metrics and daily series)
landformer_paths.py           data locations
src/                          model, data pipeline, training and metrics
  models/swin_transformer.py      temporal Swin Transformer encoder and regression heads
  models/spatial_attention.py     spatial graph attention module
  data/                           dataset construction, normalisation, spatial splitting
  training/                       trainer, losses, grid-aware samplers
configs/models/               configurations of the models reported in the paper
configs/evaluation/           test configurations
experiments/                  ensemble evaluation for the coverage and resolution experiments
analysis/                     analyses of Sections 4.4-4.8 and 5
figures/                      figure scripts
```

## Installation

Python 3.13 and a CUDA-capable GPU are recommended. Install PyTorch for your CUDA version first
(https://pytorch.org), then

```bash
pip install -r requirements.txt
```

The models of the paper were trained and evaluated on a single NVIDIA RTX A5000 (24 GB).

## Data

The processed dataset is available at https://doi.org/10.5281/zenodo.19667154. It contains the ERA5 forcing on the
0.0625° grid, the 150 VIC parameter sets and the corresponding VIC outputs. ERA5 is distributed by the Copernicus
Climate Data Store, the baseline VIC parameters by Schaperow et al. (2021), and VIC 4.2.d at
https://github.com/UW-Hydro/VIC.

Expected layout (default locations, relative to the repository root):

```
data/Surr_forcing/forcing.zarr        daily forcing (prcp, tasmax, tasmin, vp, swdown, lwdown, sp, wind)
data/Surr_forcing/pram_150.nc         150 VIC parameter sets
data/Surr_obs/evap.zarr               VIC evapotranspiration
data/Surr_obs/SM.zarr                 VIC soil moisture, layer 1
data/Surr_obs/Q_real.zarr             VIC total runoff
results/                              training runs (created by main.py)
```

Data in other locations can be used by setting environment variables, which are read by `landformer_paths.py`:

| Variable | Default | Used for |
|---|---|---|
| `LANDFORMER_DATA` | `./data` | forcing, parameters and VIC outputs |
| `LANDFORMER_RESULTS` | `./results` | trained models, normalisation statistics, cell splits |
| `LANDFORMER_EXT` | `./data_ext` | Köppen–Geiger classes, national boundaries, coarse-resolution forcing |
| `LANDFORMER_VICFLOW` | `./data_vicflow` | gauge catchments used in Section 4.8 |

The YAML files refer to `data/...` and `results/` relative to the working directory; run all commands from the
repository root, or edit the `paths` and `output` sections of the configuration.

## Models

| Configuration (`configs/models/`) | Model | Section |
|---|---|---|
| `LandFormer-S_12.5M.yaml`, `LandFormer-B_61.5M.yaml`, `LandFormer-NAR.yaml` | 12.5 M, 61.5 M and 109 M parameters, no lagged inputs | 4.2 |
| `LandFormer-AR-ET.yaml`, `LandFormer-AR-Q.yaml`, `LandFormer-AR-SM.yaml` | Autoregressive, one lagged output | 4.4 |
| `LandFormer-AR-All.yaml` | Autoregressive reference configuration | 4.4–4.8 |
| `LandFormer-AR-All_GAT.yaml` | Reference configuration with spatial graph attention | 4.7 |
| `LandFormer-AR_1deg.yaml`, `LandFormer-AR_0.5deg.yaml`, `LandFormer-AR_0.25deg.yaml` | Trained at coarser resolution | 4.5 |
| `LSTM.yaml` | LSTM benchmark | 4.1 |

These are the configurations recorded by the trainer at training time; only machine-specific paths and restart
settings were removed. The `experiment_name` of each file is the name of its folder in `results/` (for example
`Large_full_lag` for LandFormer-AR-All and `TEST_HYDRODL_4` for LandFormer-NAR), which the analysis scripts use.

All models are trained on years before 2001; the training and validation periods are set in each configuration
(`data.temporal`; 1990–1995 and 1996–2000 for the reference configuration). Evaluation uses the independent
period 2001–2006, on held-out grid cells and a parameter set not used in training.

## Training

```bash
python main.py --config configs/models/LandFormer-AR-All.yaml
```

The checkpoints (`models/best_model.pth`), the normalisation statistics (`train_data_scaler.bin`), the cell splits
(`grid_locations_*.pkl`) and the training log are written to `results/<experiment_name>/`. To train
LandFormer-AR-All_GAT on the same cells as LandFormer-AR-All, copy the training and validation cell lists as described
at the top of its configuration file.

## Evaluation

```bash
python run_inference.py --config configs/evaluation/test.yml \
    --checkpoint results/Large_full_lag/models/best_model.pth --output_dir inference_output
```

Predictions start on 27 September 2001, the first day with a complete 270-day input window, and end on
31 December 2006. Outputs, named after the run folder:

```
inference_output/grid_metrics_<run>.csv          KGE, NSE, correlation, RMSE, bias and variability ratio per cell
inference_output/time_series/grid_<cell>_<run>.csv   daily predictions (pred_*) and VIC values (target_*)
```

Run this for `Large_full_lag`, `TEST_HYDRODL`, `TEST_HYDRODL_3`, `TEST_HYDRODL_4` and `Lagged_E/Q/SM` before the
analyses below. The ensembles of the training-coverage and training-resolution experiments are evaluated with
`experiments/ensemble_coverage.py` and `experiments/ensemble_resolution.py` (checkpoint lists at the top of each
script; outputs in `outputs/`).

## Analyses

| Section | Script | Description |
|---|---|---|
| 4.4 | `analysis/closed_loop/closed_loop.py` | Closed-loop (free-running) evaluation over the test period; open loop, LandFormer-NAR and persistence on the same cells |
| 4.4 | `analysis/closed_loop/error_growth.py` | Errors as a function of lead time for runs of 365 days started every 30 days |
| 4.4, 4.6, 4.7 | `analysis/consistency_persistence_gat.ipynb` | Persistence benchmark; cross-variable consistency (partitioning of precipitation, ET–soil moisture relation, runoff and antecedent wetness, timing, coupling, zero predictions); comparison with and without graph attention |
| 4.6 | `analysis/closed_loop/precipitation_partitioning.py` | Partitioning of precipitation for VIC, open loop, closed loop and LandFormer-NAR |
| 4.7 | `analysis/graph_attention/gat_test_inference.py` | Test inference of the model with graph attention |
| 4.8 | `analysis/flow_regimes/flow_regimes.py` | Skill by flow regime and for the Indus, Ganges and Brahmaputra |
| 5 | `analysis/spatial_independence/distance_to_training.py` | Test skill against the distance to the nearest training cell |
| 5 | `analysis/computational_cost/inference_benchmark.py` | Inference throughput and GPU memory |

Order of execution:

```bash
cd analysis/closed_loop
python closed_loop.py all          # creates the test-set cache and the daily arrays used by the other scripts
python error_growth.py 30
python soil_moisture_bias.py
python precipitation_partitioning.py
cd ../graph_attention && python gat_test_inference.py ../../results/Large_full_lag_GAT/models/best_model.pth GAT
cd ../flow_regimes && python flow_regimes.py && python basin_runoff_figure.py
cd ../spatial_independence && python distance_to_training.py
cd ../computational_cost && python inference_benchmark.py
```

The notebook `analysis/consistency_persistence_gat.ipynb` is run after these steps from its own folder. Each script
writes its results next to itself. The basin analysis needs the HydroBASINS v1c levels 3 and 4 for Asia
(https://www.hydrosheds.org) in `analysis/flow_regimes/hydrobasins/`.

## Figures

| Figure | Source |
|---|---|
| 2, 3 | `figures/architecture_diagrams.py`, `figures/architecture_overview.py` |
| 4 | `figures/fig04_data/*.py`, then `figures/fig04_study_area.py` (needs Natural Earth 1:10m river centrelines, GMTED2010 tiles and the Köppen–Geiger classes) |
| 5–8, 10, 11 | `figures/fig05_to_fig11.ipynb` (the LSTM and LandFormer values of Figure 5 are entered in the notebook) |
| 9 | `figures/fig09_closed_loop.py` |
| 12 | `analysis/consistency_persistence_gat.ipynb` |
| 13 | `analysis/flow_regimes/basin_runoff_figure.py` |

## Citation

Barbhuiya, S., and Gupta, V. LandFormer: A Swin Transformer Surrogate for Multi-Output Land Surface Modelling at
Continental Scale. The full reference will be added on publication.

## Contact

Siddik Barbhuiya, School of Civil and Environmental Engineering, Indian Institute of Technology Mandi.
