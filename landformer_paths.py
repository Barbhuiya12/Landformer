"""Central data locations for LandFormer.

All scripts read their input and output locations from here. Set the environment variables to point to your copies of
the data, or create the default sub-directories next to this file:

    LANDFORMER_DATA      forcing, VIC parameters and VIC outputs (Surr_forcing/, Surr_obs/)   default: ./data
    LANDFORMER_RESULTS   trained models, normalisation statistics and cell splits           default: ./results
    LANDFORMER_EXT       Koppen-Geiger classes, national boundaries, coarse-resolution forcing default: ./data_ext
    LANDFORMER_VICFLOW   gauge catchments (Data/target/Discharge/final_basins/)             default: ./data_vicflow
"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))


def _dir(var, default):
    return os.path.abspath(os.environ.get(var, os.path.join(ROOT, default)))


DATA_DIR = _dir("LANDFORMER_DATA", "data")
RESULTS_DIR = _dir("LANDFORMER_RESULTS", "results")
EXT_DIR = _dir("LANDFORMER_EXT", "data_ext")
VICFLOW_DIR = _dir("LANDFORMER_VICFLOW", "data_vicflow")
