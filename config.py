#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""User-editable paths and manuscript parameter settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

# Required input files. Place files here or edit these paths.
PRECIP_NC = DATA_DIR / "CN05.1_Pre_1961_2022_daily_025x025.nc"
CHINA_BOUNDARY = DATA_DIR / "china_boundary" / "china_dissolve.shp"
ZONE_SHP = DATA_DIR / "climate_zones" / "china_climate_zones.shp"
NINE_DASH_LINE = DATA_DIR / "china_boundary" / "china_nine_dotted_line.shp"

DIAGNOSTIC_FILES = {
    "W500": DATA_DIR / "diagnostics" / "W500_1961_2022.nc",
    "Pre": PRECIP_NC,
    "Tm": DATA_DIR / "diagnostics" / "Tm_1961_2022.nc",
    "VPD": DATA_DIR / "diagnostics" / "VPD_1961_2022.nc",
    "SWL1": DATA_DIR / "diagnostics" / "SWL1_1961_2022.nc",
    "Bowen": DATA_DIR / "diagnostics" / "Bowen_1961_2022.nc",
}
CLIMATOLOGY_DIR = DATA_DIR / "diagnostics" / "climatology"

SPI_SCALES = [5, 30, 90]
AIC_SWITCH_THRESHOLD = 2.0
BURN_IN_DAYS = 365
MIN_EVENTS = 3
N_SURROGATES = 1000
SURROGATE_QUANTILE = 95.0
BC_SAMPLE_K = 1000

BASELINE = {
    5: {"type": "rapid", "threshold": -2.5, "development_days": 15, "final_threshold": -1.0, "tau": 15},
    30: {"type": "regular", "threshold": -1.0, "duration": 30, "tau": 30},
    90: {"type": "regular", "threshold": -1.0, "duration": 90, "tau": 90},
}

THRESHOLD_SENSITIVITY = {5: [-2.0, -2.5, -3.0], 30: [-0.5, -1.0, -1.5], 90: [-0.5, -1.0, -1.5]}
TAU_SENSITIVITY = {5: [10, 15, 20], 30: [15, 30, 45], 90: [60, 90, 120]}
SEASONS = {"annual": None, "spring": [3, 4, 5], "summer": [6, 7, 8], "autumn": [9, 10, 11], "winter": [12, 1, 2]}

SOURCE_SINK_MIN_CLUSTER_FRACTION = 0.001
SOURCE_SINK_TOTAL_FRACTION = 0.01
SOURCE_SINK_FLUX_RATIO = 0.60
GI_Z_THRESHOLD = 1.96
GI_P_THRESHOLD = 0.05

LEAD_LAG_WINDOWS = {5: (-15, 30), 30: (-30, 60), 90: (-90, 180)}
SMOOTHING_WINDOWS = {5: 3, 30: 15, 90: 45}
COMPOSITE_SEASONS = {5: ["spring", "summer", "autumn", "winter"], 30: ["autumn", "winter"], 90: ["autumn", "winter"]}
