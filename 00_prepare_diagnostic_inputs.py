#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optional preprocessing helper for diagnostic NetCDF files.

The manuscript analysis used analysis-ready W500, precipitation, Tm, VPD, SWL1,
and Bowen-ratio files. This helper can derive VPD and Bowen ratio when the raw
inputs are available and variable names are supplied below. Verify ERA5 sign and
unit conventions against the exact files used in the study before release.
"""
from pathlib import Path
import numpy as np
import xarray as xr
import config as cfg


def saturation_vapor_pressure_kpa(t_c):
    return 0.6108 * np.exp(17.27 * t_c / (t_c + 237.3))


def derive_vpd(temperature_c, relative_humidity_pct):
    return saturation_vapor_pressure_kpa(temperature_c) * (1.0 - relative_humidity_pct / 100.0)


def derive_bowen(sensible_heat, latent_heat):
    return sensible_heat / latent_heat


def main():
    print("This helper is intentionally not run automatically.")
    print("Place the analysis-ready diagnostic files listed in config.DIAGNOSTIC_FILES in data/diagnostics/.")
    print("If you regenerate VPD or Bowen ratio, verify variable units and ERA5 flux sign conventions first.")


if __name__ == '__main__':
    main()
