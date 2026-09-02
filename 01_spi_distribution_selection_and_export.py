#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Select precipitation distributions and export SPI-5d, SPI-30d and SPI-90d."""
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed
import scipy.stats as st

import config as cfg
import network_core as core


def analyze_cell(ts):
    sample = ts[np.isfinite(ts)]
    positive = sample[sample > 0]
    if sample.size < 10 or positive.size < 6:
        return -1, np.nan, np.nan, np.nan, np.nan
    records = core.fit_distribution_set(positive)
    aic = np.array([r[3] for r in records], dtype=float)
    dist_id = core.conservative_best_distribution(aic, cfg.AIC_SWITCH_THRESHOLD)
    q = float(np.sum(sample == 0)) / float(sample.size)
    params = records[dist_id][0]
    n = positive.size
    p = np.clip((np.arange(1, n + 1) - 0.3175) / (n + 0.365), 1e-8, 1 - 1e-8)
    theo = core.PPF_FUNCS[dist_id](p, params)
    ppcc = float(np.corrcoef(np.sort(positive), theo)[0, 1]) if np.all(np.isfinite(theo)) else np.nan
    emp = np.sort(positive)
    F = np.clip(core.CDF_FUNCS[dist_id](emp, params), 1e-12, 1 - 1e-12)
    ks_d = float(np.max(np.abs(F - np.arange(1, n + 1) / n)))
    ad = float(-n - np.mean((2*np.arange(1,n+1)-1) * (np.log(F) + np.log(1-F[::-1]))))
    return dist_id, q, ppcc, ks_d, ad


def process_scale(precip, lat, lon, time_values, land_mask, scale):
    accumulated = core.vectorized_rolling_sum(precip, scale)
    nlat, nlon = len(lat), len(lon)
    cells = [accumulated[:, i, j] if land_mask[i, j] else np.full(accumulated.shape[0], np.nan)
             for i in range(nlat) for j in range(nlon)]
    results = Parallel(n_jobs=-1, prefer="processes")(delayed(analyze_cell)(x) for x in cells)
    dist = np.array([r[0] for r in results], dtype=np.int16).reshape(nlat, nlon)
    q = np.array([r[1] for r in results], dtype=np.float32).reshape(nlat, nlon)
    ppcc = np.array([r[2] for r in results], dtype=np.float32).reshape(nlat, nlon)
    ks = np.array([r[3] for r in results], dtype=np.float32).reshape(nlat, nlon)
    ad = np.array([r[4] for r in results], dtype=np.float32).reshape(nlat, nlon)

    spi = np.full(accumulated.shape, np.nan, dtype=np.float32)
    for i in range(nlat):
        for j in range(nlon):
            if dist[i, j] >= 0:
                spi[:, i, j] = core.spi_from_selected_distribution(accumulated[:, i, j], int(dist[i, j]))

    out = cfg.OUTPUT_DIR / "01_spi"
    out.mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset(
        {"spi": (("time","lat","lon"), spi),
         "best_dist_conservative": (("lat","lon"), dist.astype(np.float32)),
         "zero_fraction": (("lat","lon"), q),
         "selected_ppcc": (("lat","lon"), ppcc),
         "selected_ks_d": (("lat","lon"), ks),
         "selected_ad_a2": (("lat","lon"), ad)},
        coords={"time": time_values, "lat": lat, "lon": lon},
        attrs={"distribution_encoding":"0=Gamma,1=LogNormal,2=PearsonIII,3=Weibull,4=GenGamma",
               "aic_switch_threshold": cfg.AIC_SWITCH_THRESHOLD,
               "rolling_convention":"pipeline-compatible cumulative-sum indexing"})
    ds.to_netcdf(out / f"CN05.1_SPI{scale}d_BestDist.nc")
    summary = {"scale_days": scale, "valid_cells": int(np.sum(dist >= 0)),
               "median_zero_fraction": float(np.nanmedian(q)), "median_selected_ppcc": float(np.nanmedian(ppcc)),
               "median_selected_ks_d": float(np.nanmedian(ks)), "median_selected_ad_a2": float(np.nanmedian(ad))}
    return summary


def main():
    precip, lat, lon, times = core.load_precipitation(cfg.PRECIP_NC)
    land = core.create_mask_from_shapefile(lat, lon, str(cfg.CHINA_BOUNDARY), inv=True)
    if land.shape != precip.shape[1:]:
        land = np.flipud(land)
    precip = np.where(land[None, :, :], precip, np.nan)
    summaries = [process_scale(precip, lat, lon, times, land, s) for s in cfg.SPI_SCALES]
    out = cfg.OUTPUT_DIR / "01_spi"
    pd.DataFrame(summaries).to_csv(out / "gof_summary.csv", index=False)
    print(f"Saved SPI and GOF outputs to {out}")


if __name__ == "__main__":
    main()
