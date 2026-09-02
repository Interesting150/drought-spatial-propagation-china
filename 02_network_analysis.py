#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Baseline full-period and seasonal drought-network analysis."""
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

import config as cfg
import network_core as core


def case_key(scale, threshold, season, tau):
    p = cfg.BASELINE[scale]
    if p["type"] == "rapid":
        return f"CN05.1_spi{scale}_delta{threshold}_final{p['final_threshold']}_{season}_tau{tau}"
    return f"CN05.1_spi{scale}_th{threshold}_dur{p['duration']}_{season}_tau{tau}"


def load_spi(scale):
    path = cfg.OUTPUT_DIR / "01_spi" / f"CN05.1_SPI{scale}d_BestDist.nc"
    ds = xr.open_dataset(path)
    arr = ds["spi"].values.astype(np.float32)
    lat, lon, times = ds["lat"].values.astype(np.float32), ds["lon"].values.astype(np.float32), ds["time"].values
    ds.close()
    return arr, lat, lon, times


def extract_full_events(spi, scale, threshold):
    p = cfg.BASELINE[scale]
    effective = spi[cfg.BURN_IN_DAYS:] if len(spi) > cfg.BURN_IN_DAYS else spi
    if p["type"] == "rapid":
        num, starts, ends = core.fd_ex_parallel(effective, p["development_days"], threshold, p["final_threshold"])
    else:
        num, starts, ends = core.drought_ex_parallel(effective, threshold, p["duration"])
    return num.reshape(-1).astype(np.int32), starts.reshape(-1, starts.shape[-1]).astype(np.int32), num


def save_case(result, evt_num, evt_srs, evt_num_2d, lat, lon, out_dir, key, annual_evt_num=None, annual_evt_srs=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    shape = (len(lat), len(lon))
    for name in ["dc", "dc_w", "msd", "msd_w", "nd", "nd_w"]:
        core.arr_to_tiff(result[name].reshape(shape).astype(np.float32), lat, lon, str(out_dir / f"{key}_{name}.tif"))
    if result["bc"] is not None:
        core.arr_to_tiff(result["bc"].reshape(shape).astype(np.float32), lat, lon, str(out_dir / f"{key}_bc.tif"))
    core.arr_to_tiff(evt_num_2d.astype(np.float32), lat, lon, str(out_dir / f"{key}_evt_num.tif"))
    payload = {"Q_matrix":result["Q"], "q_matrix":result["q"], "AQ":result["AQ"], "Aq":result["Aq"],
               "evt_num_seasonal":evt_num, "evt_srs_seasonal":evt_srs}
    if annual_evt_num is not None:
        payload["evt_num_annual"] = annual_evt_num
        payload["evt_srs_annual"] = annual_evt_srs
    core.save_compressed_hdf5(payload, str(out_dir / f"{key}_matrices.h5"))


def main():
    summaries=[]
    for scale in cfg.SPI_SCALES:
        spi, lat, lon, times = load_spi(scale)
        p=cfg.BASELINE[scale]
        threshold=p["threshold"]
        full_num, full_srs, full_num_2d = extract_full_events(spi, scale, threshold)
        for season, months in cfg.SEASONS.items():
            evt_num, evt_srs, len_time = core.filter_events_by_season(full_num, full_srs, times, months, cfg.BURN_IN_DAYS)
            valid_nodes=int(np.sum(evt_num >= cfg.MIN_EVENTS))
            if valid_nodes < 10:
                continue
            result=core.build_network(evt_num, evt_srs, lat, lon, p["tau"], len_time,
                                      n_surrogates=cfg.N_SURROGATES, quantile=cfg.SURROGATE_QUANTILE,
                                      cache_dir=cfg.OUTPUT_DIR/"surrogate_cache", compute_bc=True,
                                      bc_sample_k=cfg.BC_SAMPLE_K)
            key=case_key(scale, threshold, season, p["tau"])
            out_dir=cfg.OUTPUT_DIR/"02_network"/season/f"spi{scale}"/f"tau{p['tau']}"
            save_case(result, evt_num, evt_srs, evt_num.reshape(len(lat),len(lon)), lat, lon, out_dir, key,
                      annual_evt_num=full_num, annual_evt_srs=full_srs)
            summaries.append({"season":season,"scale":scale,"threshold":threshold,"tau":p["tau"],
                              "valid_nodes":valid_nodes,"significant_undirected_edges":int(np.sum(result['AQ'])//2),
                              "significant_directed_entries":int(np.sum(result['Aq']!=0))})
    pd.DataFrame(summaries).to_csv(cfg.OUTPUT_DIR/"02_network"/"network_summary.csv", index=False)


if __name__ == "__main__":
    main()
