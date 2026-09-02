#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run event-threshold and maximum-lag sensitivity experiments."""
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

import config as cfg
import network_core as core


def load_spi(scale):
    ds=xr.open_dataset(cfg.OUTPUT_DIR/"01_spi"/f"CN05.1_SPI{scale}d_BestDist.nc")
    a=ds['spi'].values.astype(np.float32); lat=ds['lat'].values.astype(np.float32); lon=ds['lon'].values.astype(np.float32); times=ds['time'].values; ds.close()
    return a,lat,lon,times


def extract(spi,scale,threshold):
    p=cfg.BASELINE[scale]; a=spi[cfg.BURN_IN_DAYS:]
    if p['type']=='rapid': num,srs,_=core.fd_ex_parallel(a,p['development_days'],threshold,p['final_threshold'])
    else: num,srs,_=core.drought_ex_parallel(a,threshold,p['duration'])
    return num.reshape(-1).astype(np.int32),srs.reshape(-1,srs.shape[-1]).astype(np.int32)


def main():
    rows=[]
    for scale in cfg.SPI_SCALES:
        spi,lat,lon,times=load_spi(scale)
        for threshold in cfg.THRESHOLD_SENSITIVITY[scale]:
            full_num,full_srs=extract(spi,scale,threshold)
            for season,months in cfg.SEASONS.items():
                evt_num,evt_srs,len_time=core.filter_events_by_season(full_num,full_srs,times,months,cfg.BURN_IN_DAYS)
                if np.sum(evt_num>=cfg.MIN_EVENTS)<10: continue
                for tau in cfg.TAU_SENSITIVITY[scale]:
                    result=core.build_network(evt_num,evt_srs,lat,lon,tau,len_time,n_surrogates=cfg.N_SURROGATES,
                                              quantile=cfg.SURROGATE_QUANTILE,cache_dir=cfg.OUTPUT_DIR/"surrogate_cache",
                                              compute_bc=False)
                    shape=(len(lat),len(lon)); out=cfg.OUTPUT_DIR/"05_sensitivity"/season/f"spi{scale}"/f"tau{tau}"; out.mkdir(parents=True,exist_ok=True)
                    key=f"spi{scale}_thr{threshold}_{season}_tau{tau}"
                    core.arr_to_tiff(result['dc'].reshape(shape),lat,lon,str(out/f"{key}_dc.tif"))
                    rows.append({"scale":scale,"season":season,"threshold":threshold,"tau":tau,
                                 "mean_dc":float(np.nanmean(result['dc'])),"valid_nodes":int(np.sum(evt_num>=cfg.MIN_EVENTS))})
    root=cfg.OUTPUT_DIR/"05_sensitivity"; root.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(root/"sensitivity_summary.csv",index=False)


if __name__ == "__main__":
    main()
