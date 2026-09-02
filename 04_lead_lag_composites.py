#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compute source/sink lead-lag composite anomalies and centered moving averages."""
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import rasterio
import xarray as xr

import config as cfg
import network_core as core

ORDERED_VARS=["W500","Pre","Tm","VPD","SWL1","Bowen"]


def baseline_key(scale, season):
    p=cfg.BASELINE[scale]
    if p["type"]=="rapid":
        return f"CN05.1_spi{scale}_delta{p['threshold']}_final{p['final_threshold']}_{season}_tau{p['tau']}"
    return f"CN05.1_spi{scale}_th{p['threshold']}_dur{p['duration']}_{season}_tau{p['tau']}"


def load_data_file(path):
    return core.load_full_nc(path)


def load_pair(scale, season):
    p=cfg.BASELINE[scale]
    path=cfg.OUTPUT_DIR/"03_source_sink"/season/f"spi{scale}"/f"paired_regions_spi{scale}_tau{p['tau']}_{season}_thr{p['threshold']}.tif"
    if not path.exists(): return None
    with rasterio.open(path) as src:
        a=src.read(1).astype(np.float32)
        if src.nodata is not None: a[np.isclose(a,src.nodata)]=np.nan
    src_rc=np.where(np.isfinite(a)&(a>0.5)); snk_rc=np.where(np.isfinite(a)&(a<-0.5))
    if len(src_rc[0])==0 or len(snk_rc[0])==0: return None
    return src_rc,snk_rc


def process_case(scale, season, all_data, grid):
    pair=load_pair(scale,season)
    if pair is None: return
    p=cfg.BASELINE[scale]; key=baseline_key(scale,season)
    matrix_path=cfg.OUTPUT_DIR/"02_network"/season/f"spi{scale}"/f"tau{p['tau']}"/f"{key}_matrices.h5"
    if not matrix_path.exists(): return
    src_rc,snk_rc=pair
    with h5py.File(matrix_path,'r') as f:
        evt_srs=f['evt_srs_seasonal'][:]; evt_num=f['evt_num_seasonal'][:]
    ncols=len(grid['lon']); src_idx=src_rc[0]*ncols+src_rc[1]
    starts=[]
    for node in src_idx:
        n=int(evt_num[node])
        if n>0: starts.extend(evt_srs[node,:n]+cfg.BURN_IN_DAYS)
    lag_min,lag_max=cfg.LEAD_LAG_WINDOWS[scale]
    starts=np.unique(np.asarray(starts,dtype=np.int32))
    starts=starts[(starts>=-lag_min)&(starts<grid['ntime']-lag_max)]
    if starts.size==0: return
    src_lats,src_lons=grid['lat'][src_rc[0]],grid['lon'][src_rc[1]]
    snk_lats,snk_lons=grid['lat'][snk_rc[0]],grid['lon'][snk_rc[1]]
    times=np.arange(lag_min,lag_max+1)
    rows=[]
    for name in ORDERED_VARS:
        d=all_data[name]
        src_raw=core.extract_pixels_from_full(d['raw'],d['lat'],d['lon'],src_lats,src_lons)
        snk_raw=core.extract_pixels_from_full(d['raw'],d['lat'],d['lon'],snk_lats,snk_lons)
        src_clim=core.compute_climatology_numba(src_raw,grid['doy']); snk_clim=core.compute_climatology_numba(snk_raw,grid['doy'])
        src_an=core.compute_anomaly_numba(src_raw,grid['doy'],src_clim); snk_an=core.compute_anomaly_numba(snk_raw,grid['doy'],snk_clim)
        for region,anom in [('source',src_an),('sink',snk_an)]:
            comp=core.extract_event_windows_numba(anom,starts,lag_min,lag_max,grid['ntime'])
            mean=np.nanmean(comp,axis=0); se=np.nanstd(comp,axis=0)/np.sqrt(max(len(starts),1))
            win=cfg.SMOOTHING_WINDOWS[scale]
            mean=pd.Series(mean).rolling(win,min_periods=1,center=True).mean().values
            se=pd.Series(se).rolling(win,min_periods=1,center=True).mean().values
            rows.extend({"scale":scale,"season":season,"variable":name,"region":region,"lag_day":int(t),
                         "mean_anomaly":float(m),"standard_error":float(s),"n_events":int(len(starts))}
                        for t,m,s in zip(times,mean,se))
    out=cfg.OUTPUT_DIR/"04_lead_lag"; out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/f"lead_lag_spi{scale}_{season}.csv",index=False)


def main():
    with xr.open_dataset(cfg.DIAGNOSTIC_FILES['VPD']) as ds:
        lat=ds['lat'].values.astype(np.float32) if 'lat' in ds else ds['latitude'].values.astype(np.float32)
        lon=ds['lon'].values.astype(np.float32) if 'lon' in ds else ds['longitude'].values.astype(np.float32)
        times=ds['time'].values
    if lat[0]>lat[-1]: lat=lat[::-1]
    grid={"lat":lat,"lon":lon,"doy":pd.DatetimeIndex(times).dayofyear.values,"ntime":len(times)}
    all_data={}
    for name,path in cfg.DIAGNOSTIC_FILES.items():
        raw,lats,lons=load_data_file(path); all_data[name]={"raw":raw,"lat":lats,"lon":lons}
    for scale,seasons in cfg.COMPOSITE_SEASONS.items():
        for season in seasons: process_case(scale,season,all_data,grid)


if __name__ == "__main__":
    main()
