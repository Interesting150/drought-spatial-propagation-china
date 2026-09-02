#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Identify significant source/sink clusters and retain coherent source-sink pathways."""
from pathlib import Path
from collections import defaultdict
import h5py
import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import label, center_of_mass

import config as cfg
import network_core as core


def baseline_key(scale, season):
    p=cfg.BASELINE[scale]
    if p["type"]=="rapid":
        return f"CN05.1_spi{scale}_delta{p['threshold']}_final{p['final_threshold']}_{season}_tau{p['tau']}"
    return f"CN05.1_spi{scale}_th{p['threshold']}_dur{p['duration']}_{season}_tau{p['tau']}"


def extract_candidates(mask_condition, transform, total_valid_pixels):
    labeled, n = label(mask_condition, structure=np.ones((3,3)))
    out=[]
    for idx in range(1,n+1):
        mask=labeled==idx
        pixels=int(mask.sum())
        if pixels/total_valid_pixels >= cfg.SOURCE_SINK_MIN_CLUSTER_FRACTION:
            row,col=center_of_mass(mask)
            lon,lat=rasterio.transform.xy(transform,row,col)
            out.append({"mask":mask,"idx":np.where(mask.ravel())[0],"pixels":pixels,"lon":float(lon),"lat":float(lat)})
    return out


def process_case(scale, season):
    p=cfg.BASELINE[scale]
    key=baseline_key(scale, season)
    case_dir=cfg.OUTPUT_DIR/"02_network"/season/f"spi{scale}"/f"tau{p['tau']}"
    nd_path=case_dir/f"{key}_nd.tif"
    h5_path=case_dir/f"{key}_matrices.h5"
    if not nd_path.exists() or not h5_path.exists():
        return []
    with rasterio.open(nd_path) as src:
        nd=src.read(1).astype(np.float32); transform=src.transform; profile=src.profile
        if src.nodata is not None: nd[np.isclose(nd,src.nodata)]=np.nan
    valid=np.isfinite(nd); total=int(valid.sum())
    gi_z,gi_p=core.getis_ord_gi_star(nd[valid], core.build_queen_weights(valid))
    z=np.full(nd.shape,np.nan,np.float32); q=np.full(nd.shape,np.nan,np.float32); z[valid]=gi_z; q[valid]=gi_p
    sources=extract_candidates((z < -cfg.GI_Z_THRESHOLD)&(q<cfg.GI_P_THRESHOLD)&valid,transform,total)
    sinks=extract_candidates((z > cfg.GI_Z_THRESHOLD)&(q<cfg.GI_P_THRESHOLD)&valid,transform,total)
    if not sources or not sinks: return []
    with h5py.File(h5_path,'r') as f: Aq=f['Aq'][:]
    raw=[]
    for src in sources:
        best_idx=None; best_ratio=0.0; best_active=0
        for j,snk in enumerate(sinks):
            sub=Aq[src['idx'],:][:,snk['idx']]
            active=int(np.sum(np.sum(sub==1,axis=1)>0))
            ratio=active/src['pixels']
            if ratio>best_ratio: best_idx,best_ratio,best_active=j,ratio,active
        if best_idx is not None and best_ratio >= cfg.SOURCE_SINK_FLUX_RATIO:
            snk=sinks[best_idx]
            raw.append({"sink_idx":best_idx,"flux_ratio":best_ratio,"active_px":best_active,"src":src,"snk":snk,
                        "distance_km":core.haversine_km(src['lon'],src['lat'],snk['lon'],snk['lat'])})
    if not raw: return []
    groups=defaultdict(list)
    for pair in raw: groups[pair['sink_idx']].append(pair)
    best_key=max(groups, key=lambda k: sum(x['src']['pixels'] for x in groups[k])+groups[k][0]['snk']['pixels'])
    kept=groups[best_key]
    group_area=sum(x['src']['pixels'] for x in kept)+kept[0]['snk']['pixels']
    if group_area/total < cfg.SOURCE_SINK_TOTAL_FRACTION: return []
    region=np.full(nd.shape,np.nan,np.float32); rows=[]
    for pair_id,pair in enumerate(kept,1):
        region[pair['src']['mask']]=pair_id; region[pair['snk']['mask']]=-pair_id
        rows.append({"scale":scale,"season":season,"pair_id":pair_id,"flux_ratio":pair['flux_ratio'],
                     "distance_km":pair['distance_km'],"source_pixels":pair['src']['pixels'],"sink_pixels":pair['snk']['pixels']})
    out_dir=cfg.OUTPUT_DIR/"03_source_sink"/season/f"spi{scale}"; out_dir.mkdir(parents=True,exist_ok=True)
    out_tif=out_dir/f"paired_regions_spi{scale}_tau{p['tau']}_{season}_thr{p['threshold']}.tif"
    profile.update(dtype='float32',count=1,nodata=np.nan)
    with rasterio.open(out_tif,'w',**profile) as dst: dst.write(region,1)
    pd.DataFrame(rows).to_csv(out_dir/f"paired_regions_spi{scale}_{season}.csv",index=False)
    return rows


def main():
    all_rows=[]
    for scale,seasons in cfg.COMPOSITE_SEASONS.items():
        for season in seasons: all_rows.extend(process_case(scale,season))
    out=cfg.OUTPUT_DIR/"03_source_sink"; out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(all_rows).to_csv(out/"source_sink_summary.csv",index=False)


if __name__ == "__main__":
    main()
