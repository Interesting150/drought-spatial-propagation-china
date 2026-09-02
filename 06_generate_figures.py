#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate the network, source-sink, lead-lag, and sensitivity figures from archived outputs."""
from pathlib import Path
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio

import config as cfg


def read_tif(path):
    with rasterio.open(path) as src:
        a=src.read(1).astype(np.float32); extent=[src.bounds.left,src.bounds.right,src.bounds.bottom,src.bounds.top]
        if src.nodata is not None: a[np.isclose(a,src.nodata)]=np.nan
    return a,extent


def baseline_key(scale, season):
    p=cfg.BASELINE[scale]
    if p['type']=='rapid': return f"CN05.1_spi{scale}_delta{p['threshold']}_final{p['final_threshold']}_{season}_tau{p['tau']}"
    return f"CN05.1_spi{scale}_th{p['threshold']}_dur{p['duration']}_{season}_tau{p['tau']}"


def plot_full_period():
    fig,axes=plt.subplots(3,3,figsize=(15,11),constrained_layout=True)
    metrics=[('dc','Degree centrality'),('msd_w','Mean synchronization distance (km)'),('nd','Network divergence')]
    for r,scale in enumerate(cfg.SPI_SCALES):
        key=baseline_key(scale,'annual'); p=cfg.BASELINE[scale]; root=cfg.OUTPUT_DIR/'02_network'/'annual'/f'spi{scale}'/f"tau{p['tau']}"
        for c,(suffix,title) in enumerate(metrics):
            path=root/f"{key}_{suffix}.tif"; a,ext=read_tif(path)
            im=axes[r,c].imshow(a,origin='upper',extent=ext,aspect='auto',cmap='RdBu_r' if suffix=='nd' else 'viridis')
            axes[r,c].set_title(f"SPI-{scale}d - {title}"); fig.colorbar(im,ax=axes[r,c],shrink=.75)
    out=cfg.OUTPUT_DIR/'06_figures'; out.mkdir(parents=True,exist_ok=True); fig.savefig(out/'Figure4_full_period_network.png',dpi=300); plt.close(fig)


def plot_seasonal_networks():
    out=cfg.OUTPUT_DIR/'06_figures'; out.mkdir(parents=True,exist_ok=True)
    for scale in cfg.SPI_SCALES:
        seasons=['spring','summer','autumn','winter']; fig,axes=plt.subplots(2,4,figsize=(16,8),constrained_layout=True)
        for c,season in enumerate(seasons):
            key=baseline_key(scale,season); p=cfg.BASELINE[scale]; root=cfg.OUTPUT_DIR/'02_network'/season/f'spi{scale}'/f"tau{p['tau']}"
            for r,suffix in enumerate(['dc','nd']):
                path=root/f"{key}_{suffix}.tif"
                if not path.exists(): axes[r,c].axis('off'); continue
                a,ext=read_tif(path); im=axes[r,c].imshow(a,origin='upper',extent=ext,aspect='auto',cmap='RdBu_r' if suffix=='nd' else 'viridis')
                axes[r,c].set_title(f"{season.title()} - {'ND' if suffix=='nd' else 'DC'}"); fig.colorbar(im,ax=axes[r,c],shrink=.65)
        fig.savefig(out/f"seasonal_network_spi{scale}.png",dpi=300); plt.close(fig)


def plot_lead_lag():
    out=cfg.OUTPUT_DIR/'06_figures'; out.mkdir(parents=True,exist_ok=True)
    for csv_path in sorted((cfg.OUTPUT_DIR/'04_lead_lag').glob('lead_lag_*.csv')):
        df=pd.read_csv(csv_path); variables=df['variable'].unique(); fig,axes=plt.subplots(2,3,figsize=(15,9),constrained_layout=True)
        for ax,var in zip(axes.ravel(),variables):
            d=df[df.variable==var]
            for region in ['source','sink']:
                x=d[d.region==region]; ax.plot(x.lag_day,x.mean_anomaly,label=region); ax.fill_between(x.lag_day,x.mean_anomaly-x.standard_error,x.mean_anomaly+x.standard_error,alpha=.15)
            ax.axvline(0,color='k',lw=1); ax.axhline(0,color='k',lw=.8,ls='--'); ax.set_title(var); ax.legend()
        fig.savefig(out/(csv_path.stem+'.png'),dpi=300); plt.close(fig)



def plot_source_sink_composites():
    out = cfg.OUTPUT_DIR / "06_figures"
    out.mkdir(parents=True, exist_ok=True)
    lead_dir = cfg.OUTPUT_DIR / "04_lead_lag"
    for csv_path in sorted(lead_dir.glob("lead_lag_spi*_*.csv")):
        stem = csv_path.stem
        parts = stem.split("_")
        scale = int(parts[2].replace("spi", ""))
        season = parts[3]
        p = cfg.BASELINE[scale]
        pair_path = (cfg.OUTPUT_DIR / "03_source_sink" / season / f"spi{scale}" /
                     f"paired_regions_spi{scale}_tau{p['tau']}_{season}_thr{p['threshold']}.tif")
        if not pair_path.exists():
            continue
        pair, extent = read_tif(pair_path)
        df = pd.read_csv(csv_path)
        variables = list(df["variable"].drop_duplicates())[:6]
        fig = plt.figure(figsize=(18, 9), constrained_layout=True)
        gs = fig.add_gridspec(2, 4, width_ratios=[1.4, 1, 1, 1])
        ax_map = fig.add_subplot(gs[:, 0])
        masked = np.ma.masked_invalid(pair)
        im = ax_map.imshow(masked, origin="upper", extent=extent, aspect="auto", cmap="coolwarm")
        ax_map.set_title(f"Source-sink regions: SPI-{scale}d, {season}")
        ax_map.set_xlabel("Longitude")
        ax_map.set_ylabel("Latitude")
        fig.colorbar(im, ax=ax_map, shrink=0.75, label="Pair ID (+ source, - sink)")
        for k, var in enumerate(variables):
            ax = fig.add_subplot(gs[k // 3, 1 + k % 3])
            d = df[df.variable == var]
            for region in ["source", "sink"]:
                x = d[d.region == region]
                if x.empty:
                    continue
                ax.plot(x.lag_day, x.mean_anomaly, label=region)
                ax.fill_between(x.lag_day, x.mean_anomaly - x.standard_error,
                                x.mean_anomaly + x.standard_error, alpha=0.15)
            ax.axvline(0, color="k", lw=1)
            ax.axhline(0, color="k", lw=0.8, ls="--")
            ax.set_title(var)
            ax.set_xlabel("Lag day")
            ax.set_ylabel("Anomaly")
            ax.legend(fontsize=8)
        fig.savefig(out / f"source_sink_composite_spi{scale}_{season}.png", dpi=300)
        plt.close(fig)

def plot_sensitivity():
    files=list((cfg.OUTPUT_DIR/'05_sensitivity').glob('**/*_dc.tif'))
    if not files: return
    rows=[]
    for path in files:
        a,_=read_tif(path); stem=path.stem; parts=stem.split('_'); scale=int(parts[0].replace('spi','')); threshold=float(parts[1].replace('thr','')); season=parts[2]; tau=int(parts[3].replace('tau',''))
        rows.append((scale,threshold,season,tau,np.nanmean(a,axis=1),np.nanmean(a,axis=0)))
    out=cfg.OUTPUT_DIR/'06_figures'; out.mkdir(parents=True,exist_ok=True)
    for scale in cfg.SPI_SCALES:
        sub=[x for x in rows if x[0]==scale and x[2]=='annual'];
        if not sub: continue
        fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
        for _,thr,_,tau,latp,lonp in sub:
            axes[0].plot(latp,label=f"thr={thr}, tau={tau}"); axes[1].plot(lonp,label=f"thr={thr}, tau={tau}")
        axes[0].set_title(f"SPI-{scale}d latitude profile"); axes[1].set_title(f"SPI-{scale}d longitude profile"); axes[0].legend(fontsize=7); axes[1].legend(fontsize=7)
        fig.savefig(out/f"sensitivity_profiles_spi{scale}.png",dpi=300); plt.close(fig)


def main():
    plot_full_period(); plot_seasonal_networks(); plot_lead_lag(); plot_source_sink_composites(); plot_sensitivity()


if __name__ == '__main__':
    main()
