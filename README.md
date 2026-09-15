# Scale-dependent and seasonally modulated drought propagation across mainland China

Reproducibility code for the Water Resources Research manuscript **“Scale-Dependent and Seasonally Modulated Spatial Propagation of Rapid-Onset, Monthly-Scale, and Seasonal-Scale Meteorological Drought across Mainland China.”**

The archived Version 1.0.0 is available at Zenodo: https://doi.org/10.5281/zenodo.22249367.

## 1. Repository structure

```text
drought-spatial-propagation-china/
├── network_core.py
├── config.py
├── 00_prepare_diagnostic_inputs.py
├── 01_spi_distribution_selection_and_export.py
├── 02_network_analysis.py
├── 03_source_sink_analysis.py
├── 04_lead_lag_composites.py
├── 05_sensitivity_test.py
├── 06_generate_figures.py
├── requirements.txt
├── environment.yml
├── CITATION.cff
├── LICENSE
└── README.md
```

`network_core.py` contains the shared numerical routines. The numbered scripts implement the main analysis workflow and can be run in numerical order after the required input paths are configured in `config.py`.

## 2. Analysis workflow

1. **`00_prepare_diagnostic_inputs.py`** — prepares analysis-ready atmospheric and land-surface diagnostic variables used in the composite analysis.
2. **`01_spi_distribution_selection_and_export.py`** — calculates 5-, 30-, and 90-day precipitation accumulations, performs five-distribution fitting and AIC-based model selection, evaluates selected-model goodness of fit using KS-D, and exports SPI.
3. **`02_network_analysis.py`** — performs drought-event extraction, full-period and seasonal event assignment, Event Synchronization, surrogate significance testing, and calculation of DC, MSD, and ND.
4. **`03_source_sink_analysis.py`** — identifies significant source and sink regions using Getis-Ord Gi* and evaluates directional source-sink linkages.
5. **`04_lead_lag_composites.py`** — calculates source/sink composite anomalies over scale-specific lead-lag windows using centered moving averages.
6. **`05_sensitivity_test.py`** — evaluates sensitivity to event-extraction thresholds and maximum allowable synchronization lags.
7. **`06_generate_figures.py`** — generates reproducible figures from the analysis outputs. Figure styling may differ slightly from the final publication graphics.

The code also retains an optional Betweenness Centrality implementation for diagnostic purposes; BC is not used in the revised manuscript.

## 3. Baseline analysis settings

| Component | Baseline setting |
|---|---|
| SPI accumulation | 5, 30, and 90 days |
| Rapid-onset event | ΔSPI-5d ≤ −2.5 over 15 days, reaching SPI-5d ≤ −1.0 |
| Monthly/seasonal event parameters | SPI threshold = −1.0; duration = 30 and 90 days, respectively |
| Maximum synchronization lag | 15, 30, and 90 days |
| Surrogate tests | 1000 surrogates; 95th percentile for Q and \|q\| |
| Minimum events for a network node | 3 |
| Seasonal assignment | Events identified from the continuous record and assigned by onset month |
| Threshold sensitivity | rapid-onset: 2.0/2.5/3.0 standard-unit decline; monthly/seasonal: −0.5/−1.0/−1.5 |
| Lag sensitivity | 10/15/20; 15/30/45; 60/90/120 days |
| Candidate Gi* clusters | \|Gi* z\| > 1.96, p < 0.05; minimum cluster fraction = 0.1% |
| Source-sink flux-ratio threshold | 60% |
| Minimum combined source-sink coverage | 1% |
| Lead-lag ranges | −15:+30; −30:+60; −90:+180 days |
| Centered smoothing | 3, 15, and 45 days |

## 4. Input data

Raw third-party datasets are not redistributed in this software repository. Users should obtain the original datasets from their authoritative providers and configure local paths in `config.py`.

Required inputs include:

- CN05.1 daily precipitation at 0.25° resolution for 1961–2022;
- atmospheric and land-surface diagnostic variables used in the composite analysis, including W500, precipitation, Tm, VPD, SWL1, and Bowen ratio;
- a mainland China boundary polygon.

Climate-zone and nine-dash-line shapefiles are optional and are used for publication-style plotting.

## 5. Installation

Python 3.10+ is recommended.

### pip

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

### conda

```bash
conda env create -f environment.yml
conda activate wrr-drought-network
```

### PyCharm

Open this folder as a project, select the virtual or conda environment as the project interpreter, configure the required paths in `config.py`, and run the numbered scripts in sequence.

## 6. Outputs

Outputs are written under `outputs/` by analysis stage. Large adjacency and synchronization-matrix products may require substantial memory and disk space.

## 7. Citation and license

Archived software release:

Fei, J. (2026). *Code for Scale-Dependent and Seasonally Modulated Spatial Propagation of Rapid-Onset, Monthly-Scale, and Seasonal-Scale Meteorological Drought across Mainland China* (Version 1.0.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.22249367.

Development repository:

https://github.com/Interesting150/drought-spatial-propagation-china

License: MIT.
