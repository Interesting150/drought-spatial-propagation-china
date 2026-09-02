# Scale-dependent and seasonally modulated drought propagation across mainland China

Reproducibility code for the Water Resources Research manuscript **“Scale-Dependent and Seasonally Modulated Spatial Propagation of Rapid-Onset, Monthly-Scale, and Seasonal-Scale Meteorological Drought across Mainland China.”**

## 1. Repository structure

```text
WRR_drought_network_code_v1.0.0/
├── network_core.py
├── config.py
├── 00_prepare_diagnostic_inputs.py
├── 01_spi_distribution_selection_and_export.py
├── 02_network_analysis.py
├── 03_source_sink_analysis.py
├── 04_lead_lag_composites.py
├── 05_sensitivity_test.py
├── 06_generate_figures.py
├── demo_smoke_test.py
├── requirements.txt
├── environment.yml
├── CITATION.cff
├── LICENSE
├── ORIGINAL_TO_RELEASE_MAP.md
├── REPRODUCIBILITY_NOTES.md
├── PRE_RELEASE_CHECKLIST.md
├── DATA_ARCHIVE_CHECKLIST.md
├── data/README.md
└── outputs/README.md
```

`network_core.py` contains the shared numerical routines. The numbered scripts are directly runnable from PyCharm or from a terminal and each contains a `main()` entry point.

## 2. Analysis workflow

Run the scripts in numerical order after editing `config.py` or placing the required files under `data/`.

1. **`00_prepare_diagnostic_inputs.py`** — optional provenance/helper script for analysis-ready W500, precipitation, Tm, VPD, SWL1, and Bowen-ratio inputs. See `REPRODUCIBILITY_NOTES.md` before using it to regenerate VPD or Bowen ratio.
2. **`01_spi_distribution_selection_and_export.py`** — 5-, 30-, and 90-day precipitation accumulation; five-distribution fitting; AIC-based selection with conservative Gamma retention when ΔAIC(Gamma) ≤ 2; SPI export; selected-model GOF summaries.
3. **`02_network_analysis.py`** — drought-event extraction; full-period and onset-month seasonal assignment; Event Synchronization; Q and |q| surrogate significance; DC, MSD, ND, and optional BC outputs.
4. **`03_source_sink_analysis.py`** — Getis-Ord Gi* significant-region extraction and directional source-sink matching.
5. **`04_lead_lag_composites.py`** — source/sink composite anomalies over scale-specific lead-lag ranges with centered 3/15/45-day moving averages.
6. **`05_sensitivity_test.py`** — event-threshold and maximum-lag sensitivity experiments over the full parameter grid reported in the manuscript.
7. **`06_generate_figures.py`** — reproducible plots from archived outputs. Styling is simplified relative to the final manuscript graphics; see `REPRODUCIBILITY_NOTES.md`.

Run a small synthetic check with:

```bash
python demo_smoke_test.py
```

## 3. Manuscript parameter settings

| Component | Baseline setting |
|---|---|
| SPI accumulation | 5, 30, 90 days |
| Rapid-onset event | ΔSPI-5d ≤ −2.5 over 15 days; SPI-5d ≤ −1.0 within the development interval |
| Monthly/seasonal drought threshold | SPI < −1.0 in the supplied event-extraction implementation |
| Baseline maximum synchronization lag | 15, 30, 90 days |
| Surrogate tests | 1000 surrogates; 95th percentile for Q and |q| |
| Minimum events for a network node | 3 |
| Betweenness Centrality approximation | 1000 sampled source nodes |
| Seasonal assignment | Identify events on the continuous record; assign by onset month |
| Threshold sensitivity | rapid: 2.0/2.5/3.0 standard-unit decline; monthly/seasonal: −0.5/−1.0/−1.5 |
| Lag sensitivity | 10/15/20; 15/30/45; 60/90/120 days |
| Candidate Gi* clusters | absolute Gi* z-score > 1.96, p < 0.05; cluster ≥ 0.1% of valid land cells |
| Directional source-sink criterion | flux ratio ≥ 60% |
| Retained source-sink group | combined source + sink coverage ≥ 1% of valid land cells |
| Lead-lag ranges | −15:+30; −30:+60; −90:+180 days |
| Centered smoothing | 3, 15, 45 days |

The release deliberately preserves several implementation conventions from the supplied analysis code. These are documented in `REPRODUCIBILITY_NOTES.md` and should be reconciled with the manuscript before DOI publication.

## 4. Input data

Raw third-party datasets are not redistributed in this software archive. Configure local paths in `config.py`.

Required analysis inputs include:

- CN05.1 daily precipitation, 0.25°, 1961–2022;
- ERA5/CN05.1 analysis-ready diagnostic variables used in the composite analysis: W500, precipitation, Tm, VPD, SWL1, and Bowen ratio;
- a mainland China boundary polygon.

Climate-zone and nine-dash-line shapefiles are optional for publication-style plotting. If raw third-party data cannot be redistributed, cite their authoritative repositories and archive the processed/figure-supporting products needed to evaluate the paper in a DOI-bearing data repository. See `DATA_ARCHIVE_CHECKLIST.md`.

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

Open this folder as a project, select the virtual/conda environment as the project interpreter, edit `config.py`, and run each numbered script directly.

## 6. Outputs

Outputs are written under `outputs/` by stage. Large adjacency and matrix products can require substantial memory and disk space. For manuscript preservation, archive at least the figure-supporting processed products listed in `DATA_ARCHIVE_CHECKLIST.md`.

## 7. Citation and license

Before publication:

1. verify the software creator list in `CITATION.cff`;
2. replace the GitHub and DOI placeholders;
3. confirm the software license with the authors/institution;
4. cite the **version-specific archival DOI** in the manuscript Availability Statement and References.

See `PRE_RELEASE_CHECKLIST.md` before depositing the package.
