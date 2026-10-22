# Campi Flegrei: can ground uplift forecast earthquake rates?

Work in progress. A learning project that asks whether GPS ground deformation at Campi Flegrei
(uplift, uplift rate and acceleration across 21 stations) can forecast the **rate** of earthquakes
and the **probability of larger events** (M ≥ 3) in the following days and weeks.

This project forecasts rates and probabilities. It does not predict individual earthquakes.

## Setup

With conda (recommended):

```bash
conda env create -f environment.yml
conda activate flegrei-seismicity
```

Without conda, use Python 3.14 and `pip install -r requirements.txt`. On macOS, LightGBM also
needs OpenMP (`brew install libomp`).

## Data

Raw data lives in `data/raw/` and is not committed. Download instructions and full credits will
be added in phase 2. Sources (all CC BY 4.0):

- GPS, 21 stations, 2000–2025: De Martino et al. (2026), Zenodo, doi:10.5281/zenodo.22259337
- GPS, RITE vertical, 2000–2024: De Martino et al. (2025), Zenodo, doi:10.5281/zenodo.17132077
- De Martino et al. (2021), *Remote Sensing* 13(14):2725, doi:10.3390/rs13142725
- Earthquake catalogue: INGV Osservatorio Vesuviano, doi:10.13127/gossip
