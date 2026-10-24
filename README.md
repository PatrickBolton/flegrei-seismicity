# Campi Flegrei: can ground uplift forecast earthquake rates?

Work in progress. A learning project that asks whether GPS ground deformation at Campi Flegrei
(uplift, uplift rate and acceleration across 21 stations) can forecast the **rate** of earthquakes
and the **probability of larger events** (M ≥ 3) in the following days and weeks.

This project forecasts rates and probabilities. It does not predict individual earthquakes.

## Results (October 2026)

**Short answer: at weekly resolution, GPS uplift adds no measurable forecasting skill beyond what
past seismicity already provides.** Uplift explains the long-term growth of seismicity well, but
that growth is already visible in the earthquake counts themselves.

How this was tested ([`notebooks/03_baselines.ipynb`](notebooks/03_baselines.ipynb),
[`04_models`](notebooks/04_models.ipynb), [`05_evaluation`](notebooks/05_evaluation.ipynb)):

- **Targets:** the number of events with M ≥ 0.6 next week, and the probability of at least one
  M ≥ 3 next week.
- **Evaluation:** 312 test weeks (2020–2025), each forecast made only from data before that week
  (walk-forward); results also for the steady period 2024–2025. Scored with proper scores
  (Poisson deviance and negative binomial log score for counts; Brier score and log-loss for
  probabilities), with 95% block-bootstrap intervals for every comparison.
- **Models:** baselines (persistence, rolling means, climatology); the model of the paper we
  build on (arXiv 2503.20129), reproduced in open code and turned into a weekly forecast;
  Poisson regression and LightGBM with 19 features from past seismicity and 21 GPS stations.
- **The key test:** the same model trained with and without GPS features.

| | Best result | Effect of adding GPS |
|---|---|---|
| Weekly count | Poisson regression on recent seismicity, with a negative binomial spread: 2.4% better log score than a 4-week rolling mean (significant) | +0.2% (−0.8% to +1.1%): none |
| P(M ≥ 3 next week) | 4-week mean count converted allowing for clustering: 3–5% better Brier score than climatology | +1.2% (+0.3% to +2.6%): worse |

![Effect of adding GPS features across setups](reports/figures/gps_effect_sensitivity.png)

The conclusion holds across six alternative setups (training start, training window, a 4-week
horizon, an M ≥ 2.5 threshold). The tests could detect effects of about 1–3% of the score, the
size of the real gains made from seismicity, so this isn't a matter of too little data to see
an effect of that size. The reason: the slow part of the uplift signal duplicates the
cumulative earthquake count, while its fast part is weak, noisy, and simultaneous with seismic
bursts rather than leading them.

Other findings along the way:

- **The paper's model reproduces** (we get k₁ = 0.752, k₂ = 0.205 against the reported 0.76,
  0.2), but its parameters depend on fixing an integration constant to zero, and the curve
  underpredicts events after May 2024 by 15–36%. As a weekly forecast it tracks the yearly
  level of seismicity but misses swarms.
- **Uncertainty matters:** Poisson forecasts are far too confident (their 80% intervals held
  only 45% of weeks); a negative binomial gives honest intervals (86–90%).
- **Larger events are hard to forecast** beyond their recent base rate: the best forecasts rank
  risky weeks only modestly better than chance (ROC AUC about 0.63 in 2024–2025).

**Limitations:** weekly resolution and horizons of 1–4 weeks only; duration magnitudes (Md),
which can differ by a few tenths from ML/Mw; a catalogue that is incomplete below about Md 0.6
before 2017; few weeks with an M ≥ 3 (43 in the test period), so probability results are
uncertain; and the GPS series end in December 2025. GPS uplift may still matter on longer
horizons, for other targets, or with better deformation data (for example InSAR).

## Exploratory findings

These are the first observations from cleaning and exploring the data (catalogue downloaded 22
Oct 2026). Details, code and caveats are in
[`notebooks/01_completeness.ipynb`](notebooks/01_completeness.ipynb) and
[`notebooks/02_explore.ipynb`](notebooks/02_explore.ipynb).

![Map of located earthquakes with M ≥ 0.6 and the 21 GPS stations](reports/figures/map_events_stations.png)

**Where.** The earthquakes are tightly clustered around Solfatara, under the town of Pozzuoli and
the Gulf of Pozzuoli, mostly 0.5–3.5 km deep. Each year, half the events lie within about 1 km
of the centre; the outer edge has grown from about 2 km to about 3 km since 2023. The ground rises most
at station RITE on the Pozzuoli waterfront, about 1 km south-west of the seismic centre, and
uplift falls off with distance (from over 1 m since 2018 near the centre to 1–2 cm or less at
stations 9–10 km away).

**Which events count.** The catalogue (34,535 events since 2005) is only complete above a
magnitude of about Md 0.6 before 2017 and 0.5 after, because the seismic network grew. We count
events with M ≥ 0.6 (6,161 events), so that rates are comparable across the years. Small
magnitudes take only a few discrete values (Md comes from signal duration, apparently measured
in whole seconds), which limits how low the completeness magnitude can be measured.

![Monthly earthquake counts, 2005–2026](reports/figures/event_rate_monthly.png)

**How fast.** Events with M ≥ 0.6 were rare until about 2016 (most months had none or one). Since
2017–2018 the rate has climbed steadily, from about 7 a month in 2018 to about 180 a month in
2025, roughly doubling every 1.5 years, with large swarms on top (February 2025: 600 events at
M ≥ 0.6 in one month). 2026 so far is quieter than 2025.

**Uplift.** RITE has risen about 160 cm since January 2005, almost all after 2011. Since about
2013 the uplift has grown roughly exponentially (doubling about every 4 years, slowly
lengthening), and the uplift rate has reached 20–30 cm/year in pulses since 2023.

![Cumulative number of earthquakes against uplift at RITE](reports/figures/cumulative_count_vs_uplift.png)

**Earthquakes against uplift.** As in the paper we build on (arXiv 2503.20129), the cumulative
number of earthquakes rises smoothly with uplift: since 2012 it grows tenfold for every 73 cm of
uplift at RITE. Our event set (all events with M ≥ 0.6) gives almost the same curve as the
paper's (located events with M > 0.5). As a consequence, the same amount of uplift now produces
far more earthquakes than ten years ago: events per cm of uplift grew about 36-fold from 2014 to
2025, in step with the cumulative count.

**Week by week.** After removing the long-term trend, pulses of uplift and bursts of seismicity
happen at the same time, with at most a week's lead for the uplift. The uplift rate measured up
to the end of a week correlates only weakly with the next week's count (0.17), and not at all
four weeks ahead. So any forecasting skill from GPS is likely to be modest; the models in the
next phase will be compared against simple baselines that use past seismicity alone.

**Larger events.** There have been 91 events with M ≥ 3, all since December 2019 and most since
2023, falling in 51 of 399 weeks. They are more common than a single Gutenberg–Richter law fitted
to small events predicts, and the Gutenberg–Richter b-value has fallen from about 1.05 to about
0.8 during 2022–2023.

## Setup

With conda (recommended):

```bash
conda env create -f environment.yml
conda activate flegrei-seismicity
```

Without conda, use Python 3.14, then `pip install -r requirements.txt` and `pip install -e .`
(the second command makes the project's own code in `src/` importable). On macOS, LightGBM also
needs OpenMP (`brew install libomp`).

## Data

Raw data lives in `data/raw/` and is not committed. To download it (about 10 MB):

```bash
python scripts/download_data.py
```

This fetches:

| Folder in `data/raw/` | Contents |
|---|---|
| `catalogue/` | INGV-OV GOSSIP earthquake catalogue for Campi Flegrei, one CSV per year from 2005 |
| `gps_21_stations/` | Daily East/North/Up GPS displacements for 21 stations, 2000–2025 |
| `gps_rite/` | Daily and weekly vertical GPS displacement at station RITE, 2000–2024 |
| `osm/` | Coastline, lakes and place names from OpenStreetMap, for maps |

Each folder gets a `SOURCES.json` with every file's URL, download time (UTC) and SHA-256
checksum. The GPS files are also checked against the MD5 checksums published by Zenodo. The
catalogue is updated continuously, so a later download will contain more (and revised) events.

GPS station coordinates are in `data/reference/gps_stations.csv` (committed; see the README
there). To build the cleaned tables in `data/processed/` used by the notebooks:

```bash
python scripts/process_data.py
```

Then run the notebooks in order (`01_completeness` … `05_evaluation`). Notebooks 03–05 run
hundreds of weekly model fits; `05_evaluation` caches its forecasts in
`data/processed/forecasts/` (git-ignored). Delete that folder after changing model code, to
recompute them.

### Credits

The earthquake and GPS data are published under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

- Ricciolino, P., Lo Bascio, D., & Esposito, R. (2024). *GOSSIP - Database Sismologico Pubblico
  INGV-Osservatorio Vesuviano*. Istituto Nazionale di Geofisica e Vulcanologia (INGV).
  https://doi.org/10.13127/GOSSIP
- De Martino, P., Dolce, M., Brandi, G., Iorio, A., Scarpato, G., & Tammaro, U. (2026).
  *Continuous GPS time series of East, North and Up ground displacements in the Campi Flegrei
  caldera, Italy (2000-2025)* [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.22259337
- De Martino, P., Dolce, M., Brandi, G., Iorio, A., Romeo, A., Tammaro, U., & Scarpato, G.
  (2025). *Vertical displacement at Campi Flegrei: daily and weekly GNSS time series from station
  RITE (2000-2024)* [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.17132077
- De Martino, P., Dolce, M., Brandi, G., Scarpato, G., & Tammaro, U. (2021). The Ground
  Deformation History of the Neapolitan Volcanic Area (Campi Flegrei Caldera, Somma–Vesuvius
  Volcano, and Ischia Island) from 20 Years of Continuous GPS Observations (2000–2019).
  *Remote Sensing*, 13(14), 2725. https://doi.org/10.3390/rs13142725 (also the source of the
  station coordinates, from its Table 1).
- Map features: © [OpenStreetMap](https://www.openstreetmap.org/copyright) contributors,
  available under the [Open Database Licence (ODbL)](https://opendatacommons.org/licenses/odbl/).
