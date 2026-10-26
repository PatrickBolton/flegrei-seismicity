# Campi Flegrei: can ground uplift forecast earthquake rates?

Campi Flegrei, the volcanic caldera beneath Pozzuoli near Naples, is in a long period of unrest:
the ground has risen by about 1.6 m since 2005 and earthquakes have become far more frequent.
This project asks whether **GPS measurements of that uplift can help forecast how many
earthquakes will happen in the coming week, and how likely a larger one (magnitude 3 or more)
is.**

It forecasts *rates and probabilities*, not individual earthquakes, which cannot be predicted.
It is a learning project in machine learning, built on public data, with all the code included
so the results can be checked and extended.

**Short answer:** past earthquakes are the best guide to next week's earthquakes. Adding GPS
uplift, whether from the main station or combined over many stations, made no measurable
difference to the forecasts.

## The question, and where it comes from

A 2025 study ([Godano et al., arXiv 2503.20129](https://arxiv.org/abs/2503.20129)) showed that
the cumulative number of earthquakes at Campi Flegrei rises smoothly with the uplift at one GPS
station, and explained this with a simple "epidemic" model in which uplift loads the rock and
earthquakes trigger further earthquakes. It released no code and did not test forecasts.

This project:

1. reproduces that model in open code;
2. builds forecasting models that use the uplift at **many GPS stations** (the main station
   RITE, and combined rates from 10 others: 6 near the centre of uplift and 4 far from it),
   together with past earthquakes;
3. tests honestly whether the GPS information improves the forecasts.

## What the data show

The data are the earthquake catalogue of INGV's Vesuvius Observatory (34,535 events since 2005)
and daily GPS positions from 21 stations (2000–2025), all public (see [Credits](#credits)).

![Map of located earthquakes with magnitude 0.6 or more, and the 21 GPS stations](reports/figures/map_events_stations.png)

- **The earthquakes are tightly clustered** around the Solfatara crater, under Pozzuoli and its
  gulf, mostly 0.5–3.5 km deep. The ground rises most at station RITE on the Pozzuoli
  waterfront, and much less with distance.
- **Small earthquakes are not always recorded,** especially before the seismic network grew in
  the 2010s. To compare years fairly, the forecasts count only events of magnitude 0.6 or more,
  which the network has recorded reliably since 2005 (6,161 events).
- **Both uplift and earthquakes have accelerated.** The uplift rate rose from a few cm a year
  before 2011 to 20–30 cm a year in pulses since 2023; the number of earthquakes doubled
  roughly every 1.5 years from 2018, with large swarms on top (600 events in February 2025).
- **The 2025 study's relationship holds:** since 2012 the cumulative number of earthquakes has
  grown tenfold for every 73 cm of uplift (below). The same uplift now produces far more
  earthquakes than ten years ago.

![Cumulative number of earthquakes against the uplift at station RITE](reports/figures/cumulative_count_vs_uplift.png)

- **Week by week, the link is weak and simultaneous.** Bursts of uplift and bursts of
  earthquakes happen in the same weeks; the uplift doesn't clearly lead. That already hinted
  that GPS might not help much with forecasting next week.

## Method

- **What is forecast:** each week, (1) the number of earthquakes of magnitude 0.6 or more next
  week, and (2) the probability of at least one earthquake of magnitude 3 or more.
- **How forecasts are tested:** each of the 312 weeks from 2020 to 2025 is forecast using only
  data from before that week, exactly as a forecaster would have had to, and the forecast is
  then compared with what happened. Results are also given for 2024–2025 alone, when the level
  of activity changed least.
- **What is compared:**
  - simple benchmarks (next week like the last four weeks; the recent frequency of larger
    events);
  - the 2025 study's model, turned into a weekly forecast;
  - statistical and machine-learning models (Poisson regression and gradient-boosted trees,
    LightGBM) using 19 inputs: recent earthquake counts, and uplift and uplift rates from the GPS
    stations.
- **The key test:** each model is trained twice, with and without the GPS inputs. The
  difference in forecast quality is what GPS adds.
- **Scoring** uses standard measures of forecast quality that reward forecasts for being both
  accurate and honest about their uncertainty, with 95% uncertainty intervals for every
  comparison, so that chance differences aren't mistaken for real ones.

## Results

| Forecast | Best model | Effect of adding GPS |
|---|---|---|
| Earthquakes next week | Regression on recent earthquake counts: about 2.4% better than the simple benchmark (a real, statistically significant gain) | +0.2% worse (interval −0.8% to +1.1%): **no effect** |
| Chance of a magnitude ≥ 3 next week | Recent earthquake rate, allowing for clustering in swarms: 3–5% better than the recent frequency of such weeks, but **not significant** on the main score (only one of the two scores, over 2020–2025, shows a significant gain) | +1.2% worse (interval +0.3% to +2.6%): **slightly harmful** |

The chart below shows the effect of adding GPS in the main setup and in the variations that
change each model (different training periods, a 4-week forecast instead of 1 week, a lower
magnitude threshold). Points left of the line would mean GPS helps; the bars show 95% intervals.

![Effect of adding GPS inputs on forecast quality, in the main setup and its variations](reports/figures/gps_effect_sensitivity.png)

**In every setup, GPS adds nothing measurable.** The tests were sensitive enough to detect
gains of about 1–3%, the size of the real gains the models do make from earthquake data, so
this is not just a lack of data.

**Why:** the slow part of the uplift signal (how far the ground has risen) tells the models the
same thing as the running total of earthquakes, which they already have. The fast part (how
quickly it is rising this week) is weak, noisy, and happens at the same time as the earthquake
bursts rather than before them.

Other findings:

- **The 2025 study's model can be reproduced** (we obtain the same parameters, 0.75 and 0.21,
  against the published 0.76 and 0.2), but its parameters depend on an assumption the data
  can't check, and it underestimates the earthquakes after May 2024. As a weekly forecast it
  gets the overall level right but misses swarms.
- **Honest uncertainty matters.** Simple forecasts badly understate how variable the weekly
  numbers are (their "80% ranges" contained the true count only 45% of the time); allowing for
  swarms fixes this (86–90%).
- **Larger earthquakes are hard to forecast** beyond how often they have happened recently: the
  best forecasts pick out the riskier weeks only modestly better than chance.

**Limitations:** weekly forecasts up to four weeks ahead only; the catalogue's magnitudes are
duration magnitudes, which can differ by a few tenths from other magnitude scales; only 43 test
weeks contained an earthquake of magnitude 3 or more, so those results are uncertain; and the
GPS data end in December 2025. Uplift may still matter for longer horizons (months to years),
for other questions such as how large a swarm can grow, or with other deformation data such as
satellite radar (InSAR).

## Notebooks

The details, code and many more plots are in the notebooks, which are meant to be read in
order:

| Notebook | Contents |
|---|---|
| [`01_completeness`](notebooks/01_completeness.ipynb) | Which earthquakes are reliably recorded (magnitude of completeness), and the Gutenberg–Richter b-value over time |
| [`02_explore`](notebooks/02_explore.ipynb) | Exploration: earthquake rate, uplift, their relationship, larger events, map |
| [`03_baselines`](notebooks/03_baselines.ipynb) | Simple benchmarks, and the 2025 study's model reproduced and turned into a forecast |
| [`04_models`](notebooks/04_models.ipynb) | Inputs, regression and LightGBM models with and without GPS, larger events, explanations of the models (SHAP) |
| [`05_evaluation`](notebooks/05_evaluation.ipynb) | Full evaluation: uncertainty of forecasts, detectable effects, and the six variations |

## Reproducing the results

### Setup

With conda (recommended):

```bash
conda env create -f environment.yml
conda activate flegrei-seismicity
```

Without conda, use Python 3.14, then `pip install -r requirements.txt` and `pip install -e .`
(the second command makes the project's own code in `src/` importable). On macOS, LightGBM also
needs OpenMP (`brew install libomp`).

### Data

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

## Credits

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

## Licence

The code is released under the [MIT licence](LICENSE). The data keep their own licences, listed
above.
