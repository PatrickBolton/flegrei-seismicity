# Campi Flegrei: can ground uplift forecast earthquake rates?

Campi Flegrei, the volcanic caldera beneath Pozzuoli near Naples, is in a long period of unrest:
the ground has risen by about 1.6 m since 2005 and earthquakes have become far more frequent.
This project asks whether **GPS measurements of that uplift can help forecast how many
earthquakes will happen in the coming week, and how likely a larger one (magnitude 3 or more)
is.**

It forecasts *rates and probabilities*, not individual earthquakes, which cannot be predicted.
It is a learning project in machine learning, built on public data, with all the code included
so the results can be checked and extended.

> **This is a research exercise, not a hazard forecast.** For official information about Campi
> Flegrei, see the [INGV Vesuvius Observatory (INGV-OV)](https://www.ov.ingv.it/) and the
> Italian [Civil Protection Department](https://www.protezionecivile.gov.it/).

**Short answer:** past earthquakes are the best guide to next week's earthquakes. Adding GPS
uplift, whether from the main station or combined over many stations, made no measurable
difference to the forecasts. A simple regression on recent earthquake counts also forecasts at
least as well as two physics-based models of the unrest, including a 2026 model in which the
uplift drives the earthquake rate, and letting the uplift drive that physics model didn't
measurably improve its week-ahead forecasts either.

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
3. tests honestly whether the GPS information improves the forecasts;
4. compares these models with physics-based forecasts, including the uplift-driven model of
   Hainzl et al. (2026), and tests whether giving the physics forecast to the machine-learning
   model helps.

## Related work

- **Godano et al. (2025)**, the starting point above: the cumulative number of earthquakes as a
  function of the uplift at one GPS station.
- **Petrillo & Taroni (2025)** extend ETAS, the standard statistical model of earthquakes
  triggering further earthquakes, with a background rate driven by the strain rate measured by
  GNSS (GPS), and find that it fits 2013–2024 better (a lower AIC).
- **Hainzl, Dahm & Tramelli (2026)** combine a rate-and-state model, in which the uplift since
  1905 loads faults, with ETAS. The combined model reproduces both the long-term trend and the
  short-term clustering, and gives weekly to monthly forecasts in a pseudo-prospective test.
  Their code and data are public; this project reuses them and reproduces their fitted
  parameters.
- **Iaccarino et al. (2025)** relate deformation rates and seismic moment to the magnitudes of
  the larger earthquakes in 20 sequences from 2015 to 2024.
- **Tan et al. (2025)** used machine learning on the continuous seismic records of 2022–2025 to
  expand the catalogue from about 12,000 to more than 54,000 earthquakes. This project uses the
  standard catalogue.

What this project adds: an out-of-sample comparison, week by week on the same 312 weeks and with
the same scores, of machine-learning and physics-based forecasts, with and without the uplift,
in open code.

References:

- Godano, C., Convertito, V., Tramelli, A., & Petrillo, G. (2025). How ground deformation
  influences earthquake occurrence during the ongoing unrest at Campi Flegrei (2005–present).
  arXiv:2503.20129. https://arxiv.org/abs/2503.20129
- Petrillo, G., & Taroni, M. (2025). Adding strain rate information into a short-term
  seismicity model improves forecasting performances: the case of Campi Flegrei, Italy.
  *Seismica*, 4(2). https://doi.org/10.26443/seismica.v4i2.1908
- Hainzl, S., Dahm, T., & Tramelli, A. (2026). A deformation-driven earthquake interaction model
  for seismicity at Campi Flegrei. *Communications Earth & Environment*, 7, 244.
  https://doi.org/10.1038/s43247-026-03296-3
- Iaccarino, A. G., Picozzi, M., De Landro, G., & Spallarossa, D. (2025). Preparatory phase of
  major earthquakes during Campi Flegrei unrest (2020–2024). *Journal of Geophysical Research:
  Solid Earth*, 130, e2025JB031777. https://doi.org/10.1029/2025JB031777
- Tan, X., Tramelli, A., Gammaldi, S., Beroza, G. C., Ellsworth, W. L., & Marzocchi, W. (2025).
  A clearer view of the current phase of unrest at Campi Flegrei caldera. *Science*, 390,
  70–75. https://doi.org/10.1126/science.adw9038

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
    stations;
  - physics-based models: ETAS (earthquakes triggering earthquakes), a rate-and-state model
    driven by the uplift since 1905, and the hybrid of the two by Hainzl et al. (2026). They are
    refitted every three months on earlier data only, and each week's forecast is the average
    of 2,000 simulations of that week;
  - the regression and LightGBM models given a physics forecast as an extra input.
- **The key tests:** each machine-learning model is trained twice, with and without the GPS
  inputs, and ETAS is compared with the hybrid, which adds the uplift. The difference in
  forecast quality is what the uplift adds.
- **Scoring** uses standard measures of forecast quality that reward forecasts for being both
  accurate and honest about their uncertainty, with 95% uncertainty intervals for every
  comparison, so that chance differences aren't mistaken for real ones.

## Results

### Does GPS help the machine-learning models?

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

### How do they compare with physics-based models?

Forecasts of next week's number of earthquakes, compared with the simple benchmark (next week
like the average of the last four). Positive means better than the benchmark; 95% intervals in
brackets; * marks differences that are statistically significant.

| Model | 2020–2025 | 2024–2025 |
|---|---|---|
| Rate-and-state (uplift only) | −1.1% [−3.7, +1.5] | −3.9% [−7.6, −0.8]* |
| ETAS (earthquakes only) | +0.7% [−0.9, +2.3] | +1.7% [+0.1, +3.1]* |
| Hybrid of Hainzl et al. (uplift + earthquakes) | +1.6% [−0.6, +3.7] | +1.3% [−1.4, +3.4] |
| Regression on recent earthquakes | +2.4% [+0.8, +4.2]* | +2.8% [+0.4, +5.1]* |
| Same, with GPS inputs | +2.3% [+0.3, +4.4]* | +2.4% [−0.4, +4.8] |
| Regression on recent earthquakes + the hybrid's forecast | +2.7% [+1.1, +4.5]* | +2.9% [+0.4, +5.2]* |

- **Machine learning against physics:** the regression on recent earthquakes is at least as
  good as both physics models, and significantly better than ETAS over 2020–2025.
- **Does the uplift help the physics model?** Not measurably, a week ahead. Fitted to the whole
  period at once, the hybrid explains the 20 years of seismicity far better than ETAS, as
  Hainzl et al. and Petrillo & Taroni found. But forecasting each week from earlier data only,
  the hybrid and ETAS are within noise of each other (0.8% apart over 2020–2025, 0.4% the other
  way in 2024–2025). The uplift alone (rate-and-state) is worse than the simple benchmark.
- **Physics forecast as an input:** adding the hybrid's forecast to the regression gives the
  best model overall, but only 0.3% better than the regression alone over 2020–2025, and no
  better in 2024–2025. The physics forecast is 97–98% predictable from the recent earthquake
  counts the regression already has.
- **Larger earthquakes:** no physics-based or combined forecast beats the best simple forecast
  of the chance of a magnitude ≥ 3 next week.

The same reason as for GPS applies: over 20 years the uplift explains the growth of
seismicity, but a week ahead that growth is already visible in the recent earthquakes.

Other findings:

- **The 2025 study's model can be reproduced** (we obtain the same parameters, 0.75 and 0.21,
  against the published 0.76 and 0.2), but its parameters depend on an assumption the data
  can't check, and it underestimates the earthquakes after May 2024. As a weekly forecast it
  gets the overall level right but misses swarms.
- **Hainzl et al.'s model can be reproduced exactly** from their published code and data (same
  likelihood, same parameters: for example 7.48 against their 7.5 earthquakes a year, and a
  relaxation time of 34.4 against 34.2 years).
- **Within-week aftershocks matter.** In the fitted physics models each earthquake
  triggers on average 0.8–1 further earthquakes, mostly within days, so a weekly physics forecast that ignores
  the aftershocks of the week's own earthquakes is 2–3 times too low.
- **Honest uncertainty matters.** Simple forecasts badly understate how variable the weekly
  numbers are (their "80% ranges" contained the true count only 45% of the time); allowing for
  swarms fixes this (86–90%).
- **Larger earthquakes are hard to forecast** beyond how often they have happened recently: the
  best forecasts pick out the riskier weeks only modestly better than chance.

**Limitations:** weekly forecasts up to four weeks ahead only (one week for the physics
models); the physics models are fitted to our catalogue at magnitude 0.6 rather than the
published 0.45, and their fits are poorly determined before 2022, when there were few
earthquakes; the catalogue's magnitudes are
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
| [`06_physics_baselines`](notebooks/06_physics_baselines.ipynb) | ETAS, rate-and-state and the hybrid of Hainzl et al.: implementation checked against the authors' code, then weekly forecasts compared with the models above |
| [`07_ml_vs_physics`](notebooks/07_ml_vs_physics.ipynb) | The regression and LightGBM models given the physics forecast as an input or an offset |

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

Raw data lives in `data/raw/` and is not committed. To download it (about 40 MB):

```bash
python scripts/download_data.py
```

This fetches:

| Folder in `data/raw/` | Contents |
|---|---|
| `catalogue/` | INGV-OV GOSSIP earthquake catalogue for Campi Flegrei, one CSV per year from 2005 |
| `gps_21_stations/` | Daily East/North/Up GPS displacements for 21 stations, 2000–2025 |
| `gps_rite/` | Daily and weekly vertical GPS displacement at station RITE, 2000–2024 |
| `hainzl/` | Code and data of Hainzl, Dahm & Tramelli (2026): their catalogue, an uplift series since 1905 (levelling joined to GPS), fitted parameters and forecasts. Used for the physics baselines |
| `osm/` | Coastline, lakes and place names from OpenStreetMap, for maps |

Each folder gets a `SOURCES.json` with every file's URL, download time (UTC) and SHA-256
checksum. The Zenodo files (GPS and Hainzl et al.) are also checked against the MD5 checksums published by Zenodo. The
catalogue is updated continuously, so a later download will contain more (and revised) events.

GPS station coordinates are in `data/reference/gps_stations.csv` (committed; see the README
there). To build the cleaned tables in `data/processed/` used by the notebooks:

```bash
python scripts/process_data.py
```

Then run the notebooks in order (`01_completeness` … `07_ml_vs_physics`). Notebooks 03–05 run
hundreds of weekly model fits; `05_evaluation` caches its forecasts in
`data/processed/forecasts/` (git-ignored), as do `06` and `07` (the physics forecasts take
about 15 minutes the first time). Delete that folder after changing model code, to
recompute them.

## Credits

The earthquake and GPS data, and Hainzl et al.'s code and data, are published under
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
- Hainzl, S. (2026). *Codes and files related to Hainzl, Dahm & Tramelli (2026)* [Dataset].
  Zenodo. https://doi.org/10.5281/zenodo.18413119
- Map features: © [OpenStreetMap](https://www.openstreetmap.org/copyright) contributors,
  available under the [Open Database Licence (ODbL)](https://opendatacommons.org/licenses/odbl/).

## Licence

The code is released under the [MIT licence](LICENSE). The data keep their own licences, listed
above.
