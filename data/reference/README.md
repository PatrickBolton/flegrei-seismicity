# Reference data

Small, hand-entered tables of public metadata. Unlike `data/raw/` and `data/processed/`,
these are committed, because no script downloads them.

## gps_stations.csv

Latitude and longitude (degrees north and east, WGS84) of the 21 Campi Flegrei cGPS stations
in the GPS dataset (Zenodo 10.5281/zenodo.22259337).

- **Source:** Table 1 of De Martino, P.; Dolce, M.; Brandi, G.; Scarpato, G.; Tammaro, U. (2021).
  The Ground Deformation History of the Neapolitan Volcanic Area (Campi Flegrei Caldera,
  Somma–Vesuvius Volcano, and Ischia Island) from 20 Years of Continuous GPS Observations
  (2000–2019). *Remote Sensing* 13(14), 2725. https://doi.org/10.3390/rs13142725 (CC BY 4.0).
- **Entered:** 23 Oct 2026, copied from the article's HTML table (https://www.mdpi.com/2072-4292/13/14/2725).
- **Precision:** 3 decimal places, about 100 m: enough for maps and distances at caldera scale.
- The table also lists the 4 seafloor (MEDUSA) stations and the Ischia–Procida and Somma–Vesuvius
  stations; they are not in our GPS dataset and are left out.
- The paper gives FRUL's first observation as 2008.1, but our GPS files have FRUL data from
  February 2003 (with a gap in 2005–2006). Not resolved; it doesn't affect the coordinates.
