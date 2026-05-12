# CHIRPS — South African Rainfall Analysis

A Python toolkit for downloading, processing, and visualising the CHIRPS v3 daily rainfall dataset clipped to South Africa.

---

## What is CHIRPS?

CHIRPS (Climate Hazards InfraRed Precipitation with Station) is a daily rainfall dataset produced by the [Climate Hazards Center at UC Santa Barbara](https://www.chc.ucsb.edu/). It covers all land areas between 60°S and 60°N at a resolution of 0.05° (~2.5 km grid), from 1981 to the present — over 40 years of consistent, gridded precipitation data.

Version 3, launched in early 2025, draws on over 90 sources of station data and incorporates an improved high-resolution climatology developed in partnership with the Global Precipitation Climatology Centre. It is meaningfully better at capturing extreme rainfall events than its predecessor.

The full global dataset runs to approximately 23 GB per year. This repository works with a South Africa extract, reducing this to a manageable ~2 GB compressed file for all 45 years.

---

## Why does this matter for South African insurers?

South Africa's weather station network has contracted over the decades. Gauge data in rural areas is patchy at best. CHIRPS fills that gap with a 44-year consistent record of daily rainfall across every square kilometre of the country.

The predictive value of CHIRPS for insurance claims was demonstrated at the [2023 ASSA Convention](https://www.actuarialsociety.org.za/) — *Claims Modelling with Climate Data* (Richman, Mbuvha, Perumal, Balusik & Balona, 2023) — and the dataset is used in practice by organisations such as [Arbol](https://www.arbol.io/) for parametric weather products and [FEWS Net](https://fews.net/) for early warning drought monitoring.

---

## Repository contents

| File | Description |
|---|---|
| `chirps_sa_pipeline.py` | Downloads monthly CHIRPS v3 daily files and clips them to the South African bounding box, producing a compressed NetCDF output |
| `chirps_heatmap.py` | Pixel-level heatmap dashboard comparing two user-defined periods across multiple rainfall metrics, with OpenStreetMap basemap |
| `chirps_visualise.py` | Additional visualisation utilities for exploring the clipped dataset |
| `weather_station.py` | Utilities for working with SAWS weather station data alongside CHIRPS |
| `tests.py` | Basic tests for the pipeline and processing functions |
| `wgetem` | Shell script for batch downloading raw CHIRPS files via wget |

---

## Getting started

### Prerequisites

```bash
pip install xarray netcdf4 geopandas rasterio shapely numpy pandas plotly
```

### 1. Download and clip

Edit the config section of `chirps_sa_pipeline.py` (output path, date range) then run:

```bash
python chirps_sa_pipeline.py
```

This downloads monthly daily precipitation files from the CHIRPS FTP server, clips each to the South African extent, and writes a single compressed NetCDF file (~2 GB).

### 2. Visualise

```bash
python chirps_heatmap.py
```

Produces a self-contained HTML dashboard comparing rainfall metrics across two user-defined periods, rendered on an OpenStreetMap basemap. Open the output file in any browser — no server required.

---

## Data requirements

- **Storage:** ~2 GB for the South Africa extract; ~23 GB per year for the full global dataset
- **Internet:** A stable connection is needed for the initial download
- **Shapefiles:** Province-level shapefiles (e.g. from [GADM](https://gadm.org/)) are used for masking and boundary overlays

The raw CHIRPS files and weather station data are excluded from this repository via `.gitignore`.

---

## Metrics available in the dashboard

| Metric | Description |
|---|---|
| Average daily rainfall | Mean mm/day across all days in the period |
| % days exceeding threshold | Proportion of days above a configurable mm threshold |
| 95th percentile (mean) | 95th percentile of wet-day rainfall, averaged across pixels |
| 95th percentile (max) | 95th percentile of wet-day rainfall, pixel maximum |

All metrics are computed per pixel and compared across two configurable periods, with a difference map showing change over time.

---

## Limitations

- Daily disaggregation from the five-day CHIRPS product introduces some smoothing
- Pixel-level values do not always correlate precisely with individual rain gauges
- Satellite-based retrieval can be confused by certain weather conditions (orographic effects, shallow convection)

---

## References

- Funk, C. et al. (2015). The Climate Hazards Infrared Precipitation with Stations — a new environmental record for monitoring extremes. *Scientific Data*, 2, 150066.
- Richman, Mbuvha, Perumal, Balusik & Balona (2023). Claims Modelling with Climate Data. *2023 ASSA Convention*.

---

## Licence

Apache 2.0 — see [LICENSE](LICENSE) for details.

---

*All views expressed are my own.*
