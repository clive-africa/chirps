"""
CHIRPS v3 Interactive Rainfall Dashboard
=========================================
Generates a self-contained HTML dashboard with three interactive choropleth maps:
  1. Average daily rainfall per geometry
  2. Percentage of days exceeding 50 mm per geometry
  3. 95th percentile daily rainfall per geometry

Each map has a toggle between two 20-year periods:
  • Period 1: 1981–2000
  • Period 2: 2005–2024

Edit the CONFIG section below to point to your files, then run with F5 or Ctrl+F5.
"""

import json
import os
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — edit these paths before running
# ─────────────────────────────────────────────────────────────────────────────

# Set one of NETCDF_PATH or CSV_PATH; leave the other as None
NETCDF_PATH  = r"chirps_output/chirps_v3_south_africa.nc"   # combined NetCDF
CSV_PATH     = None   # long-format zonal CSV (faster if already computed)

SHAPEFILE    = r"shapefiles/zaf_admin4.shp"      # path to your .shp file
ZONE_FIELD   = "adm4_name"                  # column name for zone labels e.g. "PROVINCE"; None = auto-number
OUTPUT_HTML  = r"chirps_output/rainfall_dashboard.html"

EXCEEDANCE_THRESHOLD = 50  # mm — days above this count as significant rainfall events

# ─────────────────────────────────────────────────────────────────────────────
# PERIODS
# ─────────────────────────────────────────────────────────────────────────────

PERIODS = {
    "1981 – 2000": ("1981-01-01", "2000-12-31"),
    "2006 – 2025": ("2006-01-01", "2025-12-31"),
}

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR SCALES
# ─────────────────────────────────────────────────────────────────────────────

COLORSCALES = {
    "avg":     "Blues",
    "pct":     "YlOrRd",
    "p95":     "Purples",
    "p95_max": "Oranges",
    "max":     "Reds",
}

TITLES = {
    "avg":     "Average Daily Rainfall (mm)",
    "pct":     f"% Days Exceeding {EXCEEDANCE_THRESHOLD} mm",
    "p95":     "95th Percentile — Province Mean (mm)",
    "p95_max": "95th Percentile — Spatial Max (mm)",
    "max":     "Maximum — Province Mean (mm)",
}

LABELS = {
    "avg":     "mm/day",
    "pct":     "% days",
    "p95":     "mm",
    "p95_max": "mm",
    "max":     "mm",
}


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_shapefile(shp_path: str, zone_field: str | None, simplify_tolerance: float = 0.005) -> gpd.GeoDataFrame:
    """
    Load shapefile and optionally simplify geometries to reduce output file size.

    simplify_tolerance: degrees. 0.005 ≈ 500m — invisible at country scale but
    typically reduces vertex count by 80–95% for detailed sub-place boundaries.
    Set to 0 to disable simplification.
    """
    os.environ["SHAPE_RESTORE_SHX"] = "YES"
    gdf = gpd.read_file(shp_path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")

    gdf["zone_id"] = range(len(gdf))

    if zone_field and zone_field in gdf.columns:
        gdf["zone_label"] = gdf[zone_field].astype(str)
    else:
        gdf["zone_label"] = [f"Zone {i+1}" for i in range(len(gdf))]

    # Simplify geometries to reduce GeoJSON size
    if simplify_tolerance > 0:
        before = sum(len(g.wkt) for g in gdf.geometry)
        gdf["geometry"] = gdf.geometry.simplify(
            tolerance=simplify_tolerance,
            preserve_topology=True,
        )
        after = sum(len(g.wkt) for g in gdf.geometry)
        print(f"  Loaded shapefile: {len(gdf)} zones")
        print(f"  Geometry simplified: {before/1e6:.1f} MB → {after/1e6:.1f} MB WKT "
              f"(tolerance={simplify_tolerance}°)")
    else:
        print(f"  Loaded shapefile: {len(gdf)} zones (no simplification)")

    return gdf


def load_from_csv(csv_path: str) -> pd.DataFrame:
    """Load pre-computed long-format zonal stats CSV."""
    df = pd.read_csv(csv_path, parse_dates=["date"])
    print(f"  Loaded CSV: {len(df):,} rows, "
          f"{df['zone_label'].nunique()} zones, "
          f"{df['date'].min().date()} to {df['date'].max().date()}")
    return df


def load_from_netcdf(nc_path: str, gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Compute zonal daily rainfall from the combined NetCDF.
    Uses a pixel-centroid-in-polygon approach — fast enough for provinces.
    For 20k sub-places use the full pipeline with exactextract instead.
    """
    import xarray as xr

    print("  Opening NetCDF...")
    ds = xr.open_dataset(nc_path, engine="netcdf4")

    # Auto-detect precip variable
    precip_var = next(
        (v for v in ds.data_vars
         if any(k in v.lower() for k in ["precip", "rain", "prcp", "chirps"])),
        list(ds.data_vars)[0]
    )
    print(f"  Precipitation variable: '{precip_var}'")

    # Normalise coordinates
    rename = {}
    for c in ds.coords:
        if c.lower() == "latitude":  rename[c] = "lat"
        if c.lower() == "longitude": rename[c] = "lon"
    if rename:
        ds = ds.rename(rename)

    lons = ds.lon.values
    lats = ds.lat.values
    times = pd.DatetimeIndex(ds.time.values)
    precip = ds[precip_var].values  # (time, lat, lon)

    # Mask fill values
    precip = np.where(precip < -999, np.nan, precip)

    # Build grid of pixel centroids
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    points = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(lon_grid.ravel(), lat_grid.ravel()),
        crs="EPSG:4326"
    )
    points["pixel_idx"] = np.arange(len(points))

    # Spatial join: which zone does each pixel belong to?
    print("  Spatial join (pixels → zones)...")
    joined = gpd.sjoin(points, gdf[["zone_id", "zone_label", "geometry"]],
                       how="left", predicate="within")
    joined = joined.dropna(subset=["zone_id"])
    joined["zone_id"] = joined["zone_id"].astype(int)

    # Group pixel indices by zone
    zone_pixels = joined.groupby("zone_id")["pixel_idx"].apply(list)

    print(f"  Computing daily zonal means for {len(times)} days...")
    records = []
    precip_flat = precip.reshape(len(times), -1)  # (time, pixels)

    for zone_id, pixels in zone_pixels.items():
        zone_label = gdf.loc[gdf["zone_id"] == zone_id, "zone_label"].iloc[0]
        zone_data = precip_flat[:, pixels]  # (time, n_pixels)
        daily_mean = np.nanmean(zone_data, axis=1)   # spatial mean per day
        daily_max  = np.nanmax(zone_data, axis=1)    # spatial max per day

        for t_idx, date in enumerate(times):
            records.append({
                "zone_id":        zone_id,
                "zone_label":     zone_label,
                "date":           date,
                "rainfall_mm":    float(daily_mean[t_idx]),
                "rainfall_max_mm": float(daily_max[t_idx]),
            })

    df = pd.DataFrame(records)
    print(f"  Built DataFrame: {len(df):,} rows")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

# Days with mean rainfall below this are treated as dry and excluded from p95
WET_DAY_THRESHOLD = 1.0  # mm


def compute_stats(df: pd.DataFrame, period_name: str, period_dates: tuple) -> pd.DataFrame:
    """
    Compute summary statistics per zone for a given time period.

    Metrics:
      avg     — mean daily rainfall over ALL days (including dry days)
      pct     — % of days where province mean exceeds EXCEEDANCE_THRESHOLD
      p95     — 95th pct of province-mean rainfall on WET days only (mean >= 1mm)
      p95_max — 95th pct of the province spatial-MAX rainfall on WET days only

    Dry days (mean < WET_DAY_THRESHOLD) are excluded from p95 and p95_max so that
    the percentile reflects event intensity rather than being diluted by zero-rain days.
    """
    start, end = period_dates
    mask = (df["date"] >= start) & (df["date"] <= end)
    sub = df[mask].copy()

    if sub.empty:
        print(f"  WARNING: No data found for period {period_name} ({start} to {end})")
        return pd.DataFrame()

    has_max_col = "rainfall_max_mm" in sub.columns

    def p95_wet_mean(x):
        wet = x[x >= WET_DAY_THRESHOLD]
        return np.nanpercentile(wet, 95) if len(wet) > 0 else np.nan

    def p95_wet_max(x):
        wet = x[x >= WET_DAY_THRESHOLD]
        return np.nanpercentile(wet, 95) if len(wet) > 0 else np.nan

    agg_dict = {
        "avg":      ("rainfall_mm", "mean"),
        "p95":      ("rainfall_mm", p95_wet_mean),
        "n_days":   ("rainfall_mm", "count"),
        "n_exceed": ("rainfall_mm", lambda x: (x > EXCEEDANCE_THRESHOLD).sum()),
        "max":      ("rainfall_mm", "max"),
    }

    if has_max_col:
        agg_dict["p95_max"] = ("rainfall_max_mm", p95_wet_max)

    stats = sub.groupby(["zone_id", "zone_label"]).agg(**agg_dict).reset_index()

    stats["pct"] = (stats["n_exceed"] / stats["n_days"] * 100).round(2)
    stats["avg"] = stats["avg"].round(2)
    stats["p95"] = stats["p95"].round(2)
    stats["max"] = stats["max"].round(2)
    if has_max_col:
        stats["p95_max"] = stats["p95_max"].round(2)
    else:
        stats["p95_max"] = np.nan

    stats["period"] = period_name
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

def build_dashboard(
    all_stats: dict[str, pd.DataFrame],
    gdf: gpd.GeoDataFrame,
    output_path: str,
) -> None:
    """Build a single self-contained HTML dashboard with plotly."""

    # Convert shapefile to GeoJSON with reduced coordinate precision
    # 5 decimal places ≈ 1m precision — more than enough for visualisation
    raw = json.loads(gdf[["zone_id", "zone_label", "geometry"]].to_json())

    def round_coords(obj, decimals=5):
        """Recursively round all coordinates to reduce GeoJSON size."""
        if isinstance(obj, list):
            if obj and isinstance(obj[0], (int, float)):
                return [round(v, decimals) for v in obj]
            return [round_coords(item, decimals) for item in obj]
        if isinstance(obj, dict):
            return {k: round_coords(v, decimals) for k, v in obj.items()}
        return obj

    geojson = round_coords(raw)

    # Compute shared colour ranges across both periods (so colours are comparable)
    combined = pd.concat(all_stats.values())
    ranges = {
        metric: (float(combined[metric].min()), float(combined[metric].max()))
        for metric in ["avg", "pct", "p95", "p95_max","max"]
        if metric in combined.columns
    }

    period_names = list(PERIODS.keys())
    metrics      = ["avg", "pct", "p95", "p95_max","max"]

    p1_name, p2_name = period_names[0], period_names[1]

    # ── Compute difference stats (later minus earlier) ─────────────────────
    diff_stats = {}
    for metric in metrics:
        s1 = all_stats[p1_name][["zone_id", "zone_label", metric]].copy()
        s2 = all_stats[p2_name][["zone_id", "zone_label", metric]].copy()
        merged = s1.merge(s2, on=["zone_id", "zone_label"], suffixes=("_p1", "_p2"))
        merged["diff"] = merged[f"{metric}_p2"] - merged[f"{metric}_p1"]
        merged["pct_chg"] = (merged["diff"] / merged[f"{metric}_p1"].abs() * 100).round(1)
        diff_stats[metric] = merged

    # ── 4 rows × 3 cols: rows = metrics, cols = period1 | period2 | change ─
    col_titles = [p1_name, p2_name, "Change (later − earlier)"]
    subplot_titles = []
    for metric in metrics:
        for ct in col_titles:
            subplot_titles.append(ct)

    fig = make_subplots(
        rows=5, cols=3,
        subplot_titles=subplot_titles,
        vertical_spacing=0.06,
        horizontal_spacing=0.015,
        specs=[
            [{"type": "choropleth"}, {"type": "choropleth"}, {"type": "choropleth"}],
            [{"type": "choropleth"}, {"type": "choropleth"}, {"type": "choropleth"}],
            [{"type": "choropleth"}, {"type": "choropleth"}, {"type": "choropleth"}],
            [{"type": "choropleth"}, {"type": "choropleth"}, {"type": "choropleth"}],
            [{"type": "choropleth"}, {"type": "choropleth"}, {"type": "choropleth"}],
        ],
    )

    geo_axes   = []
    cb_y_pos = {0: 0.93, 1: 0.72, 2: 0.51, 3: 0.30, 4: 0.09}

    sa_geo = dict(
        scope="africa",
        resolution=50,
        showland=True,
        landcolor="#f0ede6",
        showocean=True,
        oceancolor="#daeef7",
        showlakes=True,
        lakecolor="#daeef7",
        showrivers=True,
        rivercolor="#b0d4e8",
        showcountries=True,
        countrycolor="#aaaaaa",
        showcoastlines=True,
        coastlinecolor="#777777",
        lonaxis=dict(range=[15.5, 34.0]),
        lataxis=dict(range=[-35.5, -21.5]),
        bgcolor="rgba(0,0,0,0)",
    )

    for m_idx, metric in enumerate(metrics):
        row = m_idx + 1

        # ── Cols 1 & 2: individual periods ────────────────────────────────
        for p_idx, period_name in enumerate([p1_name, p2_name]):
            col   = p_idx + 1
            stats = all_stats[period_name]

            hover_text = [
                f"<b>{r['zone_label']}</b><br>"
                f"{TITLES[metric]}: {r[metric]:.2f} {LABELS[metric]}<br>"
                f"Period: {period_name}"
                for _, r in stats.iterrows()
            ]

            # Show colourbar only on col 2 (right of the pair)
            show_cb = (p_idx == 1)

            trace = go.Choropleth(
                geojson=geojson,
                locations=stats["zone_id"],
                z=stats[metric],
                featureidkey="properties.zone_id",
                colorscale=COLORSCALES[metric],
                marker=dict(line=dict(width=0)),
                zmin=ranges[metric][0],
                zmax=ranges[metric][1],
                colorbar=dict(
                    title=dict(text=LABELS[metric], font=dict(size=9)),
                    len=0.19,
                    y=cb_y_pos[m_idx],
                    yanchor="middle",
                    thickness=11,
                    x=0.645,
                    tickfont=dict(size=8),
                ) if show_cb else None,
                showscale=show_cb,
                hovertext=hover_text,
                hoverinfo="text",
                name=f"{period_name} | {TITLES[metric]}",
                showlegend=False,
            )
            fig.add_trace(trace, row=row, col=col)

            geo_idx = (m_idx * 3) + p_idx + 1
            geo_axes.append("geo" if geo_idx == 1 else f"geo{geo_idx}")

        # ── Col 3: difference (diverging scale, centred at 0) ─────────────
        diff = diff_stats[metric]

        abs_max = diff["diff"].abs().max()
        # Round to a clean number for the colour range
        abs_max = max(abs_max, 0.01)

        diff_hover = [
            f"<b>{r['zone_label']}</b><br>"
            f"Change: {r['diff']:+.2f} {LABELS[metric]}<br>"
            f"({r['pct_chg']:+.1f}%)<br>"
            f"{p1_name}: {r[f'{metric}_p1']:.2f} {LABELS[metric]}<br>"
            f"{p2_name}: {r[f'{metric}_p2']:.2f} {LABELS[metric]}"
            for _, r in diff.iterrows()
        ]

        diff_trace = go.Choropleth(
            geojson=geojson,
            locations=diff["zone_id"],
            z=diff["diff"],
            featureidkey="properties.zone_id",
            colorscale="RdBu",
            reversescale=False,      # blue = higher/wetter, red = lower/drier
            marker=dict(line=dict(width=0)),
            zmid=0,
            zmin=-abs_max,
            zmax=abs_max,
            colorbar=dict(
                title=dict(
                    text=f"Δ {LABELS[metric]}",
                    font=dict(size=9),
                ),
                len=0.19,
                y=cb_y_pos[m_idx],
                yanchor="middle",
                thickness=11,
                x=1.01,
                tickfont=dict(size=8),
                tickformat="+.1f",
            ),
            showscale=True,
            hovertext=diff_hover,
            hoverinfo="text",
            name=f"Change | {TITLES[metric]}",
            showlegend=False,
        )
        fig.add_trace(diff_trace, row=row, col=3)

        geo_idx = (m_idx * 3) + 3
        geo_axes.append("geo" if geo_idx == 1 else f"geo{geo_idx}")

    # ── Apply SA extent to all geo axes ───────────────────────────────────
    geo_updates = {geo_key: sa_geo for geo_key in geo_axes}

    # ── Row labels (rotated, left margin) ─────────────────────────────────
    row_label_y = {0: 0.93, 1: 0.72, 2: 0.51, 3: 0.30, 4: 0.09}
    annotations = [
        dict(
            text=f"<b>{TITLES[metric]}</b>",
            x=-0.01,
            y=row_label_y[m_idx],
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=11, family="Georgia, serif", color="#222222"),
            xanchor="right",
            textangle=-90,
        )
        for m_idx, metric in enumerate(metrics)
    ] + [
        # Legend for the change column
        dict(
            text="🔵 wetter  🔴 drier  in the later period",
            x=0.835, y=-0.015,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=9, color="#555555", family="Georgia, serif"),
            xanchor="center",
        ),
        dict(
            text="Source: CHIRPS v3 daily rainfall  |  0.05° resolution",
            x=0.99, y=-0.015,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=9, color="#999999", family="monospace"),
            xanchor="right",
        ),
    ]

    fig.update_layout(
        **geo_updates,
        title=dict(
            text="<b>South Africa Rainfall Analysis</b>  —  Period Comparison",
            font=dict(size=18, family="Georgia, serif", color="#1a1a2e"),
            x=0.5,
            xanchor="center",
            y=0.995,
        ),
        paper_bgcolor="#fafaf8",
        plot_bgcolor="#fafaf8",
        height=1900,
        width=1800,
        margin=dict(t=70, b=40, l=55, r=110),
        annotations=annotations,
    )

    # ── Write HTML with deduplicated GeoJSON ─────────────────────────────
    # By default plotly embeds the GeoJSON once per trace (12× for this dashboard).
    # Instead we write it once as a JS variable and patch the figure before render.
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    plotly_config = {
        "displayModeBar": True,
        "scrollZoom": True,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        "toImageButtonOptions": {
            "format": "png",
            "width": 1400,
            "height": 1400,
            "scale": 2,
        },
    }

    # Serialise figure — geojson is embedded once per trace here
    import plotly.io as pio
    fig_json = pio.to_json(fig)

    # Replace each embedded geojson copy with a sentinel string,
    # then define the geojson once as a JS variable and restore at render time.
    # This is safe because the geojson dict is the same object for all traces.
    geojson_str  = json.dumps(geojson, separators=(",", ":"))
    sentinel     = '"__SHARED_GEOJSON__"'
    fig_json_deduped = fig_json.replace(geojson_str, sentinel)

    n_replacements = fig_json.count(geojson_str)
    original_kb    = len(fig_json.encode()) / 1024
    deduped_kb     = len(fig_json_deduped.encode()) / 1024
    print(f"  GeoJSON deduplicated: {n_replacements} copies → 1  "
          f"({original_kb:.0f} KB → {deduped_kb:.0f} KB)")

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>South Africa Rainfall Dashboard</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
</head>
<body style="margin:0;padding:0;background:#fafaf8;">
<div id="plot" style="width:100%;height:100vh;"></div>
<script>
  // GeoJSON defined once — shared across all choropleth traces
  var SHARED_GEOJSON = {geojson_str};

  // Restore geojson references before rendering
  var fig = {fig_json_deduped};
  fig.data.forEach(function(trace) {{
    if (trace.geojson === "__SHARED_GEOJSON__") {{
      trace.geojson = SHARED_GEOJSON;
    }}
  }});

  Plotly.newPlot("plot", fig.data, fig.layout, {json.dumps(plotly_config)});
</script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    final_mb = Path(output_path).stat().st_size / 1e6
    print(f"\n  Dashboard saved → {output_path}  ({final_mb:.1f} MB)")
    print(f"  Requires internet connection to load Plotly from CDN")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if NETCDF_PATH is None and CSV_PATH is None:
        raise ValueError("Set either NETCDF_PATH or CSV_PATH in the CONFIG section above.")

    print("\n═══ Loading shapefile ═══")
    gdf = load_shapefile(SHAPEFILE, ZONE_FIELD)

    print("\n═══ Loading rainfall data ═══")
    if CSV_PATH is not None:
        df = load_from_csv(CSV_PATH)
    else:
        df = load_from_netcdf(NETCDF_PATH, gdf)

    df["date"] = pd.to_datetime(df["date"])

    print("\n═══ Computing statistics ═══")
    all_stats = {}
    for period_name, period_dates in PERIODS.items():
        print(f"  Period: {period_name}")
        stats = compute_stats(df, period_name, period_dates)
        if not stats.empty:
            all_stats[period_name] = stats
            print(f"    avg rainfall: {stats['avg'].min():.1f} – {stats['avg'].max():.1f} mm/day")
            print(f"    % > {EXCEEDANCE_THRESHOLD}mm:  {stats['pct'].min():.2f} – {stats['pct'].max():.2f}%")
            print(f"    95th pct:     {stats['p95'].min():.1f} – {stats['p95'].max():.1f} mm")

    if not all_stats:
        raise RuntimeError("No statistics computed — check that your data covers 1981–2000 and 2005–2024")

    print("\n═══ Building dashboard ═══")
    build_dashboard(all_stats, gdf, OUTPUT_HTML)


if __name__ == "__main__":
    main()