"""
CHIRPS v3 Pixel-Level Heatmap Dashboard
=========================================
Changes from previous version:
  - File size: plotly loaded from CDN, z-arrays rounded to 1 dp,
               boundary simplified, NaN outside SA masks data
  - Light background theme
  - Heatmap pixels masked to SA shapefile boundary (NaN outside)
  - Opacity slider controls all heatmap layers simultaneously

Edit the CONFIG section below.
"""

import os
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.ops import unary_union
import xarray as xr

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

NETCDF_PATH          = r"chirps_output/chirps_v3_south_africa.nc"
SHAPEFILE            = r"shapefiles/zaf_admin1.shp"
OUTPUT_HTML          = r"chirps_output/rainfall_heatmap.html"

EXCEEDANCE_THRESHOLD = 50   # mm
WET_DAY_THRESHOLD    = 1.0  # mm

PERIODS = {
    "1981 – 2000": ("1981-01-01", "2000-12-31"),
    "2006 – 2025": ("2006-01-01", "2025-12-31"),
}

METRICS     = ["avg", "pct", "p95", "p95_max"]
TITLES      = {
    "avg":     "Average Daily Rainfall (mm/day)",
    "pct":     f"% Days Exceeding {EXCEEDANCE_THRESHOLD} mm",
    "p95":     "95th Pct Daily Rainfall — Pixel Mean (mm)",
    "p95_max": "95th Pct Daily Rainfall — Pixel Max (mm)",
}
LABELS      = {"avg": "mm/day", "pct": "% days", "p95": "mm", "p95_max": "mm"}
COLORSCALES = {"avg": "Blues", "pct": "YlOrRd", "p95": "Purples", "p95_max": "Oranges"}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_netcdf(nc_path: str):
    print("  Opening NetCDF...")
    ds = xr.open_dataset(nc_path, engine="netcdf4")

    rename = {}
    for c in ds.coords:
        if c.lower() == "latitude":  rename[c] = "lat"
        if c.lower() == "longitude": rename[c] = "lon"
    if rename:
        ds = ds.rename(rename)

    precip_var = next(
        (v for v in ds.data_vars
         if any(k in v.lower() for k in ["precip", "rain", "prcp", "chirps"])),
        list(ds.data_vars)[0],
    )
    print(f"  Variable: '{precip_var}'")

    lons  = ds.lon.values
    lats  = ds.lat.values
    times = pd.DatetimeIndex(ds.time.values)

    print(f"  Grid : {len(lats)} lats × {len(lons)} lons | {len(times)} days")
    print(f"  Lon  : {lons.min():.2f} – {lons.max():.2f}")
    print(f"  Lat  : {lats.min():.2f} – {lats.max():.2f}")
    print(f"  Time : {times[0].date()} – {times[-1].date()}")

    precip = ds[precip_var].values.astype(np.float32)
    precip[precip < -999] = np.nan

    if lats[0] > lats[-1]:
        lats   = lats[::-1]
        precip = precip[:, ::-1, :]

    return precip, lats, lons, times


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — BUILD SA MASK
# ─────────────────────────────────────────────────────────────────────────────

def build_sa_mask(shp_path: str, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """
    Rasterise the SA shapefile onto the CHIRPS grid.
    Returns a boolean 2D array (nlat × nlon): True = inside SA.
    Pixels outside SA will be set to NaN in all metric arrays,
    which (a) clips the visual to SA and (b) shrinks the JSON
    because NaN → null (4 chars) vs a float (~8 chars).
    """
    print("  Building SA raster mask...")
    os.environ["SHAPE_RESTORE_SHX"] = "YES"
    gdf = gpd.read_file(shp_path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")

    sa_union = unary_union(gdf.geometry)

    nlat, nlon = len(lats), len(lons)
    res_lat    = abs(lats[1] - lats[0])
    res_lon    = abs(lons[1] - lons[0])

    # rasterio transform expects (west, south, east, north)
    transform = from_bounds(
        west  = float(lons.min()) - res_lon / 2,
        south = float(lats.min()) - res_lat / 2,
        east  = float(lons.max()) + res_lon / 2,
        north = float(lats.max()) + res_lat / 2,
        width  = nlon,
        height = nlat,
    )

    burned = rasterize(
        [(sa_union, 1)],
        out_shape=(nlat, nlon),
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=True,
    )

    # rasterio returns row 0 = north; flip to match ascending lats
    mask = burned[::-1, :].astype(bool)
    n_inside = mask.sum()
    print(f"  Pixels inside SA: {n_inside:,} / {nlat * nlon:,}")
    return mask


def load_boundary_simplified(shp_path: str, tolerance: float = 0.02) -> tuple:
    """
    Simplified boundary for overlay (reduces repeated coordinate lists).
    tolerance in degrees — 0.02° ≈ 2 km, invisible at dashboard scale.
    """
    os.environ["SHAPE_RESTORE_SHX"] = "YES"
    gdf = gpd.read_file(shp_path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")

    gdf["geometry"] = gdf.geometry.simplify(tolerance, preserve_topology=True)

    all_lons, all_lats = [], []
    for geom in gdf.geometry:
        polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
        for poly in polys:
            x, y = poly.exterior.coords.xy
            all_lons.extend(list(x) + [None])
            all_lats.extend(list(y) + [None])

    print(f"  Boundary coords after simplification: {len(all_lons):,}")
    return all_lons, all_lats


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — COMPUTE PIXEL STATS
# ─────────────────────────────────────────────────────────────────────────────

def compute_pixel_stats(
    precip: np.ndarray,
    times: pd.DatetimeIndex,
    period_dates: tuple,
    sa_mask: np.ndarray,
) -> dict:
    start = pd.Timestamp(period_dates[0])
    end   = pd.Timestamp(period_dates[1])
    mask  = (times >= start) & (times <= end)

    if not mask.any():
        raise ValueError(f"No data in period {period_dates[0]} – {period_dates[1]}")

    sub = precip[mask, :, :]
    n   = sub.shape[0]
    print(f"    Days in period: {n}")

    avg = np.nanmean(sub, axis=0)

    exceed = np.nansum(sub > EXCEEDANCE_THRESHOLD, axis=0)
    valid  = np.sum(~np.isnan(sub), axis=0)
    pct    = np.where(valid > 0, exceed / valid * 100, np.nan).astype(np.float32)

    nlat, nlon = sub.shape[1], sub.shape[2]
    p95 = np.full((nlat, nlon), np.nan, dtype=np.float32)
    for j in range(nlat):
        row      = sub[:, j, :]
        wet_mask = row >= WET_DAY_THRESHOLD
        for i in range(nlon):
            wet_vals = row[wet_mask[:, i], i]
            if len(wet_vals) >= 10:
                p95[j, i] = np.percentile(wet_vals, 95)

    results = {
        "avg":     avg.astype(np.float32),
        "pct":     pct,
        "p95":     p95,
        "p95_max": p95.copy(),
    }

    # ── Apply SA mask — pixels outside SA → NaN ───────────────────────────
    # This clips the visual AND shrinks JSON (null ≪ float in bytes)
    for arr in results.values():
        arr[~sa_mask] = np.nan

    # ── Round to 1 dp — biggest single file-size lever ───────────────────
    # float32 full precision in JSON: ~10 chars; 1 dp: ~4 chars → ~60% smaller
    for key in results:
        results[key] = np.round(results[key], 1)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — BUILD DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

def build_heatmap_dashboard(
    period_stats: dict,
    lats: np.ndarray,
    lons: np.ndarray,
    boundary_lons: list,
    boundary_lats: list,
    output_path: str,
) -> None:

    period_names = list(period_stats.keys())
    p1, p2       = period_names[0], period_names[1]

    # Shared colour ranges
    ranges = {}
    for metric in METRICS:
        a1  = period_stats[p1][metric]
        a2  = period_stats[p2][metric]
        all_vals = np.concatenate([a1[~np.isnan(a1)], a2[~np.isnan(a2)]])
        ranges[metric] = (float(np.nanmin(all_vals)), float(np.nanmax(all_vals))) if len(all_vals) else (0, 1)

    # Difference arrays
    diff_arrays = {}
    diff_ranges = {}
    for metric in METRICS:
        diff = np.round(period_stats[p2][metric] - period_stats[p1][metric], 1)
        diff_arrays[metric] = diff
        valid = diff[~np.isnan(diff)]
        diff_ranges[metric] = float(np.nanmax(np.abs(valid))) if len(valid) else 1.0

    col_titles = [p1, p2, "Change (later − earlier)"]
    fig = make_subplots(
        rows=4, cols=3,
        subplot_titles=[ct for _ in METRICS for ct in col_titles],
        vertical_spacing=0.06,
        horizontal_spacing=0.04,
    )

    
    n_rows      = len(METRICS)
    v_spacing   = 0.06
    row_h       = (1 - (n_rows - 1) * v_spacing) / n_rows
    row_centres = {
        i: round(1 - i * (row_h + v_spacing) - row_h / 2, 4)
        for i in range(n_rows)
    }

    cb_y        = row_centres   # was hardcoded
    row_label_y = row_centres   # was hardcoded    

    cb_len = 0.19

    # SA extent
    sa_xrange = [float(lons[~np.isnan(lons)].min()) - 0.1,
                 float(lons[~np.isnan(lons)].max()) + 0.1]
    sa_yrange = [float(lats[~np.isnan(lats)].min()) - 0.1,
                 float(lats[~np.isnan(lats)].max()) + 0.1]

    def make_heatmap(z, colorscale, zmin, zmax, cb_x, cb_title, show_cb, name, row_idx):
        return go.Heatmap(
            z=z.tolist(),          # tolist() avoids numpy float serialisation overhead
            x=lons.tolist(),
            y=lats.tolist(),
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            opacity=1.0,           # controlled by JS slider
            colorbar=dict(
                title=dict(text=cb_title, font=dict(size=9)),
                len=cb_len,
                y=cb_y[row_idx],
                yanchor="middle",
                thickness=12,
                x=cb_x,
                tickfont=dict(size=8),
                bgcolor="rgba(255,255,255,0.8)",
            ) if show_cb else None,
            showscale=show_cb,
            hovertemplate=(
                f"<b>{name}</b><br>"
                "Lon: %{x:.2f}°<br>"
                "Lat: %{y:.2f}°<br>"
                f"Value: %{{z:.1f}} {LABELS.get(name, '')}"
                "<extra></extra>"
            ),
            name=name,
        )

    def make_boundary():
        return go.Scatter(
            x=boundary_lons,
            y=boundary_lats,
            mode="lines",
            line=dict(color="#333333", width=0.8),
            hoverinfo="skip",
            showlegend=False,
            name="",
        )

    for m_idx, metric in enumerate(METRICS):
        row = m_idx + 1

        # Col 1 — Period 1
        fig.add_trace(
            make_heatmap(period_stats[p1][metric], COLORSCALES[metric],
                         ranges[metric][0], ranges[metric][1],
                         cb_x=0.30, cb_title=LABELS[metric],
                         show_cb=False, name=metric, row_idx=m_idx),
            row=row, col=1,
        )
        fig.add_trace(make_boundary(), row=row, col=1)

        # Col 2 — Period 2 (with colourbar)
        fig.add_trace(
            make_heatmap(period_stats[p2][metric], COLORSCALES[metric],
                         ranges[metric][0], ranges[metric][1],
                         cb_x=0.635, cb_title=LABELS[metric],
                         show_cb=True, name=metric, row_idx=m_idx),
            row=row, col=2,
        )
        fig.add_trace(make_boundary(), row=row, col=2)

        # Col 3 — Difference
        abs_max = diff_ranges[metric]
        fig.add_trace(
            go.Heatmap(
                z=diff_arrays[metric].tolist(),
                x=lons.tolist(),
                y=lats.tolist(),
                colorscale=[
                    [0.00, "#d73027"], [0.25, "#f46d43"], [0.45, "#fee090"],
                    [0.50, "#ffffff"],
                    [0.55, "#abd9e9"], [0.75, "#4575b4"], [1.00, "#313695"],
                ],
                zmid=0, zmin=-abs_max, zmax=abs_max,
                opacity=1.0,
                colorbar=dict(
                    title=dict(text=f"Δ {LABELS[metric]}", font=dict(size=9)),
                    len=cb_len,
                    y=cb_y[m_idx],
                    yanchor="middle",
                    thickness=12,
                    x=1.01,
                    tickfont=dict(size=8),
                    tickformat="+.1f",
                    bgcolor="rgba(255,255,255,0.8)",
                ),
                showscale=True,
                hovertemplate=(
                    "<b>Change</b><br>"
                    "Lon: %{x:.2f}°<br>"
                    "Lat: %{y:.2f}°<br>"
                    f"Δ: %{{z:+.1f}} {LABELS[metric]}"
                    "<extra></extra>"
                ),
                name=f"Δ {metric}",
            ),
            row=row, col=3,
        )
        fig.add_trace(make_boundary(), row=row, col=3)

    # ── Axis updates — constrained to SA extent ───────────────────────────
    axis_updates = {}
    for m_idx in range(4):
        for c_idx in range(3):
            n = m_idx * 3 + c_idx + 1
            xk = "xaxis"  if n == 1 else f"xaxis{n}"
            yk = "yaxis"  if n == 1 else f"yaxis{n}"
            xanchor = "x" if n == 1 else f"x{n}"
            axis_updates[xk] = dict(
                range=sa_xrange, showgrid=False, zeroline=False,
                title="", tickfont=dict(size=7), tickformat=".0f", nticks=5,
                showline=True, linecolor="#cccccc",
            )
            axis_updates[yk] = dict(
                range=sa_yrange, showgrid=False, zeroline=False,
                title="", tickfont=dict(size=7), tickformat=".0f", nticks=5,
                scaleanchor=xanchor, scaleratio=1.0,
                showline=True, linecolor="#cccccc",
            )

    annotations = [
        dict(
            text=f"<b>{TITLES[metric]}</b>",
            x=-0.01, y=row_label_y[m_idx],
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=10, family="Georgia, serif", color="#222222"),
            xanchor="right", textangle=-90,
        )
        for m_idx, metric in enumerate(METRICS)
    ] + [
        dict(
            text="🔵 wetter / higher  🔴 drier / lower  in the later period",
            x=0.83, y=-0.015, xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=9, color="#555555", family="Georgia, serif"),
            xanchor="center",
        ),
        dict(
            text="Source: CHIRPS v3 daily  |  0.05° (~5.5 km) resolution",
            x=1.0, y=-0.015, xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=8, color="#999999", family="monospace"),
            xanchor="right",
        ),
    ]

    fig.update_layout(
        **axis_updates,
        title=dict(
            text="<b>South Africa Rainfall Analysis</b>  —  Pixel-Level Heatmap",
            font=dict(size=18, family="Georgia, serif", color="#1a1a2e"),
            x=0.5, xanchor="center", y=0.998,
        ),
        paper_bgcolor="#f8f9fa",   # light grey-white
        plot_bgcolor="#ffffff",    # white plot areas
        font=dict(color="#222222"),
        height=1900,
        width=1800,
        margin=dict(t=60, b=50, l=70, r=130),
        annotations=annotations,
    )

    # ── Opacity slider — injected as custom HTML ──────────────────────────
    # Identifies all Heatmap traces by index and updates their opacity.
    opacity_js = """
<style>
  #opacity-control {
    position: fixed;
    top: 14px;
    right: 160px;
    z-index: 9999;
    background: rgba(255,255,255,0.92);
    border: 1px solid #ccc;
    border-radius: 8px;
    padding: 8px 14px;
    font-family: Arial, sans-serif;
    font-size: 13px;
    color: #333;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    display: flex;
    align-items: center;
    gap: 10px;
  }
  #opacity-control label { white-space: nowrap; font-weight: 600; }
  #opacity-slider { width: 130px; cursor: pointer; }
  #opacity-value { min-width: 34px; text-align: right; }
</style>

<div id="opacity-control">
  <label>🎚 Heatmap opacity</label>
  <input id="opacity-slider" type="range" min="0" max="100" value="100" step="1">
  <span id="opacity-value">100%</span>
</div>

<script>
(function() {
  var slider = document.getElementById('opacity-slider');
  var label  = document.getElementById('opacity-value');

  slider.addEventListener('input', function() {
    var opacity = parseFloat(this.value) / 100;
    label.textContent = this.value + '%';

    // Find the plotly div (first div with class 'plotly-graph-div')
    var gd = document.querySelector('.plotly-graph-div');
    if (!gd || !gd.data) return;

    // Collect indices of all Heatmap traces
    var updates = {};
    var heatmapIndices = [];
    gd.data.forEach(function(trace, i) {
      if (trace.type === 'heatmap') heatmapIndices.push(i);
    });

    if (heatmapIndices.length === 0) return;

    // Build opacity array for Plotly.restyle
    var opacityArr = heatmapIndices.map(function() { return opacity; });
    Plotly.restyle(gd, { opacity: opacityArr }, heatmapIndices);
  });
})();
</script>
"""

    # ── Write HTML — CDN plotly (not embedded) ────────────────────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    html_str = fig.to_html(
        include_plotlyjs="cdn",    # load plotly.js from CDN — saves ~3 MB
        full_html=True,
        config={
            "displayModeBar": True,
            "scrollZoom":     True,
            "toImageButtonOptions": {
                "format": "png", "width": 1800, "height": 1900, "scale": 2,
            },
        },
    )

    # Inject opacity control just before </body>
    html_str = html_str.replace("</body>", opacity_js + "\n</body>")

    Path(output_path).write_text(html_str, encoding="utf-8")
    print(f"\n  Dashboard saved → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n═══ Loading NetCDF ═══")
    precip, lats, lons, times = load_netcdf(NETCDF_PATH)

    print("\n═══ Loading boundary ═══")
    sa_mask = build_sa_mask(SHAPEFILE, lats, lons)
    boundary_lons, boundary_lats = load_boundary_simplified(SHAPEFILE, tolerance=0.02)

    print("\n═══ Computing pixel-level statistics ═══")
    period_stats = {}
    for period_name, period_dates in PERIODS.items():
        print(f"  Period: {period_name}")
        period_stats[period_name] = compute_pixel_stats(
            precip, times, period_dates, sa_mask
        )

    print("\n═══ Building dashboard ═══")
    build_heatmap_dashboard(
        period_stats, lats, lons,
        boundary_lons, boundary_lats,
        OUTPUT_HTML,
    )


if __name__ == "__main__":
    main()