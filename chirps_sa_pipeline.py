"""
CHIRPS v3 South Africa Pipeline
================================
1. Parse a wget script to extract CHIRPS v3 daily NetCDF URLs
2. Download each file, clip to SA bounding box, delete the original (saves disk)
3. Concatenate all clipped files into a single NetCDF
4. Compute area-weighted mean daily rainfall per polygon in any uploaded shapefile

Requirements:
    pip install xarray netCDF4 geopandas exactextract rasterio numpy pandas tqdm requests
    (exactextract is recommended for 20k+ sub-places — falls back to rasterstats if unavailable)

Usage:
    python chirps_sa_pipeline.py --wget-file chirps_urls.txt --shapefile provinces.shp
    python chirps_sa_pipeline.py --wget-file chirps_urls.txt --shapefile subplaces.shp --zone-field SP_NAME
"""



import re


from pathlib import Path


import pandas as pd
import requests
import xarray as xr
from tqdm import tqdm
import time
import geopandas as gpd
# Need dask to support soem operations
import dask

from exactextract import exact_extract
from rasterio.transform import from_bounds
from rasterio.io import MemoryFile
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# South Africa bounding box — only the mainalnd
SA_BBOX = {
    "lon_min": 16.0,
    "lon_max": 33.5,
    "lat_min": -35,  # Excludes Prince Edward Islands
    "lat_max": -21.5,
}

# Download settings
CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB streaming chunks
CONNECT_TIMEOUT = 30
READ_TIMEOUT = 300


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: PARSE WGET FILE
# ─────────────────────────────────────────────────────────────────────────────

# Thsi is avaialbel form the downlaod site.
# It provides a list of all the files to download with wget commands.
# We use this list to downlaod our data

def parse_wget_file(wget_path: str) -> list[str]:
    """
    Extract all HTTPS/HTTP URLs from a wget script.
    Handles both plain URL lists and wget -q -O ... URL formats.
    """
    urls = []
    with open(wget_path) as f:
        for line in f:
            line = line.strip()
            # Match any https:// or http:// URL (ignore wget flags)
            matches = re.findall(r'https?://\S+\.nc\b', line)
            urls.extend(matches)
    if not urls:
        raise ValueError(
            f"No .nc URLs found in {wget_path}. "
            "Make sure the file contains lines with https://.../*.nc URLs."
        )
    print(f"Found {len(urls)} NetCDF URLs to download.")
    return urls


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: DOWNLOAD → CLIP → DELETE LOOP
# ─────────────────────────────────────────────────────────────────────────────

# Usign our bouding box we need to clip the downlaod file to the bounds
# We do this after every file as teh whoel file won't fit on my PC

def clip_to_sa(ds: xr.Dataset, bbox: dict | None = SA_BBOX) -> xr.Dataset:
    """
    Subset an xarray Dataset to a bounding box.
    Robustly handles:
      - 'lat'/'lon' or 'latitude'/'longitude' coordinate names
      - Ascending or descending lat ordering
      - 0-360 vs -180/180 longitude convention
    """
    # ── 1. Normalise coordinate names ─────────────────────────────────────
    rename_map = {}
    for coord in list(ds.coords):
        if coord.lower() == "latitude":
            rename_map[coord] = "lat"
        elif coord.lower() == "longitude":
            rename_map[coord] = "lon"
    if rename_map:
        ds = ds.rename(rename_map)

    if "lat" not in ds.coords or "lon" not in ds.coords:
        raise ValueError(
            f"Cannot find lat/lon coordinates. Available: {list(ds.coords)}"
        )

    # ── 2. Normalise 0-360 longitudes to -180/180 if needed ───────────────
    if float(ds.lon.max()) > 180:
        ds = ds.assign_coords(lon=(ds.lon + 180) % 360 - 180)
        ds = ds.sortby("lon")

    # ── 3. Clip longitudes ─────────────────────────────────────────────────
    ds = ds.sel(lon=slice(bbox["lon_min"], bbox["lon_max"]))

    if ds.sizes.get("lon", 0) == 0:
        raise ValueError(
            f"Longitude clip returned no data. "
            f"File lons: {float(ds.lon.min()):.2f} to {float(ds.lon.max()):.2f}, "
            f"bbox: {bbox['lon_min']} to {bbox['lon_max']}"
        )

    # ── 4. Clip latitudes — handle ascending or descending order ──────────
    # Ran into issues the first time I ran the code

    lat_vals = ds.lat.values
    lat_ascending = float(lat_vals[0]) < float(lat_vals[-1])

    if lat_ascending:
        ds = ds.sel(lat=slice(bbox["lat_min"], bbox["lat_max"]))
    else:
        ds = ds.sel(lat=slice(bbox["lat_max"], bbox["lat_min"]))

    if ds.sizes.get("lat", 0) == 0:
        raise ValueError(
            f"Latitude clip returned no data. "
            f"File lats: {float(lat_vals.min()):.2f} to {float(lat_vals.max()):.2f}, "
            f"bbox: {bbox['lat_min']} to {bbox['lat_max']}, "
            f"lat ascending: {lat_ascending}"
        )

    return ds


def delete_with_retry(path: Path, retries: int = 5, delay: float = 0.5) -> None:
    """Delete a file with retries to handle Windows file lock delays."""
    import time
    for _ in range(retries):
        try:
            if path.exists():
                path.unlink()
            return
        except PermissionError:
            time.sleep(delay)
    print(f"  WARNING: Could not delete {path.name} after {retries} attempts")


def download_clip_delete(
    urls: list[str],
    bbox: dict | None = SA_BBOX,
    out_dir: Path ='',
    raw_dir: Path ='',
    keep_raw: bool | None = False,
) -> list[Path]:
    """
    For each URL:
      1. Download the full raw file to raw_dir (visible, named files)
      2. Open with xarray, clip to bbox
      3. Save clipped file to out_dir
      4. Delete the raw file only if clipping succeeded (unless keep_raw=True)

    Parameters
    ----------
    urls       : List of CHIRPS NetCDF URLs to download
    bbox       : Bounding box dict with lon_min/lon_max/lat_min/lat_max
    out_dir    : Directory for clipped output files
    raw_dir    : Directory for raw downloads. Defaults to out_dir/../raw
    keep_raw   : If True, never delete raw files (useful for debugging)

    Returns list of paths to successfully clipped files.
    """
    

    out_dir.mkdir(parents=True, exist_ok=True)

    # Raw files go into a dedicated subfolder so they are easy to find
    if raw_dir is None:
        raw_dir = out_dir.parent / "raw_files"
    raw_dir.mkdir(parents=True, exist_ok=True)

    clipped_paths = []
    failed = []

    print(f"  Raw downloads -> {raw_dir}")
    print(f"  Clipped files -> {out_dir}")

    for url in tqdm(urls, desc="Downloading & clipping"):
        filename = url.split("/")[-1]
        raw_path = raw_dir / filename
        clipped_path = out_dir / f"clipped_{filename}"

        # ── Resume: skip if already clipped ───────────────────────────────
        if clipped_path.exists():
            print(f"  Skipping (already clipped): {filename}")
            clipped_paths.append(clipped_path)
            continue

        # ── Step 1: Download to named file in raw_dir ─────────────────────
        max_retries = 3
        downloaded = False

        for attempt in range(1, max_retries + 1):
            try:
                response = requests.get(
                    url, stream=True,
                    timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)
                )
                response.raise_for_status()

                expected_size = int(response.headers.get("content-length", 0))

                with open(raw_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        f.write(chunk)

                # Verify the download is complete
                actual_size = raw_path.stat().st_size
                if expected_size > 0 and actual_size < expected_size * 0.99:
                    raise IOError(
                        f"Incomplete download: {actual_size:,} of "
                        f"{expected_size:,} bytes"
                    )

                size_mb = actual_size / 1024 / 1024
                print(f"  Downloaded:  {filename} ({size_mb:.1f} MB)")
                downloaded = True
                break

            except Exception as e:
                print(f"  Attempt {attempt}/{max_retries} failed: {e}")
                delete_with_retry(raw_path)
                if attempt < max_retries:
                    time.sleep(5 * attempt)

        if not downloaded:
            print(f"  FAILED: {filename} — skipping")
            failed.append(filename)
            continue

        # ── Step 2: Open with engine fallback ─────────────────────────────
        ds = None
        for engine in ["netcdf4", "h5netcdf", "scipy"]:
            try:
                ds = xr.open_dataset(raw_path, engine=engine)
                break
            except Exception as e:
                print(f"  Engine '{engine}' failed: {e}")

        if ds is None:
            print(f"  FAILED to open {filename} — raw file kept for inspection")
            failed.append(filename)
            continue

        # ── Step 3: Clip and save ──────────────────────────────────────────
        clip_ok = False
        try:
            with ds:
                ds_clipped = clip_to_sa(ds, bbox)
                ds_clipped = ds_clipped.load()

            ds_clipped.to_netcdf(clipped_path)
            ds_clipped.close()
            clipped_paths.append(clipped_path)
            clip_ok = True
            print(f"  Clipped:     {clipped_path.name}")

        except Exception as e:
            print(f"  ERROR clipping {filename}: {e}")
            failed.append(filename)
            # Remove incomplete clipped file if it was partially written
            delete_with_retry(clipped_path)

        # ── Step 4: Delete raw only if clip succeeded ─────────────────────
        if clip_ok and not keep_raw:
            delete_with_retry(raw_path)
            print(f"  Deleted raw: {filename}")
        elif not clip_ok:
            print(f"  Kept raw for inspection: {raw_path}")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'─' * 50}")
    print(f"  Clipped successfully : {len(clipped_paths)}")
    print(f"  Failed               : {len(failed)}")
    if failed:
        print("  Failed files:")
        for f in failed:
            print(f"    * {f}")
    print(f"{'─' * 50}")

    return clipped_paths


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: CONCATENATE CLIPPED FILES
# ─────────────────────────────────────────────────────────────────────────────

def concatenate_clipped(clipped_paths: list[Path], output_nc: Path| None, compress: bool | None = True ) -> xr.Dataset:
    """
    Open all clipped files with xarray's multi-file support and write a
    single combined NetCDF. Uses dask if available for memory efficiency.
    """
    if not clipped_paths:
        raise RuntimeError("No clipped files found to concatenate.")

    output_nc.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nConcatenating {len(clipped_paths)} files → {output_nc}")

    try:
        ds = xr.open_mfdataset(
            sorted(clipped_paths),
            combine="by_coords",
            engine="netcdf4",
            parallel=True,
        )
    except ImportError:
        ds = xr.open_mfdataset(
            sorted(clipped_paths),
            combine="by_coords",
            engine="netcdf4",
        )

    # Ensure time is sorted
    ds = ds.sortby("time")

    # Write with greater compression to save disk space
    if compress:
        encoding = {
            var: {
                "zlib": True,
                "complevel": 9,
                "shuffle": True,
                "dtype": "float32",
                "chunksizes": (365, 100, 100),  # (time, lat, lon) - tune to your grid
            }
            for var in ds.data_vars
        }
    else:
        encoding = {
            var: {"zlib": True, "complevel": 4}
            for var in ds.data_vars
        }
    ds.to_netcdf(output_nc, encoding=encoding)
    print(f"Saved combined NetCDF: {output_nc}")
    return ds


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: ZONAL STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def load_shapefile(shp_path: str, zone_field: str | None = None):
    """
    Load a shapefile, assign CRS if missing, and return a GeoDataFrame.
    Adds a numeric 'zone_id' and an optional label column.
    """

    gdf = gpd.read_file(shp_path)

    # Assign WGS84 if CRS is missing (as is the case with the SA provinces file)
    if gdf.crs is None:
        print("  Shapefile has no CRS — assuming EPSG:4326 (WGS84)")
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")

    # Create zone identifier
    gdf["zone_id"] = range(len(gdf))
    if zone_field and zone_field in gdf.columns:
        gdf["zone_label"] = gdf[zone_field].astype(str)
    else:
        gdf["zone_label"] = [f"zone_{i}" for i in range(len(gdf))]

    print(f"  Loaded {len(gdf)} zones from shapefile")
    return gdf


def zonal_stats_exactextract(ds: xr.Dataset, gdf, precip_var: str) -> pd.DataFrame:
    """
    Use exactextract for fast, pixel-area-weighted zonal statistics.
    Handles 20,000+ sub-places efficiently by pre-computing weights.

    exactextract handles partial pixel coverage correctly — essential for
    small polygons relative to CHIRPS's 0.05° (~5.5 km) grid cells.
    """

    # Get grid metadata
    lons = ds.lon.values
    lats = ds.lat.values
    times = pd.DatetimeIndex(ds.time.values)
    precip = ds[precip_var].values  # shape: (time, lat, lon)

    # Build rasterio transformation from grid
    res = abs(float(lons[1] - lons[0]))
    transform = from_bounds(
        float(lons.min()) - res / 2,   # left
        float(lats.min()) - res / 2,   # bottom
        float(lons.max()) + res / 2,   # right
        float(lats.max()) + res / 2,   # top
        len(lons),                      # width
        len(lats),                      # height
    )

    all_records = []

    print(f"  Computing zonal stats for {len(gdf)} zones × {len(times)} days...")
    print("  Strategy: process one day at a time")

    # Looking through each day
    for t_idx in tqdm(range(len(times)), desc="Zonal stats by day"):
        date = times[t_idx]
        # Extract 2D slice — flip lat to ensure top-down order for rasterio
        band = precip[t_idx, :, :]
        if lats[0] < lats[-1]:  # ascending lat → flip
            band = np.flipud(band)

        # Write to in-memory raster
        with MemoryFile() as mem:
            with mem.open(
                driver="GTiff",
                height=band.shape[0],
                width=band.shape[1],
                count=1,
                dtype=band.dtype,
                crs="EPSG:4326",
                transform=transform,
                nodata=-9999,
            ) as raster:
                raster.write(band, 1)

            # Map teh raster files to the shapefile polygons with exactextract
            with mem.open() as raster:
                results = exact_extract(
                    raster,
                    gdf,
                    ["mean"],
                    include_cols=["zone_id", "zone_label"],
                    output="pandas",
                )
                results["date"] = date
                all_records.append(results)

    df = pd.concat(all_records, ignore_index=True)
    df = df.rename(columns={"mean": "rainfall_mm"})
    return df

def zonal_stats_vector(ds: xr.Dataset, gdf, precip_var: str) -> pd.DataFrame:
    """
    Fast vectorized zonal statistics using pre-computed pixel weight matrices.
 
    Key insight: zone weights (which pixels overlap which zones) are identical
    for every day. Computing them once and using a matrix multiply across all
    days simultaneously reduces ~15,000 exactextract calls to just n_zones calls.
 
    Strategy
    --------
    1. Load full precip array into RAM  — fits easily at 2 GB on a 64 GB machine
    2. Call exactextract ONCE per zone to get pixel coverage fractions
    3. Build a weight matrix W of shape (n_zones, n_pixels)
    4. Reshape precip to (n_time, n_pixels)
    5. Weighted mean = (precip @ W.T) / W.sum(axis=1)  — one matrix multiply
    6. Build output DataFrame from numpy arrays (no Python loops over time)
 
    Expected time: ~2–5 minutes vs ~4 hours for the per-day loop.
    """
    from exactextract import exact_extract
    from rasterio.transform import from_bounds
    from rasterio.io import MemoryFile
    from rasterio.features import rasterize
 
    # ── 1. Load data fully into RAM ───────────────────────────────────────
    print("  Loading full precip array into RAM...")
    lons   = ds.lon.values
    lats   = ds.lat.values
    times  = pd.DatetimeIndex(ds.time.values)
    precip = ds[precip_var].load().values.astype(np.float32)  # (time, lat, lon)
    precip[precip < -999] = np.nan
 
    n_time       = len(times)
    n_lat, n_lon = len(lats), len(lons)
    n_zones      = len(gdf)
 
    size_gb = precip.nbytes / 1e9
    print(f"  Array in RAM: {n_time} days × {n_lat} lats × {n_lon} lons  ({size_gb:.2f} GB)")
 
    # Ensure lats are descending (north→south) for rasterio
    if lats[0] < lats[-1]:
        lats   = lats[::-1]
        precip = precip[:, ::-1, :]
 
    # ── 2. Build affine transform ─────────────────────────────────────────
    res = abs(float(lons[1] - lons[0]))
    transform = from_bounds(
        float(lons.min()) - res / 2,   # left  (west)
        float(lats.min()) - res / 2,   # bottom (south)
        float(lons.max()) + res / 2,   # right  (east)
        float(lats.max()) + res / 2,   # top    (north)
        n_lon,
        n_lat,
    )
 
    # ── 3. Pre-compute coverage-weighted pixel mask for every zone ─────────
    # Run exactextract ONCE per zone on a dummy raster to get pixel weights.
    # The dummy raster has sequential pixel IDs so we can map results back
    # to (row, col) positions.
    print(f"  Pre-computing pixel weights for {n_zones} zones...")
 
    pixel_ids = np.arange(n_lat * n_lon, dtype=np.float32).reshape(n_lat, n_lon)
 
    with MemoryFile() as mem:
        with mem.open(
            driver="GTiff",
            height=n_lat, width=n_lon,
            count=1, dtype=np.float32,
            crs="EPSG:4326",
            transform=transform,
        ) as raster:
            raster.write(pixel_ids, 1)
 
        with mem.open() as raster:
            # Ask exactextract for pixel-level values + coverage fractions
            ee_results = exact_extract(
                raster,
                gdf,
                ["values", "coverage"],
                include_cols=["zone_id"],
                output="pandas",
            )
 
    # Build sparse weight matrix: W[z, pixel_idx] = coverage_fraction
    # Use float32 throughout to keep memory low
    # We could genertae a 3d matrix and thern use einsum to do the weighted mean
    # # THis keeps it simple and is likely faster??
    W = np.zeros((n_zones, n_lat * n_lon), dtype=np.float32)
    
    # At this stages our differnet pixels and their coverage are lists
    # We need to map these to the correct position in the weight matrix W
    # We have the zeros because the fucntion only returns ids where there is coverage
    for _, row in ee_results.iterrows():
        z_idx    = int(row["zone_id"])
        pix_ids  = np.array(row["values"],   dtype=np.int32)
        coverage = np.array(row["coverage"], dtype=np.float32)
        W[z_idx, pix_ids] = coverage
    
    # Thsi then gives us teh total sum of every pixel's contirbution
    # # Will need to divide by this figure
    weight_sums = W.sum(axis=1)  # (n_zones,) — total coverage per zone
    print(f"  Weight matrix built: {W.nbytes / 1e6:.1f} MB")
 
    # ── 4. Vectorized weighted mean across all days ───────────────────────
    print("  Computing weighted means (matrix multiply)...")
 
    # Reshape precip: (n_time, n_pixels)
    # Avoid havign to do 3D matrix multiplication
    precip_2d = precip.reshape(n_time, -1)
 
    # Replace NaN with 0 for the weighted sum; track where NaN pixels are
    # NaN represents where there is no data (e.g. ocean pixels) — we want to exclude these from the mean
    nan_mask  = np.isnan(precip_2d).astype(np.float32)  # 1 where NaN
    precip_nz = np.where(np.isnan(precip_2d), 0.0, precip_2d)
 
    # Weighted sum: (n_time, n_zones)
    # Thsi will give us weighted sum of all the pixels by day
    # We still need to divide by the sum to get a weighted average
    weighted_sum = precip_nz @ W.T
 
    # Effective weight per zone per day (subtract NaN pixel weights)
    # We have coded these as 0 so we need to exclude these values from the SUM.
    nan_weight_sum   = nan_mask @ W.T                         # (n_time, n_zones)
    # Numpy automatically broadcasts the subtraction across zones for each day
    effective_weight = weight_sums[np.newaxis, :] - nan_weight_sum  # (n_time, n_zones)
 
    # Weighted mean — NaN where no valid pixels in zone
    zone_means = np.where(
        effective_weight > 0,
        weighted_sum / effective_weight,
        np.nan,
    )  # (n_time, n_zones)
 
    print(f"  Done. Output shape: {zone_means.shape}")
 
    # ── 5. Build output DataFrame efficiently ─────────────────────────────
    # Avoid Python loops over time — use numpy tiling instead
    zone_ids_arr    = gdf["zone_id"].values
    zone_labels_arr = gdf["zone_label"].values
 
    df = pd.DataFrame({
        "zone_id":     np.tile(zone_ids_arr, n_time),
        "zone_label":  np.tile(zone_labels_arr, n_time),
        "date":        np.repeat(times, n_zones),
        "rainfall_mm": zone_means.ravel(),   # row-major: time varies slowest
    })
 


    # ── NaN count by province ─────────────────────────────────────────────────
    print("NaN diagnostic by region\n" + "─" * 50)

    for z_idx in range(n_zones):
        zone_label = gdf.iloc[z_idx]["zone_label"]
        
        # Get pixel indices for this zone (where weight > 0)
        zone_pixel_mask = W[z_idx] > 0
        n_pixels_in_zone = zone_pixel_mask.sum()
        
        if n_pixels_in_zone == 0:
            print(f"{zone_label:<30} NO PIXELS — check shapefile/clipping")
            continue
        
        # Extract time series for just this zone's pixels
        zone_precip = precip_2d[:, zone_pixel_mask]  # (n_time, n_zone_pixels)
        
        total_values = zone_precip.size
        nan_count    = np.isnan(zone_precip).sum()
        nan_pct      = nan_count / total_values * 100
        
        # Per-day NaN rate — flag days where ALL pixels in zone are NaN
        all_nan_days = np.isnan(zone_precip).all(axis=1).sum()
        
        print(
            f"{zone_label:<30} "
            f"pixels: {n_pixels_in_zone:>5}  |  "
            f"NaN: {nan_pct:>5.1f}%  |  "
            f"fully-NaN days: {all_nan_days:>5}"
        )

    print("─" * 50)
    print(f"{'TOTAL':<30} pixels: {(W > 0).any(axis=0).sum():>5}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():


    # Output directories
    OUT_DIR = Path("chirps_output")
    CLIPPED_DIR = OUT_DIR / "clipped_daily"
    RAW_DIR = OUT_DIR / "raw_files"
    FINAL_NC = OUT_DIR / "chirps_v3_south_africa_compress.nc"
    ZONAL_CSV = OUT_DIR / "rainfall_by_zone.csv"
    WGET_FILE="wgetem"


    skip_download=False
    skip_concat=False
    shapefile="shapefiles/zaf_admin4.shp"
    zone_field="adm4_name"
    precip_var='precip'


    # ── Step 1: Parse URLs ─────────────────────────────────────────────────
    print("\n═══ STEP 1: Parse wget file ═══")
    urls = parse_wget_file(WGET_FILE)

    # ── Step 2: Download, clip, delete ────────────────────────────────────
    if not skip_download:
        print("\n═══ STEP 2: Download → Clip → Delete ═══")
        clipped_paths = download_clip_delete(
            urls, SA_BBOX, CLIPPED_DIR,
            raw_dir=RAW_DIR,
            keep_raw=False,
        )
    else:
        print("\n═══ STEP 2: Skipped (using existing clipped files) ═══")
        clipped_paths = sorted(CLIPPED_DIR.glob("clipped_*.nc"))
        print(f"Found {len(clipped_paths)} existing clipped files")

    # ── Step 3: Concatenate ────────────────────────────────────────────────
    if not skip_concat:
        print("\n═══ STEP 3: Concatenate ═══")
        # The compression argument hasn't been too successful. :(
        ds = concatenate_clipped(clipped_paths, FINAL_NC, compress=True)
    else:
        print("\n═══ STEP 3: Skipped (loading existing combined NetCDF) ═══")
        ds = xr.open_dataset(FINAL_NC, engine="netcdf4")

    # ── Step 4: Zonal statistics ───────────────────────────────────────────
    print("\n═══ STEP 4: Zonal Statistics ═══")
    gdf = load_shapefile(shapefile, zone_field=zone_field)

    # Clip the dataset further to the shapefile bounds for efficiency
    bounds = gdf.total_bounds  # (minx, miny, maxx, maxy)
    ds = ds.sel(
        lon=slice(bounds[0] - 0.1, bounds[2] + 0.1),
        lat=slice(bounds[3] + 0.1, bounds[1] - 0.1),
    )
    if ds.sizes.get("lat", 0) == 0:
        ds = ds.sel(
            lon=slice(bounds[0] - 0.1, bounds[2] + 0.1),
            lat=slice(bounds[1] - 0.1, bounds[3] + 0.1),
        )
    
    df = zonal_stats_vector(ds, gdf, precip_var)

    # Keep for comparison purposes.
    # The data reconciled really well.
    #df2 = zonal_stats_exactextract(ds, gdf, precip_var)  

    # Join df and df2 on zone_label and date, calculate difference
    # df_merged = df.merge(
    #     df2,
    #     on=["zone_label", "date"],
    #     suffixes=("_vector", "_loop")
    # )
    # df_merged["rainfall_mm_diff"] = df_merged["rainfall_mm_exactextract"] - df_merged["rainfall_mm_vector"]
    # df_merged["abs_rainfall_mm_diff"] = abs(df_merged["rainfall_mm_diff"])

    # Pivot to wide format: dates × zones
    df_wide = df.pivot_table(
        index="date",
        columns="zone_label",
        values="rainfall_mm",
    )

    # Save outputs
    ZONAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ZONAL_CSV.with_suffix("_long.csv"), index=False)
    df_wide.to_csv(ZONAL_CSV)
    print(f"\nSaved zonal stats (long format): {ZONAL_CSV.with_suffix('_long.csv')}")
    print(f"Saved zonal stats (wide format): {ZONAL_CSV}")
    print(f"\nDone. Output files are in: {OUT_DIR}/")


if __name__ == "__main__":
    main()
