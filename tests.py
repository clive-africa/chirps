import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path

# Some basic tests to help with debugging and understanding the output dataset.


OUT_DIR = Path("chirps_output")
FINAL_NC = OUT_DIR / "chirps_v3_south_africa_compress.nc"

ds = xr.open_dataset(FINAL_NC, engine="netcdf4")

# High-level summary — dimensions, variables, coordinates, attributes
print(ds)

# Check a specific variable
print(ds["precip"])

# Quick stats
print(ds["precip"].min().values)
print(ds["precip"].max().values)
print(ds["precip"].mean().values)

# Check time range
print(ds.time.values[[0, -1]])   # first and last date

# Check spatial extent
print("Lons:", float(ds.lon.min()), "to", float(ds.lon.max()))
print("Lats:", float(ds.lat.min()), "to", float(ds.lat.max()))

ds["precip"].isel(time=0).plot(
    cmap="Blues",
    vmin=0, vmax=20,
)
plt.title(str(ds.time.values[0])[:10])
plt.show()

# Plot time series for a single point (e.g. Johannesburg)
ds["precip"].sel(lon=28.05, lat=-26.2, method="nearest").plot()
plt.title("Johannesburg daily rainfall")
plt.show()

# 1. Check dimensions — are any zero?
print("Dimensions:", dict(ds.dims))

# 2. Check coordinate ranges
print("Lons:", float(ds.lon.min()), "to", float(ds.lon.max()))
print("Lats:", float(ds.lat.min()), "to", float(ds.lat.max()))
print("Time steps:", len(ds.time))

# 3. Check if data is all NaN
da = ds["precip"]
print("Total values:", da.size)
print("NaN count:", int(da.isnull().sum()))
print("Non-NaN count:", int(da.notnull().sum()))
print("Min:", float(da.min()))
print("Max:", float(da.max()))