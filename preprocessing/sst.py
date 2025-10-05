import xarray as xr
import glob
import pandas as pd

# === 1. Collect all 2020 files ===
files_2020 = sorted(glob.glob("./data/2020*.nc"))
print("Files found:", len(files_2020))

# === 2. Open dataset ===
ds_2020 = xr.open_mfdataset(files_2020, combine="by_coords", parallel=True, mask_and_scale=False)

raw = ds_2020['analysed_sst']

scale = raw.attrs.get('scale_factor', 1.0)
offset = raw.attrs.get('add_offset', 0.0)
fill_value = raw.attrs.get('_FillValue', None)

sst_c = raw.where(raw != fill_value) * scale + offset - 273.15

# === 3. Convert to dataframe ===
df = sst_c.to_dataframe().reset_index().dropna(subset=["analysed_sst"])

# === 4. Save to CSV ===
df.to_csv("sst_2020_global.csv", index=False)

print("Saved CSV with", df.shape[0], "rows")
print(df.head())