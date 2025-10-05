import os
import glob
import numpy as np
import pandas as pd
from netCDF4 import Dataset, num2date

files = sorted(glob.glob("./plank_data/PACE_OCI*.nc"))
print("Number of files found:", len(files))

data_rows = []
valid_files = 0
empty_files = 0

for i, fn in enumerate(files, start=1):
    try:
        ds = Dataset(fn, "r")
        geo = ds.groups["geophysical_data"]
        nav = ds.groups["navigation_data"]
        scan = ds.groups["scan_line_attributes"]

        chl = geo.variables["chlor_a"][:].astype(float)
        lat = nav.variables["latitude"][:].astype(float)
        lon = nav.variables["longitude"][:].astype(float)
        time_var = scan.variables["time"]

        # ---- Convert time safely ----
        try:
            times = num2date(time_var[:], units=time_var.units, only_use_cftime_datetimes=False)
            first_time = times[0]
            if hasattr(first_time, "strftime"):
                date_str = first_time.strftime("%Y-%m-%d")
            else:
                date_str = str(first_time)[:10]
        except Exception as e:
            print(f"Time conversion issue in {os.path.basename(fn)}: {e}")
            date_str = "unknown"

        # ---- Mask invalid chlorophyll ----
        fill = geo.variables["chlor_a"]._FillValue
        chl = np.where((chl == fill) | (chl <= 0), np.nan, chl)

        valid_pixels = np.sum(~np.isnan(chl))
        print(f"\nFile: {os.path.basename(fn)}")
        print(f"valid pixels: {valid_pixels:,} out of {chl.size:,}")

        if valid_pixels > 0:
            print("  min (after mask):", np.nanmin(chl), "max:", np.nanmax(chl))
            lat_flat = lat.flatten()
            lon_flat = lon.flatten()
            chl_flat = chl.flatten()

            df = pd.DataFrame({
                "time": [date_str] * len(chl_flat),
                "latitude": lat_flat,
                "longitude": lon_flat,
                "chlor_a": chl_flat
            }).dropna(subset=["chlor_a"])

            data_rows.append(df)
            valid_files += 1
        else:
            print("No valid pixels found in this file.")
            empty_files += 1

        ds.close()
        print(f"Processed {i}/{len(files)} ({valid_files} valid, {empty_files} empty)")

    except Exception as e:
        print(f"Error reading {os.path.basename(fn)}: {e}")

# ---- Combine all ----
if data_rows:
    all_data = pd.concat(data_rows, ignore_index=True)
    all_data.to_csv("merged_plank.csv", index=False)
    print(f"\nSaved merged dataset to merged_plank.csv")
    print(f"Total files processed: {len(files)} (valid: {valid_files}, empty: {empty_files})")
else:
    print(f"\nNo valid data collected from {len(files)} files.")