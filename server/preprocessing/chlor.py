import xarray as xr

ds = xr.open_dataset('V2024001_A1_WW00_chlora_2km.nc')
da = ds.chlor_a
df = da.to_dataframe().dropna(subset=['chlor_a'])
df_flat = df.reset_index().drop(columns=['altitude'])
df_flat.to_json("chlor_model", orient="records", lines=True)


