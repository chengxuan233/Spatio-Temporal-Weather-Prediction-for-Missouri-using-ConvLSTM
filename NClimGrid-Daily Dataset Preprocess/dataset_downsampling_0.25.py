import pandas as pd
import numpy as np

input_file = "G:/merged_nc_data.csv"
df = pd.read_csv(input_file)

df["time"] = pd.to_datetime(df["time"], errors='coerce')  # 让 Pandas 自动检测格式
df["year"] = df["time"].dt.year
df["month"] = df["time"].dt.month
df["day"] = df["time"].dt.day

def offset_bin(coord, offset=0.125, step=0.25):
    return offset + step * np.round((coord - offset) / step)

df["lat_new"] = offset_bin(df["lat"], offset=0.125, step=0.25)
df["lon_new"] = offset_bin(df["lon"], offset=0.125, step=0.25)


df_agg = df.groupby(["year", "month", "day", "lat_new", "lon_new"], as_index=False).agg({
    "prcp": "sum",
    "tavg": "mean"
})

df_agg.rename(columns={
    "lat_new": "lat",
    "lon_new": "lon"
}, inplace=True)

df_merged_style = df_agg[["year", "month", "day", "lon", "lat", "prcp", "tavg"]]

output_file = "G:/nc_data_0.25.csv"
df_merged_style.to_csv(output_file, index=False)

print("✅ The data has been aligned to the 0.25° grid and split out the year, month, day columns.")
print(f"output_file: {output_file}")
