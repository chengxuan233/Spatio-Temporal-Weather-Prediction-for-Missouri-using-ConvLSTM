import os
import tarfile
import xarray as xr
import pandas as pd
import gc
import time
from tqdm import tqdm


START_YEAR = 2011
END_YEAR = 2022
LAT_MIN, LAT_MAX = 33, 43
LON_MIN, LON_MAX = -105, -85
GRID_SIZE = 0.25

download_root = "G:/nclimgrid_daily"
temp_extract_path = "G:/temp_extract"
final_output_file = "G:/nc_data_0.25deg.csv"

os.makedirs(temp_extract_path, exist_ok=True)

if os.path.exists(final_output_file):
    existing_csv = pd.read_csv(final_output_file, usecols=["time"])
    existing_dates = set(existing_csv["time"].unique())
else:
    existing_dates = set()

for root, dirs, files in os.walk(download_root):
    for file in tqdm(files, desc="Processing tar.gz files"):
        if file.endswith(".tar.gz"):
            try:
                year = int(file.split('_s')[1][:4])
            except ValueError:
                print(f" Skipping {file} (Could not parse year)")
                continue

            if year < START_YEAR or year > END_YEAR:
                print(f" Skipping {file} (Year {year} not in range {START_YEAR}-{END_YEAR})")
                continue

            file_path = os.path.join(root, file)
            print(f"\n Processing file: {file_path}")

            try:
                with tarfile.open(file_path, "r:gz") as tar:
                    tar.extractall(path=temp_extract_path)
            except Exception as e:
                print(f" Failed to extract {file_path}: {e}")
                continue

            for extracted_file in tqdm(os.listdir(temp_extract_path), desc="Processing extracted NC files"):
                extracted_file_path = os.path.join(temp_extract_path, extracted_file)

                if extracted_file.endswith(".nc"):
                    print(f"\n Reading NC file: {extracted_file_path}")
                    try:
                        ds = xr.open_dataset(extracted_file_path, chunks="auto")
                        print(f" Variables in {extracted_file_path}: {list(ds.variables)}")

                        ds_filtered = ds[["prcp", "tavg"]]

                        ds_filtered = ds_filtered.sel(lat=slice(LAT_MIN, LAT_MAX), lon=slice(LON_MIN, LON_MAX))

                        for time_chunk in tqdm(ds_filtered["time"].values, desc="Saving time slices"):
                            time_str = pd.Timestamp(time_chunk).strftime('%Y-%m-%d')

                            if time_str in existing_dates:
                                print(f" Skipping time: {time_str} (already in CSV)")
                                continue

                            subset = ds_filtered.sel(time=time_chunk)
                            df = subset.to_dataframe().reset_index()

                            df = df.dropna(subset=["prcp", "tavg"])

                            if df.empty:
                                print(f" Warning: No valid data for {time_str}. Skipping...")
                                continue

                            df["lat_bin"] = ((df["lat"] - 0.125) / GRID_SIZE).round() * GRID_SIZE + 0.125
                            df["lon_bin"] = ((df["lon"] - 0.125) / GRID_SIZE).round() * GRID_SIZE + 0.125

                            df_downsampled = df.groupby(["lat_bin", "lon_bin", "time"]).agg(
                                {
                                    "prcp": "sum",
                                    "tavg": "mean",
                                }
                            ).reset_index()

                            df_downsampled.rename(columns={"lat_bin": "lat", "lon_bin": "lon"}, inplace=True)

                            df_downsampled["prcp"] = df_downsampled["prcp"].round(2)  # prcp 保留 2 位
                            df_downsampled["tavg"] = df_downsampled["tavg"].round(2)  # tavg 保留 2 位

                            df_downsampled.to_csv(final_output_file, mode='a', header=not os.path.exists(final_output_file), index=False)
                            existing_dates.add(time_str)

                        ds.close()
                        del ds
                        gc.collect()

                    except Exception as e:
                        print(f" Error reading NC file: {extracted_file_path}. Error: {e}")

                try:
                    os.remove(extracted_file_path)
                    print(f" Deleted file: {extracted_file_path}")
                except PermissionError:
                    print(f" Warning: Could not delete {extracted_file_path}. Skipping...")

print(f"\n✅ Data merged and saved to: {final_output_file}")
