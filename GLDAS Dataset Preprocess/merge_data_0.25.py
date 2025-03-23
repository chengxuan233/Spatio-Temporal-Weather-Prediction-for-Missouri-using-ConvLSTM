import pandas as pd

nc_data_file = "G:/nc_data_0.25deg.csv"
df_nc = pd.read_csv(nc_data_file)

df_nc["time"] = pd.to_datetime(df_nc["time"], errors="coerce")

df_nc["year"] = df_nc["time"].dt.year
df_nc["month"] = df_nc["time"].dt.month
df_nc["day"] = df_nc["time"].dt.day

df_nc.drop(columns=["time"], inplace=True)

nasa_data_file = "G:/nasa_dataset_0.25deg.csv"
df_nasa = pd.read_csv(nasa_data_file)

df_merged = pd.merge(df_nc, df_nasa, on=["year", "month", "day", "lon", "lat"], how="inner")

df_merged = df_merged.sort_values(by=["year", "month", "day", "lat", "lon"])

output_file = "G:/merged_climate_data.csv"
df_merged.to_csv(output_file, index=False)

print(f"The data merge is complete and the results are saved to: {output_file}")
