import pandas as pd
import numpy as np

# Load the dataset
csv_file_path = r"G:/merged_nc_data.csv"
df = pd.read_csv(csv_file_path)

# Check unique lat/lon step sizes
lat_steps = np.unique(np.diff(np.sort(df["lat"].unique())))
lon_steps = np.unique(np.diff(np.sort(df["lon"].unique())))

print("Latitude Step Sizes:", lat_steps)
print("Longitude Step Sizes:", lon_steps)
