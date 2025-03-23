import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import griddata

data_file = "G:/merged_nc_data.csv"
df = pd.read_csv(data_file)

selected_date = "2019-12-25"
df_day = df[df["time"] == selected_date].dropna(subset=["lat", "lon", "tavg"])

df_day = df_day[(df_day["tavg"] > -40) & (df_day["tavg"] < 40)]

lat_min, lat_max = 36, 41
lon_min, lon_max = -95, -88
df_day = df_day[(df_day["lat"] >= lat_min) & (df_day["lat"] <= lat_max) &
                (df_day["lon"] >= lon_min) & (df_day["lon"] <= lon_max)]

grid_res = 0.2
lat_grid = np.arange(lat_min, lat_max, grid_res)
lon_grid = np.arange(lon_min, lon_max, grid_res)
lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

tavg_grid = griddata((df_day["lon"], df_day["lat"]), df_day["tavg"],
                     (lon_mesh, lat_mesh), method='cubic')

nan_mask = np.isnan(tavg_grid)
if np.any(nan_mask):
    tavg_grid_nearest = griddata((df_day["lon"], df_day["lat"]), df_day["tavg"],
                                 (lon_mesh, lat_mesh), method='nearest')
    tavg_grid[nan_mask] = tavg_grid_nearest[nan_mask]

fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})

ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

ax.set_title(f"Rough Temperature Heatmap (tavg) on {selected_date}", fontsize=14)

ax.add_feature(cfeature.BORDERS, linewidth=1)
ax.add_feature(cfeature.STATES, edgecolor='black', linewidth=1.5)
ax.add_feature(cfeature.LAKES, color='lightblue', alpha=0.5)

cmap = plt.cm.coolwarm
heatmap = ax.pcolormesh(lon_mesh, lat_mesh, tavg_grid, cmap=cmap, shading='auto')

contour_levels = np.arange(-10, 15, 2)
contours = ax.contour(lon_mesh, lat_mesh, tavg_grid, levels=contour_levels, colors='black', linewidths=1.2)
ax.clabel(contours, inline=True, fontsize=8, fmt="%d°C", colors='black', inline_spacing=2)

cbar = plt.colorbar(heatmap, ax=ax, orientation="vertical", shrink=0.7)
cbar.set_label("Temperature (°C)")

major_cities = {
    "Kansas City": (39.0997, -94.5786),
    "St. Louis": (38.6270, -90.1994),
    "Springfield": (37.2089, -93.2923),
    "Columbia": (38.9517, -92.3341),
    "Jefferson City": (38.5767, -92.1735)
}

for city, (lat, lon) in major_cities.items():
    ax.scatter(lon, lat, color='black', marker='o', s=30, transform=ccrs.PlateCarree())
    ax.text(lon + 0.2, lat, city, fontsize=10, transform=ccrs.PlateCarree())

plt.show()
