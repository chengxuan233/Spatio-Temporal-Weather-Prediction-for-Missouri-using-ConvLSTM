# === Land Mask Generator Script ===
# ✅ Defines continental US land mask and smoothed soft region mask for attention weighting

import numpy as np
import regionmask
from regionmask.defined_regions import natural_earth_v5_0_0
from scipy.ndimage import gaussian_filter

# === Define Latitude and Longitude Ranges (match dataset resolution) ===
lats = np.round(np.arange(23.125, 53.001, 0.25), 3)
lons = np.round(np.arange(-124.875, -64.999, 0.25), 3)
lon_grid, lat_grid = np.meshgrid(lons, lats)

# === Load Boundary Maps ===
us_states = natural_earth_v5_0_0.us_states_50
mask_us_states = us_states.mask(lon_grid, lat_grid)

ocean = natural_earth_v5_0_0.ocean_basins_50
mask_ocean = ocean.mask(lon_grid, lat_grid)

# === Final Binary Mask: True = land (not ocean) within US boundaries ===
mask_us_land = (~np.isnan(mask_us_states)) & (np.isnan(mask_ocean))

# === Save Static Masks ===
np.save("land_mask_bool.npy", mask_us_land)
np.save("land_mask_lats.npy", lats)
np.save("land_mask_lons.npy", lons)

# === Define Focus Region (for soft mask) ===
TARGET_LAT_MIN = 33.0
TARGET_LAT_MAX = 43.0
TARGET_LON_MIN = -105.0
TARGET_LON_MAX = -85.0

# === Create Binary Region Mask ===
mask_region = np.zeros_like(mask_us_land, dtype=np.float32)
for i, lat in enumerate(lats):
    for j, lon in enumerate(lons):
        if (TARGET_LAT_MIN <= lat <= TARGET_LAT_MAX) and (TARGET_LON_MIN <= lon <= TARGET_LON_MAX):
            mask_region[i, j] = 1.0

# === Apply Gaussian Smoothing to Create Soft Region Mask ===
soft_mask_region = gaussian_filter(mask_region, sigma=2)
soft_mask_region = soft_mask_region / soft_mask_region.max()  # Normalize to [0, 1]

# === Save Final Mask ===
np.save("region_soft_mask.npy", soft_mask_region)

print("✅ Soft region attention mask saved as region_soft_mask.npy")
print("✅ Land mask for continental US saved successfully!")
