# stream_split_chunker_streaming_to_pt_debug.py
import csv
# Import standard libraries: csv for file parsing, os for path handling, torch for tensor operations,
# numpy for numerical processing, collections for flexible dictionaries, and datetime for date operations.
import os
import torch
import numpy as np
from collections import defaultdict
from datetime import datetime

# Setting parameters
split_names = ["train_sorted", "val_sorted", "test_sorted"]
# Define the three dataset splits to process and the output directory
# chunk_size_days defines how many days are packed into one .pt file for streaming training
save_dir = "G:/stream_split_dataset"
chunk_size_days = 100  # A small PT file every 100 days

# Read and process each split
def process_split(split_name):
# Process a single split (train/val/test): read from CSV and split into temporal chunks by date
    input_path = os.path.join(save_dir, f"{split_name}.csv")
    output_split_dir = os.path.join(save_dir, f"{split_name}_pt_chunks")
    os.makedirs(output_split_dir, exist_ok=True)

    print(f"\n🔵 Processing {split_name} split...")

    chunk_id = 0
# chunk_id keeps track of the chunk number per split; current_dates stores daily sequences
    current_dates = []
    current_data = defaultdict(list)
    last_date = None

    with open(input_path, 'r', newline='') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames

        for idx, row in enumerate(reader):
            year = int(row['year'])
            month = int(row['month'])
            day = int(row['day'])
            date = datetime(year, month, day)

            if last_date != date:
# When encountering a new day:
# - Check if current chunk reached desired day count (chunk_size_days)
# - If yes, save it and reset data structures
                print(f"📅 New day encountered: {date.strftime('%Y-%m-%d')}")
                if last_date is not None and len(current_dates) >= chunk_size_days:
                    print(f"🚀 Saving chunk {chunk_id} after reaching {len(current_dates)} days: {current_dates[0]} ~ {current_dates[-1]}")
                    save_chunk(current_dates, current_data, output_split_dir, split_name, chunk_id, fieldnames)
                    chunk_id += 1
                    current_dates = []
                    current_data = defaultdict(list)

                current_dates.append(date)
                last_date = date

            current_data[date].append(row)

            if idx % 500000 == 0 and idx > 0:
                print(f"🔵 Processed {idx} rows...")

        if current_dates:
            print(f"🚀 Saving final chunk {chunk_id} with {len(current_dates)} days: {current_dates[0]} ~ {current_dates[-1]}")
            save_chunk(current_dates, current_data, output_split_dir, split_name, chunk_id, fieldnames)

    print(f"✅ {split_name} done!")

def save_chunk(dates, data_dict, output_dir, split_name, chunk_id, fieldnames):
# Convert a temporal span (multiple days) into a spatial tensor and save as a .pt file
    dates = sorted(dates)
    print(f"📦 Saving chunk {chunk_id}: {len(dates)} days from {dates[0]} to {dates[-1]}")

    all_lat = set()
# Collect all unique latitudes and longitudes present in the data chunk
    all_lon = set()
    for day_rows in data_dict.values():
        for row in day_rows:
            all_lat.add(float(row['lat']))
            all_lon.add(float(row['lon']))

    lats = sorted(all_lat)
    lons = sorted(all_lon)
    H = len(lats)
    W = len(lons)

    lat_idx = {v: i for i, v in enumerate(lats)}
    lon_idx = {v: i for i, v in enumerate(lons)}

    feature_list = [
# Define the list of features per grid point per day (climate + calendar encodings)
        'prcp', 'tavg', 'Swnet_tavg', 'Lwnet_tavg', 'Qg_tavg',
        'Evap_tavg', 'Qsm_tavg', 'SnowDepth_tavg', 'SoilMoist_P_tavg', 'TVeg_tavg', 'TWS_tavg',
        'sin_day', 'cos_day'
    ]

    X = np.zeros((len(dates), len(feature_list), H, W), dtype=np.float32)
# Allocate tensor memory: shape = (T, C, H, W), where T=days, C=channels, H×W=grid

    for t, date in enumerate(dates):
# Fill tensor X with corresponding data from each day and each (lat, lon) grid point
        if date not in data_dict:
            continue
        doy = date.timetuple().tm_yday
        for row in data_dict[date]:
            lat = float(row['lat'])
            lon = float(row['lon'])
            if lat not in lat_idx or lon not in lon_idx:
                continue
            i = lat_idx[lat]
            j = lon_idx[lon]
            X[t, 0, i, j] = float(row['prcp'])
            X[t, 1, i, j] = float(row['tavg'])
            X[t, 2, i, j] = float(row['Swnet_tavg'])
            X[t, 3, i, j] = float(row['Lwnet_tavg'])
            X[t, 4, i, j] = float(row['Qg_tavg'])
            X[t, 5, i, j] = float(row['Evap_tavg'])
            X[t, 6, i, j] = float(row['Qsm_tavg'])
            X[t, 7, i, j] = float(row['SnowDepth_tavg'])
            X[t, 8, i, j] = float(row['SoilMoist_P_tavg'])
            X[t, 9, i, j] = float(row['TVeg_tavg'])
            X[t, 10, i, j] = float(row['TWS_tavg'])
            X[t, 11, i, j] = np.sin(2 * np.pi * doy / 365)
# sin_day and cos_day encode seasonal cycles into the model as temporal features
            X[t, 12, i, j] = np.cos(2 * np.pi * doy / 365)

    final_date = dates[-1]
    covered_coords = set()
    lats_final_day = set()
    lons_final_day = set()

    for row in data_dict[final_date]:
        lat = float(row['lat'])
        lon = float(row['lon'])
        lats_final_day.add(lat)
        lons_final_day.add(lon)
        if lat in lat_idx and lon in lon_idx:
            covered_coords.add((lat, lon))

    expected_grid_num = len(lats_final_day) * len(lons_final_day)
# Compute how many grid points should exist on the final day to verify data completeness

    if expected_grid_num == 0:
# Compute how many grid points should exist on the final day to verify data completeness
        print(f"⚠️ Final day {final_date} has no grid, dropping entire chunk {split_name}_chunk_{chunk_id}.")
        return

    if len(covered_coords) < 0.8 * expected_grid_num:
# Compute how many grid points should exist on the final day to verify data completeness
        print(f"⚠️ Last day grid incomplete in {split_name}_chunk_{chunk_id}, masking last day (covered {len(covered_coords)}/{expected_grid_num}).")
# Compute how many grid points should exist on the final day to verify data completeness
        X[-1, :, :, :] = 0

    tensor = torch.tensor(X)
    torch.save(tensor, os.path.join(output_dir, f"{split_name}_chunk_{chunk_id}.pt"))
# Save the final tensor chunk to disk as a .pt file
    start_day = dates[0].strftime("%Y-%m-%d")
    end_day = dates[-1].strftime("%Y-%m-%d")
    print(f"✅ Saved {split_name}_chunk_{chunk_id}.pt ({start_day} ~ {end_day}) with shape {tensor.shape}")

for split_name in split_names:
# Process all three splits (train/val/test) using the defined function
    process_split(split_name)

print("\n🎉 All splits chunked to .pt successfully!")
