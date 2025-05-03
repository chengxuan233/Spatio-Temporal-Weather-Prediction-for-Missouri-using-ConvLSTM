# ✅ split_chunks.py（无掩膜版本）+ 字段值清洗处理，防止空值或非法值

import csv
import os
import torch
import numpy as np
from collections import defaultdict
from datetime import datetime

split_names = ["train_sorted", "val_sorted", "test_sorted"]
save_dir = "G:/stream_split_dataset_small"
chunk_size_days = 100

def safe_float(val):
    try:
        return float(val)
    except:
        return np.nan

def process_split(split_name):
    input_path = os.path.join(save_dir, f"{split_name}.csv")
    output_split_dir = os.path.join(save_dir, f"{split_name}_pt_chunks")
    os.makedirs(output_split_dir, exist_ok=True)

    print(f"\n🔵 Processing {split_name} split...")

    chunk_id = 0
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
                if last_date is not None and len(current_dates) >= chunk_size_days:
                    save_chunk(current_dates, current_data, output_split_dir, split_name, chunk_id, fieldnames)
                    chunk_id += 1
                    current_dates = []
                    current_data = defaultdict(list)

                current_dates.append(date)
                last_date = date

            current_data[date].append(row)

        if current_dates:
            save_chunk(current_dates, current_data, output_split_dir, split_name, chunk_id, fieldnames)

    print(f"✅ {split_name} done!")

def save_chunk(dates, data_dict, output_dir, split_name, chunk_id, fieldnames):
    dates = sorted(dates)
    print(f"📦 Saving chunk {chunk_id}: {len(dates)} days from {dates[0]} to {dates[-1]}")

    all_lat = set()
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
        'prcp', 'tavg', 'Swnet_tavg', 'Lwnet_tavg', 'Qg_tavg',
        'Evap_tavg', 'Qsm_tavg', 'SnowDepth_tavg', 'SoilMoist_P_tavg',
        'TVeg_tavg', 'TWS_tavg', 'sin_day', 'cos_day']

    X = np.zeros((len(dates), len(feature_list), H, W), dtype=np.float32)

    for t, date in enumerate(dates):
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

            vals = [
                safe_float(row['prcp']),
                safe_float(row['tavg']),
                safe_float(row['Swnet_tavg']),
                safe_float(row['Lwnet_tavg']),
                safe_float(row['Qg_tavg']),
                safe_float(row['Evap_tavg']),
                safe_float(row['Qsm_tavg']),
                safe_float(row['SnowDepth_tavg']),
                safe_float(row['SoilMoist_P_tavg']),
                safe_float(row['TVeg_tavg']),
                safe_float(row['TWS_tavg']),
                np.sin(2 * np.pi * doy / 365),
                np.cos(2 * np.pi * doy / 365)
            ]
            if any(np.isnan(vals)):
                continue
            for k, val in enumerate(vals):
                X[t, k, i, j] = val

    tensor = torch.tensor(X)
    torch.save(tensor, os.path.join(output_dir, f"{split_name}_chunk_{chunk_id}.pt"))
    print(f"✅ Saved {split_name}_chunk_{chunk_id}.pt with shape {tensor.shape}")

for split_name in split_names:
    process_split(split_name)

print("\n🎉 All splits chunked to .pt successfully!")
