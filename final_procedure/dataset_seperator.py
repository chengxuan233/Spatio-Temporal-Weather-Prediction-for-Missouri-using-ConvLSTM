# safe_stream_split_by_year.py
import csv
import os
from datetime import datetime

# === Path Configuration ===
# Input raw CSV file path
raw_data_path = "G:/merged_climate_data.csv"
# Output directory for separated datasets
save_dir = "G:/stream_split_dataset_small"
os.makedirs(save_dir, exist_ok=True)

# Output file paths for train, validation, and test splits
train_path = os.path.join(save_dir, "train.csv")
val_path = os.path.join(save_dir, "val.csv")
test_path = os.path.join(save_dir, "test.csv")

# === Step 1: Write CSV headers to output files ===
with open(raw_data_path, 'r', newline='') as infile:
    reader = csv.reader(infile)
    header = next(reader)  # Read the header row from input CSV

    with open(train_path, 'w', newline='') as train_f, \
         open(val_path, 'w', newline='') as val_f, \
         open(test_path, 'w', newline='') as test_f:

        train_writer = csv.writer(train_f)
        val_writer = csv.writer(val_f)
        test_writer = csv.writer(test_f)

        # Write the same header to all three split files
        train_writer.writerow(header)
        val_writer.writerow(header)
        test_writer.writerow(header)

print(" Header written. Start fast streaming split by year...")

# === Step 2: Read input file and stream rows into the appropriate split ===
with open(raw_data_path, 'r', newline='') as infile:
    reader = csv.DictReader(infile)

    with open(train_path, 'a', newline='') as train_f, \
         open(val_path, 'a', newline='') as val_f, \
         open(test_path, 'a', newline='') as test_f:

        train_writer = csv.DictWriter(train_f, fieldnames=reader.fieldnames)
        val_writer = csv.DictWriter(val_f, fieldnames=reader.fieldnames)
        test_writer = csv.DictWriter(test_f, fieldnames=reader.fieldnames)

        for idx, row in enumerate(reader):
            year = int(row['year'])
            month = int(row['month'])
            day = int(row['day'])

            # Split the data based on the year into train/val/test
            if 2011 <= year <= 2018:
                train_writer.writerow(row)
            elif 2019 <= year <= 2020:
                val_writer.writerow(row)
            elif 2021 <= year <= 2022:
                test_writer.writerow(row)
            else:
                pass  # Skip rows with years outside expected range

            # Log progress every 500,000 rows
            if idx % 500000 == 0 and idx > 0:
                print(f" Processed {idx} rows...")

print(" All done! Train/Val/Test CSVs are ready!")
