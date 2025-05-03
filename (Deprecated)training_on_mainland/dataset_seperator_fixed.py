# safe_stream_split_by_year.py
import csv
# Import libraries: csv for reading/writing data, os for file operations, datetime for date handling
import os
from datetime import datetime

# Path configuration
raw_data_path = "G:/climate_data_cleaned.csv"
# Define paths for raw data input and output directories for train, validation, and test splits
save_dir = "G:/stream_split_dataset"
os.makedirs(save_dir, exist_ok=True)
# Create save directory if it does not exist (safe operation)

train_path = os.path.join(save_dir, "train.csv")
# Specify the output file paths for train, validation, and test datasets
val_path = os.path.join(save_dir, "val.csv")
test_path = os.path.join(save_dir, "test.csv")

# Open the target file and write the header in advance
with open(raw_data_path, 'r', newline='') as infile:
    reader = csv.reader(infile)
    header = next(reader)

    with open(train_path, 'w', newline='') as train_f, \
         open(val_path, 'w', newline='') as val_f, \
         open(test_path, 'w', newline='') as test_f:

        train_writer = csv.writer(train_f)
        val_writer = csv.writer(val_f)
        test_writer = csv.writer(test_f)

        # Write header
# Write the CSV header (column names) to all three datasets
        train_writer.writerow(header)
        val_writer.writerow(header)
        test_writer.writerow(header)

print("🔵 Header written. Start fast streaming split by year...")
# Indicate that headers have been written and prepare to stream the dataset

# Reopen the input file and allocate data while reading
with open(raw_data_path, 'r', newline='') as infile:
    reader = csv.DictReader(infile)
# Re-open the CSV file with DictReader for row-wise processing

    with open(train_path, 'a', newline='') as train_f, \
         open(val_path, 'a', newline='') as val_f, \
         open(test_path, 'a', newline='') as test_f:

        train_writer = csv.DictWriter(train_f, fieldnames=reader.fieldnames)
        val_writer = csv.DictWriter(val_f, fieldnames=reader.fieldnames)
        test_writer = csv.DictWriter(test_f, fieldnames=reader.fieldnames)

        for idx, row in enumerate(reader):
# Iterate through each row in the dataset, extracting year/month/day fields
            year = int(row['year'])
            month = int(row['month'])
            day = int(row['day'])
            # Roughly divide by year
            if 2011 <= year <= 2018:
# Principle of split:
# - 2011-2018 ➔ Training data
# - 2019-2020 ➔ Validation data
# - 2021-2022 ➔ Test data
# (Note: Roughly partitioned by year to simulate real-world prediction scenarios)
                train_writer.writerow(row)
            elif 2019 <= year <= 2020:
                val_writer.writerow(row)
            elif 2021 <= year <= 2022:
                test_writer.writerow(row)
            else:
                pass  # If there is an abnormal year, skip it (such as the error year)

            if idx % 500000 == 0 and idx > 0:
# Print progress every 500,000 rows processed to monitor splitting status
                print(f"🔵 Processed {idx} rows...")

print("🎉 All done! Train/Val/Test CSVs are ready!")
# Indicate that the entire split process has completed successfully
