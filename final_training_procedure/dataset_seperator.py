# safe_stream_split_by_year.py
import csv
import os
from datetime import datetime

# 路径配置
raw_data_path = "G:/merged_climate_data.csv"
save_dir = "G:/stream_split_dataset_small"
os.makedirs(save_dir, exist_ok=True)

train_path = os.path.join(save_dir, "train.csv")
val_path = os.path.join(save_dir, "val.csv")
test_path = os.path.join(save_dir, "test.csv")

# 打开目标文件，提前写入header
with open(raw_data_path, 'r', newline='') as infile:
    reader = csv.reader(infile)
    header = next(reader)

    with open(train_path, 'w', newline='') as train_f, \
         open(val_path, 'w', newline='') as val_f, \
         open(test_path, 'w', newline='') as test_f:

        train_writer = csv.writer(train_f)
        val_writer = csv.writer(val_f)
        test_writer = csv.writer(test_f)

        # 写入表头
        train_writer.writerow(header)
        val_writer.writerow(header)
        test_writer.writerow(header)

print("🔵 Header written. Start fast streaming split by year...")

# 重新打开输入文件，边读边分配数据
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
            # 直接根据年份粗略划分
            if 2011 <= year <= 2018:
                train_writer.writerow(row)
            elif 2019 <= year <= 2020:
                val_writer.writerow(row)
            elif 2021 <= year <= 2022:
                test_writer.writerow(row)
            else:
                pass  # 如果有异常年份，跳过（比如误差年份）

            if idx % 500000 == 0 and idx > 0:
                print(f"🔵 Processed {idx} rows...")

print("🎉 All done! Train/Val/Test CSVs are ready!")
