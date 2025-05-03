import duckdb

input_path = "G:/stream_split_dataset_small/val.csv"
sorted_path = "G:/stream_split_dataset_small/val_sorted.csv"

con = duckdb.connect()
con.execute(f"COPY (SELECT * FROM read_csv_auto('{input_path}') ORDER BY year, month, day) TO '{sorted_path}' (HEADER, DELIMITER ',')")
con.close()

print("✅ 排序完成！")
