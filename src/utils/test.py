import os
import pandas as pd

def count_total_rows(directory):
    total_rows = 0
    # 遍历目录及其子目录中的所有文件
    for root, _, files in os.walk(directory):
        for filename in files:
            # 检查文件是否为xlsx
            if filename.endswith('.xlsx'):
                filepath = os.path.join(root, filename)
                # 读取xlsx文件
                try:
                    df = pd.read_excel(filepath)
                    # 统计行数并累加
                    total_rows += len(df)
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")
    return total_rows

directory_path = '/opt/Filmdataset/demo/script_3000'
total_rows = count_total_rows(directory_path)
print(f"Total rows in all .xlsx files: {total_rows}")
