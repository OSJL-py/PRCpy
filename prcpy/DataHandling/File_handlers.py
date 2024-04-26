import pyarrow as pa
from pyarrow import csv
from ..Utilities.os_check import check_and_create_directory

def load_csv(path, delimiter=','):
    try:
        data_table = csv.read_csv(path, parse_options=csv.ParseOptions(delimiter=delimiter))
        df = data_table.to_pandas()
        return df
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None

def save_csv(df, save_path, fname="", delimiter=','):
    try:
        check_and_create_directory(save_path)

        if fname != "":
            save_path = save_path + fname

        table = pa.Table.from_pandas(df)
        csv.write_csv(table, save_path, write_options=csv.WriteOptions(delimiter=delimiter))

        print("[SUCCESS] File saved")
    except Exception as e:
        print(f"[ERROR] Failed to save file: {e}")
