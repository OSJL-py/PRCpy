import os
import re
from Utilities.sorting import *

def get_directory_file_names(dir_path):

    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"The directory {dir_path} does not exist.")

    filenames = [fname for fname in os.listdir(dir_path) if "scan" in fname and os.path.isfile(os.path.join(dir_path, fname))]
    if not filenames:
        raise FileNotFoundError("No files containing 'scan' found in the directory.")

    filenames.sort(key=natural_sort_key)

    return filenames

def get_full_paths(dir_path):
    return [os.path.join(dir_path, fname) for fname in get_directory_file_names(dir_path)]

def extract_file_params(fname, regex=r'[-]?\d+'):
    """
    Extracts the parameters from the file name using a regular expression
    """
    return re.findall(regex, fname)

def get_raw_input_vals(dir_path):
    return [float(".".join(extract_file_params(fname)[1:])) for fname in get_directory_file_names(dir_path)]
