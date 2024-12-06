import datetime as dt
import yaml
from pathlib import Path
import pandas as pd

def str_or_none(arg):
    """
    Convert a string to None if it is the string "None".
    """
    if arg is None:
        return None
    if not isinstance(arg, str):
        raise ValueError(f"Expected a string, got {arg} with type {type(arg)}")
    if arg == "None":
        return None
    return arg

def format_time(timestamp, legal_chars_only=False):
    """
    Format a timestamp to a human readable string.
    timestamp: float timestamp or str (can be "now")

    """
    if timestamp == "now":
        timestamp = dt.datetime.now().timestamp()
    if legal_chars_only:
        return dt.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d_%H-%M-%S')
    return dt.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

#################
#### Logging ####
#################

HEADER = "\033[95m"
BLUE = "\033[94m"
CYAN = "\033[96m"
GREEN = "\033[92m"
ORANGE = "\033[93m"
RED = "\033[91m"
ENDC = "\033[0m"
BOLD = "\033[1m"
UNDERLINE = "\033[4m"

def timestamp_info(text):
    """
    Print a timestamped information.
    """
    print(f"{BOLD}{format_time('now')} [INFO] {text}{ENDC}")

def timestamp_ok(text):
    """
    Print a timestamped information.
    """
    print(f"{BOLD + GREEN}{format_time('now')} [OK] {text}{ENDC}")

def timestamp_warning(text):
    """
    Print a timestamped warning.
    """
    print(f"{BOLD + ORANGE}{format_time('now')} [WARNING] {text}{ENDC}")

def timestamp_error(text):
    """
    Print a timestamped error.
    """
    print(f"{BOLD + RED}{format_time('now')} [ERROR] {text}{ENDC}")

def load_yaml(path):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data

def dump_yaml(data, path):
    with open(path, "w") as f:
        yaml.dump(data, f)


def get_list(path):
    if isinstance(path, list):
        return path
    elif path is None:
        return None
    elif not isinstance(path, (str, Path)):
        raise ValueError("path must be a string or a Path object")
    return  pd.read_csv(path, header=None, index_col=0).index.to_list()