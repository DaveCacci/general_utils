# Old UIT custom functionsfor UIT_selectorPI_operative.py
import os
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import configparser
import shutil
from scipy.signal import convolve
import glob
from typing import List, Dict, Optional

##############################################################
# FUNCTION TO FIND THE #'num_files'-MOST RECENT DATA FILES IN THE DIRECTORY 
# Depending on the distance of the timestamp written in the file name with respect to the current date of running
def find_csv_file_paths(directory_path,num_files=2):
    # Step 1: List all .ini files in the directory
    csv_files = [file for file in os.listdir(directory_path) if file.endswith(".csv")]
    if not csv_files:
        no_files = 'No .csv files in the directory'
        warning_msg = 'No .csv files in the directory'
        return None, no_files, warning_msg

    # Step 2: Extract timestamps from the file names
    timestamps = [datetime.strptime(file.split('_')[0], "%Y-%m-%d") for file in csv_files]

    # Step 3: Sort timestamps in descending order (most recent first)
    current_timestamp = datetime.now()
    closest_timestamp = min(timestamps, key=lambda x: abs(x - current_timestamp))
    sorted_timestamps = sorted(timestamps, reverse=True)

    # Step 4: Get the most recent N timestamps
    recent_timestamps = sorted_timestamps[:num_files]

    # Step 5: Load the corresponding CSV files
    csv_file_paths = [os.path.join(directory_path, f"{timestamp.strftime('%Y-%m-%d')}_R2.csv") for timestamp in recent_timestamps]

    # Check if the closest timestamp is within one day. If not, return warning
    warning_msg = "No .csv file within one day of the current timestamp. Loading the nearest .csv file." if abs(current_timestamp - closest_timestamp) > timedelta(days=1) else 'Ok loading .csv'
    
    # Load and return the .ini file
    success_msg = f"Loading .csv file: {csv_file_paths}"
    print('\n'.join(csv_file_paths))
    return csv_file_paths, success_msg, warning_msg

##############################################################
# FUNCTION TO LOAD THE CSV DATA FILES
# It returns a dataframe (empty if it fails)
def read(recent_csv_files):
    file_path = recent_csv_files.replace('\\', '/')
    
    try:
        # Read the .csv file and append the dataframe to the list
        df = pd.read_csv(file_path,encoding='ISO-8859-1',delimiter = ';', 
                   parse_dates=['Timestamp'], dayfirst=True)
        success_reading = f"Successfully read {file_path}"
        print(success_reading)
        return df,success_reading
    
    except Exception:
        success_reading = f"Failed reading {file_path}"
        return [],success_reading

# FUNCTION FOR FIR FILTERING OF DATA (moving average)
def moving_average_filter(signal, window_size):
    try:
        kernel = np.ones(window_size) / window_size
        filtered_signal = convolve(signal, kernel, mode='valid')
        return np.concatenate((np.full(window_size - 1, np.nan), filtered_signal), axis=0), 'No'
    except Exception as e:
        filtering_msg = f'{e} happend while filtering. Returning unfiltered signal'
        return signal, filtering_msg

##############################################################
# FUNCTION TO READ CSV FILES
# Used to read setpoint csv file
def read_csv_with_column_names(file_path):
    """
    Read a CSV file, check for its existence, and store it in a DataFrame.

    Parameters:
    - file_path (str): Path to the CSV file.

    Returns:
    - df (pd.DataFrame): DataFrame containing the data from the CSV file.
    """
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")
    # Read the CSV file with the specified column names from the first row
    df = pd.read_csv(file_path, header=0)  # Set header=0 to use the first row as column names
    return df

##############################################################
# FUNCTION TO LOG QUANTITIES OF INTEREST OF THE CODE RUN TO A CSV FILE
# Reads and appends current run row to dataframe 
def append_row_to_csv(file_path, new_row_dict):
    """
    Append a new row (in the form of a dictionary) to a CSV file.

    Parameters:
    - file_path (str): Path to the CSV file.
    - new_row_dict (dict): Dictionary representing the new row.

    Returns:
    - None
    """

    # Check if the file exists
    if not pd.io.common.file_exists(file_path):
        # If the file doesn't exist, create a new DataFrame with the header
        df = pd.DataFrame(columns=new_row_dict.keys())
    else:
        # If the file exists, read the existing DataFrame from the CSV
        df = pd.read_csv(file_path)

    # Append the new row to the DataFrame
    timestamp = datetime.now()
    df = df.append({'Timestamp_control':timestamp,**new_row_dict}, ignore_index=True) #Modify with '_append' on UIT machine (Pyhton <= 3.8)

    # Write the DataFrame back to the CSV file
    df.to_csv(file_path, index=False)

##############################################################
# FUNCTION TO FIND THE MOST RECENT AVAILABLE .INI FILE based on timestamp in its filename
# QUESTION: Input .ini (last available) must be taken from the last available file in "Parameter correct"...or from Bioreator/Parameter?
def load_closest_ini_file(directory_path):
    # Step 1: List all .ini files in the directory (PARAMETER_CORRECT)
    ini_files = [file for file in os.listdir(directory_path) if file.endswith(".ini")]

    if not ini_files:
        no_files = 'No .ini files in the directory'
        warning_msg = 'No .ini files in the directory'
        return None, no_files, warning_msg

    # Step 2: Extract timestamps from the file names
    timestamps = [datetime.strptime(file.split('_')[1] + '_' + file.split('_')[2].split('.')[0], "%Y-%m-%d_%H-%M-%S") for file in ini_files]

    # Step 3: Find the timestamp closest to the current timestamp
    current_timestamp = datetime.now()
    closest_timestamp = min(timestamps, key=lambda x: abs(x - current_timestamp))

    # Step 5: Load the corresponding .ini file
    closest_ini_file = f"Parameter_{closest_timestamp.strftime('%Y-%m-%d_%H-%M-%S')}.ini"
    ini_file_path = os.path.join(directory_path, closest_ini_file)
    
    # Check last modification...the time difference from now is higher than the control interval?
    file_modification_time = os.path.getmtime(ini_file_path)
    modification_datetime = datetime.fromtimestamp(file_modification_time)
    age = current_timestamp - modification_datetime
    
    # Check if the closest timestamp is within one day
    if abs(current_timestamp - closest_timestamp) > timedelta(days=1):
        warning_msg = "No .ini file within one day of the current timestamp. Loading the nearest .ini file."
        print(warning_msg)
    # Check if the closest timestamp is older than the control interval
    elif age.total_seconds() > dt+500:
        warning_msg = "No .ini file within 2.5 hours of the current timestamp. Loading the nearest .ini file."
        print(warning_msg)

    # Load and return the .ini file
    success_msg = f"Loading .ini file: {ini_file_path}"
    print(success_msg)
    return ini_file_path, success_msg, warning_msg

##############################################################
# FUNCTION TO READ THE MOST RECENT .INI FILE AND UPDATE IT
# New .ini file has to be placed in 'Remote' folder (output_directory)
def update_ini_file(input_config_file_path, input_directory, output_directory, on_seconds, off_seconds):
    # Read the original .ini file
    config = configparser.ConfigParser()
    config.read(input_config_file_path)

    # Modify the .ini file with the computed control output
    config[f'REACTOR{reactor_number}']['feedpumpontime'] = str(on_seconds)+',000000'
    config[f'REACTOR{reactor_number}']['feedpumpofftime'] = str(off_seconds)+',000000'

    # Ensure the output directory exists
    # if not os.path.exists(output_directory):
    #    raise DirectoryNotFoundError(f"The directory {output_directory} does not exist.")

    # Save the modified .ini file in the specified output directory
    output_file_name = f"Parameter_{current_time.strftime('%Y-%m-%d_%H-%M-%S')}.ini"
    output_file_path = os.path.join(output_directory, output_file_name)
    output_file_path = output_file_path.replace('\\', '/')
    with open(output_file_path, 'w') as configfile:
        config.write(configfile)

    # Make a copy of the modified .ini file in another directory to keep track of them
    #copy_file_name = file_name.replace('.ini','_copy.ini')
    copy_file_path = os.path.join(input_directory, output_file_name)
    copy_file_path = copy_file_path.replace('\\', '/')
    shutil.copyfile(output_file_path, copy_file_path)
    success_msg = f"""Parameter_{current_time.strftime('%Y-%m-%d_%H-%M-%S')}.ini has been written in {output_file_path} and {copy_file_path}"""
    return success_msg

##############################################################

def read_csv_files(strings_list: List[str],
                   directory: str = ".",
                   start_date: Optional[str] = None,
                   end_date: Optional[str] = None,
                   date_format: str = "%Y-%m-%d",
                   sort_files: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Reads and concatenates CSV files whose names contain specified strings.

    Args:
        strings_list: List of name prefixes (e.g. ['Output_selector','Header']) to look for.
        directory: Directory where CSV files are stored (default: current directory).
        start_date: Optional start date (inclusive) in a parseable format (e.g. '2024-01-31').
                    If provided together with end_date the function will attempt to read files
                    for each date in the range using the pattern "<string>_<YYYY-MM-DD>*.csv".
        end_date:   Optional end date (inclusive). Required if start_date is provided.
        date_format: Format used to format the dates when building filename patterns (default "%Y-%m-%d").
        sort_files: Whether to sort matched filenames before concatenation (default True).

    Returns:
        A dict mapping each string in strings_list to a concatenated DataFrame.
        Only entries for which at least one file was found are present in the returned dict.
    """
    result_dict = {}
    directory = os.path.abspath(directory)

    # Helper to read and concat a list of files
    def _concat_files(file_list):
        if not file_list:
            return None
        if sort_files:
            file_list = sorted(file_list)
        dfs = []
        for f in file_list:
            try:
                dfs.append(pd.read_csv(f))
            except Exception as e:
                # If you prefer to surface errors, re-raise or log here.
                print(f"Warning: failed to read '{f}': {e}")
        return pd.concat(dfs, ignore_index=True) if dfs else None

    # Date-range mode
    if start_date is not None and end_date is not None:
        start = pd.to_datetime(start_date).date()
        end = pd.to_datetime(end_date).date()
        if end < start:
            raise ValueError("end_date must be >= start_date")

        all_dates = pd.date_range(start, end).to_pydatetime()
        for string in strings_list:
            matched_files = []
            for d in all_dates:
                date_str = d.strftime(date_format)
                pattern = os.path.join(directory, f"{string}_{date_str}*.csv")
                matched_files.extend(glob.glob(pattern))
            df = _concat_files(matched_files)
            if df is not None:
                result_dict[string] = df

    else:
        # No date range supplied: match all files with "<string>_*.csv"
        for string in strings_list:
            pattern = os.path.join(directory, f"{string}_*.csv")
            matched_files = glob.glob(pattern)
            df = _concat_files(matched_files)
            if df is not None:
                result_dict[string] = df

    return result_dict

