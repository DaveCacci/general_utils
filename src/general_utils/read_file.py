import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Any, Union, Optional
from .filter_df_by_timestamp import filter_df_by_timestamp
from .create_and_change_dir import create_and_change_dir

##############################################################
def read_and_split_txt(filename, delimiter='\t', log: bool = False):
    """
    Read a text file and split all the columns present in each line.
    
    Args:
    - filename: The path to the text file.
    - delimiter: The delimiter used to separate columns (default is '\t' for tab-separated values).
    
    Returns:
    - A list of lists containing all the extracted columns.
    """
    # Initialize an empty list to store the columns
    columns = []
    
    # Open the file and read its lines
    with open(filename, 'r') as file:
        # Skip the first two lines
        for _ in range(2):
            next(file)
        # Read each line in the file
        for line in file:
            
            # Check if the line is empty
            if line.strip():  # If the stripped line is not empty
                line_columns = line.strip().split(delimiter)
                # Extend columns list with the columns from the current line
                if len(columns) < len(line_columns):
                    columns.extend([[] for _ in range(len(line_columns) - len(columns))])
                for i, column_value in enumerate(line_columns):
                    try:
                        # Convert the column value to float
                        column_value = float(column_value)
                    except ValueError:
                        # If conversion fails, keep it as a string
                        pass
                    columns[i].append(column_value)
            else:
                break  # Exit the loop if the line is empty
    if log:
        logging.info(f'Read file named: {os.path.basename(filename)}')
    
    return columns

##############################################################
def filter_and_convert_to_time(file_path=None, start_timestamp=None, end_timestamp=None, fill_option="zeros", df: pd.DataFrame = None, log: bool = False):
    """
    Read a CSV file, filter it by timestamp, and convert timestamps to seconds starting from a given start time.

    Args:
        file_path (str): Path to the CSV file.
        start_timestamp (str or pd.Timestamp, optional): Start timestamp for filtering.
        end_timestamp (str or pd.Timestamp, optional): End timestamp for filtering.
        fill_option (str): Determines the first row in case of a missing start_timestamp in the data:
                           - "zeros": Fill with 0 for all columns except timestamp.
                           - "first_row": Copy the last row before the start timestamp.
        df (pd.DataFrame, optional): DataFrame to use instead of reading from the file.                   

    Returns:
        pd.DataFrame: Filtered and processed DataFrame with timestamps converted to seconds 
        np.ndarray: Array representation of the DataFrame with timestamps converted to seconds
    """
    # Load the DataFrame
    if df is None:
        df = pd.read_csv(file_path)
        if log:
            logging.info(f'Read file named: {os.path.basename(file_path)}')


    # Ensure the 'Timestamp' column exists and is in datetime format
    if "Timestamp" not in df.columns:
        raise ValueError("The CSV file must contain a 'Timestamp' column.")

    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    # Added on 12.03.2025 to create result also when I don't want to filter but "only to read"
    start_timestamp = start_timestamp if start_timestamp else df['Timestamp'].iloc[0]
    end_timestamp = end_timestamp if end_timestamp else df['Timestamp'].iloc[-1]

    # Last row before the start timestamp
    before_start_df = df[df["Timestamp"] < start_timestamp]
    after_end_df = df[df["Timestamp"] > end_timestamp]
    if not before_start_df.empty:
        last_row_before_cut = before_start_df.iloc[-1].to_list()
    if not after_end_df.empty:
        first_row_after_cut = after_end_df.iloc[0].to_list()

    # Filter the DataFrame using the start and end timestamps
    if start_timestamp or end_timestamp:
        df = filter_df_by_timestamp(df, start_timestamp, end_timestamp)

    if df.empty:
        #df = before_start_df
        logging.info(f"{os.path.basename(file_path)}: no data found within the specified timestamp range. Using data before the start timestamp.")

    # Handle the case where the filtered DataFrame doesn't start exactly at start_timestamp
    if start_timestamp and df[(df["Timestamp"]==pd.to_datetime(start_timestamp))].empty:
        start_timestamp = pd.to_datetime(start_timestamp)
        #before_start_df = df[df["Timestamp"] < start_timestamp]

        # Handle time difference
        if not df.empty:
            time_difference = (df["Timestamp"].iloc[0] - start_timestamp).total_seconds()
            logging.info(f'{os.path.basename(file_path)}: {start_timestamp} is not in the data. The first timestamp of the filtered data is {df["Timestamp"].iloc[0]}.')

        if fill_option == "zeros":
            first_row = [0] * len(df.columns)
        elif fill_option == "first_row" and not before_start_df.empty:
            first_row = last_row_before_cut
        elif fill_option == "first_row" and before_start_df.empty:
            first_row = df.iloc[0].to_list()
        else:
            raise ValueError("fill_option must be either 'zeros' or 'first_row'.")

        first_row[0] = start_timestamp
        #df.loc[0] = first_row
        df = pd.concat([pd.DataFrame([first_row], columns=df.columns), df], ignore_index=True)
        #df.index = df.index + 1  # Shift the index

    if end_timestamp and df[(df["Timestamp"]==pd.to_datetime(end_timestamp))].empty:
        end_timestamp = pd.to_datetime(end_timestamp)

        # Handle time difference
        if not df.empty:
            time_difference = (end_timestamp - df["Timestamp"].iloc[-1]).total_seconds()
            logging.info(f'{os.path.basename(file_path)}: {end_timestamp} is not in the data. The last timestamp of the filtered data is {df["Timestamp"].iloc[-1]}.')

        if fill_option == "zeros":
            last_row = [0] * len(df.columns)
        elif fill_option == "first_row" and not after_end_df.empty:
            last_row = first_row_after_cut
        elif fill_option == "first_row" and after_end_df.empty:
            last_row = df.iloc[-1].to_list()
        else:
            raise ValueError("fill_option must be either 'zeros' or 'first_row'.")

        last_row[0] = end_timestamp
        df = pd.concat([df, pd.DataFrame([last_row], columns=df.columns)], ignore_index=True)
        df = df.reset_index(drop=True)
        #df.loc[len(df)] = last_row
        #df = df.append(last_row, ignore_index=True).reset_index(drop=True)

    # Convert timestamp to seconds starting from the first timestamp in the DataFrame
    start_time = df["Timestamp"].iloc[0]
    df["Timestamp"] = (df["Timestamp"] - start_time).dt.total_seconds()
    #print(df) #for debugging

    # Convert to a NumPy array
    array = df.to_numpy()

    return df, array, before_start_df

##############################################################
def check_and_adjust_interp_output(time_points, values, threshold=100, max_duration=300):
    """
    Check and adjust the output of an interpolation function for values exceeding a threshold for too long.

    Args:
        time_points (np.ndarray): Array of time points (in seconds).
        values (np.ndarray): Array of interpolated values corresponding to the time points.
        threshold (float): Threshold value to check against (default is 100).
        max_duration (float): Maximum allowed duration (in seconds) for values above the threshold (default is 300 seconds).

    Returns:
        np.ndarray: Adjusted values with values exceeding the threshold for too long replaced with 1e-10.
    """
    # Ensure time_points and values are numpy arrays
    time_points = np.array(time_points)
    values = np.array(values)

    # Identify indices where values exceed the threshold
    above_threshold = values > threshold

    # Group consecutive indices where the condition is True
    indices = np.where(above_threshold)[0]
    consecutive_groups = np.split(indices, np.where(np.diff(indices) != 1)[0] + 1)

    # Check duration of each group and adjust if necessary
    for group in consecutive_groups:
        if len(group) > 1:
            duration = time_points[group[-1]] - time_points[group[0]]
            if duration > max_duration:
                # Replace values exceeding the allowed duration with 1e-10
                values[group[max_duration // (time_points[1] - time_points[0]):]] = 1e-10

    return values, consecutive_groups

##############################################################
def read_excel_file(file_path, exclude_sheet=None, start_timestamp=None, end_timestamp=None, log: bool = False, save_csv: bool = False):
    """
    Read an Excel file and return a dictionary of DataFrames for each sheet, optionally filtering rows
    based on the 'timestamp' column.

    Args:
        file_path (str): Path to the Excel file.
        exclude_sheet (list, optional): List of sheet names to exclude.
        start_timestamp (str or pd.Timestamp, optional): The start timestamp for filtering.
        end_timestamp (str or pd.Timestamp, optional): The end timestamp for filtering.
        log (bool, optional): Whether to log the reading process.
        save_csv (bool, optional): Whether to save each sheet as a CSV file.

    Returns:
        dict: A dictionary where keys are sheet names and values are DataFrames.
    """
    # Read the Excel file using openpyxl engine
    xls = pd.ExcelFile(file_path, engine='openpyxl')
    exclude_sheet = exclude_sheet or []
    sheets_to_read = [sheet for sheet in xls.sheet_names if sheet not in exclude_sheet]
    
    # Read and filter each sheet
    sheet_dict = {}
    for sheet_name in sheets_to_read:
        df = pd.read_excel(xls, sheet_name, engine='openpyxl', parse_dates=True)
        filtered_df = df

        # Convert non-numeric values to NaN
        for col in df.columns:
            if col != 'Timestamp':  # Skip the 'Timestamp' column
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # # Filter by timestamp if the 'Timestamp' column exists
        # if "Timestamp" in df.columns:
        #     df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors='coerce')
        #     filtered_df = filter_df_by_timestamp(df, start_timestamp, end_timestamp)

        sheet_dict[sheet_name] = filtered_df

        # Save each sheet as a CSV file if save_csv is True
        if save_csv:
            csv_output_path = os.path.join(os.path.dirname(file_path), f"{sheet_name}.csv")
            filtered_df.to_csv(csv_output_path, index=False)
            if log:
                logging.info(f'Saved sheet "{sheet_name}" to CSV file: {csv_output_path}')

    if log:
        logging.info(f'Read file named: {os.path.basename(file_path)}')
    
    return sheet_dict

##############################################################
def group_columns_by_string(sheet_dict, string_to_search, exclude_sheet):
    '''
    Group columns in each DataFrame by a specific string.
    To be coupled with read_excel_file function.
    Parameters:
    sheet_dict (dict): Dictionary of DataFrames, where keys are sheet names and values are DataFrames.
    string_to_search (str): The string to search for in column names.
    exclude_sheet (str): The sheet name to exclude from processing.
    Returns:
    grouped_dataframes (dict): Dictionary of DataFrames with grouped columns.
    '''
    # Create a new DataFrame by grouping columns containing a certain string
    grouped_dataframes = {}
    
    for sheet_name, df in sheet_dict.items():
        if sheet_name != exclude_sheet:
            matching_columns = [col for col in df.columns if string_to_search in col]
            grouped_df = df[matching_columns]
            
            # Exclude empty dataframes
            if not grouped_df.empty:
                grouped_dataframes[sheet_name] = grouped_df
    
    return grouped_dataframes

##############################################################
def read_csv_file(file_path, log: bool = False):
    df = pd.read_csv(file_path)
    # Ensure the 'Timestamp' column exists and is in datetime format
    if "Timestamp" not in df.columns:
        raise ValueError("The CSV file must contain a 'Timestamp' column.")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    if log:
        logging.info(f'Read file named: {os.path.basename(file_path)}')
    # Filter the DataFrame using the start and end timestamps (only if it becomes too heavy to be read).
    return df

##############################################################
def cut_and_interpolate_df(df: pd.DataFrame, start_timestamp: pd.Timestamp, end_timestamp: pd.Timestamp, 
                        variables: List[str], kind: str = 'linear', fill_value = None, bounds_error: bool = False,
                        eval_times: Optional[np.ndarray] = None, log: bool = False) -> Dict[str, np.ndarray]:
    """
    Cut a DataFrame between two timestamps and interpolate specified variables.

    Args:
        df (pd.DataFrame): The input DataFrame.
        start_timestamp (pd.Timestamp): The start timestamp for cutting the DataFrame.
        end_timestamp (pd.Timestamp): The end timestamp for cutting the DataFrame.
        variables (list): List of variable names (columns) to interpolate.
        kind (str): The interpolation method (default is 'linear').
        fill_value: The fill value for missing data (default is None). Can be a string or a tuple of floats.
        eval_times (Optional[np.ndarray]): Array of time instants to evaluate the interpolation (default is None).

    Returns:
        Dict[str, np.ndarray]: A dictionary of interpolated values for each variable if eval_times is provided.
        Otherwise, a dictionary of interpolation functions for each variable.
    """
    
    # Check if the DataFrame is sorted by the 'Timestamp' column
    if not df['Timestamp'].is_monotonic_increasing:
        if log:
            logging.warning("The DataFrame is not sorted by 'Timestamp'. Sorting the DataFrame.")
        df = df.sort_values(by='Timestamp')

    # Check if start_timestamp is in the original DataFrame
    if df[df['Timestamp'] == start_timestamp].empty:
        logging.warning(f"start_timestamp {start_timestamp} is not in the DataFrame. Interpolation may result uncorrect!")
        df_cut, _, _ = filter_and_convert_to_time(df=df, start_timestamp=start_timestamp, end_timestamp=end_timestamp)

    # Cut the DataFrame between the specified timestamps
    mask = (df['Timestamp'] >= start_timestamp) & (df['Timestamp'] <= end_timestamp)
    df_cut = df.loc[mask].copy().reset_index(drop=True)

    # Create a time column in seconds since the start_timestamp
    df_cut['Time'] = (df_cut['Timestamp'] - start_timestamp).dt.total_seconds()

    # Create a dictionary to store interpolation functions
    interp_funcs = {}

    # Create interpolation functions for the specified variables
    for variable in variables:
        if variable in df_cut.columns:
            interp_funcs[variable] = interp1d(df_cut['Time'], df_cut[variable], kind=kind, fill_value=fill_value, bounds_error=bounds_error)
        else:
            raise ValueError(f"Variable '{variable}' not found in DataFrame columns.")

    # Evaluate the interpolation functions on the input array of time instants if provided
    if eval_times is not None:
        interpolated_values = {}
        if log:
            logging.info(f"Number of rows in the cut DataFrame = {len(df_cut)}.")
            logging.info(f"Last time in DataFrame = {df_cut['Time'].iloc[-1]}; last time in eval_times = {eval_times[-1]}.")
            logging.info(f'Integration step for consistent timestamp and time_points is = {(len(eval_times)-1)/((end_timestamp - start_timestamp).total_seconds()/86400*24)}. Ok?')
        for variable, func in interp_funcs.items():
            interpolated_values[variable] = func(eval_times.astype(np.float64))
        return interpolated_values

    return interp_funcs

##############################################################
