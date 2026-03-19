import pandas as pd
import numpy as np
from .replace_sheet_content import replace_sheet_content
from datetime import datetime, timedelta
import logging
import os

def filter_and_save_nominal(modelname, current_date, start_window, end_window, output_sens_names, y_df, y_df_data_off=None, var_couples_list_offline=None, directory = os.getcwd(), log=False):
    '''
    This function treats the output data from a nominal run of model evaluation (directly from y_df, not from an Excel file).
    It re-organizes the excel sheets in a way that each sheet contains the output data of a single output variable.
    The function also filters the data based on the time window defined by 'start_window' and 'end_window'.
    The last part of the function is dedicated to the treatment of the offline data, if provided.
    Args:
        modelname (str): Name of the model.
        current_date (datetime): Current date.
        start_window (datetime): Start of the time window for filtering.
        end_window (datetime): End of the time window for filtering.
        output_sens_names (list): List of output sensitivity variable names.
        y_df (pd.DataFrame): DataFrame containing the output data from the nominal run.
        y_df_data_off (pd.DataFrame, optional): DataFrame containing offline data for filtering. Defaults to None.
        var_couples_list_offline (list, optional): List of variable couples for offline data filtering. Defaults to None.
        directory (str, optional): Directory to save the output files. Defaults to os.getcwd().
        log (bool, optional): Flag to enable logging. Defaults to False.
    Returns:
        None
    Note: actually saves excel files in the specified directory: '{modelname}_Output_nom_{current_date}_{date_string}.xlsx',
          '{modelname}_Sheet_Filter_Output_{current_date}_{date_string}.xlsx', and, if offline data is provided,
          '{modelname}_Discontinuous_value_nom_{current_date}_{date_string}.xlsx'.
    Note: from 24.11.2025, "replace_sheet_content" also saves a backup of the original file before modifying it i.e. '{modelname}_Sheet_Filter_Output_{current_date}_{date_string}_beforemod.xlsx'.
    ''' 
    current_date = current_date.strftime("%d-%m-%Y")
    date_string = f'{start_window.strftime("%d%m%H")}-{end_window.strftime("%d%m%H")}'
    output_names = output_sens_names

    # Filter y_df based on the time window
    if log: 
        logging.info(f'Filtering y_df between {start_window} and {end_window}')
    y_df_filtered = y_df[(y_df['Timestamp'] <= end_window) & (y_df['Timestamp'] >= start_window)]
    # Reset index to avoid issues when adding to output_df
    y_df_filtered = y_df_filtered.reset_index(drop=True)

    # Construct the Excel file names with the current date
    output_file = os.path.join(directory, f'{modelname}_Output_nom_{current_date}_{date_string}.xlsx')
    if log and os.path.exists(output_file):
        logging.warning(f"The file {output_file} already exists and will be overridden!")

    # Initialize DataFrames
    output_df = pd.DataFrame()
    # CHECK EVERYTHING IS COHERENT after Timestamp filtering
    timestamp_column = y_df_filtered[(y_df_filtered['Timestamp']<=end_window) & (y_df_filtered['Timestamp']>=start_window)]['Timestamp']
    if log and len(y_df_filtered) != len(timestamp_column):
        logging.info(f'len(data) is {len(y_df_filtered)} but filtered data shall return a lenght of {len(timestamp_column)}')
    output_df['Timestamp'] = timestamp_column #pd.date_range(start=start_window, periods=len(y_df_filtered), freq='h')  # Add timestamp column
    for output_sens_name in output_sens_names:
        output_df[f'{output_sens_name}'] = y_df_filtered[f'{output_sens_name}']

    # Save dynamic_output_df to Excel after each iteration, creating a new sheet for each run
    with pd.ExcelWriter(output_file, engine='xlsxwriter', date_format='dd-mm-yyyy') as dynamic_writer:
        output_df.to_excel(dynamic_writer, sheet_name=f'Nominal Simulation', index=False)
        logging.info(f"Sheet Nominal Simulation saved in {output_file}")

    # Re-organize the file into one sheet per output variable
    output_file_filter=os.path.join(directory, f'{modelname}_Sheet_Filter_Output_{current_date}_{date_string}.xlsx')
    data_path=os.path.join(directory, f'{modelname}_Output_nom_{current_date}_{date_string}.xlsx')
    data = pd.read_excel(data_path)
    with pd.ExcelWriter(output_file_filter, engine='xlsxwriter', date_format='dd-mm-yyyy') as writer:
        for n in range(0, len(output_names)):
            output = pd.DataFrame()  # Inizializza un DataFrame vuoto
            output_name=output_names[n]
            output[output_name] = data.iloc[:,n+1]
            output.insert(0, 'Timestamp', timestamp_column)
            output.to_excel(writer,sheet_name=f'{output_name}',index=False)
            logging.info(f"Sheet {output_name} saved in {output_file_filter}")

    # If offline data is provided, filter the reorganized file accordingly
    if y_df_data_off is not None and not y_df_data_off.empty and var_couples_list_offline is not None and len(var_couples_list_offline) > 0:
        logging.info("y_df_data_off and var_couples_list_offline are not empty.")
        input_file = os.path.join(directory, f'{modelname}_Sheet_Filter_Output_{current_date}_{date_string}.xlsx')
        output_path = os.path.join(directory, f'{modelname}_Discontinuous_value_nom_{current_date}_{date_string}.xlsx')
        filter_on_offlinedata(input_file, output_path, y_df_data_off, var_couples_list_offline, log)
    else:
        logging.info("y_df_data_off or var_couples_list_offline is empty or not provided.")

def filter_and_save_multiple(modelname, current_date, deltap, start_window, end_window, output_sens_names, y_df_data_off=None, var_couples_list_offline=None, directory = os.getcwd(), log=False):       
    '''
    This function treats the output data from multiple runs of model evaluation out of the 'sens_local_OAT' function.
    It re-organizes the excel sheets in a way that each sheet contains the output data of a single output variable.
    The function also filters the data based on the time window defined by 'start_window' and 'end_window'.
    The last part of the function is dedicated to the treatment of the offline data, if provided.
    Args:
        modelname (str): Name of the model.
        current_date (datetime): Current date.
        deltap (list): List of delta perturbations.
        start_window (datetime): Start of the time window for filtering.
        end_window (datetime): End of the time window for filtering.
        output_sens_names (list): List of output sensitivity variable names.
        y_df_data_off (pd.DataFrame, optional): DataFrame containing offline data for filtering. Defaults to None.
        var_couples_list_offline (list, optional): List of variable couples for offline data filtering. Defaults to None.
        directory (str, optional): Directory to save the output files. Defaults to os.getcwd().
        log (bool, optional): Flag to enable logging. Defaults to False.
    Returns:
        None
    Note: actually saves excel files in the specified directory: '{modelname}_Output_{delta}_{current_date}_{date_string}.xlsx',
          '{modelname}_Sheet_Filter_Output_{delta}_{current_date}_{date_string}.xlsx', and, if offline data is provided,
          '{modelname}_Discontinuous_value_{delta}_{current_date}_{date_string}.xlsx'.
    Note: from 24.11.2025, "replace_sheet_content" also saves a backup of the original file before modifying it i.e. '{modelname}_Sheet_Filter_Output_{delta}_{current_date}_{date_string}_beforemod.xlsx'.
    Note: reads files without 'date_string' in the name, because the filtering is done inside the function based on the provided time window, but the output files are named with 'date_string' for clarity.
    '''
    current_date = current_date.strftime("%d-%m-%Y")
    date_string = f'{start_window.strftime("%d%m%H")}-{end_window.strftime("%d%m%H")}'
    output_names = output_sens_names
    for delta in deltap:
        data_path=os.path.join(directory, f'{modelname}_Output_{delta}_{current_date}_{date_string}.xlsx')
        data = pd.read_excel(data_path)
        # Filter y_df based on the time window
        if log:
            logging.info(f'Filtering data between {start_window} and {end_window} for delta {delta}')
        data_filtered = data[(data['Timestamp'] <= end_window) & (data['Timestamp'] >= start_window)]
        # Reset index to avoid issues when adding to output_df
        data_filtered = data_filtered.reset_index(drop=True)
        # Delete the Timestamp column for easier handling
        filtered_data = data_filtered.iloc[:,1:]
        # CHECK EVERYTHING IS COHERENT after Timestamp filtering
        timestamp_column = data_filtered[(data_filtered['Timestamp']<=end_window) & (data_filtered['Timestamp']>=start_window)]['Timestamp']
        if log and len(filtered_data) != len(timestamp_column):
            logging.info(f'len(data) is {len(filtered_data)} but filtered data shall return a lenght of {len(timestamp_column)}')

        # Re-organize the file into one sheet per output variable
        output_file=os.path.join(directory, f'{modelname}_Sheet_Filter_Output_{delta}_{current_date}_{date_string}.xlsx')
        if log and os.path.exists(output_file):
            logging.warning(f"The file {output_file} already exists and will be overridden!")
        with pd.ExcelWriter(output_file, engine='xlsxwriter', date_format='dd-mm-yyyy') as writer:
            for n in range(0, len(output_names)):
                output = pd.DataFrame()  # Initialize an empty DataFrame
                output_name=output_names[n]
                data_subset = pd.DataFrame()
                for i in range(0 + n, len(filtered_data.keys()) + n, len(output_names)): 
                    data_subset = pd.concat([data_subset, filtered_data.iloc[:,i:i+1]], axis=1) # Concatenate columns
                output = pd.concat([output, data_subset], axis=1)
                output.insert(0, 'Timestamp', timestamp_column) #pd.date_range(start=start_window, periods=len(filtered_data), freq='h'))
                # Write the sorted DataFrame to the Excel file
                output.to_excel(writer,sheet_name=f'{output_name}',index=False)
                logging.info(f"Sheet {output_name} saved in {output_file}")
    
    # If offline data is provided, filter the reorganized files accordingly
    if y_df_data_off is not None and not y_df_data_off.empty and var_couples_list_offline is not None and len(var_couples_list_offline) > 0:
        logging.info("y_df_data_off and var_couples_list_offline are not empty.")
        # Iterate over delta values
        for delta in deltap:
            input_file =os.path.join(directory, f'{modelname}_Sheet_Filter_Output_{delta}_{current_date}_{date_string}.xlsx')
            output_path = os.path.join(directory, f'{modelname}_Discontinuous_value_{delta}_{current_date}_{date_string}.xlsx')
            filter_on_offlinedata(input_file, output_path, y_df_data_off, var_couples_list_offline, log)
    else:
        logging.info("y_df_data_off or var_couples_list_offline is empty or not provided.")

def filter_on_offlinedata(input_file, output_path, y_df_data_off, var_couples_list_offline, log=False):
    # Define output-measurement couples
    dis_outputs = [var_couple_list[0] for var_couple_list in var_couples_list_offline]
    dis_to_meas = {
        var_couple_list[0]: var_couple_list[1]
        for var_couple_list in var_couples_list_offline
        if len(var_couple_list) > 1
    }
    index=0
    if log and os.path.exists(output_path):
        logging.warning(f"The file {output_path} already exists and will be overridden!")
    with pd.ExcelWriter(output_path, engine='xlsxwriter', date_format='dd-mm-yyyy') as writer:
        for dis_output in dis_outputs:
            data = pd.read_excel(input_file, sheet_name=dis_output)
            index +=1
            # Drop columns that are not needed for merging
            # Merge DataFrames based on 'Timestamp' column
            merged_df = pd.merge(data, y_df_data_off, on='Timestamp', how='inner')
            columns_to_save = [col for col in merged_df.columns if dis_output in col]
            measurement_col = dis_to_meas.get(dis_output)
            # 18.04.2025 Deal with the situation of Sobol: first indices are computed, then filtered on offlinedata, so no "dis_output" diciture present in the column's names (only parameter names)
            if columns_to_save == []:
                logging.warning(f"Column {dis_output} not found in the data, taking all the columns from the input file (but filtered with merging).")
                columns_to_save = data.columns.tolist()
                columns_to_save = [col for col in columns_to_save if col != 'Timestamp']
            # Keep filtering consistent with compute_error: drop NaNs while measurement is still present.
            if measurement_col in merged_df.columns:
                merged_df.dropna(subset=columns_to_save + [measurement_col], inplace=True)
            else:
                if log:
                    logging.warning(f"Measurement column for {dis_output} not found. Falling back to model-only NaN filtering.")
                merged_df.dropna(subset=columns_to_save, inplace=True)

            merged_df.drop(merged_df.columns.difference(['Timestamp']+columns_to_save), axis=1, inplace=True)
            merged_df.reset_index(drop=True,inplace=True)
            # Write the DataFrame to a new Excel file
            merged_df.to_excel(writer, sheet_name=f'{dis_output}', index=False)
            logging.info(f"Sheet {dis_output} saved in {output_path}")

    # Now, replace the content of input_file with output_path
    replace_sheet_content(input_file, output_path)
