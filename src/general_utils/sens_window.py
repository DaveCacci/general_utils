# Function to compute ranking and collinearity over a certain window of time
# 24.06.2024 Davide Carecci

# Returns ranking and collinearity to be written in excel files for the output of 'output_names_subset'. 
#If more than one string is present, it returns quantities for that "subset"
# If collinearity = True, performs all computation to output dataframe to be later plotted for collinearity assessment.
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from itertools import combinations
import time
import logging
import os

# DEFINE FUNCTIONS #######################################################################################
def compute_statistics(dataframe):
    '''Compute statistics for each column in the DataFrame.
    Parameters:
    dataframe (DataFrame): Input DataFrame with multiple columns.
    Returns:
    stats_df (DataFrame): DataFrame containing statistics for each column.
    Note: The first column of dataframe is assumed to be a timestamp and is excluded from the calculations.
    '''
    # Exclude the specified column from the DataFrame
    df_subset = dataframe.iloc[:, 1:] # To remove Timestamp column # Giulia Quarta: to be commented when analyzing subsets of outputs because no timestamp column is present!!
    # Added on 10.09.2024 -------------------------------
    df_subset = df_subset.apply(pd.to_numeric, errors='coerce')
    df_subset.fillna(0, inplace=True)  # Replace NaNs with zeros
    # ---------------------------------------------------
    # df_subset=dataframe # Uncomment this when doing subsets of outputs
    # if 'values' of the columns, nan leads to nan, else, it calculates
    mean = df_subset.mean()
    max_values = df_subset.max()
    min_values = df_subset.min()
    stdev_values = df_subset.std()
    sum_abs_div_len = (df_subset.abs().sum() / len(df_subset))
    mean_sqrt = (df_subset ** 2).mean() ** 0.5
    norm = np.linalg.norm(df_subset.values, axis=0)

    # Create a new DataFrame to store the statistics with the original series names
    stats_df = pd.DataFrame({'Mean': mean, 'Max': max_values, 'Min':min_values, 'StDev':stdev_values,'SumAbsDivLen': sum_abs_div_len,'MeanSqrt':mean_sqrt,'Norm':norm})
    return stats_df

def compute_mean_sqrt(dataframe):
    '''Compute the mean square root for each column in the DataFrame.
    Parameters:
    dataframe (DataFrame): Input DataFrame with multiple columns.
    Returns:
    stats_df (DataFrame): DataFrame containing the mean square root for each column.
    Note: The first column of dataframe is assumed to be a timestamp and is excluded from the calculations.
    '''
    df_subset = dataframe.iloc[:, 1:] # To remove Timestamp column # Giulia Quarta: to be commented when analyzing subsets of outputs because no timestamp column is present!!
    # df_subset=dataframe # Uncomment this when doing subsets of outputs
    mean_sqrt = (df_subset ** 2).mean() ** 0.5 
    
    # Create a new DataFrame to store the statistics with the original series names
    stats_df = pd.DataFrame({'MeanSqrt':mean_sqrt})
    return stats_df

def rename_columns(df, output_name: str):
    '''Rename columns of the DataFrame by taking only the common part (parameter name at the end of the strings).
    Parameters:
    df (DataFrame): Input DataFrame with original column names.
    Returns:
    df (DataFrame): DataFrame with renamed columns.
    
    Note: Assumes that the parameter name is the last part of the column name after the last underscore (see 'compute_OAT_SI.py': names are "{output}_{parameter}").
    Note: col.split('_')[-1] is for taking last part. Ok if parameter names do not contain "_"!!
    Note: '_'.join(col.split('_')[1:]) is for taking all but first part. Ok if output names do not contain "_"!!
    Note: 25.11.2025 use actual parameter names if both parameter and output names contain "_"!! Added a warning in sens_window.
    Note: currently implemented as removing output_name + "_" from the beginning of each column name.
    '''
    # Define new columns as old column name removing output name part + "_" just after the output name
    # E.g. "Output1_ParamA" becomes "ParamA"
    # Note 01.12.2025 : modified to remove only leading occurrence of output_name + "_" to avoid issues if parameter names contain output name as substring
    new_columns = {}
    prefix = f"{output_name}_"
    for col in df.columns:
        if col.startswith(prefix):
            # remove only the leading occurrence
            new_columns[col] = col[len(prefix):]
        else:
            # remove only the first occurrence anywhere else (if present)
            new_columns[col] = col.replace(prefix, "", 1)
    return df.rename(columns=new_columns)

#############################################################################################################
def sens_window(modelname: str, current_date: datetime, sens_type: str, start_window: datetime, end_window: datetime, 
                output_names_subset: list, collinearity: bool = False, 
                local_formulation: str = 'rr', local_delta: str = '0.01', directory = os.getcwd(), log: bool = False, **kwargs):
    '''Compute ranking and collinearity analysis over a certain window of time.
    Parameters:
    modelname (str): Name of the model.
    current_date (datetime): Current date for file naming.
    sens_type (str): Type of sensitivity analysis ('sobol', 'local', 'morris').
    start_window (datetime): Start of the time window.
    end_window (datetime): End of the time window.
    output_names_subset (list): List of output names to consider.
    collinearity (bool): Whether to perform collinearity analysis.
    local_formulation (str): Local formulation type (default is 'rr'). See 'compute_OAT_SI.py' for details. Used only if sens_type is 'local'.
    local_delta (str): Local delta value (default is '0.01'). See 'compute_OAT_SI.py' for details. 
                       It can also be '+0.01' or '-0.01' if sensitivity was not computed with the Central Difference method.
                       Used only if sens_type is 'local'.
    directory (str): Directory where files are located (default is current working directory).
    log (bool): Whether to enable logging.
    **kwargs: Additional keyword arguments.
        remove_too_low_magnitudes (bool): Whether to remove parameters with very low sensitivity magnitudes before collinearity (default is False).
        absolute_threshold (float): absolute treshold to remove unsensitivie parameters from ranking and before collinearity. (default is 0.01).
        relative_threshold (float): relative treshold to remove unsensitivie parameters from ranking and before collinearity. (default is 0.1).
    Returns:
    sorted_columns (DataFrame): Sorted sensitivity columns (sorted by MeanSqrt).
    stats_sorted (DataFrame): Sorted statistics (sorted by MeanSqrt).
    results_df (DataFrame, optional): Collinearity analysis results (if collinearity is True).

    Note: save to list and then to dataframe 'num_combinations' for each num_cols for logging purposes?
    Note: it is suggested to keep "local_formulation" as "rr" for best result interpretability.
    Note: what meaning has collinearity analysis and the absolute magnitude of sensitivity indices if sens_type is not 'local'? 
          Be careful when running "parameter_choice.py" and interpreting results!!
    '''
    current_date = current_date.strftime("%d-%m-%Y")
    date_string = f'{start_window.strftime("%d%m%H")}-{end_window.strftime("%d%m%H")}'

    # Extract inputs from kwargs and assign default values
    remove_too_low_magnitudes = kwargs.get('remove_too_low_magnitudes', False)
    at = kwargs.get('absolute_threshold', 0.01)
    rt = kwargs.get('relative_threshold', 0.1)
    
    # COMPUTE RANKING #######################################################################################
    # Filter within the current window of time
    # Note: there must be a timestamp column at the beginning of each sheet (i.e. each output)!
    if sens_type == 'sobol':
        file = os.path.join(directory, f'{modelname}_first_indices_{sens_type}_{current_date}.xlsx')
    elif sens_type == 'local':
        file = os.path.join(directory, f'{modelname}_Sensitivity_{sens_type}_{local_formulation}_{local_delta}.xlsx') # REMOVED DATE to ensure 'sens_window' can read also across 00:00
        if local_formulation != 'rr' and log:
            logging.warning(f'Local formulation is not "rr" but {local_formulation}...be careful in interpreting the ranking if output and parameter values have different orders of magnitude!')
    elif sens_type == 'morris':
        file = os.path.join(directory, f'{modelname}_mu_star_{sens_type}_{current_date}.xlsx')
    df_list = []
    output_string = ''.join(output_names_subset) # Put output_names all together
    # Read and filter each sheet, then combine them
    for output_name in output_names_subset:
        df = pd.read_excel(file,sheet_name = output_name)
        filtered_df = df[(df["Timestamp"] >= start_window) & (df["Timestamp"] <= end_window)]
        df_list.append(filtered_df)
    combined_df = pd.concat([rename_columns(df, output_name) for df, output_name in zip(df_list, output_names_subset)], ignore_index=True)
    if log:
        logging.warning('If both parameter and output names contain the "_" symbol, the current implementation may not work correctly!\n'
                        'Modify this function and "rename_columns" with the parameter names as input.')
    # Replace NaN values with 0 and count the number of substitutions
    nan_count_before = combined_df.isna().sum().sum()
    combined_df.fillna(0, inplace=True)
    nan_count_after = combined_df.isna().sum().sum()
    nan_substitutions = nan_count_before - nan_count_after
    # Log the number of NaN substitutions
    if log:
        logging.warning(f"Number of NaN values replaced with 0: {nan_substitutions}")
 
    # Compute mean sqrt for each column i.e. all parameters
    mean_sqrt = compute_mean_sqrt(combined_df)
    # Re-order sensitivity matrix columns according to mean_sqrt values
    sorted_columns = combined_df[mean_sqrt.sort_values(by='MeanSqrt', ascending=False).index]
    # Added on 10.09.2024 for debugging purposes -------------------------------
    if log:
        combined_df_output_file = os.path.join(directory, f'{modelname}_last_combined_df_{current_date}_{output_string}_{date_string}.xlsx')
        if os.path.exists(combined_df_output_file):
            logging.warning(f"The file {combined_df_output_file} already exists and will be overridden!")
        combined_df.to_excel(combined_df_output_file, index=False) # Output:combined_df for checking purposes
    
    if log:
        sorted_columns_output_file = os.path.join(directory, f'{modelname}_last_sorted_columns_{current_date}_{output_string}_{date_string}.xlsx')
        if os.path.exists(sorted_columns_output_file):
            logging.warning(f"The file {sorted_columns_output_file} already exists and will be overridden!")
        sorted_columns.to_excel(sorted_columns_output_file, index=False) # Output:sorted_columns for checking purposes
        logging.info("File Excel unified and saved correctly.")
    # ------------------------------------------------------------------------------
    # Compute all the lumping statistics for each column i.e. all parameters
    stats = compute_statistics(combined_df)
    # Extract index names from combined_df columns
    index = combined_df.columns.tolist()
    index = index[1:]
    stats.insert(0, 'Index', index)
    stats_sorted=stats.sort_values(by='MeanSqrt',ascending=False)
    output_file_sort = os.path.join(directory, f'{modelname}_StatsSorted_{sens_type}_{current_date}_{output_string}_{date_string}.xlsx')
    if log and os.path.exists(output_file_sort):
        logging.warning(f"The file {output_file_sort} already exists and will be overridden!")
    stats_sorted.to_excel(output_file_sort, index=False) # Output:ranking sorted by msqr
    #Filter out parameter with too low magnitude
    if remove_too_low_magnitudes:
        absolute_threshold = at
        relative_threshold = rt*stats_sorted['MeanSqrt'].max()
        stats_sorted_filtered = stats_sorted[~((stats_sorted['MeanSqrt'] < absolute_threshold) | (stats_sorted['MeanSqrt'] < relative_threshold))]
        indexes_to_eliminate = stats_sorted[(stats_sorted['MeanSqrt'] < absolute_threshold) | (stats_sorted['MeanSqrt'] < relative_threshold)].index
        if log:
            logging.info(f'"remove_too_low_magnitudes" is True: len(stats_sorted) is {len(stats_sorted)}, whereas len(stats_sorted_filtered) is {len(stats_sorted_filtered)}')
            logging.info(f'Eliminated parameters (too low magnitude) are {indexes_to_eliminate.tolist()}')
        if log:
            if os.path.exists(output_file_sort):
                logging.warning(f"The file {output_file_sort} already exists and will be overridden!")
            stats_sorted_filtered.to_excel(output_file_sort, index=False) # Output:ranking sorted by msqr

    # COLLINEARITY ANALYSIS #######################################################################################
    if collinearity==True:
        # Start timer for execution time measurement
        start_time = time.time()
        
        # Prepare output file for collinearity results
        outputfile_path = os.path.join(directory, f'{modelname}_CollIdx_{sens_type}_{current_date}_{output_string}_{date_string}.xlsx')
        if log and os.path.exists(outputfile_path):
            logging.warning(f"The file {outputfile_path} already exists and will be overridden!")

        with pd.ExcelWriter(outputfile_path, engine='xlsxwriter') as writer:
            if remove_too_low_magnitudes:
                if log:
                    logging.info(f'Collinearity analysis is performed only on parameters with sufficient sensitivity magnitude: {len(stats_sorted_filtered)} parameters considered.')
                sorted_columns = sorted_columns.iloc[:, 0:len(stats_sorted_filtered)] # Filter out columns for too low magnitude parameters
            else:
                if log:
                    logging.info(f'Collinearity analysis is performed on all the first 20 parameters ordered by MeanSqrt: min(20,{len(stats_sorted)}) parameters considered.')
                # Note: limit to first 20 parameters for computational reasons (as found by Giulia Quarta testing)
                sorted_columns = sorted_columns.iloc[:, 0:20] # [1:21] if Timestamp column present, but actually eliminated before by 'compute_statistics' function
                # Read the Excel file into a DataFrame
                # df = pd.read_excel(file_path, sheet_name=key, usecols=lambda x: x != 0)
                # df = df.iloc[:, 1:21]
            
            # Normalization of the sensitivity matrix
            norms = np.linalg.norm(sorted_columns, axis=0)
            sorted_columns_norm = sorted_columns / norms
            # Initialize initial_matrix
            initial_matrix = sorted_columns_norm.values
            # Get the column names
            column_names = sorted_columns_norm.columns.tolist()
            # Define an empty list to store the results
            results = []
            # Get the number of rows and columns of the initial matrix
            rows, cols = initial_matrix.shape
            # Iterate over the number of columns in the combinations
            for num_cols in range(2, cols + 1):
                # Compute the number of combinations for the current number of columns
                num_combinations = len(list(combinations(range(cols), num_cols)))
                if log:
                    logging.info(f'Computing collinearity for combinations of {num_cols} columns ({num_combinations} combinations)...')
                # Generate combinations of column indices
                for cols_comb in combinations(range(cols), num_cols):
                    # Extract the submatrix based on the selected columns
                    submatrix = initial_matrix[:, list(cols_comb)]
                    # Calculate the product of the transpose of the submatrix with itself
                    product = np.dot(submatrix.T, submatrix)
                    # Calculate the eigenvalues of the product
                    eigenvalues = np.linalg.eigvals(product)
                    # Select the minimum eigenvalue
                    min_eigenvalue = min(eigenvalues)
                    # Compute the collinearity index
                    collinearity_index = 1 / (min_eigenvalue ** 0.5)
                    # Get the column names for the combination
                    comb_column_names = [column_names[idx] for idx in cols_comb]
                    # Store the combination and its minimum eigenvalue in the results list
                    results.append({
                        'Columns': comb_column_names,
                        'Subset dimension': num_cols,
                        'Minimum Eigenvalue': min_eigenvalue,
                        'Collinearity Index': collinearity_index
                    })
            # Convert the results list to a DataFrame
            results_df = pd.DataFrame(results)
            # Write the results DataFrame to the ExcelWriter object on a separate sheet
            results_df.to_excel(writer, sheet_name=output_string, index=False)
        
        # Record end time for execution time measurement
        end_time = time.time()
        # Calculate the total duration of execution
        duration = end_time - start_time
        logging.info(f"Duration of collinearity analysis was: {duration} seconds")
        #####################################################################
        # Add here part related to choice i.e. computations for all rows of 'results_df'.
        return sorted_columns, stats_sorted, results_df
    return sorted_columns, stats_sorted