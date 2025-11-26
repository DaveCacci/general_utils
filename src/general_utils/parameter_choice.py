import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import os
import time

def parameter_choice(modelname: str, current_date: datetime, sens_type: str, start_window: datetime, end_window: datetime, 
                     output_names_subset: list, collinearity_range: list = [0,20], 
                     distance_from_first: float = 0.3, criteria: str = 'maxsum', directory: str = os.getcwd(), log: bool = False):
    '''
    Select a subset of parameters based on collinearity and sensitivity criteria.
    Reads collinearity index and sensitivity statistics from Excel files, filters the data based on the provided collinearity range,
    computes additional statistics, and selects parameters according to the specified criteria.
    Parameters:
    > modelname (str): Name of the model.
    > current_date (datetime): Current date for file naming.
    > sens_type (str): Type of sensitivity analysis ('local', 'sobol', etc.).
    > start_window (datetime): Start of the time window for analysis. Used here only for file naming.
    > end_window (datetime): End of the time window for analysis. Used here only for file naming.
    > output_names_subset (list): List of output names to consider.
    > collinearity_range (list): Range [LB, UB] to filter the rows of the MinEig file (to avoid too long computations).
    > distance_from_first (float): Scalar in (0,1) that cuts again the parameter subsets to select only the subsets 
      that have parameters characterized by similar magnitude metric i.e. the distance from the highest one is 
      at most (distance_from_first*100)% of the highest 'MaxMeanSqrt' inside that subset 
      i.e. distance from the most sensitive parameter and the less sensitive parameter is constrained.
    > criteria (str): Criteria for parameter selection. Either 'maxsum' or 'maxmean'.
    > directory (str): Directory where the input files are located.
    > log (bool): Whether to enable logging.
    Returns:
    > elements (list): List of selected parameter names.
    > data_lastcut (DataFrame): DataFrame with statistics of the selected subsets.

    Note: what meaning has collinearity analysis and the absolute magnitude of sensitivity indices if sens_type is not 'local'? 
          Be careful when setting "collinearity_range" and interpreting results!!
    '''
    current_date = current_date.strftime("%d-%m-%Y")
    date_string = f'{start_window.strftime("%d%m%H")}-{end_window.strftime("%d%m%H")}'
    # Read collinearity index ------------------------------------------------
    output_string = ''.join(output_names_subset) # Put output_names all together
    file = os.path.join(directory, f'{modelname}_CollIdx_{sens_type}_{current_date}_{output_string}_{date_string}.xlsx')
    data_all = pd.read_excel(file)

    # Filter data_all based on collinearity_range ----------------------------------------------
    while True:
        data = data_all[(data_all['Collinearity Index'] >= collinearity_range[0]) & (data_all['Collinearity Index'] <= collinearity_range[1])]
        if 1 <= len(data) <= 100000:
            break
        if len(data) > 100000:
            collinearity_range[0] += 1
            collinearity_range[1] -= 1
        elif len(data) < 1:
            collinearity_range[0] -= 1
            collinearity_range[1] += 1
    if log:
        logging.info(f'len(data) between {collinearity_range[0]} and {collinearity_range[1]} is {len(data)}')
        logging.info(f'Saving pre-cut subset statistics to Excel...')
        data.to_excel(os.path.join(directory, f'{modelname}_CollIdx_FirstCut_{sens_type}_{current_date}_{output_string}_{date_string}.xlsx'), index=False)
        
    # Compute mean_sqrt e mean_mean_sqrt ----------------------------------------------
    param_list = data['Columns']
    length = len(param_list)
    mean_sqrt_list = []
    mean_mean_sqrt_list = []

    sens_data = os.path.join(directory, f'{modelname}_StatsSorted_{sens_type}_{current_date}_{output_string}_{date_string}.xlsx')
    sens_data = pd.read_excel(sens_data)
    new_columns = sens_data.iloc[:, 0].values
    df_dropped = sens_data.drop(columns=sens_data.columns[0])
    df_transposed = df_dropped.T
    df_transposed.columns = new_columns

    mean_sqrt = np.zeros(length)
    var_sqrt = np.zeros(length)
    max_sqrt = np.zeros(length)
    min_sqrt = np.zeros(length)
    elements_list = []

    for i in range(length):
        input_string = param_list.iloc[i]
        clean_string = input_string.strip("[]").replace("'", "").replace(" ", "")
        elements = clean_string.split(",")
        elements_list.append(elements)

    # This takes a lot of time!! How to optimize it??
    # Start timer for execution time measurement
    start_time = time.time()
    logging.info(f"Starting computation of lumped metrics for each subset of parameters...")
    for i in range(length):
        mean_sqrt[i] = df_transposed.loc['MeanSqrt', elements_list[i]].sum()
        var_sqrt[i] = df_transposed.loc['MeanSqrt', elements_list[i]].var() 
        max_sqrt[i] = df_transposed.loc['MeanSqrt', elements_list[i]].max()
        min_sqrt[i] = df_transposed.loc['MeanSqrt', elements_list[i]].min()
    # Record end time for execution time measurement
    end_time = time.time()
    # Calculate the total duration of execution
    duration = end_time - start_time
    logging.info(f"Duration of lumped metrics computation for each subset was: {duration} seconds")

    data['SumMeanSqrt'] = mean_sqrt
    data['LenSubset'] = [len(element) for element in elements_list]
    data['MeanMeanSqrt'] = data['SumMeanSqrt']/data['LenSubset']
    data['VarMeanSqrt'] = var_sqrt
    data['MaxMeanSqrt'] = max_sqrt
    data['MinMeanSqrt'] = min_sqrt
    data['RangeMeanSqrt'] = data['MaxMeanSqrt'] - data['MinMeanSqrt']
    
    # Save computed statistics to Excel
    data.to_excel(os.path.join(directory, f'{modelname}_CollIdx_FirstCut_Stats_{sens_type}_{current_date}_{output_string}_{date_string}.xlsx'), index=False)
    
    # Cut again based on distance_from_first ----------------------------------------------
    while True:
        data_lastcut = data[data['RangeMeanSqrt'] <= distance_from_first*data['MaxMeanSqrt']]
        if len(data_lastcut) > 1:
            break
        else:
            if distance_from_first < 0.3:
                distance_from_first += 0.05
            else:
                break
    if log:
        logging.info(f'Data (len = {len(data_lastcut)}) are cut with an UB to the variance with respect to the most sensitive parameter in each subset of {distance_from_first}')     
        logging.info(f'Saving post-cut subset statistics to Excel...')
        data_lastcut.to_excel(os.path.join(directory, f'{modelname}_CollIdx_LastCut_Stats_{sens_type}_{current_date}_{output_string}_{date_string}.xlsx'), index=False)

    # Actually select the parameters based on criteria (Parameter Subset Selection (PSS)) ----------------------------------------------
    if len(data_lastcut) == 0: # What to do if len(data_lastcut) == 0?? In this case, select the most sensitive parameter overall
        elements = [sens_data['Index'][0]]
    else:
        if criteria == 'maxsum':
            input_string = data_lastcut.loc[data_lastcut['SumMeanSqrt'].idxmax()]['Columns']
        elif criteria == 'maxmean':
            input_string = data_lastcut.loc[data_lastcut['MeanMeanSqrt'].idxmax()]['Columns']
        else:
            raise ValueError("Criteria must be 'maxsum' or 'maxmean'")
        clean_string = input_string.strip("[]").replace("'", "").replace(" ", "")
        elements = clean_string.split(",")
    
    return elements, data_lastcut

def parameter_choice_ranking(modelname: str, stats_sorted: pd.DataFrame, distance_from_first: float, directory: str = os.getcwd(), log: bool = False):
    '''
    Select a subset of parameters based on their sensitivity ranking.
    Parameters:
    > stats_sorted (DataFrame): DataFrame containing sensitivity statistics sorted by sensitivity. Ouput of "sens_window.py" function!!
    > distance_from_first (float): Scalar in (0,1) that cuts again the parameter subsets to select only the subsets 
      that have parameters characterized by similar magnitude metric i.e. the distance from the highest one is 
      at most (distance_from_first*100)% of the highest 'MeanSqrt' inside that subset 
      i.e. distance from the most sensitive parameter and the less sensitive parameter is constrained.
    > directory (str): Directory where the output files will be located.
    > log (bool): Whether to enable logging.
    Returns:
    > elements (list): List of selected parameter names.
    '''
    # If I want to select only the FIRST MOST SENSITIVE, select the first element of the output of this function
    
    stats_sorted['DistanceMeanSqrt'] = stats_sorted['MeanSqrt'][0] - stats_sorted['MeanSqrt']
    while True:
        stats_cut = stats_sorted[stats_sorted['DistanceMeanSqrt'] <= distance_from_first*stats_sorted['MeanSqrt'][0]]
        if len(stats_cut) > 1:
            break
        else:
            if distance_from_first < 0.3:
                distance_from_first += 0.05
            else:
                break
    if log:
        logging.info(f'Sorted stats (len = {len(stats_cut)}) are cut with an UB to the variance with respect to the most sensitive parameter of {distance_from_first}')  
        logging.info(f'Saving post-cut ranking statistics to Excel...')
        stats_cut.to_excel(os.path.join(directory, f'{modelname}_StatsSorted_Cut.xlsx'))
    
    elements = list(stats_cut['Index'].values)
    return elements