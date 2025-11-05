from compute_error import compute_error
import pandas as pd
import numpy as np
from typing import Callable, Dict, List, Any, Union
import logging

def cumulative_error(y_df: pd.DataFrame, 
                     y_df_data_on: pd.DataFrame, 
                     y_df_data_off:pd.DataFrame, 
                     var_couples_list: List, 
                     start_timestamp_error, end_timestamp_error, 
                     weighting_type: str = 'none', weighting_param: list = []) -> Dict:
    """
    Compute cumulative weighted errors between model predictions and measured data (both online and offline).
    Errors are evaluated over the common Timestamps between model and measurements.

    This function calculates various error metrics (weighted least squares, normalized errors,
    R², and MARE) for multiple output variables by comparing model predictions against 
    measured data. It supports different temporal weighting schemes and handles both 
    online and offline measurement data.
    
    Parameters
    ----------
    y_df : pd.DataFrame
        DataFrame containing model predictions (outputs).
    y_df_data_on : pd.DataFrame
        DataFrame containing online measured data with timestamps.
    y_df_data_off : pd.DataFrame
        DataFrame containing offline measured data with timestamps.
    var_couples_list : List
        List of tuples defining variable mappings and weights. Each tuple contains:
        - [0]: str, model output variable name
        - [1]: str, corresponding measured variable name
        - [2]: float, weight factor (w_y) for non-normalized (only scaled) error
        - [3]: float, weight factor (w_y_norm) for normalized error
    start_timestamp_error : timestamp-like
        Start timestamp for error calculation window.
    end_timestamp_error : timestamp-like
        End timestamp for error calculation window.
    weighting_type : str
        Temporal weighting scheme for measurements over time:
        - 'log': sigmoid (logistic) weighting
        - 'exp': exponential decay weighting
        - 'none': uniform weighting
        Passed to compute_error function.
    weighting_param : list
        Parameters for the selected weighting_type:
        - For 'log': [steepness, quantile_of_center]
        - For 'exp': [decay_rate_in_days]
        - For 'none': []
        Passed to compute_error function.
    
    Returns
    -------
    tuple of (Dict, Dict, Dict, Dict, Dict)
        f_obj : Dict[str, float]
            Least squares error for each output variable (scaled with the mean of measurements just for scaling).
            Scaled Mean Squared Error (scaled-MSE). Scaled to make different outputs comparable.
            Each output variable's scaled-MSE is weighted according to specified weights.
        f_obj_norm : Dict[str, float]
            Weighted least squares (WLS) error for each output variable (normalized by measurement for each t time index).
            Relative Mean Squared Error (RMSE). Each output variable's RMSE is weighted according to specified weights.
            This is actually a true implementation of WLS.
        merged_df_dict : Dict[str, pd.DataFrame]
            Dictionary of DataFrames containing detailed error metrics for each output.
            Output of compute_error function for each variable.
        R2 : Dict[str, float]
            Coefficient of determination (R²) for each output variable, rounded to 4 decimals.
            Measures the correlation between predictions and measurements.
        MAREpc : Dict[str, float]
            Mean Absolute Relative Error in percentage for each output, rounded to 4 decimals.
    
    Notes
    -----
    - The function automatically selects between y_df_data_on and y_df_data_off based on
      which DataFrame contains the measured variable.
    - This function cumulates and weights errors over all specified output variables.
    - Weighted errors are computed using diagonal weight matrices.
    - Empty measurement sets result in zero error values to avoid division by zero.
    - All least squares errors are normalized by the number of measurements for comparability.
    - All least squares errors are scaled by some factor to make different outputs comparable.
    
    See Also
    --------
    compute_error : Computes individual errors between model and measured data
    """ 
    
    # Extract variable information from var_couples_list
    output_names = [var_couples_list[i][0] for i in range(len(var_couples_list))]  # Model output variable names
    w_y = [var_couples_list[i][2] for i in range(len(var_couples_list))]  # Weights for scaled errors
    w_y_norm = [var_couples_list[i][3] for i in range(len(var_couples_list))]  # Weights for normalized errors
    
    # Compute errors for each output variable by comparing model predictions with measurements
    merged_df_dict = {}
    for i in range(len(output_names)):
        # Select appropriate measurement dataset (online or offline) based on variable availability
        if var_couples_list[i][1] in y_df_data_on.columns:
            merged_df_dict[output_names[i]]  = compute_error(y_df, y_df_data_on, var_couples_list[i], 
                                                             start_timestamp_error, end_timestamp_error, 
                                                             weighting_type, weighting_param)
        else:
            merged_df_dict[output_names[i]]  = compute_error(y_df, y_df_data_off, var_couples_list[i], 
                                                             start_timestamp_error, end_timestamp_error, 
                                                             weighting_type, weighting_param)

    # Initialize dictionaries for output metrics
    f_obj = {}        # weighted and scaled MSE
    f_obj_norm = {}   # weighted RMSE (for WLS)
    R2 = {}           # Coefficient of determination (R²)
    MAREpc = {}       # Mean absolute relative error in percentage terms (MARE%)
    
    # Compute weighted errors and performance metrics for each output variable
    for i, output_name in enumerate(output_names):
        output_name = output_names[i]
        
        # --- Weight and cumulate scaled squared errors. Compute MSE ---
        # Create diagonal weight matrix for this output
        w_array = w_y[i] * np.ones(len(merged_df_dict[output_name]['err_sqr']))
        w_diag = np.diag(w_array)
        
        # Apply weights to squared errors
        weighted_array = np.dot(w_diag, merged_df_dict[output_name]['err_sqr'])
        weighted_array_df = pd.DataFrame(weighted_array, columns=[f'{output_name}-WLS'])
        
        # Add weighted errors to the merged dataframe
        df_concatenated = pd.concat([merged_df_dict[output_name], weighted_array_df], axis=1)
        merged_df_dict[output_name] = df_concatenated
        
        # Compute total weighted error, normalized by number of measurements
        f_obj[output_name] = 0 if merged_df_dict[output_name]['err_sqr'].empty else \
                             df_concatenated[f'{output_name}-WLS'].sum() / len(merged_df_dict[output_name]['err_sqr'])

        # --- Weight and cumulate normalized squared errors. Compute RMSE ---
        # Create diagonal weight matrix for normalized errors
        w_array = w_y_norm[i] * np.ones(len(merged_df_dict[output_name]['err_sqr_norm']))
        w_diag = np.diag(w_array)
        
        # Apply weights to normalized squared errors
        weighted_array = np.dot(w_diag, merged_df_dict[output_name]['err_sqr_norm'])
        weighted_array_df = pd.DataFrame(weighted_array, columns=[f'{output_name}_norm-WLS'])
        
        # Add normalized weighted errors to the merged dataframe
        df_concatenated = pd.concat([merged_df_dict[output_name], weighted_array_df], axis=1)
        merged_df_dict[output_name] = df_concatenated
        
        # Compute total normalized weighted error, normalized by number of measurements
        f_obj_norm[output_name] = 0 if merged_df_dict[output_name]['err_sqr'].empty else \
                                   df_concatenated[f'{output_name}_norm-WLS'].sum() / len(merged_df_dict[output_name]['err_sqr'])
        
        # --- Compute Coefficient of determination (R²) ---
        # Measure correlation between predictions and measurements
        corr_matrix = np.corrcoef(merged_df_dict[output_name][var_couples_list[i][1]], 
                                  merged_df_dict[output_name][output_name])
        corr = corr_matrix[0, 1]
        R_sq = corr ** 2
        R2[output_name] = round(R_sq, 4)

        # --- Cumulate absolute normalized errors. Compute Mean Absolute Relative Error in percentage terms (MARE%) ---
        # Average of absolute normalized errors, expressed as percentage
        marepc = 0 if merged_df_dict[output_name]['err_norm'].empty else \
                 merged_df_dict[output_name]['err_norm'].abs().sum() / len(merged_df_dict[output_name]['err_norm']) * 100
        MAREpc[output_name] = round(marepc, 4)
        
    return f_obj,f_obj_norm,merged_df_dict, R2, MAREpc