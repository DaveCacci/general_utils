import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def compute_error(df_output, df_meas, var_couple, start_timestamp_error, end_timestamp_error, 
                  weighting_type: str = 'none', weighting_param: list = []) -> pd.DataFrame:
    """
    Compute detailed error metrics between model predictions and measured data for a specific output variable.
    
    This function processes model predictions and measurement data to calculate various error metrics
    with optional temporal weighting. It performs data alignment, filtering, and computes both 
    relative and normalized errors suitable for parameter estimation and model validation.
    
    Parameters
    ----------
    df_output : pd.DataFrame
        DataFrame containing model predictions (outputs) with a 'Timestamp' column.
        Must include the model output variable specified in var_couple[0].
    df_meas : pd.DataFrame
        DataFrame containing measured data with a 'Timestamp' column.
        Must include the measured variable specified in var_couple[1].
    var_couple : list
        List defining variable mapping. Structure:
        - [0]: str, model output variable name (column in df_output)
        - [1]: str, corresponding measured variable name (column in df_meas)
        Additional elements (weights) may be present but are not used in this function.
    start_timestamp_error : timestamp-like
        Start timestamp for error calculation window. Data before this time is excluded.
    end_timestamp_error : timestamp-like
        End timestamp for error calculation window. Data after this time is excluded.
    weighting_type : str
        Temporal weighting scheme for measurements over time. Options:
        - 'log': Sigmoid (logistic) weighting - emphasizes recent measurements gradually
        - 'exp': Exponential decay weighting - emphasizes recent measurements exponentially
        - 'none': Uniform weighting - all measurements weighted equally
        Default is 'none'.
    weighting_param : list
        Parameters for the selected weighting_type:
        - For 'log': [steepness, quantile_of_center]
          * steepness (float): Controls the sharpness of the sigmoid transition (higher = sharper)
          * quantile_of_center (float): Quantile (0-1) where sigmoid reaches midpoint (e.g., 0.5)
        - For 'exp': [decay_rate_in_days]
          * decay_rate_in_days (float): Time constant for exponential decay (higher = faster forgetting)
        - For 'none': [] (empty list)
        Default is an empty list.
    
    Returns
    -------
    merged_df : pd.DataFrame
        DataFrame containing aligned timestamps and comprehensive error metrics:
        
        Columns include:
        - 'Timestamp': Timestamps where both model and measurement data exist
        - var_couple[0]: Model output values
        - var_couple[1]: Measured values
        - 'time_diff': Time difference from most recent timestamp (seconds) [if weighted]
        - 'normalized_time_diff': Normalized time difference [0-1] [if log weighting]
        - 'weights': Temporal weights applied to each measurement
        - 'err': Relative error normalized by mean of measurements, weighted
          Formula: (prediction - measurement) / mean(measurements) * weight
        - 'err_sqr': Squared relative error (used for least squares optimization)
        - 'err_norm': Relative error normalized by individual measurements, weighted
          Formula: (prediction - measurement) / measurement * weight
        - 'err_sqr_norm': Squared normalized relative error
    
    Notes
    -----
    - Only timestamps present in both df_output and df_meas are retained (inner join).
    - Rows with NaN values in either the model output or measurement are dropped.
    - Division by zero is avoided: when measurement is zero, error is set to zero.
    - Temporal weighting is applied after error calculation, affecting both 'err' and 'err_norm'.
    - The 'err' metric normalizes by the mean of all measurements (suitable for aggregation).
    - The 'err_norm' metric normalizes by individual measurements (percentage error per point).
    - For 'log' weighting: weight increases from 0 to 1 as time approaches present.
    - For 'exp' weighting: weight = exp(-decay_rate * time_from_present).
    
    Examples
    --------
    >>> var_couple = ['CH4_model', 'CH4_measured']
    >>> merged = compute_error(model_df, meas_df, var_couple, 
    ...                        start='2025-01-01', end='2025-01-31',
    ...                        weighting_type='exp', weighting_param=[0.3])
    >>> total_error = merged['err_sqr'].sum()
    
    See Also
    --------
    cumulative_error : Aggregates errors across multiple output variables
    """ 
    # --- Step 1: Filter data within the specified time window ---
    df_output_interval = df_output[(df_output['Timestamp'] >= start_timestamp_error) & 
                                    (df_output['Timestamp'] <= end_timestamp_error)]
    df_meas_interval = df_meas[(df_meas['Timestamp'] >= start_timestamp_error) & 
                               (df_meas['Timestamp'] <= end_timestamp_error)]
    
    # --- Step 2: Align model and measurement data by timestamp ---
    # Inner join: keep only timestamps where both model output and measurement exist
    merged_df = pd.merge(df_output_interval, df_meas_interval, on='Timestamp', how='inner')
    
    # Keep only relevant columns: Timestamp and the two variables of interest
    merged_df.drop(merged_df.columns.difference(['Timestamp', var_couple[0], var_couple[1]]), 
                   axis=1, inplace=True)
    
    # Remove any rows with missing values and reset the index
    merged_df.dropna(inplace=True)
    merged_df.reset_index(drop=True, inplace=True)
    
    # --- Step 3: Calculate temporal weights based on weighting scheme ---
    # Weights determine the relative importance of older vs. newer measurements
    
    if weighting_type == 'log' and len(weighting_param) == 2:
        # Logistic (sigmoid) weighting: smooth transition from old to recent data
        # Recent measurements get higher weights (approaching 1)
        
        # Calculate time difference from most recent timestamp (in seconds)
        merged_df['time_diff'] = (merged_df['Timestamp'].max() - merged_df['Timestamp']).dt.total_seconds()
        
        # Normalize time differences to [0, 1] range (0 = oldest, 1 = most recent)
        merged_df['normalized_time_diff'] = (merged_df['time_diff'].max() - merged_df['time_diff']) / \
                                            (merged_df['time_diff'].max() - merged_df['time_diff'].min())
        
        # Apply sigmoid function: weight = 1 / (1 + exp(-k * (t - t_0)))
        k = weighting_param[0]  # Steepness: higher values create sharper transition
        t_0 = merged_df['normalized_time_diff'].quantile(weighting_param[1])  # Center point of sigmoid
        merged_df['weights'] = 1 / (1 + np.exp(-k * (merged_df['normalized_time_diff'] - t_0)))
        
    elif weighting_type == 'exp' and len(weighting_param) == 1:
        # Exponential decay weighting: exponentially decreasing importance for older data
        # weight = exp(-decay_rate * time_from_present)
        
        # Calculate time difference from most recent timestamp (in seconds)
        merged_df['time_diff'] = (merged_df['Timestamp'].max() - merged_df['Timestamp']).dt.total_seconds()
        
        # Convert decay rate from days to seconds and apply exponential decay
        decay_rate = weighting_param[0] / (24 * 3600)  # Convert days to seconds
        merged_df['weights'] = np.exp(-decay_rate * merged_df['time_diff'])
        
    else:
        # Uniform weighting: all measurements equally important
        merged_df['weights'] = np.ones(len(merged_df[var_couple[1]]))

    # --- Step 4: Compute errors (scaled by mean of measurements) ---
    # This error metric is suitable for aggregation across multiple time points
    # Formula: err = (prediction - measurement) / mean(all_measurements)
    # Division by zero protection: set error to 0 when measurement is 0
    merged_df['err'] = np.where(merged_df[var_couple[1]] != 0, 
                                 (merged_df[var_couple[0]] - merged_df[var_couple[1]]) / merged_df[var_couple[1]].mean(), 
                                 0)
    
    # Apply temporal weights to the relative errors
    merged_df['err'] = merged_df['err'] * merged_df['weights']
    
    # Square the weighted errors (for least squares optimization)
    merged_df['err_sqr'] = merged_df['err'] ** 2
    
    # --- Step 5: Compute normalized relative errors (percentage error per measurement) ---
    # This error metric shows the percentage deviation at each time point
    # Formula: err_norm = (prediction - measurement) / measurement
    merged_df['err_norm'] = np.where(merged_df[var_couple[1]] != 0, 
                                      (merged_df[var_couple[0]] - merged_df[var_couple[1]]) / merged_df[var_couple[1]], 
                                      0)
    
    # Apply temporal weights to the normalized errors
    merged_df['err_norm'] = merged_df['err_norm'] * merged_df['weights']
    
    # Square the weighted normalized errors
    merged_df['err_sqr_norm'] = merged_df['err_norm'] ** 2
    
    return merged_df