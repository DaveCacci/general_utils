"""
Linear Uncertainty Propagation Module

This module implements first-order uncertainty propagation (linear approximation)
to compute confidence intervals for model outputs based on parameter uncertainties.
Uses the delta method from statistical theory.

Mathematical Background:
    For output y = f(θ₁, θ₂, ..., θₙ), the variance is approximated by:
    
    Diagonal approximation (ignores correlations):
        σ²_y ≈ Σᵢ (∂y/∂θᵢ)² × σ²_θᵢ
    
    Full covariance propagation (accounts for correlations):
        Var(y) = S × COV × S^T
        where S is the sensitivity matrix [n_timepoints × n_parameters]
        and COV is the full parameter covariance matrix [n_parameters × n_parameters]
    
    This assumes:
    - Small parameter uncertainties (linear approximation is valid)
    - Normal distribution of parameters
    - For diagonal method: parameters are uncorrelated

Typical workflow:
    1. Run FIM analysis to obtain parameter covariance matrix and standard deviations
    2. Compute absolute-absolute ('aa') sensitivities matching FIM grad_type
    3. Call lin_unc_prop() with either:
       - stdev_param_dict only (diagonal approximation, faster but ignores correlations)
       - cov_matrix for full propagation (recommended when parameters are correlated)
    
Important:
    The covariance matrix should be cov = σ² × FIM^(-1) (2nd return value from fim.py function output!)
    NOT the raw FIM or pre_cov. This already includes error variance scaling.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
import logging
from .sens_window import rename_columns

def lin_unc_prop(output_expected_dict: dict, stdev_param_dict: dict, grad: dict, 
                  confidence_level: float = 0.95, cov_matrix: np.ndarray = None, log: bool = False) -> dict:
    """
    Compute confidence intervals for model outputs using linear uncertainty propagation.
    
    This function applies first-order Taylor series expansion (delta method) to propagate
    parameter uncertainties to output uncertainties. Supports both diagonal approximation
    (faster, ignores correlations) and full covariance propagation (more accurate).
    
    Mathematical Formulas:
        Diagonal approximation (when cov_matrix=None):
            σ²_y(t) = Σᵢ (∂y/∂θᵢ)² × σ²_θᵢ
        
        Full covariance propagation (when cov_matrix is provided):
            Var(y) = S × COV × S^T
            where S[t,i] = ∂y(t)/∂θᵢ is the sensitivity matrix
        
        Confidence Interval:
            y ± z_α/2 × σ_y
            where z_α/2 = norm.ppf(1 - (1-α)/2) for confidence level α
    
    Args:
        output_expected_dict: Dictionary of DataFrames containing expected (nominal) output values.
                             Keys are output variable names. Each DataFrame must contain a column
                             with the same name as the key containing the output time series.
                             Nominally structure from the outputs of "Sensitivity_local_OAT.ipynb" (nominal simulation) 
                             ("_beforemod.xlsx" file i.e. unfiltered with respect to the available offline measurement points).
        stdev_param_dict: Dictionary mapping parameter names to their standard deviations (scalars).
                         Obtained from sqrt(diag(cov)) where cov is from fim.py function output.
                         Used only when cov_matrix=None (diagonal approximation).
        grad: Dictionary of sensitivity matrices (MUST be Absolute-Absolute 'aa' type).
              Keys are output names. Values are DataFrames with:
              - First column: timestamp (will be excluded)
              - Remaining columns: sensitivities ∂y/∂θᵢ for each parameter
              Column names (except first) must match parameter names in stdev_param_dict.
              CRITICAL: Use same grad_type as in FIM() computation (fim.py function)!
              Nominal structure from "Sensitivity_local_OAT.ipynb" sensitivity results 
              ("_beforemod.xlsx" file i.e. unfiltered with respect to the available offline measurement points).
        confidence_level: Confidence level for intervals (default: 0.95 for 95% CI).
                         Common values: 0.90 (90%), 0.95 (95%), 0.99 (99%).
        cov_matrix: Optional full parameter covariance matrix [n_parameters × n_parameters].
                   If provided, uses full covariance propagation accounting for correlations.
                   MUST be the 'cov' output from FIM(): cov = σ² × FIM^(-1)
                   NOT the raw FIM or pre_cov matrix!
                   Nominally, the second output of the FIM() function in fim.py.
                   If None, uses diagonal approximation (faster but ignores correlations).
        log: If True, enables logging for debugging purposes.
    
    Returns:
        dict: Dictionary with output names as keys and tuples (y_max, y_min) as values:
              - y_max: Upper bound of confidence interval (numpy array, shape: [n_timepoints])
              - y_min: Lower bound of confidence interval (numpy array, shape: [n_timepoints])
    
    Raises:
        ValueError: If output_expected_dict and grad keys don't match
        ValueError: If parameter names in stdev_param_dict not found in grad columns
    
    Notes:
        - Diagonal approximation (cov_matrix=None) is faster but assumes uncorrelated parameters
        - Full propagation (cov_matrix provided) is recommended when parameter correlation exists
        - Check correlation from FIM output: if off-diagonal terms in cov are large, use full method
        - Linear approximation assumes small parameter uncertainties; may be inaccurate otherwise
        - Assumes normal distribution of parameters for confidence interval calculation
    
    Example (diagonal approximation):
        >>> fim_mat, cov, cov_export, stdev_param = FIM(grad, merged_df, ...)
        >>> stdev_dict = {'k1': stdev_param[0], 'k2': stdev_param[1]}
        >>> bounds = lin_unc_prop(outputs, stdev_dict, sensitivities)
        >>> upper_bound, lower_bound = bounds['concentration']
    
    Example (full covariance propagation):
        >>> fim_mat, cov, cov_export, stdev_param = FIM(grad, merged_df, ...)
        >>> bounds = lin_unc_prop(outputs, {}, sensitivities, cov_matrix=cov)
        >>> upper_bound, lower_bound = bounds['concentration']
    """
    ###############################################################
    # Compute output uncertainties using first-order propagation
    ###############################################################
    # Check that output_expected_dict and grad have matching keys and lengths
    if set(output_expected_dict.keys()) != set(grad.keys()):
        raise ValueError("Keys of output_expected_dict and grad must match.")
    if len(output_expected_dict) != len(grad):
        raise ValueError("output_expected_dict and grad must have the same number of outputs.")
    # Validate that all parameters in dict exist in gradient DataFrames
    for output_name in grad.keys():
        # Parameter columns exclude the first column (assumed timestamp/index)
        param_keys = list(stdev_param_dict.keys())
        matching_cols = [col for col in grad[output_name].columns if any(param_key in col for param_key in param_keys)]
        if len(matching_cols) != len(stdev_param_dict):
            raise ValueError(f'For output "{output_name}", length mismatch: found {len(matching_cols)} matching columns but stdev_param_dict has {len(stdev_param_dict)} entries.')
    # Calculate z-score for given confidence level
    # E.g., confidence_level=0.95 → z_score≈1.96 (covers central 95% of normal distribution)
    z_score = norm.ppf(1 - (1 - confidence_level) / 2)
    if log:
        logging.info(f"Using z-score: {z_score} for confidence level: {confidence_level}")
    
    stdev_output_dict = {}
    for output_name in output_expected_dict.keys():
        if cov_matrix is not None:
            ###############################################################
            # METHOD 1: Full covariance propagation (accounts for correlations)
            ###############################################################
            # Full covariance propagation: Var(y) = S × COV × S^T
            # where S is [n_timepoints × n_parameters] sensitivity matrix
            # and COV is [n_parameters × n_parameters] parameter covariance
            S = grad[output_name].iloc[:,1:].values  # Sensitivity matrix (exclude timestamp column)
            
            # Compute variance for each time point: diag(S @ COV @ S^T)
            # This accounts for parameter correlations through off-diagonal COV terms
            var_output = np.diag(S @ cov_matrix @ S.T)
            
            # Compute output standard deviation: σ_y(t) = sqrt(Var(y(t)))
            stdev_out = np.sqrt(var_output)
        else:
            ###############################################################
            # METHOD 2: Diagonal approximation (faster, ignores correlations)
            ###############################################################
            # Initialize variance accumulator for this output
            stdev_output = 0
            
            # Sum variance contributions from all parameters
            # Formula: σ²_y = Σᵢ (∂y/∂θᵢ)² × σ²_θᵢ
            # This is equivalent to assuming COV is diagonal
            for param_name in stdev_param_dict:
                # Extract sensitivity gradient for this parameter: ∂y/∂θᵢ
                sensitivity = grad[output_name].filter(like=param_name).values.squeeze()
                
                # Add variance contribution: (∂y/∂θᵢ)² × σ²_θᵢ
                stdev_output = sensitivity**2 * stdev_param_dict[param_name]**2 + stdev_output
            
            # Compute output standard deviation: σ_y = sqrt(σ²_y)
            stdev_out = np.sqrt(stdev_output)
        
        ###############################################################
        # Compute confidence interval bounds
        ###############################################################        
        
        # Extract expected (nominal) output values at each time point
        y = output_expected_dict[output_name].filter(like=output_name).values.squeeze()
        
        # Compute upper confidence bound: y(t) + z_α/2 × σ_y(t)
        # This represents the (1 - α/2) percentile of the normal distribution
        y_max = y + z_score * stdev_out
        
        # Compute lower confidence bound: y(t) - z_α/2 × σ_y(t)
        # This represents the α/2 percentile of the normal distribution
        y_min = y - z_score * stdev_out
        
        # Store bounds as tuple (upper, lower) for this output
        stdev_output_dict[output_name] = (y_max, y_min)
    
    # Return dictionary of confidence bounds for all outputs
    return stdev_output_dict