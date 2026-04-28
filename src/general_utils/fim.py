import numpy as np
from scipy.linalg import block_diag
import warnings
from typing import Optional
from scipy.stats import chi2
import scipy.sparse as sp
from scipy.sparse.linalg import inv, spsolve
from scipy.sparse import csc_matrix
import pandas as pd
import logging

class NegativeEigenvalue(Warning):
    """
    Warning raised when a supposedly positive definite matrix has a negative eigenvalue.
    
    This typically occurs due to:
    - High conditioning number leading to numerical errors
    - Insufficient data for parameter identifiability
    - Numerical precision issues in eigenvalue decomposition
    """

class HighConditioningNumber(Warning):
    """
    Warning raised when a positive definite matrix has a large conditioning number.
    
    High conditioning numbers indicate:
    - Potential numerical instability during matrix inversion
    - Parameters may be nearly collinear (identifiability issues)
    - Eigenvalue decomposition may be unreliable
    
    Rule of thumb: condition number > 1/epsilon is problematic
    """

def enforce_pos(matrix: np.ndarray, epsilon=10 ** (-6)) -> np.ndarray:
    """
    Enforce positive definite condition on a matrix through regularization.
    
    This function adds a small diagonal term to ensure the matrix is numerically
    positive definite, which is crucial for reliable matrix inversion and
    covariance computation in Fisher Information Matrix calculations.

    If the smallest eigenvalue e0 of matrix is <= 0, returns matrix + (eps - e0) * Id
    where Id is the identity matrix.

    Args:
        matrix: Square matrix to regularize (typically pre-computed FIM)
        epsilon: Regularization parameter (default: 1e-6). Controls minimum relative
                eigenvalue as fraction of maximum eigenvalue

    Returns:
        Regularized matrix guaranteed to be positive definite
        
    Warnings:
        NegativeEigenvalue: If eigendecomposition found negative eigenvalue
        HighConditioningNumber: If conditioning number > 1/epsilon

    Note:
        Regularization was reduced on 03.10.2024 to avoid over-modifying FIM.
        The current implementation preserves FIM structure while ensuring invertibility.
    """
    # Compute eigenvalues in ascending order (smallest first)
    eigs = np.linalg.eigvalsh(matrix)

    # Normalize by maximum eigenvalue to assess relative magnitudes
    max_eigval = np.max(np.abs(eigs))
    # Note: Normalization by max_eigval is not applied to avoid changing matrix scale

    # Handle negative eigenvalues by shifting spectrum
    if eigs[0] < 0:
        warnings.warn(
            f"""Eigendecomposition of matrix found negative eigenvalue {eigs[eigs<0]}
            Setting smallest eigenvalue to {epsilon * max_eigval}""",
            category=NegativeEigenvalue,
        )
        # Add diagonal term to shift smallest eigenvalue to epsilon * max_eigval
        # This preserves matrix structure while ensuring positive definiteness
        matrix = matrix + (max_eigval * epsilon - eigs[0]) * np.eye(matrix.shape[0])
        # Alternative (commented): matrix = matrix + epsilon * np.eye(matrix.shape[0])
        # Note: Current approach modified on 03.10.2024 to minimize FIM distortion
        
        # Update eigenvalue estimates (for reference, not used further)
        eigs = eigs - eigs[0] + epsilon * max_eigval

        # Return regularized matrix (assumes negative eigenvalues were small)
        return matrix

    # Check for high conditioning number (ratio of largest to smallest eigenvalue)
    # High condition number indicates potential numerical instability
    cond = max_eigval / np.min(eigs)

    if cond > (1 / epsilon):
        warnings.warn(
            f"Conditioning number is {cond} > {1/epsilon}.",
            category=HighConditioningNumber,
        )

        # Calculate diagonal term to reduce conditioning number
        # Formula ensures new condition number ~= 1/epsilon
        to_add = max_eigval * (epsilon - 1 / cond) / (1 - epsilon)
        # Alternative (commented): to_add = epsilon
        # Note: Current approach modified on 03.10.2024 to minimize FIM distortion

        # Apply Tikhonov regularization: A_reg = A + λI
        matrix = matrix + max_eigval * to_add * np.eye(matrix.shape[0])
        # Alternative (commented): matrix = matrix + to_add * np.eye(matrix.shape[0])
        return matrix

    return matrix

def FIM(grad: dict, merged_df_dict: dict, var_couples_list: list, constant_error: dict, uncertain_param_dict: dict,
        error_variance: Optional[float] = None, grad_type: str = 'aa', weighted=True, reg=True, epsilon=1e-6) -> tuple:
    """
    Compute Fisher Information Matrix and parameter covariance from sensitivity data.
    
    This function implements weighted least squares parameter estimation theory to compute
    the Fisher Information Matrix (FIM), which quantifies the information content of data
    about model parameters. The covariance matrix is obtained as the inverse of FIM.
    
    Mathematical Background:
        For weighted least squares: FIM = S^T * W * S
        where S is the sensitivity matrix and W is the weight matrix (inverse of error covariance)
        Parameter covariance: COV = σ² * FIM^(-1)
    
    Args:
        grad: Dictionary of sensitivity matrices (one DataFrame per output variable)
              Keys must match output_names in var_couples_list. First column is timestamp.
              CRITICAL: Ordering of grad.keys() must match var_couples_list!
        merged_df_dict: Dictionary of DataFrames containing both simulated outputs and measurements
                       Keys are output variable names
        var_couples_list: List of tuples [(output_name, measurement_name), ...]
                         Defines correspondence between model outputs and measurements
        constant_error: Dictionary mapping measurement names to relative error coefficients (alpha)
                       Used to compute measurement variance: σ_i² = (alpha * measurement_i)²
        error_variance: Fixed error variance (σ²) if known, or None to estimate from residuals
        uncertain_param_dict: Dictionary of uncertain parameters {name: [nominal_value, ...]}
                             Used for scaling when grad_type='ar'
        grad_type: Sensitivity type - 'aa' (absolute-absolute) or 'ar' (absolute-relative). 
                   Affects parameter uncertainty scaling.
        weighted: If True, use weighted least squares (recommended). If False, ordinary least squares.
                  Practically, if refers to the weighting of the pre_fim computation.
        reg: If True, apply regularization via enforce_pos() to ensure FIM invertibility
        epsilon: Regularization parameter passed to enforce_pos() if reg=True
    
    Returns:
        tuple: (fim_mat, cov, cov_export, stdev_param)
            - fim_mat: Fisher Information Matrix (numpy array)
            - cov: Parameter covariance matrix (numpy array)
            - cov_export: Covariance matrix as labeled DataFrame
            - stdev_param: Standard deviations of parameters (numpy array)
    
    Note:
        - Residuals are computed as: simulated_output - measurement
        - Small constant (1e-6) added to measurements to prevent division by zero
        - Function prints diagnostic information about parameter uncertainties
    Note: use this function on the outputs of the "Sensitivity_loca_OAT.ipynb" notebook!!
          It is of primary importance that the ordering of grad.keys() matches var_couples_list!!
          It is also important, if error_variance is None, that the keys and timestamps in merged_df_dict 
          match those in grad!!

    Note 28/04/2026: futher dvelopment considering the 'rr' grad_type to enhence the numerical stability of the FIM and related matrix inversion.
            If grad_type=='rr', compute FIM weighted by df['stdev_meas'] = constant_error[meas_name]. Then scale COV by D = diag(np.array([value[0] for value in uncertain_param_dict.values()])).
    """
    # Return info on grad_type
    if grad_type not in ['aa', 'ar']:
        raise ValueError("grad_type must be 'aa' or 'ar'")
    else:
        logging.info(f'Computing FIM with grad_type={grad_type}. Make sure the"grad" input is consistent with this choice.')
    ##############################################################
    # Extract measurement and output names from var_couples_list
    meas_names = [var_couples_list[i][1] for i in range(len(var_couples_list))]
    output_names = [var_couples_list[i][0] for i in range(len(var_couples_list))]
    
    ###############################################################
    # STEP 1: Construct block diagonal weight matrix W = diag(1/σ₁², 1/σ₂², ...)
    # Each block corresponds to one output variable's measurement error covariance
    ###############################################################

    mat_var_meas_all = {}
    weights_vector = []
    concatenated_res = []
    for output_name, meas_name in zip(output_names,meas_names):
        if output_name in merged_df_dict.keys():
            
            # Access DataFrame containing both simulation and measurement data
            df = merged_df_dict[output_name]
            
            # Compute measurement standard deviation: σ_i = alpha * measurement_i
            # Add small constant (1e-6) to prevent division by zero
            df['stdev_meas'] = constant_error[meas_name]*(df[meas_name]+1e-6)
            df['var_meas'] = df['stdev_meas']**2
            
            # Extract variance values for this output
            var_meas = df['var_meas'].values
            
            # Create diagonal covariance matrix for this output
            mat_var_meas = np.diag(var_meas)
            
            # Invert to get weight matrix for this output (W_i = Σ_i^(-1))
            inv_mat_var_meas = np.linalg.inv(mat_var_meas)
            
            # Store inverse covariance (weight) matrix
            mat_var_meas_all[output_name] = inv_mat_var_meas
            weights_vector.append(var_meas)
            
            # Extract residuals: e_i = y_sim - y_meas
            concatenated_res.append(merged_df_dict[output_name][output_name]-merged_df_dict[output_name][meas_name])

    # Assemble block diagonal weight matrix W from individual output weight matrices
    matrices = list(mat_var_meas_all.values())
    
    # Convert to sparse format for computational efficiency
    matrices_sp = [sp.csr_matrix(matrix) for matrix in matrices]
    
    # Construct block diagonal: W = diag(W_1, W_2, ..., W_n)
    block_diag_matrix = sp.block_diag(matrices_sp, format='csr')
    
    # Concatenate all measurement variances into single vector
    weights_vector = np.concatenate(weights_vector)
    
    # Concatenate all residuals into single vector
    res_all = np.concatenate(concatenated_res)

    ###############################################################
    # STEP 2: Assemble sensitivity matrix S from gradient data
    # Each row corresponds to one time point, columns are parameters
    ###############################################################

    concatenated_columns = []
    for key in grad.keys():
        if key in output_names:
            # Extract sensitivity data, excluding first column (assumed to be timestamp)
            extracted_data = grad[key].iloc[:,1:]
            
            # Ensure all data is numeric (convert any non-numeric to NaN)
            extracted_data = extracted_data.apply(pd.to_numeric, errors='coerce')
            
            # Convert to float64 for numerical stability
            extracted_data = extracted_data.astype('float64')
            concatenated_columns.append(extracted_data)
    
    # Stack all sensitivity data into single matrix S [n_measurements × n_parameters]
    result_array = np.concatenate(concatenated_columns)
    
    # Print diagnostic: parameter uncertainty estimates from diagonal of S^T*S
    # These are approximate - true uncertainties require full covariance calculation
    for i,key in enumerate(uncertain_param_dict.keys()):
        logging.info(f'Standard deviation of {key} shall be related to {(1/np.sum(result_array[:,i]**2))**0.5}')
    
    # Convert to sparse format for efficient matrix multiplication
    result_array_sp = sp.csr_matrix(result_array)

    ###############################################################
    # STEP 3: Compute Fisher Information Matrix
    # Weighted: FIM = S^T * W * S
    # Unweighted: FIM = S^T * S
    ###############################################################

    if weighted:
        # Weighted Fisher Information Matrix: FIM = S^T * W * S
        pre_fim = result_array_sp.T @ block_diag_matrix @ result_array_sp
    else:
        # Ordinary (unweighted) Fisher Information Matrix: FIM = S^T * S
        pre_fim = result_array_sp.T @ result_array_sp
    
    # Convert back to dense array for further processing
    pre_fim = csc_matrix(pre_fim).toarray()
    
    # Apply regularization if requested to ensure numerical stability
    if reg:
        pre_fim = enforce_pos(pre_fim,epsilon)

    # Compute preliminary covariance as FIM inverse: COV = FIM^(-1)
    # Using sparse solver for numerical efficiency
    pre_cov = spsolve(csc_matrix(pre_fim), sp.identity(pre_fim.shape[0], format='csc')).toarray()

    ###############################################################
    # STEP 4: Estimate or apply error variance σ²
    ###############################################################

    if error_variance is None:
        # Estimate error variance from residuals
        # Formula: σ² = Σ(e_i² / w_i) / (n - p)
        # where n = number of data points, p = number of parameters
        if weighted:
            # If pre_fim was already weighted, do not weight again i.e. overwrite weights_vector with ones
            # This effectively computes unweighted variance estimate
            weights_vector = np.ones(len(weights_vector))
        
        # Degrees of freedom = n_measurements - n_parameters
        sigma2_estim = np.sum((res_all**2) / weights_vector) / (
            block_diag_matrix.shape[0] - pre_cov.shape[0])
        logging.info(f'Number of parameters: {pre_cov.shape[0]}')
    else:
        # Use provided error variance
        sigma2_estim = error_variance
    
    logging.info(f'Estimated error variance: sigma2_estim={sigma2_estim}')

    ###############################################################
    # STEP 5: Scale covariance and FIM by error variance
    ###############################################################
    
    # Final covariance matrix: COV = σ² * FIM^(-1)
    cov = pre_cov * sigma2_estim
    
    # Final FIM (normalized by error variance)
    fim_mat = pre_fim / sigma2_estim
    
    ###############################################################
    # STEP 6: Extract parameter uncertainties and format output
    ###############################################################
    
    # Create labeled DataFrame for covariance matrix export
    col_names = uncertain_param_dict.keys()
    cov_export = pd.DataFrame(cov, index=col_names, columns=col_names)
    
    # Extract diagonal elements (parameter variances)
    cov_diag = np.diag(cov)
    
    # Compute parameter standard deviations: σ_θ = sqrt(COV_ii)
    stdev_param = np.sqrt(cov_diag)
    
    # Scale standard deviations based on gradient type
    # For 'ar' (absolute-relative): multiply by nominal parameter values
    # to convert relative uncertainties to absolute
    if grad_type=='ar':
        scale = np.array([v[0] for v in uncertain_param_dict.values()], dtype=float)
        # Equivalent to D @ cov @ D but cheaper: elementwise scale by outer product
        cov = (scale[:, None] * cov) * scale[None, :]
        cov_export = pd.DataFrame(cov, index=uncertain_param_dict.keys(), columns=uncertain_param_dict.keys())
        stdev_param = stdev_param * scale
    
    # Print approximate parameter uncertainties (diagonal-only estimate)
    # This is faster but ignores parameter correlations
    logging.info(f'Parameter std dev (diagonal approximation): {dict(zip(list(uncertain_param_dict.keys()),sigma2_estim*(1/np.diag(pre_fim))**0.5))}')
    
    return fim_mat, cov, cov_export, stdev_param
    