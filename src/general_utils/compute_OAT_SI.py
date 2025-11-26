# COMPUTE SENSITIVITY INDEX
# In this case is the sensitivity (y_mod_1% - y_mod_-1%)/2% !!
import logging
import os
import pandas as pd
import numpy as np
import shutil

def actual_SI_computation(output_1: np.ndarray, output_2: np.ndarray, mean_value: np.ndarray, uncertain_param_names: list, 
                          uncertain_param_values: list, formulation: str, delta_plus: float, delta_minus: float, log: bool = False) -> np.ndarray:
    '''
    Helper function to compute sensitivity indices based on the specified formulation.
    Parameters:
    - output_1: Model outputs for perturbed parameter value (e.g., +delta).
    - output_2: Model outputs for perturbed parameter value (e.g., -delta).
    - mean_value: Mean value of the nominal output.
    - uncertain_param_names: List of names of uncertain parameters.
    - uncertain_param_values: List of nominal values of uncertain parameters.
    - formulation: Sensitivity index formulation ('rr' for Relative-Relative, 'aa' for Absolute-Absolute, 'ar' for Absolute-Relative, 'ra' for Relative-Absolute).
    - delta_plus: Positive delta value used in perturbation.
    - delta_minus: Negative delta value used in perturbation.
    Note: when computing SI with respect to nominal outputs, use delta_minus=0 or delta_plus=0 accordingly.
    Returns:
    - sensitivity_list: Computed sensitivity indices as a NumPy array.
    '''
    sensitivity_list = []

    for i in range(len(uncertain_param_names)):
        if output_2.shape[1] != output_1.shape[1]:
            if log:
                logging.info(f"Output shapes: output_1 {output_1.shape}, output_2 {output_2.shape}. Consiering output_2 as nominal output, taking only first column.")
            numerator = (output_1[:, i] - output_2[:, 0])
        else:
            numerator = (output_1[:, i] - output_2[:, i])
        
        if formulation == 'aa':
            sens_aa = numerator / (uncertain_param_values[i]*(delta_plus-delta_minus))
            sensitivity_list.append(sens_aa)
        elif formulation == 'rr':
            sens_rr = numerator / (delta_plus-delta_minus) /mean_value     
            sensitivity_list.append(sens_rr)
        elif formulation == 'ar':
            sens_ar = numerator / (delta_plus-delta_minus)
            sensitivity_list.append(sens_ar)
        elif formulation == 'ra':
            sens_ra = numerator / (uncertain_param_values[i]*(delta_plus-delta_minus))
            sensitivity_list.append(sens_ra)
        else:
            raise ValueError("Formulation must be either 'aa' (Absolute-Absolute), 'rr' (Relative-Relative), 'ar' (Absolute-Relative), or 'ra' (Relative-Absolute).")
    sensitivity_list = np.transpose(sensitivity_list)
    return sensitivity_list

def compute_OAT_SI(modelname: str, current_date: str, date_string: str, abs_deltap: float, uncertain_param_names: list, uncertain_param_values: list, output_names: list, 
                   formulation='rr', delta_method = 'CD', directory = os.getcwd(), log=True):
    '''
    Compute One-At-a-Time Sensitivity Indices based on precomputed model outputs.
    Parameters:
    - modelname: Name of the model.
    - current_date: Current date string for file naming.
    - date_string: Date string for file naming.
    - abs_deltap: Absolute percentage change in parameters (e.g., 0.01 for 1%).
    - uncertain_param_names: List of names of uncertain parameters.
    - uncertain_param_values: List of nominal values of uncertain parameters.
    - output_names: List of output variable names to compute sensitivity for.
    - formulation: Sensitivity index formulation ('rr' for Relative-Relative, 'aa' for Absolute-Absolute, 'ar' for Absolute-Relative, 'ra' for Relative-Absolute).
    - delta_method: Method for delta computation ('FD+' for Finite Difference Plus, 'FD-' for Finite Difference Minus, 'CD' for Central Difference).
    - directory: Directory where input/output files are located.
    - log: Boolean flag to enable logging.
    Returns:
    - None (saves sensitivity indices to Excel files).
    Note: extend to allow for the direct filename input instead of constructing it inside the function?
    '''
    # Declare output file names
    string = f'{abs_deltap}' if delta_method == 'CD' else (f'+{abs_deltap}' if delta_method == 'FD+' else (f'-{(abs_deltap)}' if delta_method == 'FD-' else 'error'))
    output_file_formulation = os.path.join(directory, f'{modelname}_Sensitivity_local_{formulation}_{string}_{current_date}_{date_string}.xlsx')
    if log and os.path.exists(output_file_formulation):
        logging.warning(f"The file {output_file_formulation} already exists and will be overridden!")
    
    output_file_copy = os.path.join(directory, f'{modelname}_Sensitivity_local_{formulation}_{string}.xlsx') # REMOVED DATE AND DATE_STRING to ensure 'sens_window' can read also across 00:00 for online application
    if log and os.path.exists(output_file_copy):
        logging.warning(f"The file {output_file_copy} already exists and will be overridden!")

    with pd.ExcelWriter(output_file_formulation, engine='xlsxwriter', date_format='dd-mm-yyyy') as writer1:
        for output_name in output_names:
            columns = [f'{output_name}_{param_string}' for param_string in uncertain_param_names]
            # Read the nominal output file
            path=os.path.join(directory, f'{modelname}_Sheet_Filter_Output_{current_date}_{date_string}.xlsx')
            output_nom = pd.read_excel(path,sheet_name=output_name)
            if log:
                logging.info(f"Read nominal output file {path}")
            output_nom=output_nom.drop(columns='Timestamp')
            output_nom=output_nom.values
            mean_value=output_nom.mean()  #SC del brun è la media sugli output modified e non i nominali...DC: contrary
        
            # Compute sensitivity indices for each output variable
            if delta_method == 'FD+':
                logging.info("Using Finite Difference method for sensitivity computation, delta_plus with respect to nominal.")
                # Carica il file Excel degli output per deltap=1%
                delta_plus = abs_deltap
                path_mod = os.path.join(directory, f'{modelname}_Sheet_Filter_Output_{delta_plus}_{current_date}_{date_string}.xlsx')
                output_mod_plus = pd.read_excel(path_mod, sheet_name=output_name)
                timestamp_col=output_mod_plus['Timestamp']
                output_mod_plus=output_mod_plus.drop(columns='Timestamp')
                length=output_mod_plus.shape[0]
                output_mod_plus=output_mod_plus.values

                sensitivity_list = actual_SI_computation(output_mod_plus, output_nom, mean_value, uncertain_param_names, uncertain_param_values, formulation, delta_plus=delta_plus, delta_minus=0, log=log)

            elif delta_method == 'FD-':
                logging.info("Using Finite Difference method for sensitivity computation, delta_minus with respect to nominal.")
                # Carica il file Excel degli output per deltap=-1%
                delta_minus = -abs_deltap
                path_mod = os.path.join(directory, f'{modelname}_Sheet_Filter_Output_{delta_minus}_{current_date}_{date_string}.xlsx')
                output_mod_minus = pd.read_excel(path_mod, sheet_name=output_name)
                timestamp_col=output_mod_minus['Timestamp']
                output_mod_minus=output_mod_minus.drop(columns='Timestamp')
                length=output_mod_minus.shape[0]
                output_mod_minus=output_mod_minus.values

                sensitivity_list = actual_SI_computation(output_mod_minus, output_nom, mean_value, uncertain_param_names, uncertain_param_values, formulation, delta_plus=0, delta_minus=delta_minus, log=log)

            elif delta_method == 'CD':
                logging.info("Using Central Difference method for sensitivity computation.")
                # Carica il file Excel degli output per deltap=1%
                delta_plus = abs_deltap
                path_1 = os.path.join(directory, f'{modelname}_Sheet_Filter_Output_{delta_plus}_{current_date}_{date_string}.xlsx')
                output_mod_plus = pd.read_excel(path_1, sheet_name=output_name)
                timestamp_col=output_mod_plus['Timestamp']
                output_mod_plus=output_mod_plus.drop(columns='Timestamp')
                length=output_mod_plus.shape[0]
                output_mod_plus=output_mod_plus.values
                # Carica il file Excel degli output per deltap=-1%
                delta_minus = -abs_deltap
                path_1 = os.path.join(directory, f'{modelname}_Sheet_Filter_Output_{delta_minus}_{current_date}_{date_string}.xlsx')
                output_mod_minus = pd.read_excel(path_1, sheet_name=output_name)
                timestamp_col=output_mod_minus['Timestamp']
                output_mod_minus=output_mod_minus.drop(columns='Timestamp')
                length=output_mod_minus.shape[0]
                output_mod_minus=output_mod_minus.values

                sensitivity_list = actual_SI_computation(output_mod_plus, output_mod_minus, mean_value, uncertain_param_names, uncertain_param_values, formulation, delta_plus=delta_plus, delta_minus=delta_minus, log=log)
            
            else:
                raise ValueError("Delta method must be either 'FD+' (Finite Difference Plus), 'FD-' (Finite Difference Minus), or 'CD' (Central Difference).")
            
            # Save sensitivity indices to Excel
            sensitivity_df = pd.DataFrame(sensitivity_list, columns=columns)
            sensitivity_df.insert(0, 'Timestamp', timestamp_col)   
            sensitivity_df.to_excel(writer1, sheet_name=output_name, index=False)
            if log:
                logging.info(f'Sensitivity local with {delta_method} method and "{formulation}" formulation for {output_name} saved to {output_file_formulation}')
            
    # Copy to a fine without current_date and date_string in filename for online applications
    shutil.copy(output_file_formulation, output_file_copy)
    if log:
        logging.info(f'Sensitivity local with {delta_method} method and "{formulation}" formulation for all outputs copied and saved to {output_file_copy}')