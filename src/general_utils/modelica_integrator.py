import sys
import os
import subprocess
import DyMat
import psutil
from datetime import datetime, timedelta
import numpy as np
from .create_dataframe import create_dataframe, create_dataframe_sec
from .save_df import*
import logging
#import pandas as pd
import glob
import time

def terminate_running_process(process_name):
    for process in psutil.process_iter(['pid', 'name']):
        if process.info['name'] == process_name:
            logging.warning(f"Terminating existing process: {process_name}, PID: {process.info['pid']}")
            process.terminate()

            # Wait for the process to terminate
            try:
                process.wait(timeout=10)
            except psutil.TimeoutExpired:
                logging.warning(f"Process {process_name} did not terminate gracefully. Killing.")
                process.kill()

def process_array(input_array, substitution_array):
    # Check for np.nan values in the input array
    #np.nan_mask = np.isnp.nan(input_array)
    np.nan_mask = np.isnan(input_array)

    # Substitute np.nan values with corresponding values from the substitution array
    input_array[np.nan_mask] = substitution_array[np.nan_mask]

    return input_array

def is_string_in_file(file_path, target_string):
    with open(file_path, 'r') as file:
        for line in file:
            if target_string in line:
                return True
    return False

def modelica_integrator(mo_path: str, model_name: str, folder_name: str, param_dict: dict, param_scale_dict: dict, x0_dict: dict, time_interval: float, start_time: int, stop_time: int, tolerance: float, 
                    omc_path: str = None, results_sample_interval: int = None, timestamp_start: datetime = datetime.now(), x0_extract_names: list = None, path_to_x0_extract: list = None, path_to_outputs_extract: list = None,
                    outputs_extract_names: list = None, log_datetime: datetime = "", return_dymat: bool = False, log: bool = True,
                    round_params: int = 8):
    '''
    A function to run Modelica simulations from Python using OpenModelica compiler (omc).
    It creates a temporary folder to store the results, writes a .mos script with the simulation commands,
    runs the simulation, and extracts the results into pandas DataFrames.
    > mo_path: string of the path where the modelica file is located
    > ModelName: string of the model name inside the modelica file
    > folder_name: string of the temporary folder to save results (within the current working directory)
    > param_dict contains the keys and values to be set inside the Modelica model
    > param_scale_dict are the scales of the parameters (the values in param_dict are DIVIDED by these scales before being set in the model)
    > x0_dict is to set initial conditions
    > time_interval: float of seconds for the simulation intervals
    > start_time: integer of seconds for the simulation start time #delta_seconds #It starts where I cut the combi for AM2H...
    > stop_time: integer of seconds for the simulation stop time #interval_pilot*np.ceil((startime_pilot+3600*(t_span[1]+(delta_seconds-startime_pilot)/86400)*24)/interval_pilot) #????????
    > tolerance: float for the simulation tolerance
    > omc_path: string of the path where the omc.exe is located. If None, a standard path is used.
    > results_sample_interval: integer of seconds for the sampling interval of the results. If None, it is set to time_interval.
    > timestamp_start: datetime for the start timestamp of the simulation results. To create dataframe only. Default is datetime.now().
    > x0_extract_names: list of strings with the names of the states to extract final values from the simulation.
    > path_to_x0_extract: list of strings with the "internal" paths to the states to extract final values from the simulation.
    > path_to_outputs_extract: list of strings with the "internal" paths to the outputs to extract from the simulation.
    > outputs_extract_names: list of strings with the names of the outputs to extract from the simulation. Must have at least one element.
    > log_datetime: datetime to append to the log and result files.
    > log: boolean to enable or disable logging of info messages.
    > return_dymat: boolean to return also the DyMat file object along with the outputs and final states.
    > round_params: integer to round the parameter values when setting them in the model.
    Returns:
    -------
    > y_df: pandas DataFrame containing the outputs of interest with timestamps as index.
    > final_states_dict: dictionary containing the final states extracted from the simulation.
    > mat_file: DyMat file object containing the simulation results.

    Note: In Modelica, there MUST be present inside 'model_name' a declaration of the param_dict.keys(), for each key, as:
        "parameter Real param_dict_key = my_nominal_value;"
        Then, to propagate param_dict.values() set by this code inside the sub-models present inside 'model_name', 
        override the values of the true parameter names present in each sub-model as: 
        "Library.Sub_Model my_submodel(true_param_name = param_dict_key);"
        This is the only viable way to override from Python the "my_nominal_value" specified in Modelica with the value present in "param_dict" for the "param_dict_key" key.   
    Note: to set 'boolean' or non-numeric parameters, set them as strings in x0_dict, e.g. {"my_boolean_param": "true"} or {"my_string_param": '"my_string_value"'}.
    Note: if time_interval is less than 1, a different function shall be used to create the dataframe with timestamps.
        This is to ensure that the timestamps are generated correctly for high-frequency simulations i.e. sub-second intervals.
    Note: The .mos and .log files are created inside the temporary folder. Check them for debugging purposes.
    Note: User cannot extract constant and parameters as outputs (issues with y_df creation)! To extract them, use return_dymat=True and get them from the DyMat file externally.
    Note: when dealing with control actions in the order of 1e2 g/day or higher, it is recommended to set the parameter 'round_params' to max 2 to avoid numerical issues when setting parameters in Modelica.
    '''
    # Change temporaily the directory to save results
    original_dir = os.getcwd()  # Store the current directory
    temp_dir = os.path.join(original_dir, folder_name)
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    os.chdir(temp_dir)  # Change to the save directory
    if log:
        logging.info(f"Current working directory is now: {os.getcwd()}")
    
    # Set default values for optional parameters and perform checks
    # If omc_path is provided, update the system PATH
    if omc_path is None:
        standard_omc_path = "C:/Program Files/OpenModelica1.21.0-64bit/bin/omc.exe"
        if not os.path.exists(standard_omc_path):
            logging.warning(f"Standard OMC path {standard_omc_path} does not exist. Provide a valid omc_path.")
        else:
            omc_path = standard_omc_path
    # If results_sample_interval is not provided, set it to time_interval
    if results_sample_interval is None:
        results_sample_interval = time_interval
    # Return an error if x0_extract_names and path_to_x0_extract have different lengths
    if len(x0_extract_names) != len(path_to_x0_extract):
        raise ValueError("x0_extract_names and path_to_x0_extract must have the same length.")
    # Return an error if param_dict and param_scale_dict have different lengths
    if len(param_dict) != len(param_scale_dict):
        raise ValueError("param_dict and param_scale_dict must have the same length.")
    # Return a warning if x0_dict length is different from x0_extract_names length
    if len(x0_dict) != len(x0_extract_names):
        logging.warning("x0_dict length is different from x0_extract_names length. Make sure this is intentional.")
    # Return an error if outputs_extract_names and path_to_outputs_extract have different lengths
    if len(outputs_extract_names) != len(path_to_outputs_extract):
        raise ValueError("outputs_extract_names and path_to_outputs_extract must have the same length.")
    # Return an error if the length of outputs_extract_names is zero
    if len(outputs_extract_names) == 0:
        raise ValueError("outputs_extract_names cannot be empty, must have at least one element.")
    # Return a warning if time_interval or results_sample_interval are less than 1 second
    if time_interval < 1:
        logging.warning("time_interval is less than 1 second. This function may not handle sub-second intervals correctly in the dataframe creation.")
    if results_sample_interval < 1:
        logging.warning("results_sample_interval is less than 1 second. This function may not handle sub-second intervals correctly in the dataframe creation.")

    # Define absolute paths for results and script
    absolute_path_results_original = str(model_name)+f'_res.mat' # Where the results will be stored
    absolute_path_results_new = str(model_name)+f'_res_{log_datetime}.mat' # Where the results will be stored
    absolute_path_script = os.path.join(temp_dir, f'{model_name}_{log_datetime}.mos') # Where the .mos script will be stored i.e. setup OMC + simulate commands
    absolute_path_log = os.path.join(temp_dir, f'{model_name}_log_{log_datetime}.txt') # Where the Modelica log file will be stored

    # WRITE .mos: initialize an empty string ---------------------------------- #
    parametric = f"""loadModel(Modelica, {{"4.0.0"}}); getErrorString(); \n"""
    parametric += f"""loadFile("{mo_path}"); getErrorString(); \n"""
    parametric += f"""setCommandLineOptions(""); getErrorString(); \n"""
    # ADD one line for every calibration parameter to be set
    for key, value in param_dict.items():
        if key in param_scale_dict:
        # Concatenate the string for each iteration
            parametric += f"""setParameterValue({model_name}, {key}, {np.round(value/param_scale_dict[key],round_params)}); getErrorString(); \n"""
        else:
            logging.warning(f"Key {key} not found in the scale dictionary.")
    # ADD one line for every initial condition parameter to be set
    if all(value is None for value in x0_dict.values()) and log:
        logging.info("The initial condition dictionary has keys but no values. The default values specified in the Modelica model will be used.")
    else:
        for key, value in x0_dict.items():
            # Concatenate the string for each iteration
            parametric += f"""setParameterValue({model_name}, {key}, {value}); getErrorString(); \n"""
    # Add a final line
    parametric += f"""simulate({model_name}, startTime={start_time}, stopTime={stop_time}, numberOfIntervals={int((stop_time-start_time)/time_interval)}, tolerance = {tolerance}, simflags = "-lv=LOG_STATS"); getErrorString();"""
    with open(absolute_path_script, "w") as file:
        file.write(parametric)
    # ------------------------------------------------------------------------ #

    # RUN THE .MOS AND WRITE LOG OUTPUT IN THE .TXT --------------------------- #
    start= time.time()
    with open(absolute_path_log, "w") as file:
        temp_files = glob.glob(os.path.join(temp_dir, "*.tmp"))
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except Exception as e:
                logging.warning(f"Could not delete temporary file {temp_file}: {e}")
        terminate_running_process(f"{model_name}.exe")
        time.sleep(2)  # Wait for the process to terminate completely
        subprocess.run([omc_path, absolute_path_script], stdout = file)
    stop = time.time()
    duration = stop-start
    if log:
        logging.info(f"Modelica simulation completed in {duration:.2f} seconds.")
    # ------------------------------------------------------------------------ #

    # LOAD RESULTS AND EXTRACT VARIABLES OF INTEREST ------------------------- #
    # Change the file name of a file if exists (to avoid overwriting)
    if os.path.exists(absolute_path_results_original) and log_datetime != "":
        os.rename(absolute_path_results_original, absolute_path_results_new)
    elif os.path.exists(absolute_path_results_original) and log_datetime == "":
        absolute_path_results_new = absolute_path_results_original
    # Load results using DyMayFile(absolute_result_path) or DyMatFile("path/to/file.mat")
    mat_file = DyMat.DyMatFile(absolute_path_results_new)
    success = is_string_in_file(absolute_path_log,'The simulation finished successfully')
    if success and log:
        logging.info('Successful simulation')
    if not mat_file.names():
        raise ValueError("DyMat file is empty.")
    if not success:
        raise ValueError("Failed integrator")

    # Extract final states of interest
    final_states = []
    for i in range(len(x0_extract_names)):
        if path_to_x0_extract[i]=="":
            variable_name = f"{x0_extract_names[i]}"
        else:
            variable_name = f"{path_to_x0_extract[i]}.{x0_extract_names[i]}"
        if variable_name in mat_file.names():
            variable = mat_file.data(variable_name)[-1]
            final_states.append(variable)
        else:
            if log:
                logging.warning(f"Variable {variable_name} not found in the DyMat file.")
    final_states_dict = dict(zip(x0_dict.keys(), final_states)) # No x0_extract_names because I want to return the dictionary with the same keys as the input x0_dict

    # Extract outputs of interest for pilot
    time_simulation = mat_file.abscissa(f"{path_to_outputs_extract[0]}.{outputs_extract_names[0]}",valuesOnly=True) if path_to_outputs_extract[0] != "" else mat_file.abscissa(f"{outputs_extract_names[0]}",valuesOnly=True)

    y = []
    for i in range(len(outputs_extract_names)):
        if path_to_outputs_extract[i]=="":
            variable_name = f"{outputs_extract_names[i]}"
        else:
            variable_name = f"{path_to_outputs_extract[i]}.{outputs_extract_names[i]}"
        if variable_name in mat_file.names():
            variable = mat_file.data(variable_name)
            y.append(variable)
        else:
            if log:
                logging.warning(f"Variable {variable_name} not found in the DyMat file.")

    # ------------------------------------------------------------------------ #

    # Clean outputs
    x_data = np.arange(results_sample_interval, time_simulation[-1]+results_sample_interval, results_sample_interval) # It does not take the initial condition! 
    # Define y_data
    xy, x_ind, y_ind = np.intersect1d(time_simulation, x_data, return_indices=True)
    y_discrete = [np.take(arr,x_ind) for arr in y]
    # I have to add the first state value...only if startime=0!! Else, do not repeat the initial condition
    if start_time == 0:
        y_discrete = [np.insert(arr, 0, val) for arr, val in zip(y_discrete, [y[i][0] for i in range(len(y))])]
    y_df = create_dataframe_sec(outputs_extract_names, y_discrete, timestamp_start, results_sample_interval)

    # Return to the original directory
    os.chdir(original_dir)

    if return_dymat:
        return y_df, final_states_dict, mat_file
    else:
        return y_df, final_states_dict