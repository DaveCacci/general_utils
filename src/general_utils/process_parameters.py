import json
import logging
from datetime import datetime

# Function to read the JSON file and access different parameter categories
def load_parameters(file_path: str):
    with open(file_path, "r") as file:
        params = json.load(file)
    return params

############################################################################################################
def convert_lists_to_tuples(d):
    if isinstance(d, dict):  # if the object is a dictionary
        for key, value in d.items():
            if isinstance(value, list):  # if the value is a list, convert it to a tuple
                d[key] = tuple(value)
            elif isinstance(value, dict):  # if the value is a dictionary, recurse into it
                convert_lists_to_tuples(value)
    elif isinstance(d, list):  # if the object is a list, apply the conversion to each element
        for i in range(len(d)):
            if isinstance(d[i], list):
                d[i] = tuple(d[i])
            elif isinstance(d[i], dict):
                convert_lists_to_tuples(d[i])

############################################################################################################
# Function to update a specific category of parameters in the JSON file
def update_process_parameters(file_path: str, new_params: dict):
    with open(file_path, "r") as file:
        data = json.load(file)
    
    # Update process parameters
    data['process_parameters'].update(new_params)
    
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)
############################################################################################################
# Function to update a specific category of parameters in the JSON file
def datetime_serializer(obj):
    if isinstance(obj, datetime):
        return obj.strftime('%Y-%m-%d %H:%M:%S')
    raise TypeError(f"Type {type(obj)} not serializable")
def update_parameters(file_path: str, new_params: dict, category: str = None, log: bool = False):
    with open(file_path, "r") as file:
        data = json.load(file)
        convert_lists_to_tuples(data)

    # If category is provided, check if it exists in the JSON data
    if category:
        if category not in data:
            raise KeyError(f"Category '{category}' not found in the JSON file.")
        target_data = data[category]
    else:
        target_data = data

    # Log info if the same keys have different values
    if log:
        for key, new_value in new_params.items():
            if key in target_data and target_data[key] != new_value:
                logging.info(f"Parameter '{key}' was {target_data[key]}, now updated to {new_value}")

    # Update process parameters
    target_data.update(new_params)

    # If category is provided, update the specific category in the original data
    if category:
        data[category] = target_data

    with open(file_path, "w") as file:
        json.dump(data, file, default=datetime_serializer, indent=4)
############################################################################################################

import math

def calculate_scale(value: float) -> float:
    """
    Calculate a scale as a multiplier of 10 such that value * scale is in the range (1, 10),
    and the scale is one of the powers of 10 (1e-3, 1e-2, ..., 1e3).

    Args:
        value (float): The parameter value.

    Returns:
        float: The calculated scale.
    """
    if value <= 0:
        raise ValueError("Value must be positive to calculate scale.")

    # Find the power of 10 that brings the value into the range (1, 10)
    scale = 10 ** (-math.floor(math.log10(value)))
    
    # Ensure that value * scale falls within the range (1, 10)
    while value * scale < 1:
        scale *= 10
    
    while value * scale >= 10:
        scale /= 10

    return scale
############################################################################################################

def add_scales_to_parameters(file_path: str):
    """
    Load parameters from a JSON file, calculate scales, and update the file.

    Args:
        file_path (str): Path to the JSON file.
    """
    with open(file_path, "r") as file:
        params = json.load(file)

    updated_params = {
        key: [value, calculate_scale(value)] for key, (value, *_) in params.items()
    }

    with open(file_path, "w") as file:
        json.dump(updated_params, file, indent=4)

