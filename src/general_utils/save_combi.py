import numpy as np
import logging
import os

def save_combi(file_path, array, formats, log: bool = False):
    """
    Save an array to a text file, adapting to the number of columns and applying scientific notation.

    Parameters:
        file_path (str): Path to the file to write to.
        array (np.ndarray): Input array to save.
        formats (list of str): A list of format strings (e.g., "%.2f", "%.2e") for each column.
    """
    if log:
        logging.info(f'Saving the combi named: {os.path.basename(file_path)}, in {file_path}')
    if len(formats) != array.shape[1]:
        raise ValueError("The number of format strings must match the number of columns in the array.")
    
    # Step 1: Read the first two lines from the file
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Copy the first two lines
    first_two_lines = lines[:2]
    
    # Modify the second line to reflect the new array shape
    first_line = first_two_lines[1]
    split_line = first_line.split('(')
    split_line[1] = f"{array.shape[0]},{array.shape[1]})\n"  # Update with the new array dimensions
    first_two_lines[1] = '('.join(split_line)
    
    # Step 2: Prepare new content to write
    output_content = first_two_lines  # Start with the modified first two lines
    
    # Add the values from the array, formatting dynamically based on the specified formats
    for row in array:
        formatted_row = '\t'.join(fmt % value for fmt, value in zip(formats, row))  # Apply formatting
        output_content.append(f"{formatted_row}\n")
        
    # Step 3: Write the updated content back to the file
    with open(file_path, 'w') as output_file:
        output_file.writelines(output_content)
