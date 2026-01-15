import numpy as np
import logging

def compare_arrays(array1: np.ndarray, array2: np.ndarray):
    """
    Compare two arrays and print out the differences along with their indices.

    Args:
        array1 (np.ndarray): The first array to compare.
        array2 (np.ndarray): The second array to compare.

    Returns:
        None
    """
    if array1.shape != array2.shape:
        logging.info("Arrays have different shapes and cannot be compared element-wise.")
        return

    differences = array1 != array2
    indices = np.where(differences)

    if indices[0].size == 0:
        logging.info("Arrays are identical.")
    else:
        for index in zip(*indices):
            logging.info(f"Difference at index {index}: array1 = {array1[index]}, array2 = {array2[index]}")