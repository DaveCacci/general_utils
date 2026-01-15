import pandas as pd

def check_columns_empty_or_nan(df: pd.DataFrame, columns: list) -> bool:
    """
    Check if specific columns in a DataFrame are empty or contain all NaN values.

    Args:
        df (pd.DataFrame): The DataFrame to check.
        columns (list): List of column names to check.

    Returns:
        bool: True if any of the specified columns are empty or contain all NaN values, False otherwise.
    """
    for column in columns:
        if column in df.columns:
            if df[column].isna().all():
                return True
        else:
            return True  # If the column does not exist, consider it as empty
    return False