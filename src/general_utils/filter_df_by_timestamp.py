import pandas as pd
import numpy as np
import logging
from scipy.interpolate import interp1d
from typing import Callable, Dict, List, Any, Union, Optional

def filter_df_by_timestamp(df, start_timestamp=None, end_timestamp=None):
    """
    Filter a DataFrame by the 'timestamp' column based on the provided start and end timestamps.

    Args:
        df (pd.DataFrame): The input DataFrame.
        start_timestamp (str or pd.Timestamp, optional): The start timestamp. Rows earlier than this will be excluded.
        end_timestamp (str or pd.Timestamp, optional): The end timestamp. Rows later than this will be excluded.

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    if "Timestamp" in df.columns:
        # Ensure 'timestamp' column is in datetime format
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        # Apply filtering if start or end timestamp is provided
        if start_timestamp:
            start_timestamp = pd.to_datetime(start_timestamp)
            df = df[df["Timestamp"] >= start_timestamp]
        if end_timestamp:
            end_timestamp = pd.to_datetime(end_timestamp)
            df = df[df["Timestamp"] <= end_timestamp]
    return df