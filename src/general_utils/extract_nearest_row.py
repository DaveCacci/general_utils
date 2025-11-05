# FUNCTION TO EXTRACT ROW CORRESPONDING TO CURRENT TIME OF CONTROL RUN
def extract_nearest_row(dataframe,current_time):
    """
    Extract the row from a DataFrame corresponding to the nearest current timestamp, rounded to the hour period.
    Args:
    - dataframe (pd.DataFrame): DataFrame with timestamp column.
    Returns:
    - nearest_row (pd.Series): Series containing the row data.
    """
    # Ensure that the DataFrame has a timestamp column
    if 'Timestamp' not in dataframe.columns:
        raise ValueError("DataFrame must have a 'timestamp' column.")

    # Get the current timestamp rounded to the hour period
    current_time = current_time.replace(microsecond=0, second=0, minute=0)

    # Find the index of the nearest timestamp in the DataFrame
    nearest_index = (dataframe['Timestamp'] - current_time).abs().idxmin()

    # Extract the row corresponding to the nearest timestamp
    nearest_row = dataframe.loc[nearest_index]

    return nearest_row