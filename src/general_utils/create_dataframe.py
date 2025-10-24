import pandas as pd

# Convert Modelica output list to dataframe with timestamp column
def create_dataframe(names, data,start_time):
    """
    Create a DataFrame from a list of names and a list of arrays.

    Parameters:
    - names (list): List of column names.
    - data (list): List of arrays, where each array corresponds to a column.

    Returns:
    - pd.DataFrame: Resulting DataFrame.
    """
    if len(names) != len(data):
        raise ValueError("Number of names and data arrays must be the same.")

    data_dict = dict(zip(names, data))
    df = pd.DataFrame(data_dict)
    start_time = pd.Timestamp(start_time)
    timestamps = pd.date_range(start=start_time, periods=len(df), freq='H')  # Add timestamp column
    timestamps = timestamps.strftime('%d.%m.%Y %H:%M:%S')
    df.insert(loc=0, column='Timestamp', value=timestamps)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'],format='%d.%m.%Y %H:%M:%S')

    return df

def create_dataframe_sec(names, data,start_time, frequency: int):
    """
    frequency: integer to specify in seconds the frequency of timestamp generation
    """
    
    if len(names) != len(data):
        raise ValueError("Number of names and data arrays must be the same.")

    data_dict = dict(zip(names, data))
    df = pd.DataFrame(data_dict)
    start_time = pd.Timestamp(start_time)
    timestamps = pd.date_range(start=start_time, periods=len(df), freq=f'{frequency}S')  # Add timestamp column
    timestamps = timestamps.strftime('%d.%m.%Y %H:%M:%S')
    df.insert(loc=0, column='Timestamp', value=timestamps)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'],format='%d.%m.%Y %H:%M:%S')

    return df