import numpy as np
from .create_dataframe import*
import pandas as pd
from datetime import datetime, timedelta

def sample_and_create_df(x, output_discrete_names: list, time_points_discrete, frequency, start_timestamp, modelica_from0: bool = False):
    # x is a list of arrays for example stored in a dictionary. If array (time, states) is given as input, make sure to convert it to a list of arrays, after transposing it.
    # frequency is a float number representing the number of hours to sample and create dataframe

    if isinstance(x, dict):
        x = list(x.values())

    time_points_discrete_seconds = time_points_discrete*86400
    if modelica_from0:
        x_data = np.arange(3600*frequency,time_points_discrete_seconds[-1]+3600*frequency,3600*frequency)
    else:
        x_data = np.arange(0,time_points_discrete_seconds[-1]+3600*frequency,3600*frequency)
    #Define y_data
    xy, x_ind, y_ind = np.intersect1d(np.round(time_points_discrete_seconds), x_data, return_indices=True) # Need round...
    y_sampled = [np.take(arr,x_ind) for arr in x]
    # I have to add the first state value if modelica_from0
    if modelica_from0:
        y_sampled = [np.insert(arr, 0, val) for arr, val in zip(y_sampled, [x[i][0] for i in range(len(x))])]
    y_df = create_dataframe_sec(output_discrete_names, y_sampled, start_timestamp, 3600*frequency) # SAME AS START_TIMESTAMP_ERROR!!
    
    return y_df

import pandas as pd
from datetime import datetime, timedelta

###########################################################
def sample_df(df, freq):

    sample_timestamps = []
    current_time = df['Timestamp'].min()
    while current_time <= df['Timestamp'].max():
        sample_timestamps.append(current_time)
        current_time += timedelta(hours=freq)
    # Filter the DataFrame based on the created list of timestamps
    df_filtered = df[df['Timestamp'].isin(sample_timestamps)]
    
    return df_filtered