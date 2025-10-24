import pandas as pd
import os
import logging

def save_df(df, new_rows, path, log: bool = False):
    if log:
        logging.info(f'Saving the dataframe named: {os.path.basename(path)}, in {path}...')
    for _, new_row in new_rows.iterrows():
        new_timestamp = new_row['Timestamp']

        if new_timestamp in df['Timestamp'].values:
            old_row = df.loc[df['Timestamp'] == new_timestamp]
            
            old_row_dict = old_row.drop(columns=['Timestamp']).iloc[0].to_dict()
            new_row_dict = new_row.drop(labels='Timestamp').to_dict()
            if log:
                logging.info(f"Duplicate timestamp found: {new_timestamp}. Old values: {old_row_dict}, New values: {new_row_dict}. Replacing the row.")

            # Find the index of the existing row and update the value
            df.loc[df['Timestamp'].values == new_timestamp] = new_row.values
        else:
            # Append the new row to the DataFrame
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        
    df['Timestamp'] = df['Timestamp'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
    columns_to_round = df.columns.difference(['Timestamp', 'time'], sort=False) #Round everything but for Timestamp and time columns if present
    df[columns_to_round] = df[columns_to_round].round(5)
    df.to_csv(path, mode='w', header=True, index=False, sep=',', float_format='%g')

def save_df_with_check(new_rows, path, log: bool = False):
    if os.path.exists(path):
        df = pd.read_csv(path)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    else:
        df = pd.DataFrame(columns=list(new_rows.columns))
    if log:
        logging.info(f'Saving the dataframe named: {os.path.basename(path)}, in {path}...')
    for _, new_row in new_rows.iterrows():
        new_timestamp = new_row['Timestamp']

        if new_timestamp in df['Timestamp'].values:
            old_row = df.loc[df['Timestamp'] == new_timestamp]
            
            old_row_dict = old_row.drop(columns=['Timestamp']).iloc[0].to_dict()
            new_row_dict = new_row.drop(labels='Timestamp').to_dict()
            if log:
                logging.info(f"Duplicate timestamp found: {new_timestamp}. Old values: {old_row_dict}, New values: {new_row_dict}. Replacing the row.")

            # Find the index of the existing row and update the value
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df.loc[df['Timestamp'].values == new_timestamp] = new_row.values
        else:
            # Append the new row to the DataFrame
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
    # Ensure the Timestamp column is in datetime format
    df['Timestamp'] = df['Timestamp'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
    
    columns_to_round = df.columns.difference(['Timestamp', 'time'], sort=False) #Round everything but for Timestamp and time columns if present
    df[columns_to_round] = df[columns_to_round].round(5)
    df.to_csv(path, mode='w', header=True, index=False, sep=',', float_format='%g')

def save_df_with_check_merge(new_rows, path, log: bool = False):
    if os.path.exists(path):
        df = pd.read_csv(path)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    else:
        df = pd.DataFrame(columns=list(new_rows.columns))
    if log:
        logging.info(f'Saving the dataframe named: {os.path.basename(path)}, in {path}...')

    # Ensure new_rows['Timestamp'] is in datetime format
    new_rows['Timestamp'] = pd.to_datetime(new_rows['Timestamp'])
    # Identify duplicate rows based on 'Timestamp'
    duplicate_rows = new_rows[new_rows['Timestamp'].isin(df['Timestamp'])]
    if log and not duplicate_rows.empty:
        for _, duplicate_row in duplicate_rows.iterrows():
            old_row = df.loc[df['Timestamp'] == duplicate_row['Timestamp']]
            old_row_dict = old_row.drop(columns=['Timestamp']).iloc[0].to_dict()
            new_row_dict = duplicate_row.drop(labels='Timestamp').to_dict()
            logging.info(f"Duplicate timestamp found: {duplicate_row['Timestamp']}. Old values: {old_row_dict}, New values: {new_row_dict}. Replacing the row.")
    
    # Merge the new rows with the existing DataFrame
    df = pd.concat([df, new_rows]).drop_duplicates(subset='Timestamp', keep='last')
    
    # Round numeric columns (except 'Timestamp' and 'time')
    columns_to_round = df.columns.difference(['Timestamp', 'time'], sort=False)
    df[columns_to_round] = df[columns_to_round].round(5)
    
    # Save the DataFrame to CSV
    df.to_csv(path, mode='w', header=True, index=False, sep=',', float_format='%g')