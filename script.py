import pandas as pd
import numpy as np

def merge_columns_with_conditions(df):
    def custom_merge(row):
        # Replace NaN with ';' and join using space
        return ''.join(row.fillna(';'))
    
    # Apply the custom merge to each row
    df['merged'] = df.apply(custom_merge, axis=1)
    return df

def clean_data(input_file_path, output_file_path):
    fb_dataframe = pd.read_excel(input_file_path, header=None)
    print(f"Data file summary : {fb_dataframe.head()}")

    # Merge all columns into a single column separated by semicolons
    df_combined = merge_columns_with_conditions(fb_dataframe)

    # Define column names (these are hypothetical, adjust as per the actual data)
    column_names = ['ID', 'Profile', 'Phone', 'Birthday', 'Name', 'Hometown', 'Country', 'Status', 'Update', 'Email']

    
    # Split the data into separate columns with the semicolon separator
    df_splitted = df_combined['merged'].str.split(';', expand=True)

    print("\n Splitted data Summary\n ",df_splitted.head())

    df_splitted.replace('', np.nan, inplace=True)
    df_splitted = df_splitted.dropna(axis='columns', how='all')

    print("\n AFTER DROPPING \n ",df_splitted.head())

    df_splitted.columns = column_names[:len(df_splitted.columns)]  # Assign column names

    df_splitted.to_excel(output_file_path, index=False)  # Write to excel file

inpit_file_path = './100K FACEBOOK.xlsx'
output_file_path = './cleaned_data.xlsx'
clean_data(inpit_file_path, output_file_path)

print("SCRIPT ENDED SUCCESSFULLY")
