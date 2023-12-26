import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

    return df_splitted

def visualization(data):
    # Convert Birthday and Update columns to datetime
    data['Birthday'] = pd.to_datetime(data['Birthday'], errors='coerce')
    data['Update'] = pd.to_datetime(data['Update'], errors='coerce')

    # Extracting month from Birthday for distribution
    data['Birthday Month'] = data['Birthday'].dt.month

    # Count the number of birthdays per month
    birthday_month_distribution = data['Birthday Month'].value_counts().sort_index()
    
    # Extracting unique values and their counts for Hometown
    hometown_distribution = data['Hometown'].value_counts()

    # Preparing data for Status Update Timeline
    data['Update Date'] = data['Update'].dt.date
    update_timeline = data['Update Date'].value_counts().sort_index()

    # Setting up the matplotlib figure
    plt.figure(figsize=(18, 6))

    # Plotting Birthday Distribution
    plt.subplot(1, 3, 1)
    sns.barplot(x=birthday_month_distribution.index, y=birthday_month_distribution.values, palette="viridis")
    plt.title('Birthday Distribution per Month')
    plt.xlabel('Month')
    plt.ylabel('Number of Birthdays')
    plt.xticks(np.arange(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

    # Plotting Hometown Distribution (Top 10)
    plt.subplot(1, 3, 2)
    sns.barplot(x=hometown_distribution.head(10).values, y=hometown_distribution.head(10).index, palette="rocket")
    plt.title('Top 10 Hometown Distribution')
    plt.xlabel('Number of Profiles')
    plt.ylabel('Hometown')

    # Plotting Status Update Timeline
    plt.subplot(1, 3, 3)
    plt.plot(update_timeline.index, update_timeline.values, color='teal')
    plt.title('Status Update Timeline')
    plt.xlabel('Date')
    plt.ylabel('Number of Updates')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


inpit_file_path = './100K FACEBOOK.xlsx'
output_file_path = './cleaned_data.xlsx'
cleaned_data = clean_data(inpit_file_path, output_file_path)
visualization(cleaned_data)


print("SCRIPT ENDED SUCCESSFULLY")
