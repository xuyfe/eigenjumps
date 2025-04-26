import pandas as pd
import numpy as np
import re
from datetime import datetime
import os

def clean_log_file(file_path):
    # Initialize lists to store data
    timestamps = []
    data_arrays = []
    is_release = []
    is_landing = []
    
    # Get the file name without extension
    file_name = os.path.basename(file_path).split('.')[0]
    
    # Read the file line by line
    with open(file_path, 'r') as file:
        for line in file:
            # Skip empty lines
            if not line.strip():
                continue
            
            # Check for release event
            if "Release at" in line:
                release_match = re.search(r'Release at (\d+)', line)
                if release_match:
                    release_time = int(release_match.group(1))
                    # Add a row with release event
                    timestamps.append(datetime.fromtimestamp(release_time/1000).isoformat())
                    data_arrays.append([np.nan] * 100)  # Placeholder array with NaN values
                    is_release.append(1)
                    is_landing.append(0)
                continue
            
            # Check for landing event
            if "Landing at" in line:
                landing_match = re.search(r'Landing at (\d+)', line)
                if landing_match:
                    landing_time = int(landing_match.group(1))
                    # Add a row with landing event
                    timestamps.append(datetime.fromtimestamp(landing_time/1000).isoformat())
                    data_arrays.append([np.nan] * 100)  # Placeholder array with NaN values
                    is_release.append(0)
                    is_landing.append(1)
                continue
            
            # Look for data lines with timestamps and arrays
            if "LOG" in line and "[" in line and "]" in line:
                # Extract timestamp
                timestamp_match = re.search(r'\[(.*?)\]', line)
                if timestamp_match:
                    timestamp = timestamp_match.group(1)
                    
                    # Extract the array of numbers
                    array_match = re.search(r'\[(.*?)\]$', line)
                    if array_match:
                        # Convert string of numbers to list of floats
                        try:
                            numbers = [float(x.strip()) for x in array_match.group(1).split(',')]
                            timestamps.append(timestamp)
                            data_arrays.append(numbers)
                            is_release.append(0)
                            is_landing.append(0)
                        except ValueError:
                            continue

    # Convert to DataFrame
    df = pd.DataFrame(data_arrays, columns=[f'value_{i}' for i in range(len(data_arrays[0]))])
    
    # Add timestamp column using mixed format to handle different timestamp formats
    df['timestamp'] = pd.to_datetime(timestamps, format='mixed')
    
    # Extract only the time component
    df['time'] = df['timestamp'].dt.time
    
    # Add file name column
    df['file_name'] = file_name
    
    # Add release and landing columns
    df['is_release'] = is_release
    df['is_landing'] = is_landing
    
    # Reorder columns to have time, file_name, and event columns first
    cols = ['time', 'file_name', 'is_release', 'is_landing'] + [col for col in df.columns if col not in ['timestamp', 'time', 'file_name', 'is_release', 'is_landing']]
    df = df[cols]
    
    return df

def main():
    # Read and clean the data
    file_path = 'txt_files/cleaned/AnnieGu.txt'
    df = clean_log_file(file_path)
    
    # Display basic information about the cleaned data
    print("\nDataFrame Info:")
    print(df.info())
    
    print("\nFirst few rows:")
    print(df.head())
    
    # Print the number of rows with release and landing events
    print(f"\nNumber of release events: {df['is_release'].sum()}")
    print(f"Number of landing events: {df['is_landing'].sum()}")
    print(f"Total number of rows: {len(df)}")
    
    # Save to CSV
    output_path = 'cleaned_data.csv'
    df.to_csv(output_path, index=False)
    print(f"\nCleaned data saved to {output_path}")

if __name__ == "__main__":
    main() 