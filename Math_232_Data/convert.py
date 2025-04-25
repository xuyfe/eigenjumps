import pandas as pd
import re
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

VALID_MIN_FLIGHT_TIME = 0.3
VALID_MAX_FLIGHT_TIME = 0.82

def convert_txt_to_df(file_path):
    # Read the file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Initialize lists to store data
    timestamps = []
    data_arrays = []
    release_indices = []
    landing_indices = []
    jump_pairs = []
    flight_times = []  # Store flight times
    current_index = 0

    # Regular expression pattern to match timestamp and array
    pattern = r'LOG  \[(.*?)\] \[(.*?)\]'
    flight_time_pattern = r'[Ff]light [Tt]ime: ([\d.]+)'  # Make case insensitive

    # Parse each line
    i = 0
    curr_jump_pair = {}
    while i < len(lines):
        line = lines[i]
        # check if the line contains "Release" or "Landing"
        if "Release" in line or "Landing" in line:
            if "Release" in line:
                release_indices.append(current_index)
                print(f"\nRelease detected at index {current_index}")
                curr_jump_pair['release_index'] = current_index
            else:
                landing_indices.append(current_index)
                # Add debug prints
                print(f"\nLanding detected at index {current_index}")
                if 'release_index' in curr_jump_pair:
                    curr_jump_pair['landing_index'] = current_index
                    if i + 1 < len(lines):
                        print(f"Next line: {lines[i+1].strip()}")
                    # Check next line for flight time
                    if i + 1 < len(lines) and re.search(r'[Ff]light [Tt]ime:', lines[i + 1]):
                        match = re.search(flight_time_pattern, lines[i + 1])
                        if match:
                            flight_time = float(match.group(1))
                            flight_times.append((len(landing_indices) - 1, flight_time))
                            print(f"Flight time detected: {flight_time}")
                            curr_jump_pair['flight_time'] = flight_time
                        else:
                            print("No flight time match found in the line")
                    else:
                        print("No flight time line found after landing")
            i += 1
            if 'release_index' in curr_jump_pair and 'landing_index' in curr_jump_pair and 'flight_time' in curr_jump_pair: 
                jump_pairs.append(curr_jump_pair)
                curr_jump_pair = {}
            continue
        
        match = re.search(pattern, line)
        if match:
            timestamp = match.group(1)
            data_str = match.group(2)
            
            # Convert string array to list of floats
            data = [float(x) for x in data_str.split(', ')]
            
            timestamps.append(timestamp)
            data_arrays.append(data)
            current_index += 1
        i += 1

    # filter out invalid jumps based on flight time
    valid_jump_pairs = [pair for pair in jump_pairs if VALID_MIN_FLIGHT_TIME <= pair['flight_time'] <= VALID_MAX_FLIGHT_TIME]

    print("\nDEBUG - Valid jumps:", valid_jump_pairs)

    # Create DataFrame
    df = pd.DataFrame(data_arrays, index=timestamps)
    df = df.reset_index()
    
    # Name columns
    sensor_columns = [f'Sensor_{i+1}' for i in range(len(data_arrays[0]))]
    df.columns = ['Timestamp'] + sensor_columns
    
    # Convert timestamp strings to datetime objects
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Add boolean columns for Release and Landing events
    df['is_release'] = False
    df['is_landing'] = False
    df['in_flight'] = False
    
    # Create valid release-landing pairs
    valid_release_landing_pairs = []
    
    print("\nValid Release-Landing Pairs (in chronological order):")
    for idx in range(len(valid_jump_pairs)):
        # Get the corresponding release and landing indices
        if idx < len(release_indices) and idx < len(landing_indices):
            release_idx = valid_jump_pairs[idx]['release_index']
            landing_idx = valid_jump_pairs[idx]['landing_index']
            
            print(f"DEBUG - Processing jump {idx}: Release={release_idx}, Landing={landing_idx}")
            
            if release_idx < landing_idx:  # Verify release comes before landing
                valid_release_landing_pairs.append((release_idx, landing_idx))
                flight_time = valid_jump_pairs[idx]['flight_time']
                print(f"Release at {release_idx} -> Landing at {landing_idx} (Flight time: {flight_time:.2f}s)")
    
    print(f"\nDEBUG - Number of valid pairs found: {len(valid_release_landing_pairs)}")
    
    # Set the boolean values and in_flight status for valid pairs
    for release_idx, landing_idx in valid_release_landing_pairs:
        print(f"DEBUG - Setting flags for release={release_idx}, landing={landing_idx}")
        df.loc[release_idx, 'is_release'] = True
        df.loc[landing_idx, 'is_landing'] = True
        df.loc[release_idx:landing_idx, 'in_flight'] = True
    
    # Reorder columns
    column_order = ['Timestamp', 'is_release', 'is_landing', 'in_flight'] + sensor_columns
    df = df[column_order]

    # Print verification
    print("\nVerification of in_flight status:")
    flight_periods = df[df['in_flight'] == True]
    print(f"Total rows marked as in_flight: {len(flight_periods)}")
    print(f"Number of True values in is_release: {df['is_release'].sum()}")
    print(f"Number of True values in is_landing: {df['is_landing'].sum()}")

    return df, valid_release_landing_pairs


def store_df_to_csv(df, file_path):
    # store the df to a csv file
    df.to_csv(file_path, index=False)



def plot_one_sensor(df, sensor_name, peaks=None, jump_cycles=None):
    # plot the column with the release and landing indices
    plt.plot(df['Timestamp'], df[sensor_name])

    # plot the release and landing indices using boolean indexing
    release_data = df[df['is_release']]
    landing_data = df[df['is_landing']]
    
    plt.scatter(release_data['Timestamp'], release_data[sensor_name], color='red', label='Release')
    plt.scatter(landing_data['Timestamp'], landing_data[sensor_name], color='blue', label='Landing')
    
    if peaks is not None:
        plt.scatter(df['Timestamp'].iloc[peaks], df[sensor_name].iloc[peaks], color='green')
    if jump_cycles is not None:
        for cycle in jump_cycles:
            plt.plot(cycle['Timestamp'], cycle[sensor_name])
            plt.scatter(cycle['Timestamp'].iloc[0], cycle[sensor_name].iloc[0], color='orange')
            plt.scatter(cycle['Timestamp'].iloc[-1], cycle[sensor_name].iloc[-1], color='purple')

    # get max and min values of the column AFTER the first release and before the last landing
    first_release_idx = df[df['is_release']].index[0]  # Get first True index
    last_landing_idx = df[df['is_landing']].index[-1]  # Get last True index
    # get max and min values of the column AFTER the first release and before the last landing
    max_value = df[sensor_name].iloc[first_release_idx:last_landing_idx].max()
    min_value = df[sensor_name].iloc[first_release_idx:last_landing_idx].min()
    plt.ylim(min_value - 0.5, max_value + 0.5)
    # set x axis limits to the first and last release and landing indices
    first_release = df['Timestamp'].iloc[first_release_idx]
    last_landing = df['Timestamp'].iloc[last_landing_idx]
    plt.xlim(first_release, last_landing)

    # set title
    plt.title(sensor_name)
    plt.xlabel('Time')
    plt.ylabel(sensor_name)
    plt.legend()
    plt.grid(True)
    plt.show()

# test the function above
df, valid_release_landing_pairs = convert_txt_to_df('data/AnnieGu.txt')
store_df_to_csv(df, 'data/AnnieGu.csv')

# plot first sensor
plot_one_sensor(df, "Sensor_1")

def plot_list_of_sensors(df, sensor_names):
    # Calculate number of rows and columns for the grid
    n_sensors = len(sensor_names)
    n_cols = 3 
    n_rows = (n_sensors + n_cols - 1) // n_cols 
    
    # Create figure and subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    fig.suptitle('Sensor Data with Release and Landing Points', fontsize=16)
    
    # Flatten axes array for easier iteration
    axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
    
    # Plot each sensor
    for i, (sensor_name, ax) in enumerate(zip(sensor_names, axes)):
        # Plot sensor data
        ax.plot(df['Timestamp'], df[sensor_name], label=sensor_name)
        
        # Plot release and landing points
        release_data = df[df['is_release']]
        landing_data = df[df['is_landing']]
        
        ax.scatter(release_data['Timestamp'], 
                  release_data[sensor_name], 
                  color='red', 
                  label='Release')
        ax.scatter(landing_data['Timestamp'], 
                  landing_data[sensor_name], 
                  color='blue', 
                  label='Landing')
        
        # Customize subplot

        # get max and min values of the column AFTER the first release and before the last landing
        first_release_idx = df[df['is_release']].index[0]  # Get first True index
        last_landing_idx = df[df['is_landing']].index[-1]  # Get last True index

        # get max and min values of the column AFTER the first release and before the last landing
        max_value = df[sensor_name].iloc[first_release_idx:last_landing_idx].max()
        min_value = df[sensor_name].iloc[first_release_idx:last_landing_idx].min()

        ax.set_ylim(min_value - 0.5, max_value + 0.5)

        # set x axis limits to the first and last release and landing indices
        first_release = df['Timestamp'].iloc[first_release_idx]
        last_landing = df['Timestamp'].iloc[last_landing_idx]
        ax.set_xlim(first_release, last_landing)

        # set title
        ax.set_title(sensor_name)
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.grid(True)
        ax.legend()
        
        # Rotate x-axis labels for better readability
        ax.tick_params(axis='x', rotation=45)
    
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()


def find_data_peaks(df, num_jumps=10, column='Sensor_1', window_size=50):
    """
    Extract exactly num_jumps peaks from the data, representing each jump
    
    Parameters:
    - df: pandas DataFrame containing the time series data
    - num_jumps: number of jumps to detect (default: 10)
    - column: name of the column to analyze (default: 'Sensor_1')
    - window_size: size of window for peak detection (default: 50)
    
    Returns:
    - peak_indices: indices where peaks occur
    - peak_properties: dictionary containing properties of the peaks
    """
    
    data = df[column].values
    
    # Start with a high prominence and gradually decrease until we find exactly num_jumps peaks
    prominence = np.max(data) - np.min(data)
    step = prominence / 100
    
    while prominence > 0:
        peak_indices, peak_properties = find_peaks(data, 
                                                 prominence=prominence,
                                                 distance=window_size)  # Minimum distance between peaks
        
        if len(peak_indices) == num_jumps:
            break
        elif len(peak_indices) < num_jumps:
            prominence -= step
        else:
            prominence += step
            step /= 2
    
    # Sort peaks by prominence to get the num_jumps most significant peaks
    if len(peak_indices) > num_jumps:
        prominences = peak_properties['prominences']
        # Get indices of num_jumps highest prominences
        top_prominence_idx = np.argsort(prominences)[-num_jumps:]
        peak_indices = peak_indices[top_prominence_idx]
        # Update peak properties
        for key in peak_properties:
            peak_properties[key] = peak_properties[key][top_prominence_idx]
    
    # Sort peaks by time
    peak_indices.sort()
    
    # Plot to visualize the peaks
    plt.figure(figsize=(15, 6))
    plt.plot(df['Timestamp'], data, 'b-', label='Data')
    
    # Plot peaks
    plt.plot(df['Timestamp'].iloc[peak_indices], 
             data[peak_indices], "ro", label='Peaks')
    
    # Add Release/Landing points if they exist
    if 'Is_Release' in df.columns:
        release_points = df[df['Is_Release']].index
        plt.plot(df['Timestamp'].iloc[release_points], 
                data[release_points], "bo", label='Release')
    
    if 'Is_Landing' in df.columns:
        landing_points = df[df['Is_Landing']].index
        plt.plot(df['Timestamp'].iloc[landing_points], 
                data[landing_points], "go", label='Landing')
    
    # Add jump numbers to the plot
    for i, idx in enumerate(peak_indices):
        plt.annotate(f'Jump {i+1}', 
                    (df['Timestamp'].iloc[idx], data[idx]),
                    xytext=(10, 10), textcoords='offset points')
    
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.ylim(-1.7, 1.7)
    plt.title(f'Detected {num_jumps} Jumps')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Create a summary DataFrame for the jumps
    jump_summary = pd.DataFrame({
        'Jump_Number': range(1, num_jumps + 1),
        'Timestamp': df['Timestamp'].iloc[peak_indices],
        'Peak_Value': data[peak_indices],
        'Peak_Index': peak_indices
    })
    
    print("\nJump Summary:")
    print(jump_summary)
    
    return peak_indices, peak_properties, jump_summary


def test_window_sizes(df, window_sizes=[20, 50, 100, 150], num_jumps=10, column='Sensor_1'):
    """
    Test different window sizes for peak detection and visualize the results including windows
    """
    
    fig, axes = plt.subplots(len(window_sizes), 1, figsize=(15, 5*len(window_sizes)))
    fig.suptitle('Peak Detection with Different Window Sizes', fontsize=16)
    
    data = df[column].values
    
    for i, window_size in enumerate(window_sizes):
        # Initialize with reasonable prominence bounds
        max_prominence = np.max(data) - np.min(data)
        min_prominence = max_prominence * 0.01  # 1% of max range
        prominence = max_prominence / 2  # Start in the middle
        
        best_peaks = None
        best_properties = None
        best_num_peaks = 0
        
        # Binary search for the right prominence
        while max_prominence - min_prominence > 0.01:
            peak_indices, peak_properties = find_peaks(data, 
                                                     prominence=prominence,
                                                     distance=window_size)
            
            num_peaks = len(peak_indices)
            
            # Keep track of the best result so far
            if best_peaks is None or abs(num_peaks - num_jumps) < abs(best_num_peaks - num_jumps):
                best_peaks = peak_indices
                best_properties = peak_properties
                best_num_peaks = num_peaks
            
            # Binary search adjustment
            if num_peaks > num_jumps:
                min_prominence = prominence
                prominence = (max_prominence + prominence) / 2
            elif num_peaks < num_jumps:
                max_prominence = prominence
                prominence = (min_prominence + prominence) / 2
            else:
                break  # Found exactly num_jumps peaks
        
        peak_indices = best_peaks
        peak_properties = best_properties
        
        # If we still have too many peaks, take the most prominent ones
        if len(peak_indices) > num_jumps:
            prominences = peak_properties['prominences']
            top_prominence_idx = np.argsort(prominences)[-num_jumps:]
            peak_indices = peak_indices[top_prominence_idx]
            peak_indices.sort()  # Sort by time
            
        ax = axes[i]
        
        # Plot the main data
        ax.plot(df['Timestamp'], data, 'b-', label='Data', alpha=0.7)
        
        # Plot peaks
        ax.plot(df['Timestamp'].iloc[peak_indices], 
                data[peak_indices], "ro", label='Peaks')
        
        # Plot windows around peaks
        for peak_idx in peak_indices:
            window_start = max(0, peak_idx - window_size//2)
            window_end = min(len(data), peak_idx + window_size//2)
            
            window_times = df['Timestamp'].iloc[window_start:window_end]
            
            # Plot window as a shaded region
            ax.axvspan(window_times.iloc[0], window_times.iloc[-1], 
                      alpha=0.2, color='gray')
            
            # Add window boundary markers
            ax.axvline(x=window_times.iloc[0], color='gray', linestyle='--', alpha=0.5)
            ax.axvline(x=window_times.iloc[-1], color='gray', linestyle='--', alpha=0.5)
        
        # Add jump numbers
        for j, idx in enumerate(peak_indices):
            ax.annotate(f'Jump {j+1}', 
                       (df['Timestamp'].iloc[idx], data[idx]),
                       xytext=(10, 10), textcoords='offset points')
        
        ax.set_title(f'Window Size = {window_size} data points\nFound {len(peak_indices)} peaks')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()

    # Print time differences between peaks
    print("\nTime differences between consecutive peaks:")
    for window_size in window_sizes:
        print(f"\nWindow size = {window_size}:")
        peak_indices, _ = find_peaks(data, distance=window_size)
        if len(peak_indices) > 1:
            time_diffs = np.diff([df['Timestamp'].iloc[idx] for idx in peak_indices])
            for j, diff in enumerate(time_diffs):
                print(f"Between peaks {j+1} and {j+2}: {diff}")


def extract_jump_cycles(df, peak_indices, column='Sensor_1', window_size=50):
    data = df[column].values
    jump_cycles = []
    
    # Calculate grid dimensions
    n_cols = 3  # Number of columns in the grid
    n_rows = (len(peak_indices) + n_cols - 1) // n_cols  # Ceiling division for number of rows
    
    # Create figure with a reasonable size
    fig = plt.figure(figsize=(15, 5*n_rows))
    
    # Create grid of subplots
    gs = fig.add_gridspec(n_rows, n_cols, hspace=0.4, wspace=0.3)
    axes = []
    for row in range(n_rows):
        for col in range(n_cols):
            if row * n_cols + col < len(peak_indices):
                axes.append(fig.add_subplot(gs[row, col]))
    
    for i, peak_idx in enumerate(peak_indices):
        # Find nearest release and landing points if they exist
        release_data = df.loc[:peak_idx][df['is_release']]
        landing_data = df.loc[peak_idx:][df['is_landing']]
        
        # Look for significant change in slope before peak
        left_idx = peak_idx
        min_value = data[peak_idx]
        
        # If there's a release point before peak, start search from there
        if not release_data.empty:
            release_before_peak = release_data.index[-1]
            search_start = release_before_peak
        else:
            search_start = peak_idx
            
        # Search for left trough
        for j in range(search_start - 1, max(0, search_start - window_size), -1):
            if data[j] < min_value:
                min_value = data[j]
                left_idx = j
            if data[j] > data[j+1] + 0.1:  # Threshold for significant change
                break
                
        # Look for significant change in slope after peak
        right_idx = peak_idx
        min_value = data[peak_idx]
        
        # If there's a landing point after peak, start search from there
        if not landing_data.empty:
            landing_after_peak = landing_data.index[0]
            search_end = landing_after_peak
        else:
            search_end = peak_idx
            
        # Search for right trough
        for j in range(search_end + 1, min(len(data), search_end + window_size)):
            if data[j] < min_value:
                min_value = data[j]
                right_idx = j
            if j < len(data)-1 and data[j+1] > data[j] + 0.1:
                break
        
        # Add buffer around the detected region
        buffer = window_size // 4
        cycle_start = max(0, left_idx - buffer)
        cycle_end = min(len(data), right_idx + buffer)
        
        # Create cycle data
        cycle_df = df.iloc[cycle_start:cycle_end+1].copy()
        cycle_df['Time_From_Peak'] = (cycle_df['Timestamp'] - 
                                    df['Timestamp'].iloc[peak_idx]).dt.total_seconds()
        
        jump_cycles.append(cycle_df)
        
        # Plotting
        ax = axes[i]
        ax.plot(cycle_df['Time_From_Peak'], cycle_df[column], 'b-', label='Data')
        ax.plot(0, data[peak_idx], 'ro', label='Peak')
        
        # Mark the detected start and end points
        ax.axvline(x=cycle_df['Time_From_Peak'].iloc[left_idx-cycle_start], 
                  color='g', linestyle='--', label='Jump Start')
        ax.axvline(x=cycle_df['Time_From_Peak'].iloc[right_idx-cycle_start], 
                  color='r', linestyle='--', label='Jump End')
        
        # Add release/landing points with vertical lines if they exist
        release_points = cycle_df[cycle_df['is_release']]
        landing_points = cycle_df[cycle_df['is_landing']]
        
        if not release_points.empty:
            ax.plot(release_points['Time_From_Peak'], 
                   release_points[column], 'rx', label='Release',
                   markersize=10, markeredgewidth=2)
            ax.axvline(x=release_points['Time_From_Peak'].iloc[0], 
                      color='r', linestyle=':', alpha=0.5)
        if not landing_points.empty:
            ax.plot(landing_points['Time_From_Peak'], 
                   landing_points[column], 'bx', label='Landing',
                   markersize=10, markeredgewidth=2)
            ax.axvline(x=landing_points['Time_From_Peak'].iloc[0], 
                      color='b', linestyle=':', alpha=0.5)
        
        ax.set_title(f'Jump {i+1}')
        ax.set_xlabel('Time from peak (seconds)')
        ax.set_ylabel(column)
        ax.grid(True)
        if i == 0:  # Only show legend for first plot to save space
            ax.legend()
    
    plt.suptitle(f'Jump Cycles Analysis - {column}', y=1.02, fontsize=14)
    plt.tight_layout()
    plt.show()
    return jump_cycles

# test the function above
df, valid_release_landing_pairs = convert_txt_to_df('data/AnnieGu.txt')
store_df_to_csv(df, 'data/AnnieGu.csv')
peak_indices, peak_properties, jump_summary = find_data_peaks(df, num_jumps=10, column='Sensor_1', window_size=50)
jump_cycles = extract_jump_cycles(df, peak_indices, column='Sensor_1', window_size=50)

