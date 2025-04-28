import pandas as pd
import re
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import glob
import os
VALID_MIN_FLIGHT_TIME = 0.3
VALID_MAX_FLIGHT_TIME = 0.82
NUM_SENSORS = 80
DROP_SENSORS_THRESHOLD = 0.2
SUM_THRESHOLD = 20
HALF_WINDOW_SIZE = 90
def pool_df(df, pool_type='sum'):
    # sum across all sensors
    if pool_type == 'sum':
        df['sum'] = df.iloc[:, 5:].sum(axis=1)
    elif pool_type == 'mean':
        df['mean'] = df.iloc[:, 5:].mean(axis=1)
    elif pool_type == 'median':
        df['median'] = df.iloc[:, 5:].median(axis=1)
    else:
        raise ValueError(f"Invalid pool type: {pool_type}")
    return df

def convert_txt_to_df(file_path):
    # Initialize lists to store data
    data_arrays = []
    jump_sets = []

    # Regular expression pattern to match timestamp and array
    pattern = r'LOG  \[(.*?)\] \[(.*?)\]'
    flight_time_pattern = r'[Ff]light [Tt]ime: ([\d.]+)'  # Make case insensitive

    with open(file_path, 'r') as file:
        lines = file.readlines()


    calibration_index = 0
    # find the line with the calibration pattern
    for line in lines:
        if 'Calibration' in line:
            calibration_index = lines.index(line) + 5
            break
    
    assert calibration_index != 0, "Calibration index not found"
    # starting from calibration index, read all the lines
    current_jump_set = {}
    sliced_lines = lines[calibration_index:]
    for i, line in enumerate(sliced_lines):
        # check if the line contains "Release" or "Landing"
        if "Release" in line or "Landing" in line:
            to_amend = data_arrays.pop()
            if "Release" in line:
                print("RELEASE: ", to_amend[0])
                current_jump_set["release_time"] = to_amend[0]
                to_amend[1] = True
                to_amend[2] = False
            elif "Landing" in line and "release_time" in current_jump_set:
                print("LANDING: ", to_amend[0])
                current_jump_set["landing_time"] = to_amend[0]
                # search for flight time in the next line
                flight_time_found = False
                for offset in range(1, 4):  # Look ahead up to 3 lines
                    if i + offset < len(sliced_lines):
                        print(f"Checking line {i+offset}: {sliced_lines[i+offset].strip()}")
                        flight_time_match = re.search(flight_time_pattern, sliced_lines[i + offset])
                        if flight_time_match:
                            print("Found flight time:", flight_time_match.group(1))
                            current_jump_set["flight_time"] = float(flight_time_match.group(1))
                            flight_time_found = True
                            break
                if not flight_time_found:
                    print("No flight time found for this landing.")
                to_amend[1] = False
                to_amend[2] = True

                # append the current jump set to the list and clear
                jump_sets.append(current_jump_set)
                current_jump_set = {}
            data_arrays.append(to_amend)
        else:
            # convert the line to a list of floats
            match = re.search(pattern, line)
            if match:
                timestamp = match.group(1)
                data_str = match.group(2)
                metadata = [timestamp, False, False, False]
                data = [float(x) for x in data_str.split(', ')]
                metadata.extend(data)
                data_arrays.append(metadata)

    columns = ['Timestamp', 'is_release', 'is_landing', 'in_flight'] + [f'Sensor_{i+1}' for i in range(NUM_SENSORS)]
    df = pd.DataFrame(data_arrays, columns=columns)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

        # add the "in_flight" column to the data_arrays
    # get rows between release and landing
    for jump_set in jump_sets:
        release_time = pd.to_datetime(jump_set['release_time'])
        landing_time = pd.to_datetime(jump_set['landing_time'])
        jump_indices = df[(df['Timestamp'] >= release_time) & (df['Timestamp'] <= landing_time)].index
        df.loc[jump_indices, 'in_flight'] = True

    # pool the df
    df = pool_df(df, pool_type='sum')
    return df, jump_sets

def print_mean_of_sensors(df):
    for i in range(NUM_SENSORS):
        print(f"Sensor {i+1} mean: {df[f'Sensor_{i+1}'].mean()}")
    return


def find_valid_jump_set(df, jump_sets, sum_threshold):
    valid_jump_sets, invalid_jump_sets = [], []
    for jump_set in jump_sets:
        # Get the time window for this jump
        release_time = pd.to_datetime(jump_set['release_time'])
        landing_time = pd.to_datetime(jump_set['landing_time'])
        # Filter the DataFrame for this window
        window_df = df[(df['Timestamp'] >= release_time) & (df['Timestamp'] <= landing_time)]
        # Check if any value in 'sum' is below the threshold
        if (window_df['sum'] < sum_threshold).any():
            valid_jump_sets.append(jump_set)
        else:
            invalid_jump_sets.append(jump_set)
    return valid_jump_sets, invalid_jump_sets

def clear_invalid_jump_sets(df, invalid_jump_sets):
    for invalid_jump_set in invalid_jump_sets:
        # set the release and landing times to false for both
        df.loc[df['Timestamp'] == invalid_jump_set['release_time'], 'is_release'] = False
        df.loc[df['Timestamp'] == invalid_jump_set['landing_time'], 'is_landing'] = False
        # set the in_flight to false for the jump set
        df.loc[(df['Timestamp'] >= invalid_jump_set['release_time']) & (df['Timestamp'] <= invalid_jump_set['landing_time']), 'in_flight'] = False
    return df

def clean_df(df, jump_sets):
    # drop sensors
    min_avg = 1000
    for i in range(NUM_SENSORS):
        col = f'Sensor_{i+1}'
        avg = df[col].mean()
        if -DROP_SENSORS_THRESHOLD < avg < DROP_SENSORS_THRESHOLD:
            df[col] = 0
        if avg < min_avg:
            min_avg = avg

    valid_jump_sets, invalid_jump_sets = find_valid_jump_set(df, jump_sets, SUM_THRESHOLD)
    df = clear_invalid_jump_sets(df, invalid_jump_sets)
    return df, valid_jump_sets

def find_jump_cycles(df, valid_jump_sets):
    # find the jump cycles
    jump_cycles = []
    #find the closest midpoint between the release and landing
    for i, jump_set in enumerate(valid_jump_sets):
        release_time = pd.to_datetime(jump_set['release_time'])
        landing_time = pd.to_datetime(jump_set['landing_time'])
        midpoint = pd.to_datetime((release_time.value + landing_time.value) // 2)

        # If your DataFrame is timezone-aware (e.g., UTC):
        if hasattr(df['Timestamp'].iloc[0], 'tzinfo') and df['Timestamp'].iloc[0].tzinfo is not None:
            midpoint = midpoint.tz_localize('UTC')

        closest_timestamp = df[df['Timestamp'] > midpoint]['Timestamp'].min()
        # take 90 indices before and after the closest timestamp
        middle_idx = df.index[df['Timestamp'] == closest_timestamp][0]
        window_size = HALF_WINDOW_SIZE  # or whatever number of rows you want
        start_idx = max(0, middle_idx - window_size)
        end_idx = min(len(df) - 1, middle_idx + window_size)

        jump_cycle = df.iloc[start_idx:end_idx+1]

        # transpose the jump cycle
        jump_cycle = jump_cycle.T

        # add a column at the front with the jump number
        jump_cycle.insert(0, 'Jump_Number', i+1)

        # RENAME COLUMNS
        n_steps = jump_cycle.shape[1] - 1  # exclude 'Jump_Number'
        jump_cycle.columns = ['Jump_Number'] + [f'time_step_{j+1}' for j in range(n_steps)]

        jump_cycles.append(jump_cycle)
    return jump_cycles

def format_jumps_csv(file_path, jump_cycles: list[pd.DataFrame]):
    # export the summed one first
    # create file name
    file_name = file_path.split('/')[-1]
    file_name = file_name.split('.')[0]
    summed_file_path = os.path.join(os.path.dirname(file_path), file_name + '_jumps_summed.csv')
    all_sensors_file_path = os.path.join(os.path.dirname(file_path), file_name + '_all_sensors.csv')

    summed_jump_cycles = [cycle.loc["sum", :] for cycle in jump_cycles]
    summed_jump_cycles_df = pd.DataFrame(summed_jump_cycles)
    summed_jump_cycles_df.to_csv(summed_file_path, index=False)

    # export the separate sensors too
    # pd.concat(jump_cycles, axis=0).to_csv(all_sensors_file_path, index=False)
    return summed_jump_cycles_df

def plot_jump_cycles(summed_jump_cycles_df: pd.DataFrame):
    print("========== PLOTTING JUMP CYCLES ==========")
    print(summed_jump_cycles_df.head())
    print("========== SHAPE ==========")
    print(summed_jump_cycles_df.shape)
    print("========== COLUMNS ==========")
    print(summed_jump_cycles_df.columns)
    # plot each of the jump cycles in a grid
    n_rows = 3
    n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten()
    time_cols = [col for col in summed_jump_cycles_df.columns if col.startswith('time_step_')]

    for i, ax in enumerate(axes):
        if i < len(summed_jump_cycles_df):
            ax.plot(time_cols, summed_jump_cycles_df.iloc[i][time_cols])
            ax.set_title(f'Jump Cycle {i+1}')
        else:
            ax.axis('off')  # Hide unused subplots

    plt.tight_layout()
    plt.show()

def store_df_to_csv(df_idk, file_path):
    # store the df to a csv file
    df = pd.DataFrame(df_idk)
    print("REACHING HERE: ", type(df))
    df.to_csv(file_path, index=False)
    return


def plot_one_sensor(df, sensor_name, peaks=None, jump_cycles=None):
    # plot the column with the release and landing indices
    plt.plot(df['Timestamp'], df[sensor_name])

    # plot the release and landing indices using boolean indexing
    release_data = df[df['is_release']]
    landing_data = df[df['is_landing']]
    
    print(release_data)
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


def convert_and_plot(file_path):
    # process and clean
    df, jump_sets = convert_txt_to_df(file_path)
    cleaned_df, valid_jump_sets = clean_df(df.copy(), jump_sets)
    df = pool_df(cleaned_df, pool_type='sum')

    # find jump cycles
    jump_cycles = find_jump_cycles(df, valid_jump_sets)
    summed_jump_cycles_df = format_jumps_csv(file_path, jump_cycles)

    # plot the jump cycles
    plot_jump_cycles(summed_jump_cycles_df)

    # plot the one sensor
    # plot_one_sensor(df, 'sum', jump_cycles=jump_cycles)



# delete files that have _filtered in the name
for file_path in glob.glob("data/*x"):
    os.remove(file_path)

# plot the one sensor
#df_annie, valid_jump_sets = clean_df(df_annie, jump_sets)
#print(valid_jump_sets)
#jump_cycles = find_jump_cycles(df_annie, valid_jump_sets)
# with open('data/AnnieGu_jumps_summed.csv', 'w') as file:
#     for cycle in jump_cycles:
#         file.write(cycle.to_csv(index=False))
# summed_jump_cycles_df = format_jumps_csv('data/NoahJung.txt', jump_cycles)
#plot_jump_cycles(summed_jump_cycles_df)

# df_colin, jump_sets = convert_txt_to_df('data/ColinSloan.txt')
# print_mean_of_sensors(df_colin)
# cleaned_df_colin = clean_df(df_colin.copy())
# print_mean_of_sensors(cleaned_df_colin)
# 
# df_colin = pool_df(df_colin, pool_type='sum')
# cleaned_df_colin = pool_df(cleaned_df_colin, pool_type='sum')
# 
# plot_one_sensor(df_colin, 'sum')
# plot_one_sensor(cleaned_df_colin, 'sum')


# 
# df_caroline, jump_sets = convert_txt_to_df('data/Caroline.txt')
# df_caroline = clean_df(df_caroline)
# df_caroline = pool_df(df_caroline, pool_type='sum')
# store_df_to_csv(df_caroline, 'data/Caroline.csv')
# plot_one_sensor(df_caroline, 'sum')
# 
# df_chu, jump_sets = convert_txt_to_df('data/Chu.txt')
# df_chu = clean_df(df_chu)
# df_chu = pool_df(df_chu, pool_type='sum')
# store_df_to_csv(df_chu, 'data/Chu.csv')
# plot_one_sensor(df_chu, 'sum')
