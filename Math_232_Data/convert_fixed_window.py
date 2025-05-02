import pandas as pd
import re
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import glob
import os
from sklearn.metrics.pairwise import cosine_similarity

# CONSTANTS
VALID_MIN_FLIGHT_TIME = 0.3
VALID_MAX_FLIGHT_TIME = 0.82
NUM_SENSORS = 80
DROP_SENSORS_THRESHOLD = 0.3
COSINE_SIMILARITY_THRESHOLD = 0.96
SUM_THRESHOLD = 20
HALF_WINDOW_SIZE = 85
HALF_WINDOW_SIZE = 85


# FUNCTIONS
def pool_df(df, pool_type='sum'):
    # sum across all sensors
    if pool_type == 'sum':
        df['sum'] = df.iloc[:, 5:].sum(axis=1)
        df['sum_top'] = df.iloc[:, 5:45].sum(axis=1)
        df['sum_bottom'] = df.iloc[:, 45:].sum(axis=1)
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
                #print("RELEASE: ", to_amend[0])
                current_jump_set["release_time"] = to_amend[0]
                to_amend[1] = True
                to_amend[2] = False
            elif "Landing" in line and "release_time" in current_jump_set:
                #print("LANDING: ", to_amend[0])
                current_jump_set["landing_time"] = to_amend[0]
                # search for flight time in the next line
                flight_time_found = False
                for offset in range(1, 4):  # Look ahead up to 3 lines
                    if i + offset < len(sliced_lines):
                        #(f"Checking line {i+offset}: {sliced_lines[i+offset].strip()}")
                        flight_time_match = re.search(flight_time_pattern, sliced_lines[i + offset])
                        if flight_time_match:
                            #print("Found flight time:", flight_time_match.group(1))
                            current_jump_set["flight_time"] = float(flight_time_match.group(1))
                            flight_time_found = True
                            break
                if not flight_time_found:
                    #print("No flight time found for this landing.")
                    pass
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


def find_valid_jump_set(df, jump_sets, sum_threshold, min_separation=60):
    valid_jump_sets, invalid_jump_sets = [], []
    last_valid_landing_idx = None

    for jump_set in jump_sets:
        # Get the time window for this jump
        release_time = pd.to_datetime(jump_set['release_time'])
        landing_time = pd.to_datetime(jump_set['landing_time'])
        # Filter the DataFrame for this window
        window_df = df[(df['Timestamp'] >= release_time) & (df['Timestamp'] <= landing_time)]

        # Find the index of the release in the DataFrame
        release_idx = df.index[df['Timestamp'] == release_time][0]

        # Check for minimum separation from last valid jump
        if last_valid_landing_idx is not None and (release_idx - last_valid_landing_idx) < min_separation:
            invalid_jump_sets.append(jump_set)
            continue

        # Check if any value in 'sum' is below the threshold
        if (window_df['sum'] < sum_threshold).any():
            valid_jump_sets.append(jump_set)
            # Update last_valid_landing_idx
            landing_idx = df.index[df['Timestamp'] == landing_time][0]
            last_valid_landing_idx = landing_idx
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


def filter_highest_pairwise_similar_jump_cycles(summed_jump_cycles_df, similarity_threshold=0.9):
    # 1. Extract only the time_step columns
    time_cols = [col for col in summed_jump_cycles_df.columns if col.startswith('time_step_')]
    data = summed_jump_cycles_df[time_cols].values

    # 2. Compute pairwise cosine similarity matrix
    sim_matrix = cosine_similarity(data)

    # 3. For each jump, find the highest similarity to any other jump (excluding self)
    np.fill_diagonal(sim_matrix, -np.inf)  # Exclude self-comparison
    max_similarities = sim_matrix.max(axis=1)

    # 4. Keep only those with max similarity above the threshold
    keep_mask = max_similarities >= similarity_threshold
    filtered_df = summed_jump_cycles_df[keep_mask].reset_index(drop=True)

    print("========== MAX PAIRWISE SIMILARITIES ==========")
    print(max_similarities)
    print(f"Kept {keep_mask.sum()} out of {len(summed_jump_cycles_df)} jump cycles with max pairwise similarity >= {similarity_threshold}")
    return filtered_df, max_similarities

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
#
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
        window_size = HALF_WINDOW_SIZE # or whatever number of rows you want
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
    summed_file_path_top = "../Math_232_Data/jump_data_clean/" + file_name + '_jumps_summed_top.csv'
    summed_file_path_bottom = "../Math_232_Data/jump_data_clean/" + file_name + '_jumps_summed_bottom.csv'
    # all_sensors_file_path = os.path.join(os.path.dirname(file_path), file_name + '_all_sensors.csv')

    summed_jump_cycles_top = [cycle.loc["sum_top", :] for cycle in jump_cycles]
    summed_jump_cycles_bottom = [cycle.loc["sum_bottom", :] for cycle in jump_cycles]
    summed_jump_cycles_df_top = pd.DataFrame(summed_jump_cycles_top)
    summed_jump_cycles_df_bottom = pd.DataFrame(summed_jump_cycles_bottom)
    summed_jump_cycles_df_top.to_csv(summed_file_path_top, index=False)
    summed_jump_cycles_df_bottom.to_csv(summed_file_path_bottom, index=False)

    # export the separate sensors too
    # pd.concat(jump_cycles, axis=0).to_csv(all_sensors_file_path, index=False)
    return summed_jump_cycles_df_top, summed_jump_cycles_df_bottom

def plot_jump_cycles(summed_jump_cycles_df: pd.DataFrame, name=None, separate=False):
    # print("========== PLOTTING JUMP CYCLES ==========")
    # print(summed_jump_cycles_df.head())
    # print("========== SHAPE ==========")
    # print(summed_jump_cycles_df.shape)
    # print("========== COLUMNS ==========")
    # print(summed_jump_cycles_df.columns)
    time_cols = [col for col in summed_jump_cycles_df.columns if col.startswith('time_step_')]

    if separate:
        n_cycles = len(summed_jump_cycles_df)
        n_cols = 3
        n_rows = (n_cycles + n_cols - 1) // n_cols  # Ceiling division

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten()

        for i, ax in enumerate(axes):
            if i < n_cycles:
                ax.plot(time_cols, summed_jump_cycles_df.iloc[i][time_cols])
                ax.set_title(f'Jump Cycle {i+1}')
            else:
                ax.axis('off')  # Hide unused subplots
        plt.tight_layout()
    else:
        plt.figure(figsize=(12, 6))
        for i, row in summed_jump_cycles_df.iterrows():
            plt.plot(time_cols, row[time_cols], label=f'Jump Cycle {i+1}')
            plt.xlabel("")
            plt.xticks([])
        # plt.title("Summed Jump Cycles")
        plt.xlabel("Time Step")
        plt.ylabel("Summed Value")
        plt.legend()
        plt.tight_layout()

    if name is not None:
        plt.suptitle(name)
    plt.show()

def store_df_to_csv(df_idk, file_path):
    # store the df to a csv file
    df = pd.DataFrame(df_idk)
    #("REACHING HERE: ", type(df))
    df.to_csv(file_path, index=False)
    return


def plot_one_sensor(df, sensor_name, peaks=None, jump_cycles=None):
    # plot the column with the release and landing indices
    plt.plot(df['Timestamp'], df[sensor_name])

    # plot the release and landing indices using boolean indexing
    release_data = df[df['is_release']]
    landing_data = df[df['is_landing']]
    
    # print(release_data)
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

def convert(file_path):
    # process and clean
    df, jump_sets = convert_txt_to_df(file_path)
    cleaned_df, valid_jump_sets = clean_df(df.copy(), jump_sets)
    df = pool_df(cleaned_df, pool_type='sum')

    # find jump cycles
    jump_cycles = find_jump_cycles(df, valid_jump_sets)
    summed_jump_cycles_df_top, summed_jump_cycles_df_bottom = format_jumps_csv(file_path, jump_cycles)
    return summed_jump_cycles_df_top, summed_jump_cycles_df_bottom

def save_filtered_df(df, file_path):
    # save the filtered df to a csv file
    df.to_csv(file_path, index=False)
    return

def convert_and_plot(file_path):
    summed_jump_cycles_df_top, summed_jump_cycles_df_bottom = convert(file_path)
    filtered_df_top, max_similarities_top = filter_highest_pairwise_similar_jump_cycles(summed_jump_cycles_df_top, similarity_threshold=COSINE_SIMILARITY_THRESHOLD)
    filtered_df_bottom, max_similarities_bottom = filter_highest_pairwise_similar_jump_cycles(summed_jump_cycles_df_bottom, similarity_threshold=COSINE_SIMILARITY_THRESHOLD)
    # plot the jump cycles
    # plot_jump_cycles(filtered_df_top, name=file_path.split('/')[-1])
    # plot_jump_cycles(filtered_df_bottom, name=file_path.split('/')[-1])
    # save the filtered df to a csv file
    save_filtered_df(filtered_df_top, file_path.replace(".txt", "_filtered_top.csv"))
    save_filtered_df(filtered_df_bottom, file_path.replace(".txt", "_filtered_bottom.csv"))
    # save the max similarities to a txt file
    with open(file_path.replace(".txt", "_max_similarities_top.txt"), "w") as f:
        f.write(str(max_similarities_top))
    with open(file_path.replace(".txt", "_max_similarities_bottom.txt"), "w") as f:
        f.write(str(max_similarities_bottom))
    return


# Example usage:
# summed_jump_cycles_df = convert("data/Garrett.txt")
# filtered_df, max_similarities = filter_highest_pairwise_similar_jump_cycles(summed_jump_cycles_df, similarity_threshold=0.95)
# save_filtered_df(filtered_df, "data/Garrett_filtered.csv")
# plot_jump_cycles(filtered_df)
