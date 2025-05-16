import pandas as pd
import re
import os
from typing import List, Tuple

# Data Loading Constants
MAX_FLIGHT_TIME_LOOKAHEAD = 3
NUM_SENSORS = 80

class DataLoader:
    def __init__(self, txt_path: str):
        self.txt_path = txt_path
        self.df, self.jump_sets = self.process_txt()
        self.error_path = os.path.join(os.path.dirname(txt_path), "error.txt")

    def process_txt(self) -> Tuple[pd.DataFrame, List[dict]]:
        '''
        Process the txt file into:
            1) Pandas dataframe
            2) List of jump sets (each jump set is a dictionary with the following keys:
                - release_time
                - landing_time
                - flight_time (can be None if not found)
                - estimated_vertical (can be None if not found)
        '''


        # Get the calibrated rows
        rows = self.get_calibrated_rows()
        if rows is None:
            self.log_error(f"No calibrated rows found for {self.txt_path}")
            return None, None

        # Initialize lists to store data
        dataframe_rows = []
        jump_sets = []
        current_jump = {}

        # Process each row
        for i, line in enumerate(rows):
            # Handle non-release/landing lines
            if not ("Release" in line or "Landing" in line):
                row = self.process_regular_row(line)
                if row:
                    dataframe_rows.append(row)
                else:
                    self.log_error(f"Error processing row {i}: {line}")

            # Handle release/landing lines
            else:
                previous_row = dataframe_rows.pop()
                if "Release" in line:
                    previous_row[1] = True # Set release flag to true
                    previous_row[2] = False # Set landing flag to false

                    # Track the release time
                    current_jump["release_time"] = previous_row[0]


                elif "Landing" in line and "release_time" in current_jump: # Sanity check: if you've landed you should have released lol
                    # Log landing flag
                    previous_row[1] = False
                    previous_row[2] = True

                    # Track the landing time
                    current_jump["landing_time"] = previous_row[0]

                    # Search for flight time in the next line
                    landing_end_index = min(i + MAX_FLIGHT_TIME_LOOKAHEAD, len(rows))
                    flight_time_match, estimated_vertical = self.process_data_rows(rows[i + 1:landing_end_index])
                    
                    # VALID FLIGHT TIME FOUND
                    if not flight_time_match:
                        self.log_error(f"No flight time found for landing at line {i}")
                    else:
                        current_jump["flight_time"] = float(flight_time_match)                    

                    # VALID ESTIMATED VERTICAL FOUND
                    if not estimated_vertical:
                        self.log_error(f"No estimated vertical found for landing at line {i}")
                    else:
                        current_jump["estimated_vertical"] = float(estimated_vertical)

                    # Append and reset the current jump
                    jump_sets.append(current_jump)
                    current_jump = {}

                dataframe_rows.append(previous_row)

        columns = ['Timestamp', 'is_release', 'is_landing', 'in_flight'] + [f'Sensor_{i+1}' for i in range(NUM_SENSORS)]
        df = pd.DataFrame(dataframe_rows, columns=columns)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])

        # add the "in_flight" column to the dataframe
        clean_df = self.add_in_flight_column(df, jump_sets)
        return clean_df, jump_sets
    
    def add_in_flight_column(self, df: pd.DataFrame, jump_sets: List[dict]) -> pd.DataFrame:
        for jump_set in jump_sets:
            release_time = pd.to_datetime(jump_set['release_time'])
            landing_time = pd.to_datetime(jump_set['landing_time'])
            jump_indices = df[(df['Timestamp'] >= release_time) & (df['Timestamp'] <= landing_time)].index
            df.loc[jump_indices, 'in_flight'] = True
        return df

    def process_regular_row(self, row: str) -> List[float] | None:
        LOG_PATTERN = r'LOG  \[(.*?)\] \[(.*?)\]'
        
        match = re.search(LOG_PATTERN, row)
        if match:
            timestamp = match.group(1)
            sensor_data_str = match.group(2)
            metadata = [timestamp, False, False, False]
            sensor_data = [float(x) for x in sensor_data_str.split(', ')]
            return metadata + sensor_data
        else:
            return None
            
    def process_data_rows(self, rows: List[str]) -> Tuple[float, float] | Tuple[None, None]:
        FLIGHT_TIME_PATTERN = r'[Ff]light [Tt]ime: ([\d.]+)'  # Make case insensitive
        ESTIMATED_VERTICAL_PATTERN = r'Estimated [Vv]ertical: ([\d.]+)'
        
        flight_time, estimated_vertical = None, None
        for row in rows:
            match = re.search(FLIGHT_TIME_PATTERN, row)
            if match:
                flight_time = float(match.group(1))
            
            else:
                match = re.search(ESTIMATED_VERTICAL_PATTERN, row)
                if match:
                    estimated_vertical = float(match.group(1))
        
        return flight_time, estimated_vertical


    def get_calibrated_rows(self) -> List[str] | None:
        with open(self.txt_path, 'r') as file:
            lines = file.readlines()

        calibration_index = 0

        # find the line with the calibration pattern
        for line in lines:
            if 'Calibration' in line:
                calibration_index = lines.index(line) + 5
                break
        
        if calibration_index == 0:
            return None
        
        else:
            return lines[calibration_index:]

    def log_error(self, message: str) -> None:
        with open(self.error_path, "a") as f:
            f.write(f"{message}\n")
        
        return
