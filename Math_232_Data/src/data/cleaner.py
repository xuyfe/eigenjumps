import pandas as pd
from typing import List

VALID_MIN_FLIGHT_TIME = 0.3
VALID_MAX_FLIGHT_TIME = 1.0

NUM_SENSORS = 80

DROP_SENSORS_THRESHOLD = 0.3
MIN_JUMP_SEPARATION_MS = 2000 #2 seconds

COSINE_SIMILARITY_THRESHOLD = 0.96
MIN_SUM_THRESHOLD = 20
MAX_SUM_THRESHOLD = 200
HALF_WINDOW_SIZE = 85

POOL_TYPES = ["sum", "mean", "median"]

class DataCleaner:
    def __init__(self, df: pd.DataFrame, jump_sets: List[dict], pool_type: str = "sum"):
        """
        REMEMBER: each "clean" function responsible for modifying:
        - the df
        - the jump_sets
        """
        self.df = df.copy()  # Create a copy to prevent modifying the original DataFrame
        self.jump_sets = jump_sets
        self.pool_type = pool_type if pool_type in POOL_TYPES else "sum"

    def clean_all(self):
        self.clean_inactive_sensors()
        self.clean_invalid_flight_times()
        self.clean_double_jumps()
        self.clean_invalid_pressures()
        return

    def clean_inactive_sensors(self):
        sensor_cols = self.df.columns[8:]
        df = self.df
        for i in range(NUM_SENSORS):
            col = f'Sensor_{i+1}'
            avg = df[col].mean()
            if -DROP_SENSORS_THRESHOLD < avg < DROP_SENSORS_THRESHOLD:
                df[col] = 0
        if self.pool_type == "sum":
            # add a "cleaned_sum" column
            df['cleaned_sum'] = df[sensor_cols].sum(axis=1)
        elif self.pool_type == "mean":
            # add a "cleaned_mean" column
            df['cleaned_mean'] = df[sensor_cols].mean(axis=1)
        elif self.pool_type == "median":
            # add a "cleaned_median" column
            df['cleaned_median'] = df[sensor_cols].median(axis=1)
        self.df = df
        return
    
    def clean_double_jumps(self):
        """
        Clean double jumps by checking for minimum separation between jumps.        
        """
        jump_sets = self.jump_sets
        valid_jump_sets, invalid_jump_sets = [], []
        last_valid_landing_time = None

        for jump in jump_sets:
            # Get release and landing times
            release_time = pd.to_datetime(jump['release_time'])
            landing_time = pd.to_datetime(jump['landing_time'])

            # Check for minimum separation from last valid jump
            if last_valid_landing_time is not None and (release_time - last_valid_landing_time) < MIN_JUMP_SEPARATION_MS:
                invalid_jump_sets.append(jump)
                continue
            
            else:
                valid_jump_sets.append(jump)
                last_valid_landing_time = landing_time
        
        self.update_df_and_jump_sets(valid_jump_sets, invalid_jump_sets)
        return
    
    def clean_invalid_pressures(self):
        """
        Delete jumps whose sum is above or below the max/min threshold
        """
        valid_jump_sets, invalid_jump_sets = [], []
        for jump_set in self.jump_sets:
            release_time = pd.to_datetime(jump_set['release_time'])
            landing_time = pd.to_datetime(jump_set['landing_time'])

            # Get the window of the jump set
            window_df = self.df[(self.df['Timestamp'] >= release_time) & (self.df['Timestamp'] <= landing_time)]
            
            # Check if the sum is above or below the max/min threshold
            if (window_df['sum'] > MAX_SUM_THRESHOLD).any() or (window_df['sum'] < MIN_SUM_THRESHOLD).any():
                invalid_jump_sets.append(jump_set)
            else:
                valid_jump_sets.append(jump_set)
        
        self.update_df_and_jump_sets(valid_jump_sets, invalid_jump_sets)
        return

    def clean_invalid_flight_times(self):
        """
        Delete jumps whose flight time is below the min or above the max threshold
        """
        valid_jump_sets, invalid_jump_sets = [], []
        for jump_set in self.jump_sets:
            flight_time = jump_set['flight_time']

            # Check if the flight time is below the min or above the max threshold
            if flight_time < VALID_MIN_FLIGHT_TIME or flight_time > VALID_MAX_FLIGHT_TIME:
                invalid_jump_sets.append(jump_set)
            else:
                valid_jump_sets.append(jump_set)
        
        self.update_df_and_jump_sets(valid_jump_sets, invalid_jump_sets)
        return
    
    def update_df_and_jump_sets(self, valid_jump_sets: List[dict], invalid_jump_sets: List[dict]):
        """
        Takes invalid jump sets, unmarks them in the df and deletes them from jump_sets
        """
        df = self.df

        # UPDATE THE DF
        for invalid_jump_set in invalid_jump_sets:
            # Set the release and landing times to false for both
            df.loc[df['Timestamp'] == invalid_jump_set['release_time'], 'is_release'] = False
            df.loc[df['Timestamp'] == invalid_jump_set['landing_time'], 'is_landing'] = False
            
            # Set the in_flight to false for the jump set
            df.loc[(df['Timestamp'] >= invalid_jump_set['release_time']) & (df['Timestamp'] <= invalid_jump_set['landing_time']), 'in_flight'] = False
            
            # Set the jump_index to -1 for the jump set
            df.loc[(df['Timestamp'] >= invalid_jump_set['release_time']) & (df['Timestamp'] <= invalid_jump_set['landing_time']), 'jump_index'] = -1
        self.df = df

        # UPDATE THE JUMP SETS
        self.jump_sets = valid_jump_sets

        return
    
