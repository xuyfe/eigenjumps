import pytest
import pandas as pd
from src import DataLoader, DataCleaner
import os

# Test data path
TEST_DATA_PATH = "/Users/srwang/Documents/MATH232/Math-232-Project/Math_232_Data/data/AnnieGu.txt"

# Constants
MIN_JUMP_SEPARATION_MS = 2000
VALID_MIN_FLIGHT_TIME = 0
VALID_MAX_FLIGHT_TIME = 10000
MAX_SUM_THRESHOLD = 200
MIN_SUM_THRESHOLD = 20

@pytest.fixture
def data_loader():
    """Fixture to create a DataLoader instance for testing"""
    return DataLoader(TEST_DATA_PATH)

@pytest.fixture
def data_cleaner(data_loader):
    """Fixture to create a DataCleaner instance for testing"""
    return DataCleaner(data_loader.df, data_loader.jump_sets)

class TestDataLoader:
    def test_loader_initialization(self, data_loader):
        """Test that DataLoader initializes correctly"""
        assert data_loader.df is not None
        assert data_loader.jump_sets is not None
        assert len(data_loader.jump_sets) > 0
        assert len(data_loader.df) > 0

    def test_dataframe_structure(self, data_loader):
        """Test that the dataframe has the correct structure"""
        df = data_loader.df
        # Check required columns exist
        required_columns = ['Timestamp', 'is_release', 'is_landing', 'in_flight', 'jump_index']
        for col in required_columns:
            assert col in df.columns
        
        # Check sensor columns exist
        assert all(f'Sensor_{i+1}' in df.columns for i in range(80))
        
        # Check data types
        assert pd.api.types.is_datetime64_any_dtype(df['Timestamp'])
        assert pd.api.types.is_bool_dtype(df['is_release'])
        assert pd.api.types.is_bool_dtype(df['is_landing'])
        assert pd.api.types.is_bool_dtype(df['in_flight'])
        assert pd.api.types.is_integer_dtype(df['jump_index'])

    def test_jump_sets_structure(self, data_loader):
        """Test that jump sets have the correct structure"""
        for jump in data_loader.jump_sets:
            assert 'release_time' in jump
            assert 'landing_time' in jump
            assert 'flight_time' in jump
            assert 'estimated_vertical' in jump
            
            # Check that times are valid
            assert pd.to_datetime(jump['release_time']) < pd.to_datetime(jump['landing_time'])
            assert isinstance(jump['flight_time'], (int, float))
            assert isinstance(jump['estimated_vertical'], (int, float))

class TestDataCleaner:
    def test_cleaner_initialization(self, data_cleaner):
        """Test that DataCleaner initializes correctly"""
        assert data_cleaner.df is not None
        assert data_cleaner.jump_sets is not None
        assert len(data_cleaner.jump_sets) > 0

    def test_clean_all(self, data_cleaner):
        """Test the complete cleaning process"""
        original_df_len = len(data_cleaner.df)
        original_jump_sets_len = len(data_cleaner.jump_sets)
        
        data_cleaner.clean_all()
        
        # Verify that cleaning didn't completely destroy the data
        assert len(data_cleaner.df) > 0
        assert len(data_cleaner.jump_sets) > 0
        
        # Verify that some data was cleaned (either rows or jumps were removed)
        assert len(data_cleaner.df) <= original_df_len
        assert len(data_cleaner.jump_sets) <= original_jump_sets_len

    def test_clean_inactive_sensors(self, data_cleaner):
        """Test that ONLY inactive sensors are cleared and no other cleaning occurs"""
        # Store original state
        original_df = data_cleaner.df.copy()
        original_jump_sets = data_cleaner.jump_sets.copy()
        
        # Get the sensor columns
        sensor_cols = [col for col in original_df.columns if col.startswith('Sensor_')]
        
        # Calculate which sensors should be inactive (average between -0.3 and 0.3)
        inactive_sensors = []
        for col in sensor_cols:
            avg = original_df[col].mean()
            if -0.3 < avg < 0.3:  # Using the DROP_SENSORS_THRESHOLD from cleaner.py
                inactive_sensors.append(col)
        
        # Run only the inactive sensors cleaning
        data_cleaner.clean_inactive_sensors()
        
        # 1. Verify that ONLY inactive sensors were zeroed out
        for col in sensor_cols:
            if col in inactive_sensors:
                # Inactive sensors should be zeroed
                assert (data_cleaner.df[col] == 0).all(), f"Active sensor {col} was incorrectly zeroed"
            else:
                # Active sensors should remain unchanged
                assert data_cleaner.df[col].equals(original_df[col]), f"Inactive sensor {col} was not zeroed"
        
        # 2. Verify that jump_sets were not modified
        assert data_cleaner.jump_sets == original_jump_sets, "Jump sets were unexpectedly modified"
        
        # 3. Verify that no other columns were added except the expected cleaned_sum/mean/median
        expected_new_columns = {
            'sum': 'cleaned_sum',
            'mean': 'cleaned_mean',
            'median': 'cleaned_median'
        }
        new_columns = set(data_cleaner.df.columns) - set(original_df.columns)
        assert len(new_columns) == 1, f"Unexpected number of new columns: {new_columns}"
        assert new_columns.pop() == expected_new_columns[data_cleaner.pool_type], \
            f"Unexpected new column. Expected {expected_new_columns[data_cleaner.pool_type]}"
        
        # 4. Verify that non-sensor columns remain unchanged
        non_sensor_cols = [col for col in original_df.columns if not col.startswith('Sensor_')]
        for col in non_sensor_cols:
            assert data_cleaner.df[col].equals(original_df[col]), \
                f"Non-sensor column {col} was unexpectedly modified"
        
        # 5. Verify that the cleaned sum/mean/median is calculated correctly
        if data_cleaner.pool_type == "sum":
            cleaned_col = 'cleaned_sum'
            expected_values = data_cleaner.df[sensor_cols].sum(axis=1)
        elif data_cleaner.pool_type == "mean":
            cleaned_col = 'cleaned_mean'
            expected_values = data_cleaner.df[sensor_cols].mean(axis=1)
        elif data_cleaner.pool_type == "median":
            cleaned_col = 'cleaned_median'
            expected_values = data_cleaner.df[sensor_cols].median(axis=1)
        
        assert data_cleaner.df[cleaned_col].equals(expected_values), \
            f"Cleaned {data_cleaner.pool_type} values are incorrect"

    def test_clean_invalid_flight_times(self, data_cleaner):
        """Test cleaning of invalid flight times"""
        original_jump_sets = data_cleaner.jump_sets.copy()
        data_cleaner.clean_invalid_flight_times()
        
        # Verify that invalid flight times were removed
        for jump in data_cleaner.jump_sets:
            assert VALID_MIN_FLIGHT_TIME <= jump['flight_time'] <= VALID_MAX_FLIGHT_TIME

    def test_clean_double_jumps(self, data_cleaner):
        """Test cleaning of double jumps"""
        original_jump_sets = data_cleaner.jump_sets.copy()
        data_cleaner.clean_double_jumps()
        
        # Verify that jumps are properly separated
        for i in range(len(data_cleaner.jump_sets) - 1):
            current_landing = pd.to_datetime(data_cleaner.jump_sets[i]['landing_time'])
            next_release = pd.to_datetime(data_cleaner.jump_sets[i + 1]['release_time'])
            time_diff = (next_release - current_landing).total_seconds() * 1000  # Convert to milliseconds
            assert time_diff >= MIN_JUMP_SEPARATION_MS

    def test_clean_invalid_pressures(self, data_cleaner):
        """Test cleaning of invalid pressures"""
        original_jump_sets = data_cleaner.jump_sets.copy()
        data_cleaner.clean_invalid_pressures()
        
        # Verify that remaining jumps have valid pressure ranges
        for jump in data_cleaner.jump_sets:
            release_time = pd.to_datetime(jump['release_time'])
            landing_time = pd.to_datetime(jump['landing_time'])
            window_df = data_cleaner.df[(data_cleaner.df['Timestamp'] >= release_time) & 
                                      (data_cleaner.df['Timestamp'] <= landing_time)]
            assert not (window_df['sum'] > MAX_SUM_THRESHOLD).any()
            assert not (window_df['sum'] < MIN_SUM_THRESHOLD).any()
