import pytest
import pandas as pd
from src import DataLoader, DataCleaner

# Test data path
TEST_DATA_PATH = "/Users/srwang/Documents/MATH232/Math-232-Project/Math_232_Data/data/AnnieGu.txt"

@pytest.fixture
def data_loader():
    """Creates a DataLoader instance for testing"""
    return DataLoader(TEST_DATA_PATH)

@pytest.fixture
def data_cleaner(data_loader):
    """Creates a DataCleaner instance for testing"""
    return DataCleaner(data_loader.df.copy(), data_loader.jump_sets.copy())

def test_loader_basic(data_loader):
    """Basic test to check if DataLoader loads data correctly"""
    # Check if data was loaded
    assert data_loader.df is not None
    assert data_loader.jump_sets is not None
    
    # Check if we have some data
    assert len(data_loader.df) > 0
    assert len(data_loader.jump_sets) > 0

def test_cleaner_basic(data_cleaner):
    """Basic test to check if DataCleaner works"""
    # Store original data
    original_df_len = len(data_cleaner.df)
    original_jump_sets_len = len(data_cleaner.jump_sets)
    
    # Clean the data
    data_cleaner.clean_invalid_flight_times()
    

    # save the cleaned data
    data_cleaner.df.to_csv("cleaned_data_1.csv", index=False)
    
    # Check if we still have data after cleaning
    assert len(data_cleaner.df) > 0
    assert len(data_cleaner.jump_sets) > 0
