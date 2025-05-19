from src import DataLoader
from src import DataCleaner

ANNIE_PATH = "/Users/srwang/Documents/MATH232/Math-232-Project/Math_232_Data/data/AnnieGu.txt"

def test_loader():
    loader = DataLoader(ANNIE_PATH)
    print(loader.df.head())
    print(loader.jump_sets)

    assert loader.df is not None
    assert loader.jump_sets is not None
    assert len(loader.jump_sets) > 0
    assert len(loader.df) > 0

def test_cleaner():
    raw_data = DataLoader(ANNIE_PATH)
    cleaned_data = DataCleaner(raw_data.df, raw_data.jump_sets)
    cleaned_data.clean_all()
    print(cleaned_data.df.head())
    print(cleaned_data.jump_sets)
    # save to csv
    cleaner.df.to_csv(loader.csv_path, index=False)

if __name__ == "__main__":
    test_cleaner()