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
    loader = DataLoader(ANNIE_PATH)
    cleaner = DataCleaner(loader.df, loader.jump_sets)
    cleaner.clean_all()
    print(cleaner.df.head())
    print(cleaner.jump_sets)

if __name__ == "__main__":
    test_loader()