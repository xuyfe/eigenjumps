import pandas as pd
from IPython.display import display

file_path = "Math_232_Data/AnnieGu.txt"

# Read the text file into a DataFrame
df = pd.read_csv(file_path, delimiter='\t')  # Assuming tab-delimited, adjust if needed

# Drop the first column
df = df.drop(df.columns[0], axis=1)

display(df)
