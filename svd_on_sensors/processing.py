import numpy as np
import pandas as pd

# Path to your file
file_path = 'collin_matrix.csv'

valid_rows = []

with open(file_path, 'r') as f:
    for line in f:
        # Remove BOM (byte order mark) if present
        line = line.lstrip('\ufeff')

        # Split the line by comma
        parts = line.strip().split(',')

        # Skip 'LOG' and timestamp, keep the rest
        readings = parts[2:]

        # Remove empty strings and convert to float
        numbers = []
        for x in readings:
            if x.strip() != '':
                try:
                    numbers.append(float(x))
                except ValueError:
                    pass  # Ignore non-numeric junk

        # Only keep rows with at least 80 numbers
        if len(numbers) >= 80:
            valid_rows.append(numbers[:80])  # Take exactly 80 numbers

# Turn into a matrix
matrix = np.array(valid_rows)
print(matrix)

# Optional: Save the result
# matrix.to_csv('extracted_matrix.csv', index=False)
