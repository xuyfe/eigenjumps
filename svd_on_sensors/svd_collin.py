import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# Set numpy print options
np.set_printoptions(suppress=True, precision=2)

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
A_input = np.array(valid_rows)

# Work with the matrix directly
A = A_input.copy()

# Trim to one Jump
A = A[1300:1573, :]

# Normalize matrix (optional depending if you want mean-centered)
total_sum = np.sum(A)
N = A - total_sum / A.size


# Compute SVD
U, S, VT = np.linalg.svd(N, full_matrices=False)

# Plot heatmap with fixed scale
plt.figure(figsize=(12, 6))
img = plt.imshow(np.outer(U[:, 3], VT[3, :]),
                cmap='OrRd',
                interpolation='nearest')

plt.colorbar(img, label='Normalized Value', shrink=0.8)
plt.title("Normalized Matrix Heatmap (One Jump)")
plt.xlabel("Column Index")
plt.ylabel("Row Index (Trimmed: 1300-1573)")

# Optional: Add grid lines for clarity
plt.grid(True, color='gray', linestyle=':', linewidth=0.3, alpha=0.3)

plt.tight_layout()
plt.show()

print(f'Singular matrix: {S}')
# Low-rank reconstructions
# Rebuild using only first k singular values/vectors

def rank_k_approx(U, S, VT, k):
    """Compute rank-k approximation."""
    return np.outer(U[:, k - 1], VT[k - 1, :])
# min(U.shape[0], VT.shape[1])
ranks = [1, 2, 3, 4, 5]  # Full-Rank is min(m,n)
reconstructed_images = [rank_k_approx(U, S, VT, k) for k in ranks]
ranks.append(-1)
reconstructed_images.append(A)

# Plot settings
plt.figure(figsize=(12, 8))  # 2 rows, 3 columns

rows = [1345, 1382, 1406, 1468, 1502]
rows_offset = [45, 82, 106, 168, 202]
for i, (recon, k) in enumerate(zip(reconstructed_images, ranks), 1):
    first_row_reshaped = recon[45, :].reshape(8, 10)

    plt.subplot(2, 3, i)  # 2 rows, 3 columns
    img = plt.imshow(first_row_reshaped, cmap='OrRd', interpolation='nearest')
    plt.colorbar(img, fraction=0.046, pad=0.04)
    title = f"Rank-{k}" if k != ranks[-1] else "Full-Rank"
    plt.title(title)
    plt.xticks([])
    plt.yticks([])

plt.tight_layout()
plt.show()