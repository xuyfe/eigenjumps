import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

# Normalize matrix (optional depending if you want mean-centered)
total_sum = np.sum(A)
N = A - total_sum / A.size

# Compute SVD
U, S, VT = np.linalg.svd(N, full_matrices=False)

# Low-rank reconstructions
# Rebuild using only first k singular values/vectors

def rank_k_approx(U, S, VT, k):
    """Compute rank-k approximation."""
    return U[:, :k] @ np.diag(S[:k]) @ VT[:k, :]

# Now generate several approximations
ranks = [1, 2, 5, 10]  # You can add more ranks here!

reconstructed_images = [rank_k_approx(U, S, VT, k) for k in ranks]

# Plot original and compressed images
fig, axs = plt.subplots(1, len(reconstructed_images) + 1, figsize=(15, 5))

# Plot original
axs[0].imshow(A, cmap='gray')
axs[0].set_title('Original Image')
axs[0].axis('off')

# Plot reconstructions
for idx, (img, rank) in enumerate(zip(reconstructed_images, ranks)):
    axs[idx + 1].imshow(img, cmap='gray')
    axs[idx + 1].set_title(f'Rank-{rank} Approximation')
    axs[idx + 1].axis('off')

plt.tight_layout()
plt.show()
