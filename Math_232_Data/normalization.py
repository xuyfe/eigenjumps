import pandas as pd
import numpy as np
from convert_fixed_window import plot_jump_cycles
from scipy.fft import fft
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
FILE_PATH = "data/NoahJung_filtered.csv"
df = pd.read_csv(FILE_PATH)
#plot_jump_cycles(df)

print(df.head())

def return_all_vectors(df):
    vectors = []
    for row in df.iterrows():
        vector = row.to_numpy()
        vectors.append(vector)
    return vectors

    
def get_time_step_vectors(df):
    # Select only columns that start with 'time_step_'
    time_cols = [col for col in df.columns if col.startswith('time_step_')]
    # Extract as a 2D numpy array (each row is a vector)
    vectors = df[time_cols].to_numpy()
    return vectors

# Example usage:
# vectors = get_time_step_vectors(summed_jump_cycles_df)

    

def kmeans_cluster_vectors(vectors, n_clusters):
    """
    Groups vectors into n_clusters using K-Means clustering.

    Parameters:
        vectors (array-like): 2D array or DataFrame of shape (n_samples, n_features)
        n_clusters (int): Number of clusters to form

    Returns:
        labels (np.ndarray): Cluster label for each vector
        kmeans (KMeans): Fitted KMeans object (for centroids, etc.)
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(vectors)

    return labels, kmeans


def fft_features(jump):
    jump = np.array(jump)

    fft_coeffs = np.abs(fft(jump))  # full spectrum: length 170
    fft_features = fft_coeffs[:20]  # keep first 20 (low-frequency components)
    return fft_features

# Example usage:
# vectors = np.array([[...], [...], ...])  # Your data here
# labels, kmeans = kmeans_cluster_vectors(vectors, n_clusters=3)

def plot_clusters(vectors, labels, kmeans):
    # plot the clusters
    plt.scatter(vectors[:, 0], vectors[:, 1], c=labels)
    # plot the centroids
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', s=100, alpha=0.7)
    plt.show()


people = ["NoahJung", "SkylarWang", "Owen", "ConnorFlood", "AnnieGu", "Charles", "Chu", "Caroline"]
num_clusters = len(people)

cumulative_vectors = []
for person in people:
    df = pd.read_csv(f"data/{person}_filtered.csv")
    vectors = get_time_step_vectors(df)
    fft_vectors = [fft_features(vector) for vector in vectors]
    cumulative_vectors.extend(fft_vectors)

print(len(cumulative_vectors))

cumulative_vectors = np.array(cumulative_vectors)
labels, kmeans = kmeans_cluster_vectors(cumulative_vectors, n_clusters=num_clusters)
plot_clusters(cumulative_vectors, labels, kmeans)

    # do a k-nearest neighbors
