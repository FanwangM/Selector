import numpy as np
import pandas as pd
from selector.methods.tests.common import generate_synthetic_data

# generate random data points belonging to one cluster - pairwise distance matrix
_, _, arr_dist = generate_synthetic_data(
    n_samples=100,
    n_features=2,
    n_clusters=1,
    pairwise_dist=True,
    metric="euclidean",
    random_state=42,
)

# generate random data points belonging to multiple clusters - class labels and pairwise distance matrix
_, class_labels_cluster, arr_dist_cluster = generate_synthetic_data(
    n_samples=100,
    n_features=2,
    n_clusters=3,
    pairwise_dist=True,
    metric="euclidean",
    random_state=42,
)

# Coordinates to be tested
coordinates = [[0, 0], [2, 0], [0, 2], [2, 2], [-10, -10]]

# Save to CSV
pd.DataFrame(arr_dist).to_csv('arr_dist.csv', index=False)
pd.DataFrame(arr_dist_cluster).to_csv('arr_dist_cluster.csv', index=False)
pd.DataFrame(class_labels_cluster, columns=['Label']).to_csv('class_labels_cluster.csv', index=False)
pd.DataFrame(coordinates, columns=['Feature1', 'Feature2']).to_csv('simple_coordinates.csv', index=False)

# Mocked clusters
np.random.seed(42)
cluster_one = np.random.normal(0, 1, (3, 2))
cluster_two = np.random.normal(10, 1, (6, 2))
cluster_three = np.random.normal(20, 1, (10, 2))
labels_mocked = np.hstack([[0 for _ in range(3)], [1 for _ in range(6)], [2 for _ in range(10)]])
mocked_cluster_coords = np.vstack([cluster_one, cluster_two, cluster_three])

# Save mocked cluster data
pd.DataFrame(mocked_cluster_coords, columns=['Feature1', 'Feature2']).to_csv('mocked_cluster_coords.csv', index=False)
pd.DataFrame(labels_mocked, columns=['Label']).to_csv('labels_mocked.csv', index=False)
