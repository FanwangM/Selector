import streamlit as st
import numpy as np
import pandas as pd

from sklearn.metrics import pairwise_distances
from selector.methods.distance import MaxMin

# Set page configuration
st.set_page_config(
    page_title="MaxMin",
    page_icon="assets/QC-Devs.png",
)

st.title("Brute Strength - MaxMin")
st.sidebar.header("Brute Strength - MaxMin")
st.sidebar.markdown(
    """
    MaxMin is possibly the most widely used method for dissimilarity-based
    compound selection. When presented with a dataset of samples, the
    initial point is chosen as the dataset's medoid center. Next, the second
    point is chosen to be that which is furthest from this initial point.
    Subsequently, all following points are selected via the following
    logic:

    1. Find the minimum distance from every point to the already-selected ones.
    2. Select the point which has the maximum distance among those calculated
       in the previous step.

    In the current implementation, this method requires or computes the full pairwise-distance
    matrix, so it is not recommended for large datasets.

    References
    ----------
    [1] Ashton, Mark, et al., Identification of diverse database subsets using
    property‐based and fragment‐based molecular descriptions, Quantitative
    Structure‐Activity Relationships 21.6 (2002): 598-604.
    """
)

# File uploader for feature matrix or distance matrix (required)
matrix_file = st.file_uploader("Upload a feature matrix or distance matrix", type=["csv", "xlsx", "npz", "npy"])

# Load data from matrix file
if matrix_file is not None:
    if matrix_file.name.endswith(".csv"):
        matrix = pd.read_csv(matrix_file).values
    elif matrix_file.name.endswith(".xlsx"):
        matrix = pd.read_excel(matrix_file).values
    elif matrix_file.name.endswith(".npz"):
        matrix = np.load(matrix_file)["arr_0"]
    elif matrix_file.name.endswith(".npy"):
        matrix = np.load(matrix_file)

    st.write("Matrix uploaded successfully!")

    # Input for number of points to select (required)
    num_points = st.number_input("Number of points to select", min_value=1, step=1)

    # Input for cluster label list (optional)
    label_file = st.file_uploader("Upload a cluster label list (optional)", type=["csv", "xlsx"])
    labels = None
    if label_file is not None:
        if label_file.name.endswith(".csv"):
            labels = pd.read_csv(label_file).values.flatten()
        elif label_file.name.endswith(".xlsx"):
            labels = pd.read_excel(label_file).values.flatten()

        st.write("Cluster labels uploaded successfully!")

    if st.button("Run MaxMin Algorithm"):
        # Check if the input matrix is a feature matrix or a distance matrix
        if matrix.shape[0] == matrix.shape[1]:
            # Distance matrix
            selector = MaxMin()
            selected_ids = selector.select(matrix, size=num_points, labels=labels)
        else:
            # Feature matrix
            selector = MaxMin(lambda x: pairwise_distances(x, metric="euclidean"))
            selected_ids = selector.select(matrix, size=num_points, labels=labels)

        st.write("Selected indices:", selected_ids)
