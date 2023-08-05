import numpy as np

def zscore(matrix: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Calculate the z-scores of a matrix along the specified axis.

    Args:
        matrix (numpy.ndarray): The input matrix.
        axis (int, optional): The axis along which to calculate the z-sscores. Defaults to 0.

    Returns:
        numpy.ndarray: The matrix with z-scores along the specified axis.
    """
    mean = np.mean(matrix, axis=(0, 1), keepdims=True)
    std = np.std(matrix, axis=(0, 1), keepdims=True)
    normalized_matrix = (matrix - mean) / std
    return normalized_matrix