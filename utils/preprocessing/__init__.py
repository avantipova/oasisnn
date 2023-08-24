import numpy as np

def zscore(matrix: np.ndarray, axis: int | tuple[int, ...] = 0) -> np.ndarray:
    """
    Calculate the z-scores of a matrix along the specified axis.

    Args:
        matrix (numpy.ndarray): The input matrix.
        axis (Union[int, tuple[int, ...]] optional): The axis along which to calculate the z-scores. Defaults to 0.

    Returns:
        numpy.ndarray: The matrix with z-scores along the specified axis.
    """
    mean = np.mean(matrix, axis=axis, keepdims=True)
    std = np.std(matrix, axis=axis, keepdims=True)
    normalized_matrix = (matrix - mean) / std
    return normalized_matrix


def balance(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Balance the dataset by undersampling the classes to the size of the smallest class.

    This function takes an input feature array `X` and its corresponding target array `Y`, and performs
    class-wise undersampling to balance the dataset. It ensures that each class has the same number of
    samples as the smallest class. The function returns balanced feature and target arrays.

    Args:
        X (np.ndarray): Input feature array of shape (n_samples, n_features).
        Y (np.ndarray): Target array of shape (n_samples,), containing class labels.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays: balanced feature array `X_balanced`
        and balanced target array `Y_balanced`.

    Example:
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        Y = np.array([0, 1, 0, 1])
        X_balanced, Y_balanced = balance(X, Y)
    """
    classes, classes_samples = np.unique(Y, return_counts=True)
    smallest_class = classes[np.argsort(classes_samples)][0]
    samples = classes_samples.min()
    X_list, Y_list = list(), list()
    stat = {class_: 0 for class_ in classes}

    for x, y in zip(X, Y):
        if y != smallest_class and stat[y] >= samples:
            continue
        else:
            Y_list.append(y)
            X_list.append(x)
            stat[y] += 1

    return np.array(X_list), np.array(Y_list)
