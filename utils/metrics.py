import torch
import numpy as np
import pandas as pd


def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate the accuracy of predictions.

    Args:
        predictions (torch.Tensor): The predicted scores from the model.
        targets (torch.Tensor): The true target labels.

    Returns:
        float: The accuracy of the predictions.
    """
    # Convert one-hot encoded labels to class indices
    targets = torch.argmax(targets, dim=1)

    # Compute predicted class indices
    _, predicted = torch.max(predictions, dim=1)

    # Compute accuracy
    correct = torch.sum(predicted == targets)
    total = targets.shape[0]
    accuracy = correct.item() / total

    return accuracy


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int | str] = None):
    """
    Compute a confusion matrix and return it as a pandas DataFrame.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        labels (list, optional): List of labels to include in the confusion matrix.
                                If None, all unique labels present in y_true and y_pred will be used.

    Returns:
        pd.DataFrame: Confusion matrix as a DataFrame.
    """
    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)

    if labels is None:
        labels = sorted(set(np.concatenate([y_true, y_pred], axis=0)))

    conf_matrix_df = pd.DataFrame(conf_matrix, index=labels, columns=labels)

    return conf_matrix_df