import torch


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