import torch


def mse(prediction, true_value):
    """
    Calculates the mean squared error for a models prediction

    Args:
        prediction: model predictions
        true_value: true labels

    Returns:
        - MSE score
    """
    return torch.mean(torch.pow(prediction - true_value, 2))


def mae(prediction, true_value):
    """
    Calculates the mean absolute error for a models prediction

    Args:
        prediction: model predictions
        true_value: true labels

    Returns:
        - MAE score
    """
    return torch.mean(torch.abs(prediction - true_value))


def maximum_absolute_error(prediction, true_value):
    """
    Calculates the maximum absolute error for a models prediction

    Args:
        prediction: model predictions
        true_value: true labels

    Returns:
        - MaxAE score
    """
    return torch.max(torch.abs(prediction - true_value))
