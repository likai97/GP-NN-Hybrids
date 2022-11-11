import numpy as np
import torch


def prepare_train_test_split(X, y, train_idx, test_idx) -> object:
    """
    Performs train test split on an dataframe of features and vector of labels

    Args:
        X: feature dataframe
        y: labels vector
        train_idx: train indeces
        test_idx: test indeces

    Returns:
        - 1 Train Dataframe
        - 1 Test Dataframe
        - 1 array with train labels
        - 1 array with test labels

    Exp:
    >>> indices = np.random.permutation(data.shape[0])
    >>> split = int(np.floor(0.8 * data.shape[0]))
    >>> training_idx, test_idx = indices[:split], indices[split:]
    """
    train_x = X[train_idx]
    train_y = y[train_idx]

    test_x = X[test_idx]
    test_y = y[test_idx]
    return train_x, test_x, train_y, test_y

# Source: github.com/Bjarten/early-stopping-pytorch
class EarlyStoppingWithModelSave:
    """Early stops the training if validation loss doesn't improve after a given patience. Best model can be reloaded"""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

class EarlyStopping():
    """
    Args:
        patience: How long to wait after last time validation loss improved.
        delta: Minimum change in the monitored quantity to qualify as an improvement.
    """
    def __init__(self, patience=2, delta=1e-8):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.early_stop = False
        self.best_score = None

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

