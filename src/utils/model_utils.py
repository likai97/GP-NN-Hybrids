import torch

class NNFeatureExtractor(torch.nn.Sequential):
    """
    This module creates a sequential neural network given a number of layers, max number of neurons in the first layer
    and the number of output neurons. The design follows the search space proposed in the Auto-Pytorch Tabular paper
    (https://arxiv.org/pdf/2006.13799.pdf).

    Args:
        input_dim: Input dimension of the data
        n_max: Maximuim number of neurons
        n_layers: Number of layers of the Neural Network
        n_out: Number of output neurons
        p: Dropout probability

    Examples:
    >>> import torch
    >>> feature_extractor = NNFeatureExtractor(input_dim=3, n_max=1024, n_layers=4, n_out=2)
    """

    def __init__(self, input_dim, n_max, n_layers, n_out, p=0.0):
        super(NNFeatureExtractor, self).__init__()

        in_features = input_dim
        out_features = n_max
        for i in range(n_layers):
            self.add_module(f'linear{i + 1}', torch.nn.Linear(round(in_features), round(out_features)))
            if i != n_layers - 1:
                self.add_module(f'ReLu{i + 1}', torch.nn.ReLU())
                self.add_module(f'Dropout{i + 1}', torch.nn.Dropout(p))
            in_features = out_features
            out_features = out_features - (n_max - n_out) / (n_layers - 1)


def initiliaze_feature_extractor(X, y, feature_extractor, batch_size=2048, lr=0.001, loss_fn=torch.nn.MSELoss()):
    """
    Initiliaze feature extractor with Adam on the training inputs

    Args:
        X: Train inputs
        y: Train labels
        feature_extractor: NN feature extractor
        batch_size: Batch Size
        lr: Adam learning rate
        loss_fn: Loss function to be used to train

    Examples:
    >>> feature_extractor = initiliaze_feature_extractor(X=X_train, y=y_train, feature_extractor=feature_extractor)
    """
    if len(X) < 10000:
        epochs = 25
    elif len(X) < 100000:
        epochs = 10
    else:
        epochs = 5

    train_dataset = torch.utils.data.TensorDataset(X, y)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    input_dims = feature_extractor._modules[next(reversed(feature_extractor._modules))].out_features
    feature_extractor = torch.nn.Sequential(
        feature_extractor,
        torch.nn.ReLU(),
        torch.nn.Linear(input_dims, 1)
    )

    optimizer = torch.optim.Adam([
        {'params': feature_extractor.parameters()},
    ], lr=lr)

    if torch.cuda.is_available():
        feature_extractor = feature_extractor.cuda()

    for epoch in range(epochs):
        for batch, (X_batch, y_batch) in enumerate(train_loader):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = feature_extractor(X_batch)
            output = output.flatten()
            # Calc loss and backprop gradients
            loss = loss_fn(output, y_batch)
            loss.backward()
            optimizer.step()
        # print results for first and last epoch
        if epoch == 0 or epoch + 1 == epochs:
            print(f"{epoch + 1}/{epochs} Pretrained Model - loss: {loss}")

    return feature_extractor[0]
