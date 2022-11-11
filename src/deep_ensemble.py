import torch
import torch.nn.functional as F

class DeepEnsembleBaseLearner(torch.nn.Sequential):
    """
    MLP whose outputs are mean and variance.

    Args:
        input_dim: Input dimension of the data
        n_max: Maximuim number of neurons
        n_layers: Number of layers of the Neural Network

    Examples:
    >>> import torch
    >>> feature_extractor = NNFeatureExtractor(input_dim=3, n_max=1024, n_layers=4)
    """

    def __init__(self, input_dim, n_max, n_layers):
        super(DeepEnsembleBaseLearner, self).__init__()

        self.input_dim = input_dim
        self.n_max = n_max
        self.n_layers = n_layers
        self.n_out = 2

        in_features = input_dim
        out_features = n_max

        self.layers = torch.nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(torch.nn.Linear(round(in_features), round(out_features)))
            in_features = out_features
            out_features = out_features - (n_max - self.n_out) / (n_layers - 1)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        out = self.layers[-1](x)
        mean, variance = torch.split(out, 1, dim=1)
        variance = F.softplus(variance) + 1e-6
        return mean, variance


class DeepEnsemble(torch.nn.Module):
    """
    MLP whose outputs are mean and variance.

    Args:
        num_models: Number of Neural Networks to train as an Ensemble
        input_dim: Input dimension of the data
        n_max: Maximuim number of neurons
        n_layers: Number of layers of the Neural Network

    Examples:
    >>> de_model = DeepEnsemble(
    >>>        num_models=5,
    >>>        input_dim=100,
    >>>        n_max=1024,
    >>>        n_layers=4
    >>>    )

    """

    def __init__(self, num_models=5, input_dim=1, n_max=1000, n_layers=4):
        super(DeepEnsemble, self).__init__()
        self.num_models = num_models

        for i in range(self.num_models):
            model = DeepEnsembleBaseLearner(input_dim=input_dim,
                                            n_max=n_max,
                                            n_layers=n_layers)
            setattr(self, 'Model_' + str(i + 1), model)

    def forward(self, x):
        # connect layers
        means = []
        variances = []
        for i in range(self.num_models):
            model = getattr(self, 'Model_' + str(i + 1))
            mean, var = model(x)
            means.append(mean)
            variances.append(var)
        means = torch.stack(means)
        mean = means.mean(dim=0)
        variances = torch.stack(variances)
        variance = (variances + means.pow(2)).mean(dim=0) - mean.pow(2)
        return mean, variance