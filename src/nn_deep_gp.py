import torch
import gpytorch
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood

from gpytorch.models.deep_gps import DeepGP

from .deep_gp import DeepGPHiddenLayer

class NNDGPRegression(DeepGP):
    def __init__(self, feature_extractor, output_dims, num_inducing=128):
        """
        Combines DKL and DGP as proposed by Jankowiak et al 2020 (https://arxiv.org/abs/2002.09112)

        Args:
            feature_extractor: NN feature extrtactor
            output_dims: list with dimensionality for each hidden layer
            num_inducing: Number of inducing points in each GP
        """
        # input dims into our kernel are the number of out features of the NN feature extractor
        out_dims_nn = feature_extractor._modules[next(reversed(feature_extractor._modules))].out_features

        # L hidden layers of a L+1-layer GP
        output_dims.append(None)  # The last layer has None output_dims

        # As in Salimbeni et al. 2017 linear mean for the hidden layers
        means = (len(output_dims) - 1) * ['linear'] + ['constant']  # The last layer with constant mean

        hidden_layers = torch.nn.ModuleList([DeepGPHiddenLayer(
            input_dims=out_dims_nn,
            output_dims=output_dims[0],
            mean_type=means[0],
            num_inducing=num_inducing,
        )])

        for layer in range(1, len(output_dims)):
            hidden_layers.append(DeepGPHiddenLayer(
                input_dims=hidden_layers[-1].output_dims,
                output_dims=output_dims[layer],
                mean_type=means[layer],
                num_inducing=num_inducing,
            ))

        super().__init__()

        self.hidden_layers = hidden_layers
        self.likelihood = GaussianLikelihood()

        self.feature_extractor = feature_extractor

        # Scale the NN values
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

    def forward(self, inputs):
        projected_x = self.feature_extractor(inputs)
        projected_x = self.scale_to_bounds(projected_x)  # Scale the NN values

        output = self.hidden_layers[0](projected_x)
        for layer in self.hidden_layers[1:]:
            output = layer(output)
        return output

    def predict(self, test_loader):
        with torch.no_grad():
            mus = []
            variances = []
            lls = []
            for x_batch, y_batch in test_loader:
                preds = self.likelihood(self(x_batch))
                mus.append(preds.mean)
                variances.append(preds.variance)
                lls.append(self.likelihood.log_marginal(y_batch, self(x_batch)))

        return torch.cat(mus, dim=-1), torch.cat(variances, dim=-1), torch.cat(lls, dim=-1)
