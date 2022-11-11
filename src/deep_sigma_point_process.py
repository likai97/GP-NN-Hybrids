import torch

import gpytorch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.variational import VariationalStrategy
from gpytorch.variational import MeanFieldVariationalDistribution
from gpytorch.models.deep_gps.dspp import DSPPLayer, DSPP
import gpytorch.settings as settings

class DSPPHiddenLayer(DSPPLayer):
    """
    Implementation of a deep sigma point process layer as proposed by Jankowiak et al. 2020
     (https://arxiv.org/abs/2002.09112).

    Args:
        input_dims: Input Dimension
        output_dims: list with dimensionality for each hidden layer
        num_inducing: Number of inducing points
        inducing_points: Optional starting inducing points
        mean_type: GP mean type, e.g. "constant" or "linear"
        Q: Number of Quadrature points
    """
    def __init__(self, input_dims, output_dims, num_inducing=300, inducing_points=None, mean_type='constant', Q=8):
        if inducing_points is None:
            if output_dims is None:
                inducing_points = torch.randn(num_inducing, input_dims)
            else:
                inducing_points = torch.randn(output_dims, num_inducing, input_dims)
        else:
            num_inducing = inducing_points.size(-2)

        # Mean field / diagonal covariance structure.
        variational_distribution = MeanFieldVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=torch.Size([output_dims]) if output_dims is not None else torch.Size([])
        )

        # Standard variational inference by Hensman et al. 2013
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        batch_shape = torch.Size([]) if output_dims is None else torch.Size([output_dims])

        super(DSPPHiddenLayer, self).__init__(variational_strategy, input_dims, output_dims, Q)

        if mean_type == 'constant':
            # Constant mean for the final output layer.
            self.mean_module = ConstantMean(batch_shape=batch_shape)
        elif mean_type == 'linear':
            # As in Salimbeni et al. 2017, linear mean for the hidden layers
            self.mean_module = LinearMean(input_dims, batch_shape=batch_shape)

        self.covar_module = ScaleKernel(MaternKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
                                        batch_shape=batch_shape, ard_num_dims=None)

    def forward(self, x, mean_input=None, **kwargs):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class DSPPRegression(DSPP):
    def __init__(self, train_x_shape, output_dims, num_inducing, Q=3):
        """
        Implementation of a deep sigma point process layer as proposed by Jankowiak et al. 2020
         (https://arxiv.org/abs/2002.09112).

        Args:
            train_x_shape: Input Dimension
            output_dims: list with dimensionality for each hidden layer
            num_inducing: Number of inducing points
            Q: Number of Quadrature points
        """

        output_dims.append(None)  # The last layer has None output_dims

        # As in Salimbeni et al. 2017  linear mean for the hidden layer
        means = (len(output_dims) - 1) * ['linear'] + ['constant']  # The last layer with constant mean

        hidden_layers = torch.nn.ModuleList([DSPPHiddenLayer(
            input_dims=train_x_shape[-1],
            output_dims=output_dims[0],
            mean_type=means[0],
            num_inducing=num_inducing,
            Q=Q
        )])

        for layer in range(1, len(output_dims)):
            hidden_layers.append(DSPPHiddenLayer(
                input_dims=hidden_layers[-1].output_dims,
                output_dims=output_dims[layer],
                mean_type=means[layer],
                num_inducing=num_inducing,
                Q=Q
            ))

        likelihood = GaussianLikelihood()

        super().__init__(Q)
        self.likelihood = likelihood
        self.hidden_layers = hidden_layers

    def forward(self, inputs, **kwargs):
        output = self.hidden_layers[0](inputs)
        for layer in self.hidden_layers[1:]:
            output = layer(output)
        return output

    def predict(self, loader):
        with settings.fast_computations(log_prob=False, solves=False), torch.no_grad():
            mus, variances, lls = [], [], []
            for x_batch, y_batch in loader:
                preds = self.likelihood(self(x_batch, mean_input=x_batch))
                mus.append(preds.mean.cpu())
                variances.append(preds.variance.cpu())

                # Compute test log probability. The output of a DSPP is a weighted mixture of Q Gaussians,
                # with the Q weights specified by self.quad_weight_grid. The below code computes the log probability of each
                # test point under this mixture.

                # Step 1: Get log marginal for each Gaussian in the output mixture.
                base_batch_ll = self.likelihood.log_marginal(y_batch, self(x_batch))

                # Step 2: Weight each log marginal by its quadrature weight in log space.
                deep_batch_ll = self.quad_weights.unsqueeze(-1) + base_batch_ll

                # Step 3: Take logsumexp over the mixture dimension, getting test log prob for each datapoint in the batch.
                batch_log_prob = deep_batch_ll.logsumexp(dim=0)
                lls.append(batch_log_prob.cpu())

        return torch.cat(mus, dim=-1), torch.cat(variances, dim=-1), torch.cat(lls, dim=-1)
