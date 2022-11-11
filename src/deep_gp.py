import torch
import gpytorch
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import RBFKernel, ScaleKernel, MaternKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood

from gpytorch.models.deep_gps import DeepGPLayer, DeepGP



class DeepGPHiddenLayer(DeepGPLayer):
    """
    Implementation of a deep gaussian process layer using doubly stochastic variational inference as proposed by
    Salimbeni et al. 2017 (https://arxiv.org/abs/1705.08933).

    Args:
        input_dims: Input Dimensions
        output_dims: list with dimensionality for each hidden layer
        num_inducing: Size of the variational distribution (size of variational mean)
        mean_type: GP mean type, e.g. "constant" or "linear"
        kernel_type: GP kernel type, e.g. RBF or Matern
    """
    def __init__(self, input_dims, output_dims, num_inducing=128, mean_type='constant', kernel_type='rbf'):
        if output_dims is None:
            inducing_points = torch.randn(num_inducing, input_dims)
            batch_shape = torch.Size([])
        else:
            inducing_points = torch.randn(output_dims, num_inducing, input_dims)
            batch_shape = torch.Size([output_dims])

        # multivariate normal distribution with a full covariance matrix
        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )

        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        super(DeepGPHiddenLayer, self).__init__(variational_strategy, input_dims, output_dims)

        if mean_type == 'constant':
            self.mean_module = ConstantMean(batch_shape=batch_shape)
        else:
            self.mean_module = LinearMean(input_dims)

        if kernel_type == 'rbf':
            self.covar_module = ScaleKernel(
                RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
                batch_shape=batch_shape, ard_num_dims=None
            )
        elif kernel_type == 'matern0.5':
            self.covar_module = ScaleKernel(
                MaternKernel(nu=0.5, batch_shape=batch_shape, ard_num_dims=input_dims),
                batch_shape=batch_shape, ard_num_dims=None
            )
        elif kernel_type == 'matern1.5':
            self.covar_module = ScaleKernel(
                MaternKernel(nu=1.5, batch_shape=batch_shape, ard_num_dims=input_dims),
                batch_shape=batch_shape, ard_num_dims=None
            )
        else:
            print("Not supported Kernel")
            raise
        # self.linear_layer = Linear(input_dims, 1)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def __call__(self, x, *other_inputs, **kwargs):
        """
        Follows the idea of Duvenaud et al (https://arxiv.org/abs/1402.5836) and implements a skip connection.
        To each layer, the input data will be passed in addition to the output of the previous layer.
        """
        if len(other_inputs):
            if isinstance(x, gpytorch.distributions.MultitaskMultivariateNormal):
                x = x.rsample()

            processed_inputs = [
                inp.unsqueeze(0).expand(gpytorch.settings.num_likelihood_samples.value(), *inp.shape)
                for inp in other_inputs
            ]

            x = torch.cat([x] + processed_inputs, dim=-1)

        return super().__call__(x, are_samples=bool(len(other_inputs)))


class DeepGPRegression(DeepGP):
    def __init__(self, train_x_shape, output_dims, num_inducing=128, kernel_type='rbf'):
        """
        DGP Module which passes forward through the various DGP Layers
        Inspired by the archtiecture of Deep Neural Network, the idea is to stack multiple GPs horizontally
         and vertically.

        Args:
            train_x_shape: Train data shape
            output_dims: list with dimensionality for each hidden layer
            num_inducing: Size of the variational distribution (size of variational mean)
            kernel_type: GP kernel type, e.g. RBF or Matern

        Exp:
        # 2-layer DGP with 5 GPs in each layer
        >>> import torch
        >>> model = DeepGPRegression(train_x_shape=X_train.shape, output_dims=[5, 5])
        """
        # L hidden layers of a L+1-layer GP
        output_dims.append(None)  # The last layer has None output_dims

        # As in Salimbeni et al. 2017 finds that using a linear mean for the hidden layer improves performance
        means = (len(output_dims) - 1) * ['linear'] + ['constant']  # The last layer with constant mean

        hidden_layers = torch.nn.ModuleList([DeepGPHiddenLayer(
            input_dims=train_x_shape[-1],
            output_dims=output_dims[0],
            num_inducing=num_inducing,
            mean_type=means[0],
            kernel_type=kernel_type
            )])

        for layer in range(1, len(output_dims)):
            hidden_layers.append(DeepGPHiddenLayer(
                input_dims=hidden_layers[-1].output_dims,
                output_dims=output_dims[layer],
                num_inducing=num_inducing,
                mean_type=means[layer],
                kernel_type=kernel_type
                ))

        super().__init__()

        self.hidden_layers = hidden_layers
        self.likelihood = GaussianLikelihood()

    def forward(self, inputs):
        output = self.hidden_layers[0](inputs)
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