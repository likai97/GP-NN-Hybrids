import torch
import gpytorch
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from scipy.cluster.vq import kmeans2

class DKLRegression(gpytorch.models.ExactGP):
    """
    This class implements a GP, whose kernel is parameterized by a neural network. (https://arxiv.org/abs/1511.02222)
    To scale this to large data settings, the models uses Structured Kernel Interpolation (or KISS-GP) which was
    introduced Wilson et al. 2015 (http://proceedings.mlr.press/v37/wilson15.html)

    Args:
        train_x: train features
        train_y: train labels
        likelihood: A GPytorch likelihood, e.g. Gaussian
        feature_extractor: NN Class of feature extractor
        mean_type: GP mean type, e.g. "constant" or "linear"
        kernel_type: GP kernel type, e.g. RBF or SpectralMixture
        num_mixtures: If spectral mixture kernel specify, number of mixture components
        grid_size: Integer of grid_size for KISS-GP, if None a good grid_size is determined from data

    Exp:
    >>> from src.utils.model_utils import NNFeatureExtractor
    >>> feature_extractor = NNFeatureExtractor(input_dim=3, n_max=1024, n_layers=4, n_out=2)
    >>> likelihood = gpytorch.likelihoods.GaussianLikelihood()
    >>> model = DKLRegression(X_train, y_train, likelihood, feature_extractor, grid_size=100)
    """

    def __init__(self, train_x, train_y, likelihood, feature_extractor, mean_type="constant", kernel_type="rbf",
                 num_mixtures=4, nu=1.5, grid_size=None):

        # input dims into our kernel are the number of out features of the NN feature extractor
        input_dims = feature_extractor._modules[next(reversed(feature_extractor._modules))].out_features

        super(DKLRegression, self).__init__(train_x, train_y, likelihood)

        # NN Feature extractor
        self.feature_extractor = feature_extractor.cpu()
        # Scale the input data to the kernel so that it lies in between the lower and upper bounds
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(0., 1.)

        # define GP mean
        if mean_type == "constant":
            self.mean_module = gpytorch.means.ConstantMean()
        elif mean_type == "linear":
            self.mean_module = gpytorch.means.LinearMean(input_dims)
        else:
            print("Not supported mean type")
            raise

        # Set grid size automatically if not given
        if not grid_size:
            grid_size = gpytorch.utils.grid.choose_grid_size(train_x, ratio=1.0, kronecker_structure=True)

        # Use KISS-GP covariance module
        if kernel_type == "rbf":
            self.covar_module = gpytorch.kernels.GridInterpolationKernel(
                gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel(ard_num_dims=input_dims)
                ),
                num_dims=input_dims, grid_size=grid_size
            )
        elif kernel_type == "matern0.5":
            # initialize SpectralMixture Kernel from data
            self.covar_module = gpytorch.kernels.GridInterpolationKernel(
                gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.MaternKernel(nu=0.5, ard_num_dims=input_dims)
                ),
                num_dims=input_dims, grid_size=grid_size
            )
        elif kernel_type == "matern1.5":
            # initialize SpectralMixture Kernel from data
            self.covar_module = gpytorch.kernels.GridInterpolationKernel(
                gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=input_dims)
                ),
                num_dims=input_dims, grid_size=grid_size
            )
        elif kernel_type == "spectral":
            # initialize SpectralMixture Kernel from data
            self.base_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=num_mixtures, ard_num_dims=input_dims)
            init_x = self.feature_extractor(train_x.cpu())
            init_x = self.scale_to_bounds(init_x)
            self.base_module.initialize_from_data(init_x, train_y.cpu())

            self.covar_module = gpytorch.kernels.GridInterpolationKernel(
                self.base_module,
                num_dims=input_dims, grid_size=grid_size
            )
        else:
            print("Not supported Kernel")
            raise

    def forward(self, x):
        # We're first putting our data through a deep net (feature extractor)
        projected_x = self.feature_extractor(x)
        projected_x = self.scale_to_bounds(projected_x)  # scale the output data of the NN

        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class VDKLRegression(gpytorch.models.ApproximateGP):
    """
    This class implements a GP, whose kernel is parameterized by a neural network. (https://arxiv.org/abs/1511.02222)
    To scale this to large data settings, the models uses Variational Inference as defined by Hensman et al. 2015
    (http://proceedings.mlr.press/v38/hensman15.html).

    Args:
        train_x: train features
        train_y: train labels
        feature_extractor: NN Class of feature extractor
        mean_type: GP mean type, e.g. "constant" or "linear"
        kernel_type: Integer of grid_size for KISS-GP, if None a good grid_size is determined from data
        num_mixtures: Number of spectral mixtures
        num_inducing: A GPytorch likelihood, e.g. Gaussian
        minit: Method of initialization: Either random or with kmeans. Random chooses random training points as initilization

    Exp:
    >>> from src.utils.model_utils import NNFeatureExtractor
    >>> feature_extractor = NNFeatureExtractor(input_dim=3, n_max=1024, n_layers=4, n_out=2)
    >>> likelihood = gpytorch.likelihoods.GaussianLikelihood()
    >>> input_dim = X_train.size(-1)
    >>> model = VDKLRegression(train_x, train_y, feature_extractor, num_inducing=1000)
    """

    def __init__(self, train_x, train_y, feature_extractor, mean_type="constant",
                 kernel_type="rbf", num_mixtures=None, num_inducing=200, minit="kmeans"):

        # input dims into our kernel are the number of out features of the NN feature extractor
        embedding_dim = feature_extractor._modules[next(reversed(feature_extractor._modules))].out_features

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
        )

        # define GP mean
        if minit == "random":
            indices = torch.randperm(len(train_x))[:num_inducing]
            inducing_points = train_x[indices]
        elif minit == "kmeans":
            inducing_points = train_x[torch.randperm(train_x.size(0))[0:num_inducing], :]
            inducing_points = inducing_points.clone().data.cpu().numpy()
            inducing_points = torch.Tensor(kmeans2(train_x.data.cpu().numpy(),
                                           inducing_points, minit='matrix')[0])
        else:
            print("Not supported initialization method")
            raise

        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        super(VDKLRegression, self).__init__(variational_strategy)

        self.feature_extractor = feature_extractor
        # This module will scale the NN features so that they're nice values
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(0., 1.)

        # define GP mean
        if mean_type == "constant":
            self.mean_module = gpytorch.means.ConstantMean()
        elif mean_type == "linear":
            self.mean_module = gpytorch.means.LinearMean(embedding_dim)
        else:
            print("Not supported mean type")
            raise

        # Use RBF covariance module
        if kernel_type == "rbf":
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=embedding_dim)
            )
        elif kernel_type == "matern0.5":
            # initialize SpectralMixture Kernel from data
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=0.5, ard_num_dims=embedding_dim)
            )
        elif kernel_type == "matern1.5":
            # initialize SpectralMixture Kernel from data
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=embedding_dim)
            )
        elif kernel_type == "spectral":
            self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=num_mixtures,
                                                                       ard_num_dims=embedding_dim)
            #initialize SpectralMixture Kernel
            init_x = self.feature_extractor(train_x)
            init_x = self.scale_to_bounds(init_x)
            self.covar_module.initialize_from_data(init_x, train_y)
        else:
            print("Not supported Kernel")
            raise

    def forward(self, x):
        # We're first putting our data through a deep net (feature extractor)
        projected_x = self.feature_extractor(x)
        projected_x = self.scale_to_bounds(projected_x)  # Make the NN values "nice"

        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


