import pandas as pd
import numpy as np
import math
import openml
import os
import time

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import KFold, train_test_split

import torch
from torch.utils.data import TensorDataset, DataLoader
import gpytorch

from src.utils.metric_utils import mse, mae, maximum_absolute_error
from src.utils.model_utils import NNFeatureExtractor, initiliaze_feature_extractor
from src.utils.train_utils import EarlyStoppingWithModelSave

from src.deep_kernel_learning import VDKLRegression
from src.deep_kernel_learning import DKLRegression
from src.deep_sigma_point_process import DSPPRegression
from src.deep_ensemble import DeepEnsemble
from src.deep_gp import DeepGPRegression
from src.nn_deep_gp import NNDGPRegression

import gc

def clear_cuda_memory(*args):
    for i in args:
        del i
    gc.collect()
    torch.cuda.empty_cache()


def VDKL_NLL(n_splits_outer=5,
             n_splits_inner=4,
             file=None,
             PPV=True,
             dataset_id=216,
             epochs_outer=1000,
             patience_outer=10):
    datasets = {287: "wine_quality", 216: "elevators", 42225: "diamonds", 4549: "buzz", 44027: "year"}

    if file == "k2204":
        file_path = './data/K2204_net_present_values.csv'
        df = pd.read_csv(file_path, sep=';', index_col=0)

        # prepare dataset
        X = df.iloc[:, 0:3]
        if PPV:
            y = df.iloc[:, -1]
            dataset_name = "k2204_PPV"
        else:
            y = df.iloc[:, -2]
            dataset_name = "k2204_BPV"
    elif file == "r1_08":
        file_path = '../trail.X/data/R1_08/r1_08.parquet'
        df = pd.read_parquet(file_path, engine='pyarrow')

        # prepare dataset
        X = df.iloc[:, 1:10]
        if PPV:
            y = df.iloc[:, 10]
            dataset_name = "r1_08_PPV"
        else:
            y = df.iloc[:, 11]
            dataset_name = "r1_08_BPV"

        # Feature Transformation
        X["GeburtsDatum"] = X["GeburtsDatum"].str[:4].astype(int) - 1985
        X['geschlechtVP1'] = np.where(X["geschlechtVP1"] == 'MAENNLICH', 1, 0)
        # Ordinal Encoding
        X['zahlweiseExkasso'] = pd.Categorical(X.zahlweiseExkasso,
                                               categories=['MONATLICH', 'VIERTELJAEHRLICH', 'HALBJAEHRLICH',
                                                           'JAEHRLICH'], ordered=True)
        ord = OrdinalEncoder(categories=[['MONATLICH', 'VIERTELJAEHRLICH', 'HALBJAEHRLICH', 'JAEHRLICH']])
        X['zahlweiseExkasso'] = ord.fit_transform(X[['zahlweiseExkasso']])
    elif file == "r1_08_small":
        file_path = '../trail.X/data/R1_08/r1_08.parquet'
        df = pd.read_parquet(file_path, engine='pyarrow')
        df = df.sample(600000, random_state=73)

        # prepare dataset
        X = df.iloc[:, 1:10]
        if PPV:
            y = df.iloc[:, 10]
            dataset_name = "r1_08_PPV"
        else:
            y = df.iloc[:, 11]
            dataset_name = "r1_08_BPV"

        # Feature Transformation
        X["GeburtsDatum"] = X["GeburtsDatum"].str[:4].astype(int) - 1985
        X['geschlechtVP1'] = np.where(X["geschlechtVP1"] == 'MAENNLICH', 1, 0)
        # Ordinal Encoding
        X['zahlweiseExkasso'] = pd.Categorical(X.zahlweiseExkasso,
                                               categories=['MONATLICH', 'VIERTELJAEHRLICH', 'HALBJAEHRLICH',
                                                           'JAEHRLICH'], ordered=True)
        ord = OrdinalEncoder(categories=[['MONATLICH', 'VIERTELJAEHRLICH', 'HALBJAEHRLICH', 'JAEHRLICH']])
        X['zahlweiseExkasso'] = ord.fit_transform(X[['zahlweiseExkasso']])
    else:
        # Get dataset from OpenML
        dataset = openml.datasets.get_dataset(dataset_id)

        X, y, categorical_indicator, attribute_names = dataset.get_data(
            target=dataset.default_target_attribute, dataset_format="dataframe"
        )
        dataset_name = datasets[dataset_id]

    # File Names
    savedir = './data/experiments/VDKL'
    nested_resampling_infos_file = savedir + '/nested_resampling_infos_VDKL_' + dataset_name + '.csv'

    kf_outer = KFold(n_splits=n_splits_outer, shuffle=True, random_state=73)
    kf_inner = KFold(n_splits=n_splits_inner, shuffle=True, random_state=69)

    # One hot encoding of categorical variables
    ohe = OneHotEncoder(drop='if_binary')
    categorical_features = X.select_dtypes(include=['category', 'object']).columns

    # perform one-hot encoding on categorical columns column
    encoder_df = pd.DataFrame(ohe.fit_transform(X[categorical_features]).toarray())

    # merge one-hot encoded columns back with original DataFrame
    X = X.join(encoder_df)
    X.drop(categorical_features, axis=1, inplace=True)

    X = torch.Tensor(X.values)
    y = torch.Tensor(y.values)

    print(f'Starting Nested Resampling for {dataset_name}')

    split_nr = 1
    # Outer Nested Resampling Split
    for train_outer, test_outer in kf_outer.split(np.array(X)):
        nested_resampling_infos = pd.read_csv(nested_resampling_infos_file)

        X_train_outer, X_test_outer, y_train_outer, y_test_outer = X[train_outer], X[test_outer], y[train_outer], y[
            test_outer]

        # Scale Features and Labels
        scaler = MinMaxScaler()
        X_train_outer = scaler.fit_transform(X_train_outer)
        X_test_outer = scaler.transform(X_test_outer)

        y_scaler = MinMaxScaler()
        y_train_outer = y_scaler.fit_transform(y_train_outer.reshape(-1, 1)).flatten()

        # Turn back to tensors
        X_train_outer = torch.from_numpy(X_train_outer).float()
        X_test_outer = torch.from_numpy(X_test_outer).float()

        y_train_outer = torch.from_numpy(y_train_outer).float()

        if torch.cuda.is_available():
            X_train_outer, X_test_outer, y_train_outer, y_test_outer = X_train_outer.cuda(), X_test_outer.cuda(), \
                                                                       y_train_outer.cuda(), y_test_outer.cuda()

        # refit model
        best_trial = nested_resampling_infos.iloc[split_nr - 1]

        # Refit with best trial
        # best trial
        n_max_best = int(best_trial['n_max'])
        n_layers_best = int(best_trial['n_layers'])
        n_out_best = int(best_trial['n_out'])
        mean_type_best = best_trial['mean_type']
        kernel_type_best = best_trial['kernel_type']
        max_epochs = int(best_trial['MAX_EPOCHS'] + 1)
        # Only need num_mixtures if kernel_type is spectral mixture
        if kernel_type_best == "spectral":
            num_mixtures_best = int(best_trial['num_mixtures'])
        else:
            num_mixtures_best = None
        lr_best = best_trial['lr']
        batch_size_best = int(best_trial['batch_size'])
        num_inducing_best = int(best_trial['num_inducing'])
        initialize_fe_best = best_trial['initialize_fe']
        minit_best = best_trial['minit']

        train_dataset = TensorDataset(X_train_outer, y_train_outer)
        train_loader = DataLoader(train_dataset, batch_size=batch_size_best, shuffle=True)

        test_dataset = TensorDataset(X_test_outer, y_test_outer)
        test_loader = DataLoader(test_dataset, batch_size=batch_size_best, shuffle=False)

        best_nn = NNFeatureExtractor(input_dim=X_train_outer.size(-1), n_max=n_max_best, n_layers=n_layers_best,
                                     n_out=n_out_best, p=0)
        if initialize_fe_best:
            best_nn = initiliaze_feature_extractor(X_train_outer, y_train_outer, best_nn)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = VDKLRegression(X_train_outer, y_train_outer, best_nn,
                               num_inducing=num_inducing_best, mean_type=mean_type_best,
                               kernel_type=kernel_type_best, num_mixtures=num_mixtures_best, minit=minit_best)

        if torch.cuda.is_available():
            model = model.cuda()
            likelihood = likelihood.cuda()

        # Use the adam optimizer
        optimizer = torch.optim.AdamW([
            {'params': model.parameters()},
            {'params': likelihood.parameters()},
        ], lr=lr_best)

        # "Loss" for GPs - the marginal log likelihood
        early_stopping = EarlyStoppingWithModelSave(patience=patience_outer)
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(train_loader.dataset))
        start = time.time()
        for epoch in range(epochs_outer):
            model.train()
            likelihood.train()
            epoch_loss = []
            for i, (X_batch, y_batch) in enumerate(train_loader):
                # Zero gradients from previous iteration
                optimizer.zero_grad()
                # Output from model
                output = model(X_batch)
                # Calc loss and backprop gradients
                loss = -mll(output, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())

            epoch_loss = sum(epoch_loss) / len(epoch_loss)

            if epoch % 5 == 0:
                print(f"{epoch}/{epochs_outer} - loss: {epoch_loss}")

            early_stopping(epoch_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        end = time.time()

        # load the last checkpoint with the best model
        model.load_state_dict(torch.load('checkpoint.pt'))

        # Predictions
        model.eval()
        likelihood.eval()
        test_means = torch.tensor([0.])
        test_vars = torch.tensor([0.])

        # Make predictions by feeding model through likelihood
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                predictions = likelihood(model(x_batch))
                test_means = torch.cat([test_means, predictions.mean.detach().cpu()])
                test_vars = torch.cat([test_vars, predictions.variance.detach().cpu()])

        test_means = test_means[1:]
        test_vars = test_vars[1:]

        predictions_orig = torch.from_numpy(
            y_scaler.inverse_transform(test_means.cpu().reshape(-1, 1)).flatten()).float()

        mse_score = mse(y_test_outer.cpu(), predictions_orig).item()
        mae_score = mae(y_test_outer.cpu(), predictions_orig).item()
        max_score = maximum_absolute_error(y_test_outer.cpu(), predictions_orig).item()

        # NLL
        # NLL Var(Y)=Var(aX+b)=a^2*Var(X)
        y_min = y_scaler.data_min_
        y_max = y_scaler.data_max_
        var_transformed = test_vars * (y_max - y_min) ** 2
        mean_transformed = predictions_orig
        nll = torch.nn.GaussianNLLLoss()
        nll_score = nll(mean_transformed, y_test_outer.cpu(), var_transformed).item()
        train_time = end - start

        nested_resampling_info = pd.DataFrame({'dataset_id': [dataset_id], 'dataset_name': [dataset_name],
                                               'outer_run': [split_nr], 'n_max': [n_max_best],
                                               'n_layers': [n_layers_best], 'n_out': [n_out_best],
                                               'mean_type': [mean_type_best], 'kernel_type': [kernel_type_best],
                                               'lr': [lr_best], 'batch_size': [batch_size_best],
                                               'num_inducing': [num_inducing_best], 'MAX_EPOCHS': [epoch],
                                               'initialize_fe': [initialize_fe_best], 'minit': [minit_best],
                                               'mse': [mse_score], 'mae': [mae_score],
                                               'max_absolute_error': [max_score],
                                               'nll_score': [nll_score], 'train_time': [train_time]})

        nested_resampling_infos = pd.concat([nested_resampling_infos, nested_resampling_info])

        split_nr += 1

        nested_resampling_infos.to_csv(nested_resampling_infos_file, index=False)

        os.remove("checkpoint.pt")

        # Memory Tracking
        clear_cuda_memory(X_train_outer, X_test_outer, y_train_outer, y_test_outer,
                          loss, optimizer, model, likelihood)


def DKL_NLL(n_splits_outer=5,
            n_splits_inner=4,
            file=None,
            PPV=True,
            dataset_id=216):
    datasets = {287: "wine_quality", 216: "elevators", 42225: "diamonds", 4549: "buzz", 44027: "year"}

    if file == "k2204":
        file_path = './data/K2204_net_present_values.csv'
        df = pd.read_csv(file_path, sep=';', index_col=0)

        # prepare dataset
        X = df.iloc[:, 0:3]
        if PPV:
            y = df.iloc[:, -1]
            dataset_name = "k2204_PPV"
        else:
            y = df.iloc[:, -2]
            dataset_name = "k2204_BPV"
    elif file == "r1_08":
        file_path = '../trail.X/data/R1_08/r1_08.parquet'
        df = pd.read_parquet(file_path, engine='pyarrow')

        # prepare dataset
        X = df.iloc[:, 1:10]
        if PPV:
            y = df.iloc[:, 10]
            dataset_name = "r1_08_PPV"
        else:
            y = df.iloc[:, 11]
            dataset_name = "r1_08_BPV"

        # Feature Transformation
        X["GeburtsDatum"] = X["GeburtsDatum"].str[:4].astype(int) - 1985
        X['geschlechtVP1'] = np.where(X["geschlechtVP1"] == 'MAENNLICH', 1, 0)
        # Ordinal Encoding
        X['zahlweiseExkasso'] = pd.Categorical(X.zahlweiseExkasso,
                                               categories=['MONATLICH', 'VIERTELJAEHRLICH', 'HALBJAEHRLICH',
                                                           'JAEHRLICH'], ordered=True)
        ord = OrdinalEncoder(categories=[['MONATLICH', 'VIERTELJAEHRLICH', 'HALBJAEHRLICH', 'JAEHRLICH']])
        X['zahlweiseExkasso'] = ord.fit_transform(X[['zahlweiseExkasso']])
    elif file == "r1_08_small":
        file_path = '../trail.X/data/R1_08/r1_08.parquet'
        df = pd.read_parquet(file_path, engine='pyarrow')
        df = df.sample(600000, random_state=73)

        # prepare dataset
        X = df.iloc[:, 1:10]
        if PPV:
            y = df.iloc[:, 10]
            dataset_name = "r1_08_PPV"
        else:
            y = df.iloc[:, 11]
            dataset_name = "r1_08_BPV"

        # Feature Transformation
        X["GeburtsDatum"] = X["GeburtsDatum"].str[:4].astype(int) - 1985
        X['geschlechtVP1'] = np.where(X["geschlechtVP1"] == 'MAENNLICH', 1, 0)
        # Ordinal Encoding
        X['zahlweiseExkasso'] = pd.Categorical(X.zahlweiseExkasso,
                                               categories=['MONATLICH', 'VIERTELJAEHRLICH', 'HALBJAEHRLICH',
                                                           'JAEHRLICH'], ordered=True)
        ord = OrdinalEncoder(categories=[['MONATLICH', 'VIERTELJAEHRLICH', 'HALBJAEHRLICH', 'JAEHRLICH']])
        X['zahlweiseExkasso'] = ord.fit_transform(X[['zahlweiseExkasso']])
    else:
        # Get dataset from OpenML
        dataset = openml.datasets.get_dataset(dataset_id)

        X, y, categorical_indicator, attribute_names = dataset.get_data(
            target=dataset.default_target_attribute, dataset_format="dataframe"
        )
        dataset_name = datasets[dataset_id]

    # File Names
    savedir = './data/experiments/DKL'
    nested_resampling_infos_file = savedir + '/nested_resampling_infos_DKL_' + dataset_name + '.csv'

    kf_outer = KFold(n_splits=n_splits_outer, shuffle=True, random_state=73)
    kf_inner = KFold(n_splits=n_splits_inner, shuffle=True, random_state=69)

    # One hot encoding of categorical variables
    ohe = OneHotEncoder(drop='if_binary')
    categorical_features = X.select_dtypes(include=['category', 'object']).columns

    # perform one-hot encoding on categorical columns column
    encoder_df = pd.DataFrame(ohe.fit_transform(X[categorical_features]).toarray())

    # merge one-hot encoded columns back with original DataFrame
    X = X.join(encoder_df)
    X.drop(categorical_features, axis=1, inplace=True)

    X = torch.Tensor(X.values)
    y = torch.Tensor(y.values)

    print(f'Starting Nested Resampling for {dataset_name}')

    split_nr = 1
    # Outer Nested Resampling Split
    for train_outer, test_outer in kf_outer.split(np.array(X)):
        nested_resampling_infos = pd.read_csv(nested_resampling_infos_file)

        X_train_outer, X_test_outer, y_train_outer, y_test_outer = X[train_outer], X[test_outer], y[train_outer], y[
            test_outer]

        # Scale Features and Labels
        scaler = MinMaxScaler()
        X_train_outer = scaler.fit_transform(X_train_outer)
        X_test_outer = scaler.transform(X_test_outer)

        y_scaler = MinMaxScaler()
        y_train_outer = y_scaler.fit_transform(y_train_outer.reshape(-1, 1)).flatten()

        # Turn back to tensors
        X_train_outer = torch.from_numpy(X_train_outer).float()
        X_test_outer = torch.from_numpy(X_test_outer).float()

        y_train_outer = torch.from_numpy(y_train_outer).float()

        if torch.cuda.is_available():
            X_train_outer, X_test_outer, y_train_outer, y_test_outer = X_train_outer.cuda(), X_test_outer.cuda(), \
                                                                       y_train_outer.cuda(), y_test_outer.cuda()

        # refit model
        best_trial = nested_resampling_infos.iloc[split_nr - 1]

        # Refit with best trial
        # best trial
        n_max_best = int(best_trial['n_max'])
        n_layers_best = int(best_trial['n_layers'])
        n_out_best = int(best_trial['n_out'])
        p_best = best_trial['p']
        mean_type_best = best_trial['mean_type']
        kernel_type_best = best_trial['kernel_type']
        max_epochs = int(best_trial['MAX_EPOCHS'] + 1)
        # Only need num_mixtures if kernel_type is spectral mixture
        if kernel_type_best == "spectral":
            num_mixtures_best = best_trial['num_mixtures']
        else:
            num_mixtures_best = None
        lr_best = best_trial['lr']
        grid_size_best = int(best_trial['grid_size'])
        initialize_fe_best = best_trial['initialize_fe']

        best_nn = NNFeatureExtractor(input_dim=X_train_outer.size(-1), n_max=n_max_best, n_layers=n_layers_best,
                                     n_out=n_out_best, p=p_best)
        if initialize_fe_best:
            best_nn = initiliaze_feature_extractor(X_train_outer, y_train_outer, best_nn)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = DKLRegression(X_train_outer, y_train_outer, likelihood, best_nn,
                              mean_type=mean_type_best, kernel_type=kernel_type_best,
                              num_mixtures=num_mixtures_best, grid_size=grid_size_best)

        if torch.cuda.is_available():
            model = model.cuda()
            likelihood = likelihood.cuda()

        # Use the adam optimizer
        optimizer = torch.optim.AdamW([
            {'params': model.feature_extractor.parameters()},
            {'params': model.covar_module.parameters()},
            {'params': model.mean_module.parameters()},
            {'params': model.likelihood.parameters()},
        ], lr=lr_best)

        start = time.time()
        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        for epoch in range(max_epochs):
            model.train()
            likelihood.train()
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(X_train_outer)
            # Calc loss and backprop gradients
            loss = -mll(output, y_train_outer)
            loss.backward()
            optimizer.step()

            # Early stopping due to numeric stability issues. See explanation in objective_time_prune
            if math.isnan(loss.item()):
                break

            if epoch % 5 == 0:
                print(f"{epoch}/{max_epochs} - loss: {loss}")

        end = time.time()
        # Predictions
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var():
            predictions = likelihood(model(X_test_outer))

        predictions_orig = torch.from_numpy(
            y_scaler.inverse_transform(predictions.mean.cpu().reshape(-1, 1)).flatten()).float()

        mse_score = mse(y_test_outer.cpu(), predictions_orig).item()
        mae_score = mae(y_test_outer.cpu(), predictions_orig).item()
        max_score = maximum_absolute_error(y_test_outer.cpu(), predictions_orig).item()

        # NLL
        # NLL Var(Y)=Var(aX+b)=a^2*Var(X)
        var = predictions.variance.detach().cpu()
        y_min = y_scaler.data_min_
        y_max = y_scaler.data_max_
        var_transformed = var * (y_max - y_min) ** 2
        mean_transformed = predictions_orig
        nll = torch.nn.GaussianNLLLoss()
        nll_score = nll(mean_transformed, y_test_outer.cpu(), var_transformed).item()
        train_time = end - start

        nested_resampling_info = pd.DataFrame({'dataset_id': [dataset_id], 'dataset_name': [dataset_name],
                                               'outer_run': [split_nr], 'n_max': [n_max_best],
                                               'n_layers': [n_layers_best], 'n_out': [n_out_best], 'p': [p_best],
                                               'mean_type': [mean_type_best],
                                               'kernel_type': [kernel_type_best], 'num_mixtures': [num_mixtures_best],
                                               'lr': [lr_best], 'grid_size': [grid_size_best],
                                               'initialize_fe': [initialize_fe_best],
                                               'MAX_EPOCHS': [epoch], 'mse': [mse_score],
                                               'mae': [mae_score], 'max_absolute_error': [max_score],
                                               'nll_score': [nll_score], 'train_time': [train_time]})

        nested_resampling_infos = pd.concat([nested_resampling_infos, nested_resampling_info])

        split_nr += 1

        nested_resampling_infos.to_csv(nested_resampling_infos_file, index=False)

        # Memory Tracking
        clear_cuda_memory(X_train_outer, X_test_outer, y_train_outer, y_test_outer,
                          loss, optimizer, model, likelihood)


def DSPP_NLL(n_splits_outer=5,
             n_splits_inner=4,
             file=None,
             PPV=True,
             dataset_id=216,
             epochs_outer=1000,
             patience_outer=10):
    # dict map id to name
    # OpenML Dataset to Study
    datasets = {287: "wine_quality", 216: "elevators", 42225: "diamonds", 4549: "buzz", 44027: "year"}

    if file == "k2204":
        file_path = './data/K2204_net_present_values.csv'
        df = pd.read_csv(file_path, sep=';', index_col=0)

        # prepare dataset
        X = df.iloc[:, 0:3]
        if PPV:
            y = df.iloc[:, -1]
            dataset_name = "k2204_PPV"
        else:
            y = df.iloc[:, -2]
            dataset_name = "k2204_BPV"
    elif file == "r1_08":
        file_path = '../trail.X/data/R1_08/r1_08.parquet'
        df = pd.read_parquet(file_path, engine='pyarrow')

        # prepare dataset
        X = df.iloc[:, 1:10]
        if PPV:
            y = df.iloc[:, 10]
            dataset_name = "r1_08_PPV"
        else:
            y = df.iloc[:, 11]
            dataset_name = "r1_08_BPV"

        # Feature Transformation
        X["GeburtsDatum"] = X["GeburtsDatum"].str[:4].astype(int) - 1985
        X['geschlechtVP1'] = np.where(X["geschlechtVP1"] == 'MAENNLICH', 1, 0)
        # Ordinal Encoding
        X['zahlweiseExkasso'] = pd.Categorical(X.zahlweiseExkasso,
                                               categories=['MONATLICH', 'VIERTELJAEHRLICH', 'HALBJAEHRLICH',
                                                           'JAEHRLICH'], ordered=True)
        ord = OrdinalEncoder(categories=[['MONATLICH', 'VIERTELJAEHRLICH', 'HALBJAEHRLICH', 'JAEHRLICH']])
        X['zahlweiseExkasso'] = ord.fit_transform(X[['zahlweiseExkasso']])
    elif file == "r1_08_small":
        file_path = '../trail.X/data/R1_08/r1_08.parquet'
        df = pd.read_parquet(file_path, engine='pyarrow')
        df = df.sample(600000, random_state=73)

        # prepare dataset
        X = df.iloc[:, 1:10]
        if PPV:
            y = df.iloc[:, 10]
            dataset_name = "r1_08_PPV"
        else:
            y = df.iloc[:, 11]
            dataset_name = "r1_08_BPV"

        # Feature Transformation
        X["GeburtsDatum"] = X["GeburtsDatum"].str[:4].astype(int) - 1985
        X['geschlechtVP1'] = np.where(X["geschlechtVP1"] == 'MAENNLICH', 1, 0)
        # Ordinal Encoding
        X['zahlweiseExkasso'] = pd.Categorical(X.zahlweiseExkasso,
                                               categories=['MONATLICH', 'VIERTELJAEHRLICH', 'HALBJAEHRLICH',
                                                           'JAEHRLICH'], ordered=True)
        ord = OrdinalEncoder(categories=[['MONATLICH', 'VIERTELJAEHRLICH', 'HALBJAEHRLICH', 'JAEHRLICH']])
        X['zahlweiseExkasso'] = ord.fit_transform(X[['zahlweiseExkasso']])
    else:
        # Get dataset from OpenML
        dataset = openml.datasets.get_dataset(dataset_id)

        X, y, categorical_indicator, attribute_names = dataset.get_data(
            target=dataset.default_target_attribute, dataset_format="dataframe"
        )
        dataset_name = datasets[dataset_id]

    # File Names
    savedir = './data/experiments/DSPP'
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    nested_resampling_infos_file = savedir + '/nested_resampling_infos_DSPP_' + dataset_name + '.csv'

    kf_outer = KFold(n_splits=n_splits_outer, shuffle=True, random_state=73)
    kf_inner = KFold(n_splits=n_splits_inner, shuffle=True, random_state=69)

    # One hot encoding of categorical variables
    ohe = OneHotEncoder(drop='if_binary')
    categorical_features = X.select_dtypes(include=['category', 'object']).columns

    # perform one-hot encoding on categorical columns column
    encoder_df = pd.DataFrame(ohe.fit_transform(X[categorical_features]).toarray())

    # merge one-hot encoded columns back with original DataFrame
    X = X.join(encoder_df)
    X.drop(categorical_features, axis=1, inplace=True)

    X = torch.Tensor(X.values)
    y = torch.Tensor(y.values)

    print(f'Starting Nested Resampling for {dataset_name}')

    split_nr = 1

    for train_outer, test_outer in kf_outer.split(np.array(X)):
        # Create a new dataset to track HPO progress, if it doesn't already exist
        try:
            nested_resampling_infos = pd.read_csv(nested_resampling_infos_file)
        except FileNotFoundError:
            nested_resampling_infos = pd.DataFrame({'dataset_id': [], 'dataset_name': [], 'outer_run': [],
                                                    'n_dspp_layers': [], 'n_dspp_out': [], 'batch_size': [], 'lr': [],
                                                    'num_inducing': [], 'num_quadrature_sites': [], 'beta': [],
                                                    'MAX_EPOCHS': [], 'mse': [],
                                                    'mae': [], 'max_absolute_error': []})

        X_train_outer, X_test_outer, y_train_outer, y_test_outer = X[train_outer], X[test_outer], y[train_outer], y[
            test_outer]

        # Scale Features and Labels
        scaler = MinMaxScaler()
        X_train_outer = scaler.fit_transform(X_train_outer)
        X_test_outer = scaler.transform(X_test_outer)

        y_scaler = MinMaxScaler()
        y_train_outer = y_scaler.fit_transform(y_train_outer.reshape(-1, 1)).flatten()

        # Turn back to tensors
        X_train_outer = torch.from_numpy(X_train_outer).float()
        X_test_outer = torch.from_numpy(X_test_outer).float()

        y_train_outer = torch.from_numpy(y_train_outer).float()

        if torch.cuda.is_available():
            X_train_outer, X_test_outer, y_train_outer, y_test_outer = X_train_outer.cuda(), X_test_outer.cuda(), \
                                                                       y_train_outer.cuda(), y_test_outer.cuda()

        # refit model
        best_trial = nested_resampling_infos.iloc[split_nr - 1]

        # Refit with best trial
        # best trial
        n_layers_best = int(best_trial['n_dspp_layers'])
        n_out_best = int(best_trial['n_dspp_out'])
        batch_size_best = int(best_trial['batch_size'])
        lr_best = best_trial['lr']
        num_inducing_best = int(best_trial['num_inducing'])
        num_quadrature_sites_best = int(best_trial['num_quadrature_sites'])
        beta_best = best_trial['beta']

        train_dataset = TensorDataset(X_train_outer, y_train_outer)
        train_loader = DataLoader(train_dataset, batch_size=batch_size_best, shuffle=True)

        test_dataset = TensorDataset(X_test_outer, y_test_outer)
        test_loader = DataLoader(test_dataset, batch_size=batch_size_best, shuffle=False)

        # initialize model
        output_dims = [n_out_best] * n_layers_best

        model = DSPPRegression(
            X_train_outer.shape,
            num_inducing=num_inducing_best,
            output_dims=output_dims,
            Q=num_quadrature_sites_best
        )

        if torch.cuda.is_available():
            model = model.cuda()

        # Use the adam optimizer
        optimizer = torch.optim.AdamW([
            {'params': model.parameters()},
        ], lr=lr_best)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.DeepPredictiveLogLikelihood(model.likelihood, model,
                                                        num_data=X_train_outer.size(0), beta=beta_best)
        early_stopping = EarlyStoppingWithModelSave(patience=patience_outer)
        start = time.time()
        for epoch in range(epochs_outer):
            # set to train mode
            model.train()
            epoch_loss = []
            for batch, (X_batch, y_batch) in enumerate(train_loader):
                # Zero gradient
                optimizer.zero_grad()
                # Output from model
                output = model(X_batch)
                # Calc loss and backprop gradients
                loss = -mll(output, y_batch)
                loss.backward()
                # adjust learning weights
                optimizer.step()
                epoch_loss.append(loss.item())

            epoch_loss = sum(epoch_loss) / len(epoch_loss)

            if epoch % 5 == 0:
                print(f"{epoch}/{epochs_outer} - loss: {epoch_loss}")

            early_stopping(epoch_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        end = time.time()

        # load the last checkpoint with the best model
        model.load_state_dict(torch.load('checkpoint.pt'))
        # Predictions
        model.eval()

        # Make predictions
        with torch.no_grad():
            means, vars, ll = model.predict(test_loader)
            # `means` currently contains the predictive output from each Gaussian in the mixture.
            # To get the total mean output, we take a weighted sum of these means over the quadrature weights.
            weights = model.quad_weights.unsqueeze(-1).exp().cpu()
        predictions = (weights * means).sum(0).detach().numpy()
        # variance https://en.wikipedia.org/wiki/Mixture_distribution
        variance = (weights * (vars + means.pow(2))).sum(0) - (weights * means).sum(0).pow(2)
        # rescale predictions to original format
        predictions_orig = torch.from_numpy(
            y_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()).float()

        mse_score = mse(y_test_outer.cpu(), predictions_orig).item()
        mae_score = mae(y_test_outer.cpu(), predictions_orig).item()
        max_score = maximum_absolute_error(y_test_outer.cpu(), predictions_orig).item()

        # NLL Var(Y)=Var(aX+b)=a^2*Var(X)
        y_min = y_scaler.data_min_
        y_max = y_scaler.data_max_
        var_transformed = variance.detach() * (y_max - y_min) ** 2
        mean_transformed = predictions_orig
        nll = torch.nn.GaussianNLLLoss()
        nll_score = nll(mean_transformed, y_test_outer.cpu(), var_transformed).item()
        train_time = end - start

        nested_resampling_info = pd.DataFrame({'dataset_id': [dataset_id], 'dataset_name': [dataset_name],
                                               'outer_run': [split_nr], 'n_dspp_layers': [n_layers_best],
                                               'n_dspp_out': [n_out_best], 'batch_size': [batch_size_best],
                                               'lr': [lr_best], 'num_inducing': [num_inducing_best],
                                               'num_quadrature_sites': [num_quadrature_sites_best],
                                               'beta': [beta_best],
                                               'MAX_EPOCHS': [epoch], 'mse': [mse_score],
                                               'mae': [mae_score], 'max_absolute_error': [max_score],
                                               'nll_score': [nll_score], 'train_time': [train_time]})

        nested_resampling_infos = pd.concat([nested_resampling_infos, nested_resampling_info])

        split_nr += 1

        nested_resampling_infos.to_csv(nested_resampling_infos_file, index=False)

        os.remove("checkpoint.pt")

        del X_train_outer, X_test_outer, y_train_outer, y_test_outer, train_dataset, train_loader, test_dataset, \
            test_loader, loss, optimizer, model, means, vars, ll
        gc.collect()
        torch.cuda.empty_cache()


def DE_NLL(n_splits_outer=5,
           n_splits_inner=4,
           file=None,
           PPV=True,
           dataset_id=216,
           epochs_outer=1000,
           patience_outer=10):
    # dict map id to name
    # OpenML Dataset to Study
    datasets = {287: "wine_quality", 216: "elevators", 42225: "diamonds", 4549: "buzz", 44027: "year"}

    if file == "k2204":
        file_path = './data/K2204_net_present_values.csv'
        df = pd.read_csv(file_path, sep=';', index_col=0)

        # prepare dataset
        X = df.iloc[:, 0:3]
        if PPV:
            y = df.iloc[:, -1]
            dataset_name = "k2204_PPV"
        else:
            y = df.iloc[:, -2]
            dataset_name = "k2204_BPV"
    elif file == "r1_08":
        file_path = '../trail.X/data/R1_08/r1_08.parquet'
        df = pd.read_parquet(file_path, engine='pyarrow')

        # prepare dataset
        X = df.iloc[:, 1:10]
        if PPV:
            y = df.iloc[:, 10]
            dataset_name = "r1_08_PPV"
        else:
            y = df.iloc[:, 11]
            dataset_name = "r1_08_BPV"

        # Feature Transformation
        X["GeburtsDatum"] = X["GeburtsDatum"].str[:4].astype(int) - 1985
        X['geschlechtVP1'] = np.where(X["geschlechtVP1"] == 'MAENNLICH', 1, 0)
        # Ordinal Encoding
        X['zahlweiseExkasso'] = pd.Categorical(X.zahlweiseExkasso,
                                               categories=['MONATLICH', 'VIERTELJAEHRLICH', 'HALBJAEHRLICH',
                                                           'JAEHRLICH'], ordered=True)
        ord = OrdinalEncoder(categories=[['MONATLICH', 'VIERTELJAEHRLICH', 'HALBJAEHRLICH', 'JAEHRLICH']])
        X['zahlweiseExkasso'] = ord.fit_transform(X[['zahlweiseExkasso']])
    elif file == "r1_08_small":
        file_path = '../trail.X/data/R1_08/r1_08.parquet'
        df = pd.read_parquet(file_path, engine='pyarrow')
        df = df.sample(600000, random_state=73)

        # prepare dataset
        X = df.iloc[:, 1:10]
        if PPV:
            y = df.iloc[:, 10]
            dataset_name = "r1_08_PPV"
        else:
            y = df.iloc[:, 11]
            dataset_name = "r1_08_BPV"

        # Feature Transformation
        X["GeburtsDatum"] = X["GeburtsDatum"].str[:4].astype(int) - 1985
        X['geschlechtVP1'] = np.where(X["geschlechtVP1"] == 'MAENNLICH', 1, 0)
        # Ordinal Encoding
        X['zahlweiseExkasso'] = pd.Categorical(X.zahlweiseExkasso,
                                               categories=['MONATLICH', 'VIERTELJAEHRLICH', 'HALBJAEHRLICH',
                                                           'JAEHRLICH'], ordered=True)
        ord = OrdinalEncoder(categories=[['MONATLICH', 'VIERTELJAEHRLICH', 'HALBJAEHRLICH', 'JAEHRLICH']])
        X['zahlweiseExkasso'] = ord.fit_transform(X[['zahlweiseExkasso']])
    else:
        # Get dataset from OpenML
        dataset = openml.datasets.get_dataset(dataset_id)

        X, y, categorical_indicator, attribute_names = dataset.get_data(
            target=dataset.default_target_attribute, dataset_format="dataframe"
        )
        dataset_name = datasets[dataset_id]

    # File Names
    savedir = './data/experiments/DE'
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    nested_resampling_infos_file = savedir + '/nested_resampling_infos_DE_' + dataset_name + '.csv'

    kf_outer = KFold(n_splits=n_splits_outer, shuffle=True, random_state=73)
    kf_inner = KFold(n_splits=n_splits_inner, shuffle=True, random_state=69)

    # One hot encoding of categorical variables
    ohe = OneHotEncoder(drop='if_binary')
    categorical_features = X.select_dtypes(include=['category', 'object']).columns

    # perform one-hot encoding on categorical columns column
    encoder_df = pd.DataFrame(ohe.fit_transform(X[categorical_features]).toarray())

    # merge one-hot encoded columns back with original DataFrame
    X = X.join(encoder_df)
    X.drop(categorical_features, axis=1, inplace=True)

    X = torch.Tensor(X.values)
    y = torch.Tensor(y.values)

    print(f'Starting Nested Resampling for {dataset_name}')

    split_nr = 1

    for train_outer, test_outer in kf_outer.split(np.array(X)):
        if split_nr != 3:
            split_nr += 1
            continue

        nested_resampling_infos = pd.read_csv(nested_resampling_infos_file)

        X_train_outer, X_test_outer, y_train_outer, y_test_outer = X[train_outer], X[test_outer], y[train_outer], y[
            test_outer]

        # Scale Features and Labels
        scaler = StandardScaler()
        X_train_outer = scaler.fit_transform(X_train_outer)
        X_test_outer = scaler.transform(X_test_outer)

        # Turn back to tensors
        X_train_outer = torch.from_numpy(X_train_outer).float()
        X_test_outer = torch.from_numpy(X_test_outer).float()

        if torch.cuda.is_available():
            X_train_outer, X_test_outer, y_train_outer, y_test_outer = X_train_outer.cuda(), X_test_outer.cuda(), \
                                                                       y_train_outer.cuda(), y_test_outer.cuda()

        # refit model
        best_trial = nested_resampling_infos.iloc[split_nr-1]

        # Refit with best trial
        # best trial
        n_max_best = int(best_trial['n_max'])
        n_layers_best = int(best_trial['n_layers'])
        lr_best = best_trial['lr']
        batch_size_best = int(best_trial['batch_size'])

        train_dataset = TensorDataset(X_train_outer, y_train_outer)
        train_loader = DataLoader(train_dataset, batch_size=batch_size_best, shuffle=True)

        test_dataset = TensorDataset(X_test_outer, y_test_outer)
        test_loader = DataLoader(test_dataset, batch_size=batch_size_best, shuffle=False)

        # initialize model
        de_model_best = DeepEnsemble(
            num_models=5,
            input_dim=X_train_outer.size(-1),
            n_max=n_max_best,
            n_layers=n_layers_best
        )
        if torch.cuda.is_available():
            de_model_best = de_model_best.cuda()

        # Initialize Optimizer
        optimizers = []
        for i in range(de_model_best.num_models):
            model = getattr(de_model_best, 'Model_' + str(i + 1))
            optimizers.append(torch.optim.AdamW(params=model.parameters(), lr=lr_best))

        start = time.time()
        # Train Each Model till early stopp
        for i in range(de_model_best.num_models):
            # get Model
            model = getattr(de_model_best, 'Model_' + str(i + 1))
            # Initialize Early Stopping
            early_stopping = EarlyStoppingWithModelSave(patience=patience_outer)
            # initialize loss
            nll_loss = torch.nn.GaussianNLLLoss()

            for epoch in range(epochs_outer):
                model.train()
                epoch_loss = []
                for batch, (X_batch, y_batch) in enumerate(train_loader):
                    optimizers[i].zero_grad()
                    mean, var = model(X_batch)
                    loss = nll_loss(mean.flatten(), y_batch, var.flatten())
                    loss.backward()
                    optimizers[i].step()
                    epoch_loss.append(loss.item())

                epoch_loss = sum(epoch_loss) / len(epoch_loss)
                # Early Stopping
                early_stopping(epoch_loss, model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    model.load_state_dict(torch.load('checkpoint.pt'))
                    break

        end = time.time()

        model.eval()
        # Calculate Ensemble Prediction
        ens_means = torch.tensor([0.])
        ens_var = torch.tensor([0.])
        with torch.no_grad():
            for batch, (X_batch, y_batch) in enumerate(test_loader):
                mean, var = de_model_best(X_batch)
                ens_means = torch.cat([ens_means, mean.flatten().cpu()])
                ens_var = torch.cat([ens_var, var.flatten().cpu()])

        ens_means = ens_means[1:]
        ens_var = ens_var[1:]

        mse_score = mse(y_test_outer.cpu(), ens_means).item()
        mae_score = mae(y_test_outer.cpu(), ens_means).item()
        max_score = maximum_absolute_error(y_test_outer.cpu(), ens_means).item()
        nll = torch.nn.GaussianNLLLoss()

        ens_var = torch.where(ens_var > 0, ens_var, 1e-06)
        try:
            nll_score = nll(ens_means, y_test_outer.cpu(), ens_var).item()
        except ValueError:
            nll_score = -99
        train_time = end-start
        # Predictions
        nested_resampling_info = pd.DataFrame({'dataset_id': [dataset_id], 'dataset_name': [dataset_name],
                                               'outer_run': [split_nr], 'n_layers': [n_layers_best],
                                               'n_max': [n_max_best], 'batch_size': [batch_size_best],
                                               'lr': [lr_best], 'mse': [mse_score],
                                               'mae': [mae_score], 'max_absolute_error': [max_score],
                                               'nll_score': [nll_score], 'train_time': [train_time]})

        nested_resampling_infos = pd.concat([nested_resampling_infos, nested_resampling_info])

        split_nr += 1

        os.remove("checkpoint.pt")

        nested_resampling_infos.to_csv(nested_resampling_infos_file, index=False)


def DGP_NLL(n_splits_outer=5,
            n_splits_inner=4,
            file=None,
            PPV=True,
            dataset_id=216,
            epochs_outer=1000,
            patience_outer=10):

    # dict map id to name
    # OpenML Dataset to Study
    datasets = {287: "wine_quality", 216: "elevators", 42225: "diamonds", 4549: "buzz", 44027: "year"}

    if file == "k2204":
        file_path = './data/K2204_net_present_values.csv'
        df = pd.read_csv(file_path, sep=';', index_col=0)

        # prepare dataset
        X = df.iloc[:, 0:3]
        if PPV:
            y = df.iloc[:, -1]
            dataset_name = "k2204_PPV"
        else:
            y = df.iloc[:, -2]
            dataset_name = "k2204_BPV"
    elif file == "r1_08":
        file_path = '../trail.X/data/R1_08/r1_08.parquet'
        df = pd.read_parquet(file_path, engine='pyarrow')

        # prepare dataset
        X = df.iloc[:, 1:10]
        if PPV:
            y = df.iloc[:, 10]
            dataset_name = "r1_08_PPV"
        else:
            y = df.iloc[:, 11]
            dataset_name = "r1_08_BPV"

        # Feature Transformation
        X["GeburtsDatum"] = X["GeburtsDatum"].str[:4].astype(int) - 1985
        X['geschlechtVP1'] = np.where(X["geschlechtVP1"] == 'MAENNLICH', 1, 0)
        # Ordinal Encoding
        X['zahlweiseExkasso'] = pd.Categorical(X.zahlweiseExkasso,
                                               categories=['MONATLICH', 'VIERTELJAEHRLICH', 'HALBJAEHRLICH',
                                                           'JAEHRLICH'], ordered=True)
        ord = OrdinalEncoder(categories=[['MONATLICH', 'VIERTELJAEHRLICH', 'HALBJAEHRLICH', 'JAEHRLICH']])
        X['zahlweiseExkasso'] = ord.fit_transform(X[['zahlweiseExkasso']])
    elif file == "r1_08_small":
        file_path = '../trail.X/data/R1_08/r1_08.parquet'
        df = pd.read_parquet(file_path, engine='pyarrow')
        df = df.sample(600000, random_state=73)

        # prepare dataset
        X = df.iloc[:, 1:10]
        if PPV:
            y = df.iloc[:, 10]
            dataset_name = "r1_08_PPV"
        else:
            y = df.iloc[:, 11]
            dataset_name = "r1_08_BPV"

        # Feature Transformation
        X["GeburtsDatum"] = X["GeburtsDatum"].str[:4].astype(int) - 1985
        X['geschlechtVP1'] = np.where(X["geschlechtVP1"] == 'MAENNLICH', 1, 0)
        # Ordinal Encoding
        X['zahlweiseExkasso'] = pd.Categorical(X.zahlweiseExkasso,
                                               categories=['MONATLICH', 'VIERTELJAEHRLICH', 'HALBJAEHRLICH',
                                                           'JAEHRLICH'], ordered=True)
        ord = OrdinalEncoder(categories=[['MONATLICH', 'VIERTELJAEHRLICH', 'HALBJAEHRLICH', 'JAEHRLICH']])
        X['zahlweiseExkasso'] = ord.fit_transform(X[['zahlweiseExkasso']])
    else:
        # Get dataset from OpenML
        dataset = openml.datasets.get_dataset(dataset_id)

        X, y, categorical_indicator, attribute_names = dataset.get_data(
            target=dataset.default_target_attribute, dataset_format="dataframe"
        )
        dataset_name = datasets[dataset_id]

    # File Names
    savedir = './data/experiments/DGP'
    nested_resampling_infos_file = savedir + '/nested_resampling_infos_DGP_' + dataset_name + '.csv'

    kf_outer = KFold(n_splits=n_splits_outer, shuffle=True, random_state=73)
    kf_inner = KFold(n_splits=n_splits_inner, shuffle=True, random_state=69)

    # One hot encoding of categorical variables
    ohe = OneHotEncoder(drop='if_binary')
    categorical_features = X.select_dtypes(include=['category', 'object']).columns

    # perform one-hot encoding on categorical columns column
    encoder_df = pd.DataFrame(ohe.fit_transform(X[categorical_features]).toarray())

    # merge one-hot encoded columns back with original DataFrame
    X = X.join(encoder_df)
    X.drop(categorical_features, axis=1, inplace=True)

    X = torch.Tensor(X.values)
    y = torch.Tensor(y.values)

    print(f'Starting Nested Resampling for {dataset_name}')

    split_nr = 1

    for train_outer, test_outer in kf_outer.split(np.array(X)):
        # Create a new dataset to track HPO progress, if it doesn't already exist
        nested_resampling_infos = pd.read_csv(nested_resampling_infos_file)

        X_train_outer, X_test_outer, y_train_outer, y_test_outer = X[train_outer], X[test_outer], y[train_outer], y[
            test_outer]

        # Scale Features and Labels
        scaler = MinMaxScaler()
        X_train_outer = scaler.fit_transform(X_train_outer)
        X_test_outer = scaler.transform(X_test_outer)

        y_scaler = MinMaxScaler()
        y_train_outer = y_scaler.fit_transform(y_train_outer.reshape(-1, 1)).flatten()

        # Turn back to tensors
        X_train_outer = torch.from_numpy(X_train_outer).float()
        X_test_outer = torch.from_numpy(X_test_outer).float()

        y_train_outer = torch.from_numpy(y_train_outer).float()

        if torch.cuda.is_available():
            X_train_outer, X_test_outer, y_train_outer, y_test_outer = X_train_outer.cuda(), X_test_outer.cuda(), \
                                                                       y_train_outer.cuda(), y_test_outer.cuda()

        # refit model
        best_trial = nested_resampling_infos.iloc[split_nr-1]

        # Refit with best trial
        # best trial
        n_layers_best = int(best_trial['n_layers'])
        n_out_best = int(best_trial['n_out'])
        batch_size_best = int(best_trial['batch_size'])
        lr_best = best_trial['lr']
        num_inducing_best = int(best_trial['num_inducing'])
        num_samples_best = int(best_trial['num_samples'])
        kernel_type_best = best_trial['kernel_type']

        train_dataset = TensorDataset(X_train_outer, y_train_outer)
        train_loader = DataLoader(train_dataset, batch_size=batch_size_best, shuffle=True)

        test_dataset = TensorDataset(X_test_outer, y_test_outer)
        test_loader = DataLoader(test_dataset, batch_size=batch_size_best, shuffle=False)

        # initialize model
        output_dims = [n_out_best] * n_layers_best
        model = DeepGPRegression(train_x_shape=X_train_outer.shape, output_dims=output_dims,
                                 num_inducing=num_inducing_best, kernel_type=kernel_type_best)

        if torch.cuda.is_available():
            model = model.cuda()

        # Use the adam optimizer
        optimizer = torch.optim.AdamW([
            {'params': model.parameters()},
        ], lr=lr_best)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.DeepApproximateMLL(
            gpytorch.mlls.VariationalELBO(model.likelihood, model, X_train_outer.shape[-2]))
        early_stopping = EarlyStoppingWithModelSave(patience=patience_outer)
        # time training loop
        start = time.time()
        for epoch in range(epochs_outer):
            # set to train mode
            model.train()
            epoch_loss = []
            for batch, (X_batch, y_batch) in enumerate(train_loader):
                with gpytorch.settings.num_likelihood_samples(num_samples_best):
                    # Zero gradient
                    optimizer.zero_grad()
                    # Output from model
                    output = model(X_batch)
                    # Calc loss and backprop gradients
                    loss = -mll(output, y_batch)
                    loss.backward()
                    # adjust learning weights
                    optimizer.step()
                    epoch_loss.append(loss.item())

            epoch_loss = sum(epoch_loss) / len(epoch_loss)

            if epoch % 5 == 0:
                print(f"{epoch}/{epochs_outer} - loss: {epoch_loss}")

            early_stopping(epoch_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        end = time.time()

        # load the last checkpoint with the best model
        model.load_state_dict(torch.load('checkpoint.pt'))

        # Predictions
        model.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions, predictive_variances, test_lls = model.predict(test_loader)

        predictions_orig = torch.from_numpy(
            y_scaler.inverse_transform(predictions.mean(0).cpu().reshape(-1, 1)).flatten()).float()

        mse_score = mse(y_test_outer.cpu(), predictions_orig).item()
        mae_score = mae(y_test_outer.cpu(), predictions_orig).item()
        max_score = maximum_absolute_error(y_test_outer.cpu(), predictions_orig).item()


        # NLL Var(Y)=Var(aX+b)=a^2*Var(X)
        var = predictive_variances.mean(0).cpu()
        y_min = y_scaler.data_min_
        y_max = y_scaler.data_max_
        var_transformed = var * (y_max - y_min) ** 2
        mean_transformed = predictions_orig
        nll = torch.nn.GaussianNLLLoss()
        nll_score = nll(mean_transformed, y_test_outer.cpu(), var_transformed).item()
        train_time = end - start

        nested_resampling_info = pd.DataFrame({'dataset_id': [dataset_id], 'dataset_name': [dataset_name],
                                               'outer_run': [split_nr], 'n_layers': [n_layers_best],
                                               'n_out': [n_out_best], 'batch_size': [batch_size_best],
                                               'lr': [lr_best], 'num_inducing': [num_inducing_best],
                                               'num_samples': [num_samples_best], 'kernel_type': [kernel_type_best],
                                               'MAX_EPOCHS': [epoch], 'mse': [mse_score],
                                               'mae': [mae_score], 'max_absolute_error': [max_score],
                                               'nll_score': [nll_score], 'train_time': [train_time]})

        nested_resampling_infos = pd.concat([nested_resampling_infos, nested_resampling_info])

        split_nr += 1

        nested_resampling_infos.to_csv(nested_resampling_infos_file, index=False)

        os.remove("checkpoint.pt")

        # Memory Tracking
        del X_train_outer, X_test_outer, y_train_outer, y_test_outer, train_dataset, train_loader, test_dataset,\
            test_loader, loss, optimizer, model, predictions, predictive_variances, test_lls, output, mll
        gc.collect()
        torch.cuda.empty_cache()


def NNDGP_NLL(n_splits_outer=5,
              n_splits_inner=4,
              file=None,
              PPV=True,
              dataset_id=216,
              epochs_outer=1000,
              patience_outer=10):

    # dict map id to name
    # OpenML Dataset to Study
    datasets = {287: "wine_quality", 216: "elevators", 42225: "diamonds", 4549: "buzz", 44027: "year"}

    if file == "k2204":
        file_path = './data/K2204_net_present_values.csv'
        df = pd.read_csv(file_path, sep=';', index_col=0)

        # prepare dataset
        X = df.iloc[:, 0:3]
        if PPV:
            y = df.iloc[:, -1]
            dataset_name = "k2204_PPV"
        else:
            y = df.iloc[:, -2]
            dataset_name = "k2204_BPV"
    elif file == "r1_08":
        file_path = '../trail.X/data/R1_08/r1_08.parquet'
        df = pd.read_parquet(file_path, engine='pyarrow')

        # prepare dataset
        X = df.iloc[:, 1:10]
        if PPV:
            y = df.iloc[:, 10]
            dataset_name = "r1_08_PPV"
        else:
            y = df.iloc[:, 11]
            dataset_name = "r1_08_BPV"

        # Feature Transformation
        X["GeburtsDatum"] = X["GeburtsDatum"].str[:4].astype(int) - 1985
        X['geschlechtVP1'] = np.where(X["geschlechtVP1"] == 'MAENNLICH', 1, 0)
        # Ordinal Encoding
        X['zahlweiseExkasso'] = pd.Categorical(X.zahlweiseExkasso,
                                               categories=['MONATLICH', 'VIERTELJAEHRLICH', 'HALBJAEHRLICH',
                                                           'JAEHRLICH'], ordered=True)
        ord = OrdinalEncoder(categories=[['MONATLICH', 'VIERTELJAEHRLICH', 'HALBJAEHRLICH', 'JAEHRLICH']])
        X['zahlweiseExkasso'] = ord.fit_transform(X[['zahlweiseExkasso']])
    elif file == "r1_08_small":
        file_path = '../trail.X/data/R1_08/r1_08.parquet'
        df = pd.read_parquet(file_path, engine='pyarrow')
        df = df.sample(600000, random_state=73)

        # prepare dataset
        X = df.iloc[:, 1:10]
        if PPV:
            y = df.iloc[:, 10]
            dataset_name = "r1_08_PPV"
        else:
            y = df.iloc[:, 11]
            dataset_name = "r1_08_BPV"

        # Feature Transformation
        X["GeburtsDatum"] = X["GeburtsDatum"].str[:4].astype(int) - 1985
        X['geschlechtVP1'] = np.where(X["geschlechtVP1"] == 'MAENNLICH', 1, 0)
        # Ordinal Encoding
        X['zahlweiseExkasso'] = pd.Categorical(X.zahlweiseExkasso,
                                               categories=['MONATLICH', 'VIERTELJAEHRLICH', 'HALBJAEHRLICH',
                                                           'JAEHRLICH'], ordered=True)
        ord = OrdinalEncoder(categories=[['MONATLICH', 'VIERTELJAEHRLICH', 'HALBJAEHRLICH', 'JAEHRLICH']])
        X['zahlweiseExkasso'] = ord.fit_transform(X[['zahlweiseExkasso']])
    else:
        # Get dataset from OpenML
        dataset = openml.datasets.get_dataset(dataset_id)

        X, y, categorical_indicator, attribute_names = dataset.get_data(
            target=dataset.default_target_attribute, dataset_format="dataframe"
        )
        dataset_name = datasets[dataset_id]

    # File Names
    savedir = './data/experiments/NNDGP'
    nested_resampling_infos_file = savedir + '/nested_resampling_infos_NNDGP_' + dataset_name + '.csv'

    kf_outer = KFold(n_splits=n_splits_outer, shuffle=True, random_state=73)
    kf_inner = KFold(n_splits=n_splits_inner, shuffle=True, random_state=69)

    # One hot encoding of categorical variables
    ohe = OneHotEncoder(drop='if_binary')
    categorical_features = X.select_dtypes(include=['category', 'object']).columns

    # perform one-hot encoding on categorical columns column
    encoder_df = pd.DataFrame(ohe.fit_transform(X[categorical_features]).toarray())

    # merge one-hot encoded columns back with original DataFrame
    X = X.join(encoder_df)
    X.drop(categorical_features, axis=1, inplace=True)

    X = torch.Tensor(X.values)
    y = torch.Tensor(y.values)

    print(f'Starting Nested Resampling for {dataset_name}')

    split_nr = 1

    for train_outer, test_outer in kf_outer.split(np.array(X)):
        # Create a new dataset to track HPO progress, if it doesn't already exist
        nested_resampling_infos = pd.read_csv(nested_resampling_infos_file)

        X_train_outer, X_test_outer, y_train_outer, y_test_outer = X[train_outer], X[test_outer], y[train_outer], y[
            test_outer]

        # Scale Features and Labels
        scaler = MinMaxScaler()
        X_train_outer = scaler.fit_transform(X_train_outer)
        X_test_outer = scaler.transform(X_test_outer)

        y_scaler = MinMaxScaler()
        y_train_outer = y_scaler.fit_transform(y_train_outer.reshape(-1, 1)).flatten()

        # Turn back to tensors
        X_train_outer = torch.from_numpy(X_train_outer).float()
        X_test_outer = torch.from_numpy(X_test_outer).float()

        y_train_outer = torch.from_numpy(y_train_outer).float()

        if torch.cuda.is_available():
            X_train_outer, X_test_outer, y_train_outer, y_test_outer = X_train_outer.cuda(), X_test_outer.cuda(), \
                                                                       y_train_outer.cuda(), y_test_outer.cuda()

        # refit model
        best_trial = nested_resampling_infos.iloc[split_nr-1]

        # Refit with best trial
        # best trial
        n_max_best = int(best_trial['n_max'])
        n_layers_best = int(best_trial['n_layers'])
        n_out_best = int(best_trial['n_out'])
        n_gp_layers_best = int(best_trial['n_gp_layers'])
        n_gp_out_best = int(best_trial['n_gp_out'])
        batch_size_best = int(best_trial['batch_size'])
        lr_best = best_trial['lr']
        num_inducing_best = int(best_trial['num_inducing'])
        num_samples_best = int(best_trial['num_samples'])
        # As max epochs inferred through smaller dataset, increase it by 10%
        epochs_outer = int(best_trial['MAX_EPOCHS'] * 1.2)


        train_dataset = TensorDataset(X_train_outer, y_train_outer)
        train_loader = DataLoader(train_dataset, batch_size=batch_size_best, shuffle=True)

        test_dataset = TensorDataset(X_test_outer, y_test_outer)
        test_loader = DataLoader(test_dataset, batch_size=batch_size_best, shuffle=False)

        # initialize model
        input_dim = X_train_outer.size(-1)
        nn_best = NNFeatureExtractor(input_dim=input_dim, n_max=n_max_best,
                                     n_layers=n_layers_best, n_out=n_out_best)

        # initialize model
        output_dims = [n_gp_out_best] * n_gp_layers_best
        model = NNDGPRegression(feature_extractor=nn_best, output_dims=output_dims,
                                num_inducing=num_inducing_best)

        if torch.cuda.is_available():
            model = model.cuda()

        # Use the adam optimizer
        optimizer = torch.optim.AdamW([
            {'params': model.parameters()},
        ], lr=lr_best)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.DeepApproximateMLL(
            gpytorch.mlls.VariationalELBO(model.likelihood, model, X_train_outer.shape[-2]))
        early_stopping = EarlyStoppingWithModelSave(patience=patience_outer)
        start = time.time()
        for epoch in range(epochs_outer):
            # set to train mode
            model.train()
            epoch_loss = []
            for batch, (X_batch, y_batch) in enumerate(train_loader):
                with gpytorch.settings.num_likelihood_samples(num_samples_best):
                    # Zero gradient
                    optimizer.zero_grad()
                    # Output from model
                    output = model(X_batch)
                    # Calc loss and backprop gradients
                    loss = -mll(output, y_batch)
                    loss.backward()
                    # adjust learning weights
                    optimizer.step()
                    epoch_loss.append(loss.item())

            epoch_loss = sum(epoch_loss) / len(epoch_loss)

            if epoch % 5 == 0:
                print(f"{epoch}/{epochs_outer} - loss: {epoch_loss}")

            early_stopping(epoch_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        end = time.time()

        # load the last checkpoint with the best model
        model.load_state_dict(torch.load('checkpoint.pt'))

        # Predictions
        model.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions, predictive_variances, test_lls = model.predict(test_loader)

        predictions_orig = torch.from_numpy(
            y_scaler.inverse_transform(predictions.mean(0).cpu().reshape(-1, 1)).flatten()).float()

        mse_score = mse(y_test_outer.cpu(), predictions_orig).item()
        mae_score = mae(y_test_outer.cpu(), predictions_orig).item()
        max_score = maximum_absolute_error(y_test_outer.cpu(), predictions_orig).item()

        # NLL Var(Y)=Var(aX+b)=a^2*Var(X)
        var = predictive_variances.mean(0).cpu()
        y_min = y_scaler.data_min_
        y_max = y_scaler.data_max_
        var_transformed = var * (y_max - y_min) ** 2
        mean_transformed = predictions_orig
        nll = torch.nn.GaussianNLLLoss()
        nll_score = nll(mean_transformed, y_test_outer.cpu(), var_transformed).item()
        train_time = end - start

        nested_resampling_info = pd.DataFrame({'dataset_id': [dataset_id], 'dataset_name': [dataset_name],
                                               'outer_run': [split_nr], 'n_max': [n_max_best],
                                               'n_layers': [n_layers_best], 'n_out': [n_out_best],
                                               'n_gp_layers': [n_gp_layers_best], 'n_gp_out': [n_gp_out_best],
                                               'batch_size': [batch_size_best], 'lr': [lr_best],
                                               'num_inducing': [num_inducing_best], 'num_samples': [num_samples_best],
                                               'MAX_EPOCHS': [epoch], 'mse': [mse_score],
                                               'mae': [mae_score], 'max_absolute_error': [max_score],
                                               'nll_score': [nll_score], 'train_time': [train_time]})

        nested_resampling_infos = pd.concat([nested_resampling_infos, nested_resampling_info])

        split_nr += 1

        nested_resampling_infos.to_csv(nested_resampling_infos_file, index=False)

        os.remove("checkpoint.pt")

        # Memory Tracking
        del X_train_outer, X_test_outer, y_train_outer, y_test_outer, train_dataset, train_loader, test_dataset,\
            test_loader, loss, optimizer, model, predictions, predictive_variances, test_lls, output, mll
        gc.collect()
        torch.cuda.empty_cache()



if __name__ == "__main__":
    # DKL_NLL(n_splits_outer=5,
    #         n_splits_inner=4,
    #         file="r1_08_small",
    #         PPV=False,
    #         dataset_id=44027)
    #
    # VDKL_NLL(n_splits_outer=5,
    #          n_splits_inner=4,
    #          file="r1_08_small",
    #          PPV=False,
    #          dataset_id=44027,
    #          epochs_outer=1000,
    #          patience_outer=10)

    # NNDGP_NLL(n_splits_outer=5,
    #           n_splits_inner=4,
    #           file="None",
    #           PPV=False,
    #           dataset_id=216,
    #           epochs_outer=1000,
    #           patience_outer=5)
    #
    # NNDGP_NLL(n_splits_outer=5,
    #           n_splits_inner=4,
    #           file="None",
    #           PPV=False,
    #           dataset_id=42225,
    #           epochs_outer=1000,
    #           patience_outer=5)
    #
    # DSPP_NLL(n_splits_outer=5,
    #          n_splits_inner=4,
    #          file="r1_08_small",
    #          PPV=False,
    #          dataset_id=216,
    #          epochs_outer=1000,
    #          patience_outer=10)
    #
    # DGP_NLL(n_splits_outer=5,
    #         n_splits_inner=4,
    #         file="r1_08_small",
    #         PPV=False,
    #         dataset_id=287,
    #         epochs_outer=1000,
    #         patience_outer=10)

    DE_NLL(n_splits_outer=5,
           n_splits_inner=4,
           file="r1_08_small",
           PPV=False,
           dataset_id=44027,
           epochs_outer=1000,
           patience_outer=10)

    # DE_NLL(n_splits_outer=5,
    #        n_splits_inner=4,
    #        file="r1_08_small",
    #        PPV=True,
    #        dataset_id=216,
    #        epochs_outer=1000,
    #        patience_outer=10)
