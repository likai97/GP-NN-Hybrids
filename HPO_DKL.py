import pandas as pd
import numpy as np
import optuna
import openml
import os
import sys
import logging
import time
import math

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import KFold, train_test_split

import torch
import gpytorch

from src.utils.metric_utils import mse, mae, maximum_absolute_error
from src.utils.model_utils import NNFeatureExtractor, initiliaze_feature_extractor
from src.utils.train_utils import EarlyStopping

from src.deep_kernel_learning import DKLRegression

import gc


def clear_cuda_memory(*args):
    for i in args:
        del i
    gc.collect()
    torch.cuda.empty_cache()


def objective_time_prune(trial, X, y, kf,
                         epochs_inner, patience,
                         time_tolerance=900,
                         metric=mse):
    """
    Optuna trial. Performs n_splits CV to calculate a score for a given Hyperparameter Config

    Args:
        trial: Optuna trial
        X: features
        y: labels
        kf: Sklearn KFold Object
        epochs_inner: Number of inner epochs
        patience: Number of early stopping iterations
        time_tolerance:  Maximum number of seconds before cancel the trial
        metric: Evaluation metric to be optimized

    Returns:
        Average score of chosen metric of n_splits CV folds
    """
    # trial parameters
    n_max = trial.suggest_int("n_max", 64, 1024)  # log = True?
    n_layers = trial.suggest_int("n_layers", 2, 5)
    n_out = trial.suggest_int("n_out", 2, 4)
    p = trial.suggest_float("p", 0.0, 0.0)
    mean_type = trial.suggest_categorical("mean_type", ["constant", "linear"])
    kernel_type = trial.suggest_categorical("kernel_type", ["rbf", "matern0.5", "matern1.5", "spectral"])
    if kernel_type == "spectral":
        num_mixtures = trial.suggest_int("num_mixtures", 2, 8)
    else:
        num_mixtures = None
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    # Memory requirements scale exponentially with n_out: Reduce Grid size
    if n_out <= 2:
        grid_size = trial.suggest_int("grid_size", 10, 400)
    elif n_out == 3:
        grid_size = trial.suggest_int("grid_size", 10, 50)
    else:
        grid_size = trial.suggest_int("grid_size", 10, 20)
    initialize_fe = trial.suggest_categorical("initialize_fe", [True, False])

    print(
        f'Current Configuration: n_out:{n_out} - n_max: {n_max} - n_layer: {n_layers} - mean_type: {mean_type} - \
         kernel_type: {kernel_type} - num_mixtures: {num_mixtures} - lr: {lr} - grid_size: {grid_size}')
    logging.info(
        f'Current Configuration: n_out:{n_out} - n_max: {n_max} - n_layer: {n_layers} - mean_type: {mean_type} - \
             kernel_type: {kernel_type} - num_mixtures: {num_mixtures} - lr: {lr} - grid_size: {grid_size}')

    start_time = time.time()
    early_stopping = EarlyStopping(patience=patience)
    # Inner Nested Resampling Split
    for epoch in range(epochs_inner):
        cur_split_nr = 1
        scores = []
        for train_inner, val_inner in kf.split(X):
            # Temp model name for this cv split to load and save
            temp_model_name = './models/temp/DKL_temp_split_' + str(cur_split_nr) + '.pt'

            # Train/Val Split
            X_train_inner, X_val_inner, y_train_inner, y_val_inner = X[train_inner], X[val_inner], y[train_inner], y[
                val_inner]
            if torch.cuda.is_available():
                X_train_inner, X_val_inner, y_train_inner, y_val_inner = X_train_inner.cuda(), X_val_inner.cuda(), \
                                                                         y_train_inner.cuda(), y_val_inner.cuda()

            # initialize likelihood and model
            input_dim = X_train_inner.size(-1)
            feature_extractor = NNFeatureExtractor(input_dim=input_dim, n_max=n_max,
                                                   n_layers=n_layers, n_out=n_out, p=p)
            if initialize_fe and epoch == 0:
                feature_extractor = initiliaze_feature_extractor(X_train_inner, y_train_inner,
                                                                 feature_extractor=feature_extractor)

            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            model = DKLRegression(X_train_inner, y_train_inner, likelihood,
                                  feature_extractor, mean_type=mean_type,
                                  kernel_type=kernel_type, num_mixtures=num_mixtures,
                                  grid_size=grid_size)
            # Load existing model if not the first epoch
            if epoch != 0:
                state_dict = torch.load(temp_model_name)
                model.load_state_dict(state_dict)

            # delete feature extractor to free up memory
            del feature_extractor
            torch.cuda.empty_cache()

            if torch.cuda.is_available():
                model = model.cuda()
                likelihood = likelihood.cuda()

            # Use the adam optimizer
            optimizer = torch.optim.AdamW([
                {'params': model.feature_extractor.parameters()},
                {'params': model.covar_module.parameters()},
                {'params': model.mean_module.parameters()},
                {'params': model.likelihood.parameters()},
            ], lr=lr)

            # "Loss" for GPs - the marginal log likelihood
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

            # Train Model for 5 epochs
            for i in range(1):
                model.train()
                likelihood.train()
                # Zero gradients from previous iteration
                optimizer.zero_grad()
                # Output from model
                output = model(X_train_inner)
                # Calc loss and backprop gradients
                loss = -mll(output, y_train_inner)
                # Calculate Gradients
                loss.backward()
                optimizer.step()

            # Due to numerical stability issues which can trigger an
            # RuntimeError('CUDA error: device-side assert triggered')
            # Early stop the inner Loop and break out
            if math.isnan(loss.item()):
                logging.info("NaN loss encountered")
                break

            # Get into evaluation (predictive posterior) mode
            model.eval()
            likelihood.eval()

            # Test points are regularly spaced along [0,1]
            # Make predictions by feeding model through likelihood
            with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var():
                predictions = model(X_val_inner)
                score = metric(predictions.mean, y_val_inner)

            # Append Model Scores
            scores.append(score.cpu().item())

            # Save model
            torch.save(model.state_dict(), temp_model_name)
            # Increase cur_split_nr by 1
            cur_split_nr += 1

        else:
            # average Scores
            average_scores = np.mean(scores)

            # #HB pruner
            # trial.report(score.item(), epoch)
            # if trial.should_prune():
            #     raise optuna.TrialPruned()

            # Early Stopping
            early_stopping(average_scores)
            if early_stopping.early_stop:
                print("Early stopping")
                logging.info("Early stopping")
                break

            # Time based pruner
            train_time = time.time()
            if train_time - start_time > time_tolerance:
                print("Time Budget run out. Pruning Trial")
                logging.info("Time Budget run out. Pruning Trial")
                break

            print(f"{epoch}/{epochs_inner} - Loss: {loss} - Score: {average_scores}")
            continue
        break

    #delete temp models
    cur_split_nr = 1
    for train_inner, val_inner in kf.split(X):
        # Temp model name for this cv split to load and save
        temp_model_name = './models/temp/DKL_temp_split_' + str(cur_split_nr) + '.pt'
        os.remove(temp_model_name)
        cur_split_nr += 1

    # Memory Tracking
    logging.info("After model training")
    logging.info(torch.cuda.memory_allocated() / 1024 ** 2)
    logging.info(torch.cuda.memory_reserved() / 1024 ** 2)
    X_train_inner.detach()
    y_train_inner.detach()
    X_val_inner.detach()
    y_val_inner.detach()
    loss.detach()
    optimizer.zero_grad(set_to_none=True)
    likelihood.zero_grad(set_to_none=True)
    mll.zero_grad(set_to_none=True)
    del loss, optimizer, model, likelihood, output, mll
    gc.collect()
    torch.cuda.empty_cache()
    logging.info("After memory clearing")
    logging.info(torch.cuda.memory_allocated() / 1024 ** 2)
    logging.info(torch.cuda.memory_reserved() / 1024 ** 2)

    # Set max epochs
    # Max iteration reached
    if epoch == epochs_inner - 1:
        max_epochs = epochs_inner
    # Early stopping
    else:
        max_epochs = max(1, epoch - patience + 1)
    trial.set_user_attr("MAX_EPOCHS", int(max_epochs))

    # Have to take negative due to early stopping logic
    best_score = -early_stopping.best_score

    return best_score


def objective_train_test(trial, X, y, epochs_inner, patience, time_tolerance=1800, metric=mse):
    """
    Optuna trial. Performs n_splits CV to calculate a score for a given Hyperparameter Config

    Args:
        trial: Optuna trial
        X: features
        y: labels
                epochs_inner: Number of inner epochs
        patience: Number of early stopping iterations
        time_tolerance:  Maximum number of seconds before cancel the trial
        metric: Evaluation metric to be optimized

    Returns:
        Average score of chosen metric of n_splits CV folds
    """
    # trial parameters
    n_max = trial.suggest_int("n_max", 64, 1024)  # log = True?
    n_layers = trial.suggest_int("n_layers", 2, 5)
    n_out = trial.suggest_int("n_out", 2, 2)  #3/4 OOM error
    p = trial.suggest_float("p", 0.0, 0.0)
    mean_type = trial.suggest_categorical("mean_type", ["constant", "linear"])
    kernel_type = trial.suggest_categorical("kernel_type", ["rbf", "matern0.5", "matern1.5", "spectral"])
    if kernel_type == "spectral":
        num_mixtures = trial.suggest_int("num_mixtures", 2, 8)
    else:
        num_mixtures = None
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    # Memory requirements scale exponentially with n_out: Reduce Grid size
    if n_out <= 2:
        grid_size = trial.suggest_int("grid_size", 10, 300)
    elif n_out == 3:
        grid_size = trial.suggest_int("grid_size", 10, 30)
    else:
        grid_size = trial.suggest_int("grid_size", 10, 20)
    initialize_fe = trial.suggest_categorical("initialize_fe", [True, False])

    print(
        f'Current Configuration: n_out:{n_out} - n_max: {n_max} - n_layer: {n_layers} - mean_type: {mean_type} - \
         kernel_type: {kernel_type} - num_mixtures: {num_mixtures} - lr: {lr} - grid_size: {grid_size}')

    logging.info(
        f'Current Configuration: n_out:{n_out} - n_max: {n_max} - n_layer: {n_layers} - mean_type: {mean_type} - \
             kernel_type: {kernel_type} - num_mixtures: {num_mixtures} - lr: {lr} - grid_size: {grid_size}')

    # Track train loss and val loss
    train_losses = []
    val_mses = []

    # Model Evaluation on Train/Test Split
    X_train_inner, X_val_inner, y_train_inner, y_val_inner = train_test_split(X, y, test_size=0.25, random_state=73)

    if torch.cuda.is_available():
        X_train_inner, X_val_inner, y_train_inner, y_val_inner = X_train_inner.cuda(), X_val_inner.cuda(), \
                                                                 y_train_inner.cuda(), y_val_inner.cuda()

    # initialize likelihood and model
    input_dim = X_train_inner.size(-1)
    feature_extractor = NNFeatureExtractor(input_dim=input_dim, n_max=n_max,
                                           n_layers=n_layers, n_out=n_out, p=p)
    if initialize_fe:
        feature_extractor = initiliaze_feature_extractor(X_train_inner, y_train_inner,
                                                         feature_extractor=feature_extractor)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = DKLRegression(X_train_inner, y_train_inner, likelihood,
                          feature_extractor, mean_type=mean_type,
                          kernel_type=kernel_type, num_mixtures=num_mixtures,
                          grid_size=grid_size)

    del feature_extractor
    torch.cuda.empty_cache()

    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()

    # Use the adam optimizer
    optimizer = torch.optim.AdamW([
        {'params': model.feature_extractor.parameters()},
        {'params': model.covar_module.parameters()},
        {'params': model.mean_module.parameters()},
        {'params': model.likelihood.parameters()},
    ], lr=lr)

    start_time = time.time()
    # initialize early stopping
    early_stopping = EarlyStopping(patience=patience)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    for epoch in range(epochs_inner):
        model.train()
        likelihood.train()
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(X_train_inner)
        # Calc loss and backprop gradients
        loss = -mll(output, y_train_inner)
        # Calculate Gradients
        loss.backward()
        optimizer.step()

        # Early stopping due to numerical stability issues
        if math.isnan(loss.item()):
            logging.info("NaN loss encountered")
            break

        # Get into evaluation (predictive posterior) mode
        model.eval()
        likelihood.eval()

        # Test points are regularly spaced along [0,1]
        # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = likelihood(model(X_val_inner))
            score = metric(predictions.mean, y_val_inner).cpu().item()

        # Track losses
        train_losses.append(loss.cpu().item())
        val_mses.append(score)

        # Early Stopping
        early_stopping(score)

        if early_stopping.early_stop:
            print("Early stopping")
            logging.info("Early stopping")
            break

        if epoch % 10 == 0:
            print(f"{epoch}/{epochs_inner} - Loss: {loss} - Score: {score}")

        # Pruner
        trial.report(score, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        # Time based pruner
        train_time = time.time()
        if train_time - start_time > time_tolerance:
            print("Time Budget run out. Pruning Trial")
            logging.info("Time Budget run out. Pruning Trial")
            break

    # Set max epochs, as the maximum epoch over all inner splits
    # Max iteration reached
    if epoch == epochs_inner - 1:
        max_epochs = epochs_inner
    # Early stopping
    else:
        max_epochs = max(1, epoch - patience + 1)

    # Memory Tracking
    logging.info("After model training")
    logging.info(torch.cuda.memory_allocated() / 1024 ** 2)
    logging.info(torch.cuda.memory_reserved() / 1024 ** 2)
    X_train_inner.detach()
    y_train_inner.detach()
    X_val_inner.detach()
    y_val_inner.detach()
    optimizer.zero_grad(set_to_none=True)
    likelihood.zero_grad(set_to_none=True)
    mll.zero_grad(set_to_none=True)
    del X_train_inner, y_train_inner, X_val_inner, y_val_inner, loss, optimizer, model, likelihood, output, mll
    gc.collect()
    torch.cuda.empty_cache()
    logging.info("After memory clearing")
    logging.info(torch.cuda.memory_allocated() / 1024 ** 2)
    logging.info(torch.cuda.memory_reserved() / 1024 ** 2)

    trial.set_user_attr("MAX_EPOCHS", int(max_epochs))
    trial.set_user_attr("Train_Losses", train_losses)
    trial.set_user_attr("Val_MSE", val_mses)

    best_score = -early_stopping.best_score

    return best_score

def HPO_DKL(n_splits_outer=5,
            n_splits_inner=4,
            n_trials=40,
            epochs_inner=500,
            patience=10,
            file=None,
            PPV=True,
            dataset_id=216,
            pruner_type="None",
            time_tolerance=1200):
    """
    Optuna trial. Performs n_splits CV to calculate a score for a given Hyperparameter Config

    Args:
        n_splits_outer: Number of Outer CV Splits
        n_splits_inner: Number of Inner CV Splits
        n_trials: Number of HPO trials to conduct per Outer CV split
        epochs_inner: Number of inner epochs
        patience: Number of early stopping iterations
        file: Either "k2204", "r1_08_small", "r1_08" or "None"
        PPV: True for PPV or False for BPV if "k2204", "r1_08_small" or "r1_08"
        dataset_id: OpenML dataset ID
        pruner_type: Optuna Pruner Type, either "None" or "HB" for Hyperband
        time_tolerance: Maximum number of seconds before cancel a HPO trial

    Returns:
        Average score of chosen metric of n_splits CV folds
    """
    # dict map id to name
    # OpenML Dataset to Study
    datasets = {287: "wine_quality", 216: "elevators", 42225: "diamonds", 4549: "buzz", 44027: "year"}

    if file == "k2204":
        file_path = '../trail.X/data/k2204/K2204_net_present_values.csv'
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
    log_filename = "./log/HPO_DKL_log_" + dataset_name + ".log"
    savedir = './data/experiments/DKL'
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    study_infos_file = savedir + '/study_infos_DKL_' + dataset_name + '.csv'
    nested_resampling_infos_file = savedir + '/nested_resampling_infos_DKL_' + dataset_name + '.csv'

    logging.basicConfig(filename=log_filename, encoding='utf-8', level=logging.INFO, force=True)
    optuna.logging.enable_propagation()

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
        # Create a new dataset to track HPO progress, if it doesn't already exist
        try:
            study_infos = pd.read_csv(study_infos_file)
        except FileNotFoundError:
            study_infos = pd.DataFrame({'dataset_id': [], 'dataset_name': [], 'outer_cv_split': [],
                                        'number': [], 'value': [], 'datetime_start': [], 'datetime_complete': [],
                                        'duration': [], 'params_n_max': [], 'params_n_layers': [], 'params_n_out': [],
                                        'params_p': [],
                                        'params_mean_type': [], 'params_kernel_type': [], 'params_num_mixtures': [],
                                        'params_lr': [], 'params_grid_size': [], 'params_initialize_fe': [],
                                        'user_attrs_MAX_EPOCHS': [], 'user_attrs_Train_Losses': [],
                                        'user_attrs_Val_MSE': [], 'state': []})

        try:
            nested_resampling_infos = pd.read_csv(nested_resampling_infos_file)
        except FileNotFoundError:
            nested_resampling_infos = pd.DataFrame({'dataset_id': [], 'dataset_name': [], 'outer_run': [],
                                                    'n_max': [], 'n_layers': [], 'n_out': [], 'p': [],
                                                    'mean_type': [], 'kernel_type': [], 'num_mixtures': [],
                                                    'lr': [], 'grid_size': [], 'initialize_fe': [],
                                                    'MAX_EPOCHS': [], 'mse': [],
                                                    'mae': [], 'max_absolute_error': [],
                                                    'nll_score': [], 'train_time': []})

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

        if pruner_type == "HB":
            pruner = optuna.pruners.HyperbandPruner(min_resource=3, max_resource=epochs_inner, reduction_factor=3)
        elif pruner_type == "median":
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=20,
                interval_steps=5
            )
        else:
            pruner = optuna.pruners.NopPruner()

        study_name = "Nested Resampling DKL " + str(split_nr)
        study = optuna.create_study(direction="minimize",
                                    study_name=study_name,
                                    pruner=pruner)
        # Suggest default parameters
        study.enqueue_trial({"n_max": 1024, 'n_layers': 4, 'n_out': 2, 'mean_type': 'constant', 'kernel_type': 'rbf',
                             'lr': 0.01, 'grid_size': 300, 'initialize_fe': False})
        try:
            if len(X_train_outer) > 100000:
                study.optimize(lambda trial: objective_train_test(trial, X=X_train_outer, y=y_train_outer,
                                                                  epochs_inner=epochs_inner,
                                                                  time_tolerance=time_tolerance, patience=patience),
                               n_trials=n_trials)  # n_trials=N_TRIALS
            else:
                study.optimize(lambda trial: objective_time_prune(trial, X=X_train_outer, y=y_train_outer, kf=kf_inner,
                                                                  epochs_inner=epochs_inner,
                                                                  time_tolerance=time_tolerance, patience=patience),
                               n_trials=n_trials)  # n_trials=N_TRIALS
        except:  # most likely runtime error due to not enough memory
            logging.info(sys.exc_info()[0], "occurred.")
            logging.info("Aborting Study")

        # empty cuda cache to prevent memory issues
        torch.cuda.empty_cache()

        if torch.cuda.is_available():
            X_train_outer, X_test_outer, y_train_outer, y_test_outer = X_train_outer.cuda(), X_test_outer.cuda(), \
                                                                       y_train_outer.cuda(), y_test_outer.cuda()

        # Append HPO Info
        study_info = study.trials_dataframe()
        study_info['dataset_id'] = dataset_id
        study_info['dataset_name'] = dataset_name
        study_info['outer_cv_split'] = split_nr

        study_infos = pd.concat([study_infos, study_info])
        study_infos.to_csv(study_infos_file, index=False)

        # refit model
        best_trial = study.best_trial

        # Refit with best trial
        # best trial
        n_max_best = best_trial.params['n_max']
        n_layers_best = best_trial.params['n_layers']
        n_out_best = best_trial.params['n_out']
        p_best = best_trial.params['p']
        mean_type_best = best_trial.params['mean_type']
        kernel_type_best = best_trial.params['kernel_type']
        # Only need num_mixtures if kernel_type is spectral mixture
        if kernel_type_best == "spectral":
            num_mixtures_best = best_trial.params['num_mixtures']
        else:
            num_mixtures_best = None
        lr_best = best_trial.params['lr']
        grid_size_best = best_trial.params['grid_size']
        initialize_fe_best = best_trial.params['initialize_fe']
        # As max epochs inferred through smaller dataset, increase it by 10%
        max_epochs = int(best_trial.user_attrs['MAX_EPOCHS'] * 1.2)

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

        # mse_score = mse(y_test_outer, predictions.mean).cpu().item()
        # mae_score = mae(y_test_outer, predictions.mean).cpu().item()
        # max_score = maximum_absolute_error(y_test_outer, predictions.mean).cpu().item()

        logging.info(f"Outer Run: {split_nr}: MSE - {mse_score}, MAE - {mae_score}. MAX - {max_score}")

        nested_resampling_info = pd.DataFrame({'dataset_id': [dataset_id], 'dataset_name': [dataset_name],
                                               'outer_run': [split_nr], 'n_max': [n_max_best],
                                               'n_layers': [n_layers_best], 'n_out': [n_out_best], 'p': [p_best],
                                               'mean_type': [mean_type_best],
                                               'kernel_type': [kernel_type_best], 'num_mixtures': [num_mixtures_best],
                                               'lr': [lr_best], 'grid_size': [grid_size_best],
                                               'initialize_fe': [initialize_fe_best],
                                               'MAX_EPOCHS': [max_epochs], 'mse': [mse_score],
                                               'mae': [mae_score], 'max_absolute_error': [max_score],
                                               'nll_score': [nll_score], 'train_time': [train_time]})

        nested_resampling_infos = pd.concat([nested_resampling_infos, nested_resampling_info])
        #
        # save_path = './models/' + 'DKL_' + dataset_name + str(split_nr) + '.pt'
        # torch.save(model.state_dict(), save_path)

        split_nr += 1

        nested_resampling_infos.to_csv(nested_resampling_infos_file, index=False)

        # Memory Tracking
        logging.info("After final model training")
        logging.info(torch.cuda.memory_allocated() / 1024 ** 2)
        logging.info(torch.cuda.memory_reserved() / 1024 ** 2)
        clear_cuda_memory(X_train_outer, X_test_outer, y_train_outer, y_test_outer,
                          loss, optimizer, model, likelihood)
        logging.info("After memory clearing")
        logging.info(torch.cuda.memory_allocated() / 1024 ** 2)
        logging.info(torch.cuda.memory_reserved() / 1024 ** 2)


if __name__ == "__main__":
    HPO_DKL(n_splits_outer=5,
            n_splits_inner=4,
            n_trials=100,
            epochs_inner=400,
            patience=5,
            file="r1_08_small",
            PPV=True,
            dataset_id=44027,
            pruner_type="HB",
            time_tolerance=2400)
