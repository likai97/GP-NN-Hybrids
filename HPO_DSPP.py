import pandas as pd
import numpy as np
import optuna
import openml
import logging
import os
import sys
import time

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import KFold, train_test_split

import torch
from torch.utils.data import TensorDataset, DataLoader
import gpytorch

from src.utils.metric_utils import mse, mae, maximum_absolute_error
from src.utils.train_utils import EarlyStopping, EarlyStoppingWithModelSave

from src.deep_sigma_point_process import DSPPRegression

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
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
    num_inducing = trial.suggest_int("num_inducing", 50, 800, log=True)
    num_quadrature_sites = trial.suggest_int("num_quadrature_sites", 5, 10)  # 5-10 recommended in gpytorch
    beta = trial.suggest_categorical("beta", [0.01, 0.05, 0.2, 1.0])  # follow search space of paper

    n_dspp_layers = trial.suggest_int("n_dspp_layers", 1, 4, log=True)
    n_dspp_out = trial.suggest_int("n_dspp_out", 1, 5, log=True)  # log = True?

    print(
        f'Current Configuration: n_dspp_layers:{n_dspp_layers} - n_dspp_out: {n_dspp_out}'
        f'num_quadrature_sites: {num_quadrature_sites} - num_inducing: {num_inducing} - lr: {lr}')

    logging.info(
        f'Current Configuration: n_dspp_layers:{n_dspp_layers} - n_dspp_out: {n_dspp_out}'
        f'num_quadrature_sites: {num_quadrature_sites} - num_inducing: {num_inducing} - lr: {lr}')

    start_time = time.time()
    early_stopping = EarlyStopping(patience=patience)
    # Inner Nested Resampling Split
    for epoch in range(epochs_inner):
        cur_split_nr = 1
        scores = []
        for train_inner, val_inner in kf.split(X):
            # Temp model name for this cv split to load and save
            temp_model_name = './models/temp/DSPP_temp_split_' + str(cur_split_nr) + '.pt'

            # Train/Val Split
            X_train_inner, X_val_inner, y_train_inner, y_val_inner = X[train_inner], X[val_inner], y[train_inner], y[
                val_inner]

            if torch.cuda.is_available():
                X_train_inner, X_val_inner, y_train_inner, y_val_inner = X_train_inner.cuda(), X_val_inner.cuda(), \
                                                                         y_train_inner.cuda(), y_val_inner.cuda()

            train_inner_dataset = TensorDataset(X_train_inner, y_train_inner)
            train_inner_loader = DataLoader(train_inner_dataset, batch_size=batch_size, shuffle=True)

            test_inner_dataset = TensorDataset(X_val_inner, y_val_inner)
            test_inner_loader = DataLoader(test_inner_dataset, batch_size=batch_size, shuffle=False)

            # initialize model
            output_dims = [n_dspp_out] * n_dspp_layers
            model = DSPPRegression(train_x_shape=X_train_inner.shape, output_dims=output_dims,
                                   num_inducing=num_inducing, Q=num_quadrature_sites)

            # Load existing model if not the first epoch
            if epoch != 0:
                state_dict = torch.load(temp_model_name)
                model.load_state_dict(state_dict)

            if torch.cuda.is_available():
                model = model.cuda()

            # Use the adam optimizer
            optimizer = torch.optim.AdamW([
                {'params': model.parameters()},
            ], lr=lr)

            # "Loss" for GPs - the marginal log likelihood
            mll = gpytorch.mlls.DeepPredictiveLogLikelihood(model.likelihood, model,
                                                            num_data=X_train_inner.size(0), beta=beta)

            # Train Model for 5 epochs
            for i in range(1):
                # set to train mode
                model.train()
                for batch, (X_batch, y_batch) in enumerate(train_inner_loader):
                    # Zero gradient
                    optimizer.zero_grad()
                    # Output from model
                    output = model(X_batch)
                    # Calc loss and backprop gradients
                    loss = -mll(output, y_batch)
                    loss.backward()
                    # adjust learning weights
                    optimizer.step()

            # Get into evaluation (predictive posterior) mode
            model.eval()

            # Test points are regularly spaced along [0,1]
            # Make predictions by feeding model through likelihood
            # Make predictions
            with torch.no_grad():
                means, vars, ll = model.predict(test_inner_loader)
                weights = model.quad_weights.unsqueeze(-1).exp().cpu()
            score = metric((weights * means).sum(0), y_val_inner.cpu()).item()

            # Append Model Scores
            scores.append(score)

            # Save model
            torch.save(model.state_dict(), temp_model_name)
            # Increase cur_split_nr by 1
            cur_split_nr += 1

            del X_train_inner, y_train_inner, X_val_inner, y_val_inner, loss,\
                optimizer, model, means, vars, ll, output, mll
            gc.collect()
            torch.cuda.empty_cache()

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
            break

        # Time based pruner
        train_time = time.time()
        if train_time - start_time > time_tolerance:
            print("Time Budget run out. Pruning Trial")
            break

        print(f"{epoch}/{epochs_inner} - Score: {average_scores}")

    cur_split_nr = 1
    for _, _ in kf.split(X):
        # Temp model name for this cv split to load and save
        temp_model_name = './models/temp/DSPP_temp_split_' + str(cur_split_nr) + '.pt'
        os.remove(temp_model_name)
        cur_split_nr += 1

    # Memory Tracking
    logging.info("After model training")
    logging.info(torch.cuda.memory_allocated() / 1024 ** 2)
    logging.info(torch.cuda.memory_reserved() / 1024 ** 2)
    # X_train_inner.detach()
    # y_train_inner.detach()
    # X_val_inner.detach()
    # y_val_inner.detach()
    # loss.detach()
    # optimizer.zero_grad(set_to_none=True)
    # mll.zero_grad(set_to_none=True)
    # del X_train_inner, y_train_inner, X_val_inner, y_val_inner, loss, optimizer, model, means, vars, ll, output, mll
    # gc.collect()
    # torch.cuda.empty_cache()
    # logging.info("After memory clearing")
    # logging.info(torch.cuda.memory_allocated() / 1024 ** 2)
    # logging.info(torch.cuda.memory_reserved() / 1024 ** 2)

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
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [512, 1024, 2048])
    num_inducing = trial.suggest_int("num_inducing", 50, 800, log=True)
    num_quadrature_sites = trial.suggest_int("num_quadrature_sites", 5, 10)  # 5-10 recommended in gpytorch
    beta = trial.suggest_categorical("beta", [0.01, 0.05, 0.2, 1.0])  # follow search space of paper

    n_dspp_layers = trial.suggest_int("n_dspp_layers", 1, 3, log=True)  # Had to reduce it due to CUDA Memory error
    n_dspp_out = trial.suggest_int("n_dspp_out", 1, 4, log=True)  # had to reduce it due to CUDA Memory error

    print(
        f'Current Configuration: n_dspp_layers:{n_dspp_layers} - n_dspp_out: {n_dspp_out} - num_inducing: {num_inducing} - '
        f'num_quadrature_sites: {num_quadrature_sites} - beta: {beta} - batch_size: {batch_size} - lr: {lr}')

    logging.info(
        f'Current Configuration: n_dspp_layers:{n_dspp_layers} - n_dspp_out: {n_dspp_out} - num_inducing: {num_inducing} - '
        f'num_quadrature_sites: {num_quadrature_sites} - beta: {beta} - batch_size: {batch_size} - lr: {lr}')

    # Model Evaluation on Train/Test Split
    X_train_inner, X_val_inner, y_train_inner, y_val_inner = train_test_split(X, y, test_size=0.25, random_state=73)

    if torch.cuda.is_available():
        X_train_inner, X_val_inner, y_train_inner, y_val_inner = X_train_inner.cuda(), X_val_inner.cuda(), \
                                                                 y_train_inner.cuda(), y_val_inner.cuda()

    train_inner_dataset = TensorDataset(X_train_inner, y_train_inner)
    train_inner_loader = DataLoader(train_inner_dataset, batch_size=batch_size, shuffle=True)

    test_inner_dataset = TensorDataset(X_val_inner, y_val_inner)
    test_inner_loader = DataLoader(test_inner_dataset, batch_size=batch_size, shuffle=False)

    # initialize model
    output_dims = [n_dspp_out] * n_dspp_layers
    model = DSPPRegression(train_x_shape=X_train_inner.shape, output_dims=output_dims,
                           num_inducing=num_inducing, Q=num_quadrature_sites)

    if torch.cuda.is_available():
        model = model.cuda()

    # Use the adam optimizer
    optimizer = torch.optim.AdamW([
        {'params': model.parameters()},
    ], lr=lr)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.DeepPredictiveLogLikelihood(model.likelihood, model,
                                                    num_data=X_train_inner.size(0), beta=beta)

    start_time = time.time()
    # initialize early stopping
    early_stopping = EarlyStopping(patience=patience)
    for epoch in range(epochs_inner):
        # set to train mode
        model.train()
        for batch, (X_batch, y_batch) in enumerate(train_inner_loader):
            # Zero gradient
            optimizer.zero_grad()
            # Output from model
            output = model(X_batch)
            # Calc loss and backprop gradients
            loss = -mll(output, y_batch)
            loss.backward()
            # adjust learning weights
            optimizer.step()

        # Get into evaluation (predictive posterior) mode
        model.eval()

        # Test points are regularly spaced along [0,1]
        # Make predictions by feeding model through likelihood
        # Make predictions
        with torch.no_grad():
            means, vars, ll = model.predict(test_inner_loader)
            weights = model.quad_weights.unsqueeze(-1).exp().cpu()
        score = metric((weights * means).sum(0), y_val_inner.cpu()).item()

        # Early Stopping
        early_stopping(score)

        if early_stopping.early_stop:
            print("Early stopping")
            break

        if epoch % 5 == 0:
            print(f"{epoch}/{epochs_inner} - Loss: {loss} - Score: {score}")

        # Pruner
        trial.report(score, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        # Time based pruner
        train_time = time.time()
        if train_time - start_time > time_tolerance:
            print("Time Budget run out. Pruning Trial")
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
    loss.detach()
    optimizer.zero_grad(set_to_none=True)
    mll.zero_grad(set_to_none=True)
    del X_train_inner, y_train_inner, X_val_inner, y_val_inner, train_inner_dataset, train_inner_loader,\
        test_inner_dataset, test_inner_loader, loss, optimizer, model, means, vars, ll, output, mll
    gc.collect()
    torch.cuda.empty_cache()
    logging.info("After memory clearing")
    logging.info(torch.cuda.memory_allocated() / 1024 ** 2)
    logging.info(torch.cuda.memory_reserved() / 1024 ** 2)

    trial.set_user_attr("MAX_EPOCHS", int(max_epochs))

    best_score = -early_stopping.best_score

    return best_score


def HPO_DSPP(n_splits_outer=5,
             n_splits_inner=4,
             n_trials=100,
             epochs_inner=400,
             patience=5,
             file=None,
             PPV=True,
             dataset_id=216,
             pruner_type="None",
             time_tolerance=1800,
             epochs_outer=1000,
             patience_outer=10):
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
        epochs_outer: Number of Epochs to evaluate the best configuration
        patience_outer: Number of early stopping iterations

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
    log_filename = "./log/HPO_DSPP_log_" + dataset_name + ".log"
    savedir = './data/experiments/DSPP'
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    study_infos_file = savedir + '/study_infos_DSPP_' + dataset_name + '.csv'
    nested_resampling_infos_file = savedir + '/nested_resampling_infos_DSPP_' + dataset_name + '.csv'

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

    for train_outer, test_outer in kf_outer.split(np.array(X)):
        if split_nr <= 2 and dataset_name == "year":
            split_nr += 1
            continue
        if split_nr <= 1 and dataset_name == "r1_08_PPV":
            split_nr += 1
            continue

        # Create a new dataset to track HPO progress, if it doesn't already exist
        try:
            study_infos = pd.read_csv(study_infos_file)
        except FileNotFoundError:
            study_infos = pd.DataFrame({'dataset_id': [], 'dataset_name': [], 'outer_cv_split': [],
                                        'number': [], 'value': [], 'datetime_start': [], 'datetime_complete': [],
                                        'duration': [], 'params_n_dspp_layers': [],
                                        'params_n_dspp_out': [], 'params_batch_size': [], 'params_lr': [],
                                        'params_num_inducing': [], 'params_num_quadrature_sites': [],
                                        'params_beta': [], 'user_attrs_MAX_EPOCHS': [],
                                        'user_attrs_Train_Losses': [], 'user_attrs_Val_MSE': [], 'state': []})

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

        study_name = "Nested Resampling DSPP " + str(split_nr)
        study = optuna.create_study(direction="minimize",
                                    study_name=study_name,
                                    pruner=pruner)
        # Suggest default parameters
        study.enqueue_trial({"n_dspp_layers": 3, 'n_dspp_out': 1, 'num_inducing': 200, 'batch_size': 1024,
                             'lr': 0.001, 'num_quadrature_sites': 8, 'beta': 0.05})
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
        n_layers_best = best_trial.params['n_dspp_layers']
        n_out_best = best_trial.params['n_dspp_out']
        batch_size_best = best_trial.params['batch_size']
        lr_best = best_trial.params['lr']
        num_inducing_best = best_trial.params['num_inducing']
        num_quadrature_sites_best = best_trial.params['num_quadrature_sites']
        beta_best = best_trial.params['beta']
        # As max epochs inferred through smaller dataset, increase it by 10%
        max_epochs = int(best_trial.user_attrs['MAX_EPOCHS'] * 1.1)

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
        #variance https://en.wikipedia.org/wiki/Mixture_distribution
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

        # Memory Tracking
        logging.info("After final model training")
        logging.info(torch.cuda.memory_allocated() / 1024 ** 2)
        logging.info(torch.cuda.memory_reserved() / 1024 ** 2)
        X_train_outer.detach()
        X_test_outer.detach()
        y_train_outer.detach()
        y_test_outer.detach()
        loss.detach()
        optimizer.zero_grad(set_to_none=True)
        mll.zero_grad(set_to_none=True)
        del X_train_outer, X_test_outer, y_train_outer, y_test_outer, train_dataset, train_loader, test_dataset,\
            test_loader, loss, optimizer, model, means, vars, ll
        gc.collect()
        torch.cuda.empty_cache()
        logging.info("After memory clearing")
        logging.info(torch.cuda.memory_allocated() / 1024 ** 2)
        logging.info(torch.cuda.memory_reserved() / 1024 ** 2)


if __name__ == "__main__":
    HPO_DSPP(n_splits_outer=5,
             n_splits_inner=4,
             n_trials=100,
             epochs_inner=400,
             patience=5,
             file='r1_08_small',
             PPV=True,
             dataset_id=44027,
             pruner_type='HB',
             time_tolerance=2400,
             epochs_outer=1000,
             patience_outer=10)
