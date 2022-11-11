import pandas as pd
import numpy as np
import optuna
import openml
import logging
import os
import sys
import time

from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import KFold, train_test_split

import torch
from torch.utils.data import TensorDataset, DataLoader
import gpytorch

from src.utils.metric_utils import mse, mae, maximum_absolute_error
from src.utils.train_utils import EarlyStopping, EarlyStoppingWithModelSave

from src.deep_ensemble import DeepEnsemble


def objective(trial, X, y, kf,
              epochs_inner, patience,
              metric=mse):
    """
    Optuna trial. Performs n_splits CV to calculate a score for a given Hyperparameter Config

    Args:
        trial: Optuna trial
        X: features
        y: labels
        kf: Sklearn
        epochs_inner: Number of inner epochs
        patience: Number of early stopping iterations
        metric: Evaluation metric to be optimized

    Returns:
        Average score of chosen metric of n_splits CV folds
    """
    # trial parameters
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    n_max = trial.suggest_int("n_max", 20, 1024)  # log = True?
    n_layers = trial.suggest_int("n_layers", 2, 6)
    batch_size = trial.suggest_categorical("batch_size", [512, 1024, 2048, 4096, 8124])

    print(f'Current Configuration: n_max:{n_max} - n_layers: {n_layers} - batch_size: {batch_size} - lr: {lr}')

    logging.info(f'Current Configuration: n_max:{n_max} - n_layers: {n_layers} - batch_size: {batch_size} - lr: {lr}')

    # Inner Nested Resampling Split
    scores = []
    model_scores = []
    model_scores_best_score = []
    for train_inner, val_inner in kf.split(X):
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

        # initialize likelihood and model
        # initialize model
        de_model = DeepEnsemble(
            num_models=5,
            input_dim=X_train_inner.size(-1),
            n_max=n_max,
            n_layers=n_layers
        )
        if torch.cuda.is_available():
            de_model = de_model.cuda()

        # Initialize Optimizer
        optimizers = []
        for i in range(de_model.num_models):
            model = getattr(de_model, 'Model_' + str(i + 1))
            optimizers.append(torch.optim.AdamW(params=model.parameters(), lr=lr))

        # Train Each Model till early stopp
        for i in range(de_model.num_models):
            # get Model
            model = getattr(de_model, 'Model_' + str(i + 1))
            # Initialize Early Stopping
            early_stopping = EarlyStopping(patience=patience)
            # initialize loss
            nll_loss = torch.nn.GaussianNLLLoss()
            for epoch in range(epochs_inner):
                model.train()
                epoch_loss = []
                for batch, (X_batch, y_batch) in enumerate(train_inner_loader):
                    optimizers[i].zero_grad()
                    mean, var = model(X_batch)
                    loss = nll_loss(mean.flatten(), y_batch, var.flatten())
                    loss.backward()
                    optimizers[i].step()
                    epoch_loss.append(loss.item())

                epoch_loss = sum(epoch_loss) / len(epoch_loss)
                model.eval()
                pred_means = torch.tensor([0.])
                with torch.no_grad():
                    for batch, (X_batch, y_batch) in enumerate(test_inner_loader):
                        mean, var = model(X_batch)
                        pred_means = torch.cat([pred_means, mean.flatten().cpu()])

                    pred_means = pred_means[1:]
                    model_score = metric(pred_means, y_val_inner.cpu())

                # Early Stopping
                early_stopping(model_score)
                if early_stopping.early_stop:
                    model_scores.append(model_score)
                    model_scores_best_score.append(-early_stopping.best_score)
                    print("Early stopping")
                    break

                if epoch%10==0:
                    print(f"Epoch: {epoch}/{epochs_inner} - loss {epoch_loss} - Score {model_score}")

        # Calculate Ensemble Prediction
        ens_means = torch.tensor([0.])
        with torch.no_grad():
            for batch, (X_batch, y_batch) in enumerate(test_inner_loader):
                mean, var = de_model(X_batch)
                ens_means = torch.cat([ens_means, mean.flatten().cpu()])

        ens_means = ens_means[1:]
        score = metric(ens_means, y_val_inner.cpu())
        scores.append(score)

    # Average Scores over the inner CV Splits
    average_scores = np.mean(scores)
    logging.info(f"Model Scores:{model_scores}")
    logging.info(f"Model Scores:{model_scores_best_score}")

    return average_scores


def objective_train_test(trial, X, y, epochs_inner,
                         patience, time_tolerance=2400,
                         metric=mse):
    """
    Optuna trial. Performs n_splits CV to calculate a score for a given Hyperparameter Config

    Args:
        trial: Optuna trial
        X: features
        y: labels
        epochs_inner: Number of inner epochs
        patience: Number of early stopping iterations
        time_tolerance: Maximum number of seconds before cancel the trial
        metric: Evaluation metric to be optimized

    Returns:
        Average score of chosen metric of n_splits CV folds
    """
    # trial parameters
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    n_max = trial.suggest_int("n_max", 20, 1024)  # log = True?
    n_layers = trial.suggest_int("n_layers", 2, 6)
    batch_size = trial.suggest_categorical("batch_size", [512, 1024, 2048, 4096, 8124])

    print(f'Current Configuration: n_max:{n_max} - n_layers: {n_layers} - batch_size: {batch_size} - lr: {lr}')

    logging.info(f'Current Configuration: n_max:{n_max} - n_layers: {n_layers} - batch_size: {batch_size} - lr: {lr}')

    # Model Evaluation on Train/Test Split
    X_train_inner, X_val_inner, y_train_inner, y_val_inner = train_test_split(X, y, test_size=0.25, random_state=73)

    if torch.cuda.is_available():
        X_train_inner, X_val_inner, y_train_inner, y_val_inner = X_train_inner.cuda(), X_val_inner.cuda(), \
                                                                 y_train_inner.cuda(), y_val_inner.cuda()

    train_inner_dataset = TensorDataset(X_train_inner, y_train_inner)
    train_inner_loader = DataLoader(train_inner_dataset, batch_size=batch_size, shuffle=True)

    test_inner_dataset = TensorDataset(X_val_inner, y_val_inner)
    test_inner_loader = DataLoader(test_inner_dataset, batch_size=batch_size, shuffle=False)

    # initialize likelihood and model
    # initialize model
    de_model = DeepEnsemble(
        num_models=5,
        input_dim=X_train_inner.size(-1),
        n_max=n_max,
        n_layers=n_layers
    )
    if torch.cuda.is_available():
        de_model = de_model.cuda()

    # Initialize Optimizer
    optimizers = []
    for i in range(de_model.num_models):
        model = getattr(de_model, 'Model_' + str(i + 1))
        optimizers.append(torch.optim.AdamW(params=model.parameters(), lr=lr))

    #
    start_time = time.time()
    early_stopping = EarlyStopping(patience=patience)
    nll_loss = torch.nn.GaussianNLLLoss()
    # Train Each Model till early stopp
    for epoch in range(epochs_inner):
        model_losses = []
        for i in range(de_model.num_models):
            # get Model
            model = getattr(de_model, 'Model_' + str(i + 1))
            model.train()
            epoch_loss = []
            for batch, (X_batch, y_batch) in enumerate(train_inner_loader):
                optimizers[i].zero_grad()
                mean, var = model(X_batch)
                loss = nll_loss(mean.flatten(), y_batch, var.flatten())
                loss.backward()
                optimizers[i].step()
                epoch_loss.append(loss.item())
            epoch_loss = sum(epoch_loss) / len(epoch_loss)
            model_losses.append(epoch_loss)

        # Calculate Ensemble Prediction
        ens_means = torch.tensor([0.])
        with torch.no_grad():
            for batch, (X_batch, y_batch) in enumerate(test_inner_loader):
                mean, var = de_model(X_batch)
                ens_means = torch.cat([ens_means, mean.flatten().cpu()])

        ens_means = ens_means[1:]
        score = metric(ens_means, y_val_inner.cpu())

        if epoch % 5 == 0:
            print(f"{epoch}/{epochs_inner} - Loss: {np.mean(model_losses)} - Score: {score}")

        # Early Stopping
        early_stopping(score)
        if early_stopping.early_stop:
            print("Early stopping")
            break

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

    best_score = -early_stopping.best_score

    return best_score


def HPO_DE(n_splits_outer=5,
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
    log_filename = "./log/HPO_DE_log_" + dataset_name + ".log"
    savedir = './data/experiments/DE'
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    study_infos_file = savedir + '/study_infos_DE_' + dataset_name + '.csv'
    nested_resampling_infos_file = savedir + '/nested_resampling_infos_DE_' + dataset_name + '.csv'

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
        # Create a new dataset to track HPO progress, if it doesn't already exist
        try:
            study_infos = pd.read_csv(study_infos_file)
        except FileNotFoundError:
            study_infos = pd.DataFrame({'dataset_id': [], 'dataset_name': [], 'outer_cv_split': [],
                                        'number': [], 'value': [], 'datetime_start': [], 'datetime_complete': [],
                                        'duration': [], 'params_n_max': [], 'params_n_layers': [],
                                        'params_lr': [], 'params_batch_size': [], 'state': []})

        try:
            nested_resampling_infos = pd.read_csv(nested_resampling_infos_file)
        except FileNotFoundError:
            nested_resampling_infos = pd.DataFrame({'dataset_id': [], 'dataset_name': [], 'outer_run': [],
                                                    'n_max': [], 'n_layers': [],  'lr': [], 'batch_size': [],
                                                    'mse': [], 'mae': [], 'max_absolute_error': [],
                                                    'nll_score': [], 'train_time': []})

        X_train_outer, X_test_outer, y_train_outer, y_test_outer = X[train_outer], X[test_outer], y[train_outer], y[
            test_outer]

        # Scale Features and Labels
        scaler = StandardScaler()
        X_train_outer = scaler.fit_transform(X_train_outer)
        X_test_outer = scaler.transform(X_test_outer)

        # Turn back to tensors
        X_train_outer = torch.from_numpy(X_train_outer).float()
        X_test_outer = torch.from_numpy(X_test_outer).float()

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

        study_name = "Nested Resampling DE " + str(split_nr)
        study = optuna.create_study(direction="minimize",
                                    study_name=study_name,
                                    pruner=pruner)

        # Suggest default parameters
        study.enqueue_trial({"n_max": 200, 'n_layers': 4, 'lr': 0.001, 'batch_size': 512})
        try:
            if len(X_train_outer) > 100000:
                study.optimize(lambda trial: objective_train_test(trial, X=X_train_outer, y=y_train_outer,
                                                                  epochs_inner=epochs_inner, patience=patience,
                                                                  time_tolerance=time_tolerance),
                               n_trials=n_trials)
            else:
                study.optimize(lambda trial: objective(trial, X=X_train_outer, y=y_train_outer, kf=kf_inner,
                                                       epochs_inner=epochs_inner, patience=patience),
                               n_trials=n_trials)
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
        lr_best = best_trial.params['lr']
        batch_size_best = best_trial.params['batch_size']

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
        #Numerical Stability
        ens_var = torch.tensor(np.where(ens_var.numpy() > 0, ens_var.numpy(), 1e-06))
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

        nested_resampling_infos.to_csv(nested_resampling_infos_file, index=False)

        os.remove("checkpoint.pt")


if __name__ == "__main__":
    HPO_DE(n_splits_outer=5,
           n_splits_inner=4,
           n_trials=100,
           epochs_inner=400,
           patience=5,
           file="r1_08_small",
           PPV=False,
           dataset_id=44027,
           pruner_type="HB",
           time_tolerance=2400,
           epochs_outer=1000,
           patience_outer=10)
