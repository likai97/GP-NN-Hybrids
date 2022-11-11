import pandas as pd
import numpy as np
import logging
import openml
import time
import os
import optuna

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error
from sklearn.ensemble import RandomForestRegressor


def objective(trial, X, y, kf):
    """
    Optuna trial. Performs n_splits CV to calculate a score for a given Hyperparameter Config

    Args:
        trial: Optuna trial
        X: features
        y: labels
        kf: Sklearn Kf object

    Returns:
        Average score of chosen metric of n_splits CV folds
    """

    # configurations
    n_estimators = trial.suggest_int("n_estimators", 10, 500, log=True)
    max_depth = trial.suggest_int("max_depth", 5, 100, log=True)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 25)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 25)
    max_features = trial.suggest_categorical("max_features", ["sqrt", "auto"])

    print(
        f'Current Configuration: n_estimators:{n_estimators} - max_depth: {max_depth} - min_samples_split: {min_samples_split} - \
      min_samples_leaf: {min_samples_leaf} - max_features: {max_features}')

    model = RandomForestRegressor(n_estimators=n_estimators,
                                  max_depth=max_depth,
                                  min_samples_split=min_samples_split,
                                  min_samples_leaf=min_samples_leaf,
                                  max_features=max_features)
    # xgb cross val
    score = -cross_val_score(model, X, y, n_jobs=-1, cv=kf, scoring="neg_mean_squared_error")

    return np.mean(score)


def objective_train_test(trial, X, y, test_size=0.25, seed=73):
    """
    Optuna trial. Performs n_splits CV to calculate a score for a given Hyperparameter Config

    Args:
        trial: Optuna trial
        X: features
        y: labels
        test_size: Float test size
        Seed: Seed for reproducability

    Returns:
        Average score of chosen metric of n_splits CV folds
    """
    # configurations
    n_estimators = trial.suggest_int("n_estimators", 10, 250, log=True)
    max_depth = trial.suggest_int("max_depth", 5, 50, log=True)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 15)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 15)
    max_features = trial.suggest_categorical("max_features", ["sqrt", "auto"])

    print(
        f'Current Configuration: n_estimators:{n_estimators} - max_depth: {max_depth} - min_samples_split: {min_samples_split} - \
      min_samples_leaf: {min_samples_leaf} - max_features: {max_features}')

    # Model Evaluation on Train/Test Split
    X_train_inner, X_val_inner, y_train_inner, y_val_inner = train_test_split(X, y, test_size=test_size,
                                                                              random_state=seed)

    model = RandomForestRegressor(n_estimators=n_estimators,
                                  max_depth=max_depth,
                                  min_samples_split=min_samples_split,
                                  min_samples_leaf=min_samples_leaf,
                                  max_features=max_features,
                                  n_jobs=-1)

    model.fit(X_train_inner, y_train_inner)
    predictions = model.predict(X_val_inner)

    mse = mean_squared_error(y_val_inner, predictions)

    return mse


def HPO_RF(n_splits_outer=5,
           n_splits_inner=4,
           n_trials=40,
           file=None,
           PPV=True,
           dataset_id=216):
    """
    Optuna trial. Performs n_splits CV to calculate a score for a given Hyperparameter Config

    Args:
        n_splits_outer: Number of Outer CV Splits
        n_splits_inner: Number of Inner CV Splits
        n_trials: Number of HPO trials to conduct per Outer CV split
        file: Either "k2204", "r1_08_small", "r1_08" or "None"
        PPV: True for PPV or False for BPV if "k2204", "r1_08_small" or "r1_08"
        dataset_id: OpenML dataset ID

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
    log_filename = "./log/HPO_RF_log_" + dataset_name + ".log"
    savedir = './data/experiments/RF'
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    study_infos_file = savedir + '/study_infos_RF_' + dataset_name + '.csv'
    nested_resampling_infos_file = savedir + '/nested_resampling_infos_RF_' + dataset_name + '.csv'

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

    print(f'Starting Nested Resampling for {dataset_name}')
    split_nr = 1

    X = np.array(X)
    y = np.array(y)
    # Outer Nested Resampling Split
    for train_outer, test_outer in kf_outer.split(np.array(X)):
        # Create a new dataset to track HPO progress, if it doesn't already exist
        try:
            study_infos = pd.read_csv(study_infos_file)
        except FileNotFoundError:
            study_infos = pd.DataFrame({'dataset_id': [], 'dataset_name': [], 'outer_cv_split': [],
                                        'number': [], 'value': [], 'datetime_start': [], 'datetime_complete': [],
                                        'duration': [], 'params_max_depth': [], 'params_max_features': [],
                                        'params_min_samples_leaf': [],
                                        'params_min_samples_split': [], 'params_n_estimators': [], 'state': []})

        try:
            nested_resampling_infos = pd.read_csv(nested_resampling_infos_file)
        except FileNotFoundError:
            nested_resampling_infos = pd.DataFrame({'dataset_id': [], 'dataset_name': [], 'outer_run': [],
                                                    'max_depth': [], 'max_features': [],
                                                    'min_samples_leaf': [], 'min_samples_split': [],
                                                    'n_estimators': [], 'mse': [],
                                                    'mae': [], 'max_absolute_error': []})

        X_train_outer, X_test_outer, y_train_outer, y_test_outer = X[train_outer], X[test_outer], y[train_outer], y[
            test_outer]

        # Scale Features and Labels
        # scaler = MinMaxScaler()
        # X_train_outer = scaler.fit_transform(X_train_outer)
        # X_test_outer = scaler.transform(X_test_outer)

        # Train on Normalized Features
        y_scaler = MinMaxScaler()
        y_train_outer = y_scaler.fit_transform(y_train_outer.reshape(-1, 1)).flatten()
        # y_test_outer = y_scaler.transform(y_test_outer.reshape(-1, 1)).flatten()

        study_name = "Nested Resampling RF " + str(split_nr)
        study = optuna.create_study(direction="minimize",
                                    study_name=study_name)

        if len(X_train_outer) > 100000:
            study.optimize(lambda trial: objective_train_test(trial, X=X_train_outer, y=y_train_outer),
                           n_trials=n_trials)  # n_trials=N_TRIALS
        else:
            study.optimize(lambda trial: objective(trial, X=X_train_outer, y=y_train_outer, kf=kf_inner),
                           n_trials=n_trials)  # n_trials=N_TRIALS

        # Append HPO Info
        study_info = study.trials_dataframe()
        study_info['dataset_id'] = dataset_id
        study_info['dataset_name'] = dataset_name
        study_info['outer_cv_split'] = split_nr

        study_infos = pd.concat([study_infos, study_info])

        # refit model
        best_trial = study.best_trial

        # Refit with best trial
        # best trial
        n_estimators_best = best_trial.params['n_estimators']
        max_depth_best = best_trial.params['max_depth']
        min_samples_split_best = best_trial.params['min_samples_split']
        min_samples_leaf_best = best_trial.params['min_samples_leaf']
        max_features_best = best_trial.params['max_features']

        model = RandomForestRegressor(n_estimators=n_estimators_best,
                                      max_depth=max_depth_best,
                                      min_samples_split=min_samples_split_best,
                                      min_samples_leaf=min_samples_leaf_best,
                                      max_features=max_features_best,
                                      n_jobs=-1)

        model.fit(X_train_outer, y_train_outer)

        predictions = model.predict(X_test_outer)
        predictions = y_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

        mse_score = mean_squared_error(y_test_outer, predictions)
        mae_score = mean_absolute_error(y_test_outer, predictions)
        max_score = max_error(y_test_outer, predictions)

        logging.info(f"Outer Run: {split_nr}: MSE - {mse_score}, MAE - {mae_score}. MAX - {max_score}")

        nested_resampling_info = pd.DataFrame({'dataset_id': [dataset_id], 'dataset_name': [dataset_name],
                                               'outer_run': [split_nr], 'max_depth': [max_depth_best],
                                               'max_features': [max_features_best],
                                               'min_samples_leaf': [min_samples_leaf_best],
                                               'min_samples_split': [min_samples_split_best],
                                               'n_estimators': [n_estimators_best], 'mse': [mse_score],
                                               'mae': [mae_score], 'max_absolute_error': [max_score]})

        nested_resampling_infos = pd.concat([nested_resampling_infos, nested_resampling_info])

        split_nr += 1

        study_infos.to_csv(study_infos_file, index=False)

        nested_resampling_infos.to_csv(nested_resampling_infos_file, index=False)




if __name__ == "__main__":
    HPO_RF(n_splits_outer=5,
           n_splits_inner=4,
           n_trials=100,
           file="r1_08_small",
           PPV=True,
           dataset_id=44027)
