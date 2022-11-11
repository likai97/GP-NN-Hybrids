import xgboost as xgb
import pandas as pd
import numpy as np
import logging
import os
import openml
import optuna

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error


def objective(trial, X, y, kf, patience):
    """
    Optuna trial. Performs n_splits CV to calculate a score for a given Hyperparameter Config

    Args:
        trial: Optuna trial
        X: features
        y: labels
        kf: Sklearn Kfold object
        patience: Number of early stopping iterations

    Returns:
        Average score of chosen metric of n_splits CV folds
    """
    params = {
        "verbosity": 1,  # 0 (silent) - 3 (debug)
        "objective": "reg:squarederror",
        "max_depth": trial.suggest_int("max_depth", 4, 75),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 0.6, log=True),
        "subsample": trial.suggest_float("subsample", 0.4, 0.8, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 10.0, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 2, 100, log=True),
        "seed": 42,
        "n_jobs": -1,
    }

    print(f'Current Configuration: {params}')

    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "test-rmse")
    dtrain = xgb.DMatrix(X, y)

    #xgb cross val
    res = xgb.cv(params=params,
                 dtrain=dtrain,
                 num_boost_round=10000,
                 folds=kf,
                 metrics={'rmse'},
                 callbacks=[pruning_callback],
                 early_stopping_rounds=patience
           )

    trial.set_user_attr("n_estimators", res.shape[0])

    return res.iloc[-1, 2]


def objective_train_test(trial, X, y, patience, test_size=0.25, random_state=73):
    """
    Optuna trial. Performs n_splits CV to calculate a score for a given Hyperparameter Config

    Args:
        trial: Optuna trial
        X: features
        y: labels
        patience: Number of early stopping iterations
        test_size: Float test size
        Seed: Seed for reproducability

    Returns:
        Average score of chosen metric of n_splits CV folds
    """
    params = {
        "verbosity": 2,  # 0 (silent) - 3 (debug)
        "objective": "reg:squarederror",
        "max_depth": trial.suggest_int("max_depth", 4, 50),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 0.6, log=True),
        "subsample": trial.suggest_float("subsample", 0.4, 0.8, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 10.0, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 2, 100, log=True),
        "seed": 42,
        "n_jobs": -1,
    }

    print(f'Current Configuration: {params}')

    X_train_inner, X_val_inner, y_train_inner, y_val_inner = train_test_split(X, y, test_size=test_size,
                                                                              random_state=random_state)

    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation-rmse")
    early_stop = xgb.callback.EarlyStopping(rounds=patience,
                                            metric_name='rmse',
                                            data_name='validation')

    dtrain = xgb.DMatrix(X_train_inner, label=y_train_inner)
    dtest = xgb.DMatrix(X_val_inner, label=y_val_inner)

    model = xgb.train(params,
                      dtrain,
                      num_boost_round=1000,
                      evals=[(dtest, "validation")],
                      maximize=False,
                      callbacks=[pruning_callback,
                                 early_stop],
                      verbose_eval=False)

    preds = model.predict(dtest)
    trial.set_user_attr("n_estimators", len(early_stop.stopping_history['validation']['rmse']))

    mse = mean_squared_error(y_val_inner, preds)

    return mse


def HPO_XGB(n_splits_outer=5,
            n_splits_inner=4,
            n_trials=40,
            patience=10,
            file=None,
            PPV=True,
            dataset_id=216):
    """
    Optuna trial. Performs n_splits CV to calculate a score for a given Hyperparameter Config

    Args:
        n_splits_outer: Number of Outer CV Splits
        n_splits_inner: Number of Inner CV Splits
        n_trials: Number of HPO trials to conduct per Outer CV split
        patience: Number of early stopping iterations
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
    log_filename = "./log/HPO_XGB_log_" + dataset_name + ".log"
    savedir = './data/experiments/XGB'
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    study_infos_file = savedir + '/study_infos_XGB_' + dataset_name + '.csv'
    nested_resampling_infos_file = savedir + '/nested_resampling_infos_XGB_' + dataset_name + '.csv'

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
    for train_outer, test_outer in kf_outer.split(X):
        # Create a new dataset to track HPO progress, if it doesn't already exist
        try:
            study_infos = pd.read_csv(study_infos_file)
        except FileNotFoundError:
            study_infos = pd.DataFrame({'dataset_id': [], 'dataset_name': [], 'outer_cv_split': [],
                                        'number': [], 'value': [], 'datetime_start': [], 'datetime_complete': [],
                                        'duration': [], 'params_max_depth': [], 'params_learning_rate': [],
                                        'params_colsample_bytree': [], 'params_subsample': [], 'params_reg_alpha': [],
                                        'params_reg_lambda': [], 'params_gamma': [], 'params_min_child_weight': [],
                                        'user_attrs_n_estimators': [],
                                        'state': []})

        try:
            nested_resampling_infos = pd.read_csv(nested_resampling_infos_file)
        except FileNotFoundError:
            nested_resampling_infos = pd.DataFrame({'dataset_id': [], 'dataset_name': [], 'outer_run': [],
                                                    'max_depth': [], 'learning_rate': [], 'colsample_bytree': [],
                                                    'subsample': [], 'reg_alpha': [], 'reg_lambda': [],
                                                    'gamma': [], 'min_child_weight': [], 'n_estimators': [],
                                                    'mse': [], 'mae': [], 'max_absolute_error': []})

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

        study_name = "Nested Resampling XGB " + str(split_nr)
        study = optuna.create_study(direction="minimize",
                                    study_name=study_name)

        study.enqueue_trial({"max_depth": 46, 'learning_rate':  0.05518, 'colsample_bytree': 0.59548,
                             'subsample': 0.630578, 'reg_alpha': 1.465652, 'reg_lambda': 2.7e-08,
                             'gamma': 2.57e-05, 'min_child_weight': 63.056})

        if len(X_train_outer) > 100000:
            study.optimize(lambda trial: objective_train_test(trial, X=X_train_outer, y=y_train_outer,
                                                              patience=patience),
                           n_trials=n_trials)  # n_trials=N_TRIALS
        else:
            study.optimize(lambda trial: objective(trial, X=X_train_outer, y=y_train_outer, kf=kf_inner,
                                                   patience=patience),
                           n_trials=n_trials)  # n_trials=N_TRIALS

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
        n_estimators_best = best_trial.user_attrs['n_estimators']
        max_depth_best = best_trial.params['max_depth']
        learning_rate_best = best_trial.params['learning_rate']
        colsample_bytree_best = best_trial.params['colsample_bytree']
        subsample_best = best_trial.params['subsample']
        reg_alpha_best = best_trial.params['reg_alpha']
        reg_lambda_best = best_trial.params['reg_lambda']
        gamma_best = best_trial.params['gamma']
        min_child_weight_best = best_trial.params['min_child_weight']

        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=n_estimators_best,
            max_depth=max_depth_best,
            learning_rate=learning_rate_best,
            colsample_bytree=colsample_bytree_best,
            subsample=subsample_best,
            reg_alpha=reg_alpha_best,
            reg_lambda=reg_lambda_best,
            gamma=gamma_best,
            min_child_weight=min_child_weight_best
        )

        model.fit(X_train_outer, y_train_outer)

        predictions = model.predict(X_test_outer)
        predictions = y_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

        mse_score = mean_squared_error(y_test_outer, predictions)
        mae_score = mean_absolute_error(y_test_outer, predictions)
        max_score = max_error(y_test_outer, predictions)

        logging.info(f"Outer Run: {split_nr}: MSE - {mse_score}, MAE - {mae_score}. MAX - {max_score}")

        nested_resampling_info = pd.DataFrame({'dataset_id': [dataset_id], 'dataset_name': [dataset_name],
                                               'outer_run': [split_nr], 'max_depth': [max_depth_best],
                                               'learning_rate': [learning_rate_best],
                                               'colsample_bytree': [colsample_bytree_best],
                                               'subsample': [subsample_best], 'reg_alpha': [reg_alpha_best],
                                               'reg_lambda': [reg_lambda_best], 'gamma': [gamma_best],
                                               'min_child_weight': [min_child_weight_best],
                                               'n_estimators': [n_estimators_best], 'mse': [mse_score],
                                               'mae': [mae_score], 'max_absolute_error': [max_score]})

        nested_resampling_infos = pd.concat([nested_resampling_infos, nested_resampling_info])

        split_nr += 1

        nested_resampling_infos.to_csv(nested_resampling_infos_file, index=False)


if __name__ == "__main__":
    HPO_XGB(n_splits_outer=5,
            n_splits_inner=4,
            n_trials=100,
            patience=10,
            file="None",
            PPV=True,
            dataset_id=44027)

