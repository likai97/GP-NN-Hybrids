"""
This file
"""


from HPO_DKL import HPO_DKL
from HPO_VDKL import HPO_VDKL
from HPO_DGP import HPO_DGP
from HPO_NNDGP import HPO_NNDGP
from HPO_DSPP import HPO_DSPP
from HPO_DeepEnsemble import HPO_DE
from HPO_RF import HPO_RF
from HPO_XGB import HPO_XGB


if __name__ == "__main__":
    # Configurations
    n_splits_outer = 5  # Number of Outer CV Splits
    n_splits_inner = 4  # Number of Inner CV Splits
    n_trials = 100  # Number of HPO trials to conduct per Outer CV split
    epochs_inner = 400   # Number of inner epochs
    patience = 5  # Number of early stopping iterations
    epochs_outer = 1000  # Number of Epochs to evaluate the best configuration
    patience_outer = 10  # Number of early stopping iterations

    # Type of DKL Study
    hpo = input("Enter HPO Study: ")

    # Either "k2204", "r1_08_small", "r1_08" or "None"
    file = input("Enter file: ")
    if file == "None":
        dataset_id = int(input("Enter dataset_id: "))
    else:
        dataset_id = 1234

    # Optuna Pruner Type, either "None" or "HB" for Hyperband
    pruner_type = input("Pruner Type: ")
    # Maximum number of seconds before cancel a HPO trial
    time_tolerance = int(input("Enter time tolerance: "))

    if hpo == "DKL":
        HPO_DKL(n_splits_outer=n_splits_outer,
                n_splits_inner=n_splits_inner,
                n_trials=n_trials,
                epochs_inner=epochs_inner,
                patience=patience,
                file=file,
                PPV=True,
                dataset_id=dataset_id,
                pruner_type=pruner_type,
                time_tolerance=time_tolerance)

    elif hpo == "VDKL":
        HPO_VDKL(n_splits_outer=n_splits_outer,
                 n_splits_inner=n_splits_inner,
                 n_trials=n_trials,
                 epochs_inner=epochs_inner,
                 patience=patience,
                 file=file,
                 PPV=True,
                 dataset_id=dataset_id,
                 pruner_type=pruner_type,
                 time_tolerance=time_tolerance,
                 epochs_outer=epochs_outer,
                 patience_outer=patience_outer)

    elif hpo == "DGP":
        HPO_DGP(n_splits_outer=n_splits_outer,
                n_splits_inner=n_splits_inner,
                n_trials=n_trials,
                epochs_inner=epochs_inner,
                patience=patience,
                file=file,
                PPV=True,
                dataset_id=dataset_id,
                pruner_type=pruner_type,
                time_tolerance=time_tolerance,
                epochs_outer=epochs_outer,
                patience_outer=patience_outer)

    elif hpo == "NNDGP":
        HPO_NNDGP(n_splits_outer=n_splits_outer,
                  n_splits_inner=n_splits_inner,
                  n_trials=n_trials,
                  epochs_inner=epochs_inner,
                  patience=patience,
                  file=file,
                  PPV=True,
                  dataset_id=dataset_id,
                  pruner_type=pruner_type,
                  time_tolerance=time_tolerance,
                  epochs_outer=epochs_outer,
                  patience_outer=patience_outer)

    elif hpo == "DSPP":
        HPO_DSPP(n_splits_outer=n_splits_inner,
                 n_splits_inner=n_splits_inner,
                 n_trials=n_trials,
                 epochs_inner=epochs_inner,
                 patience=patience,
                 file=file,
                 PPV=True,
                 dataset_id=dataset_id,
                 pruner_type=pruner_type,
                 time_tolerance=time_tolerance,
                 epochs_outer=epochs_outer,
                 patience_outer=patience_outer)

    elif hpo == "DE":
        HPO_DE(n_splits_outer=n_splits_inner,
               n_splits_inner=n_splits_inner,
               n_trials=n_trials,
               epochs_inner=epochs_inner,
               patience=patience,
               file=file,
               PPV=True,
               dataset_id=dataset_id,
               pruner_type=pruner_type,
               time_tolerance=time_tolerance,
               epochs_outer=epochs_outer,
               patience_outer=patience_outer)

    elif hpo == "RF":
        HPO_RF(n_splits_outer=n_splits_inner,
               n_splits_inner=n_splits_inner,
               n_trials=n_trials,
               file=file,
               PPV=True,
               dataset_id=dataset_id)

    elif hpo == "XGB":
        HPO_XGB(n_splits_outer=n_splits_inner,
                n_splits_inner=n_splits_inner,
                n_trials=n_trials,
                patience=patience,
                file=file,
                PPV=True,
                dataset_id=dataset_id)
