"""
This script contains the functions that conduct the cross-validation and forecasting experiments.
"""

import random

import numpy as np
import pandas as pd

import xarray as xr
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from ExplainMLForecasting.utils import create_grouped_folds, upsample, weights_from_costs

def train_and_test(df, config):
    """
    Low level experiment function that samples the training and test data, 
    trains, and tests the prediction models (functions in ml_functions.py) either 
    with or without the computation of Shapley values.

    Args:
        df (pd.DataFrame): Data set with an arbitrary number of predictors and the columns crisis, 
                           crisis_id, year, and iso.
        config (Config): Configuration file that specifies the experimental setup.

    Returns:
        dict: A dictionary containing the results of the experiment.
    """

    algo_names = config.exp_algos
    nfolds = config.exp_nfolds
    hyper_folds = config.exp_hyper_folds
    rep_cv = config.exp_rep_cv

    x = df.copy()
    y = x['crisis'].values.astype(int)
    crisis_id = x['crisis_id'].values.astype(int)
    years = x.year.values

    x = x.drop(columns=['crisis', 'crisis_id', 'year', 'iso'])
    feature_names = x.columns.values
    x = np.array(x)

    output_ypred = pd.DataFrame(index=np.arange(len(x)), columns=algo_names)

    # prepare a list that contains all the results
    model_out = {i: [] for i in algo_names}
    fits_out  = {i: [] for i in algo_names}
    time_out  = {i: [] for i in algo_names}
    params_out= {i: [] for i in algo_names}

    if config.exp_do_shapley:
        output_shapley = xr.DataArray(
            np.zeros((len(algo_names), x.shape[0], x.shape[1])) * float("nan"),
            [
                ('algorithms', algo_names),
                ("instances", np.arange(x.shape[0])),
                ("features", feature_names)
            ]
        )

        output_shapley_fnull = pd.DataFrame(index=np.arange(len(x)), columns=algo_names)
    else:
        output_shapley = None
        output_shapley_fnull = None

    if config.exp_shapley_interaction:
        inter_algos = list(set(algo_names).intersection(set(["extree", "forest"])))
        output_shapley_inter = xr.DataArray(
            np.zeros((len(inter_algos), x.shape[0], x.shape[1], x.shape[1])) * float("nan"),
            [
                ('algorithms', inter_algos),
                ("instances", np.arange(x.shape[0])),
                ("features1", feature_names),("features2", feature_names)
            ]
        )
    else:
        output_shapley_inter = None

    results = {'predictions': output_ypred,
               "fits": fits_out,
               'ix_test': [],
               'ix_train': [], 
               'models': model_out,
               'parameters': params_out,
               "time": time_out,
               'shapley': output_shapley,
               "shapley_fnull": output_shapley_fnull,
               "data": [],
               'shapley_inter': output_shapley_inter}

    if config.exp_year_split is None: # Create the cross-validation folds
        if config.exp_id == "none":
            folds, _ = create_grouped_folds(
                y=y, y_group=np.arange(y.shape[0]), nfolds=nfolds, reps=1
            )
        if config.exp_id == "crisis":
            folds, _ = create_grouped_folds(
                y=y, y_group=crisis_id, nfolds=nfolds, reps=1
            )
        if config.exp_id == "year_and_crisis":
            folds, _ = create_grouped_folds(
                y=y, y_group=crisis_id, y_group_2=years, nfolds=nfolds, reps=1
            )
        if config.exp_id == "year":
            folds, _ = create_grouped_folds(
                y=y, y_group=years, nfolds=nfolds, reps=1
            )
    else:
        # If we have a year splitting training and test set:
        nfolds = 1

    # run through the folds
    for f in tqdm(np.arange(nfolds)):

        if config.exp_year_split is None:
            # obtain training and test set from the previously defined folds
            ix_train = list(folds[f][0])
            ix_test = list(folds[f][1])
        else:
            # observations before splitting year are used for training,
            # the remaining observations for testing
            ix_train = list(np.where(years <= config.exp_year_split)[0])
            ix_test = list(np.where(years > config.exp_year_split)[0])

        # a random shuffle of the order of observations.
        ix_train = np.array(random.sample(ix_train, len(ix_train)))
        ix_test = np.array(random.sample(ix_test, len(ix_test)))

        if config.exp_bootstrap == "naive":
            ix_train = np.random.choice(ix_train, size=len(ix_train), replace=True)

        if config.exp_bootstrap in ["up", "down"]:
            ix_pos = ix_train[y[ix_train] == 1]
            ix_neg = ix_train[y[ix_train] == 0]

            replacer = False
            if config.exp_bootstrap_replace == "yes":
                replacer = True # whether to sample the minority class by replacement as well

            if config.exp_bootstrap == "up":
                if len(ix_neg) > len(ix_pos):
                    ix_train = np.concatenate(
                        (
                            np.random.choice(ix_neg, size=len(ix_neg), replace=replacer),
                            np.random.choice(ix_pos, size=len(ix_neg), replace=True)
                        )
                    )

                if len(ix_pos) > len(ix_neg):
                    ix_train = np.concatenate(
                        (
                            np.random.choice(ix_pos, size=len(ix_pos), replace=replacer),
                            np.random.choice(ix_neg, size=len(ix_pos), replace=True)
                        )
                    )

            if config.exp_bootstrap == "down":
                if len(ix_neg) > len(ix_pos):
                    ix_train = np.concatenate(
                        (
                            np.random.choice(ix_pos, size=len(ix_pos), replace=replacer),
                            np.random.choice(ix_neg, size=len(ix_pos), replace=False)
                        )
                    )

                if len(ix_pos) > len(ix_neg):
                    ix_train = np.concatenate(
                        (
                            np.random.choice(ix_neg, size=len(ix_neg), replace=replacer),
                            np.random.choice(ix_pos, size=len(ix_neg), replace=False)
                        )
                    )

        results["ix_train"].append(ix_train)
        results["ix_test"].append(ix_test)

        dat = {
            'train_x' : x[ix_train, :],
            'test_x' : x[ix_test,: ],
            'train_y' : y[ix_train],
            'test_y' : y[ix_test],
            'train_crisis_id' : crisis_id[ix_train]
        }

        # The error costs (false positve, false negative) determine
        # how the instances are weighted in the training set
        if isinstance(config.exp_error_costs, dict):
            class_costs = config.exp_error_costs
        # objects are weighted, such that the weighted proportion
        # of objects contribute equally to the training set
        elif config.exp_error_costs == "balanced":
            class_costs = {0: dat["train_y"].mean(), 1: 1 - dat["train_y"].mean()}
        elif config.exp_error_costs == "0.5":
            class_costs = {0: 0.5, 1: 0.5} # each object has the same weight.
        else:
            raise ValueError("Error costs not recognized")

        if config.exp_do_upsample: # upsample training set
            group = upsample(
                dat["train_x"], dat["train_y"], group=dat["train_crisis_id"], costs=class_costs
            )
            sample_weight = compute_sample_weight(class_costs, dat["train_y"])
            cv_hyper, cv_fold_vector = create_grouped_folds(
                dat['train_y'], group, nfolds=hyper_folds, reps=rep_cv
            )
        else: # create folds for the hyperparater search. (Nested cross-validation)
            group = dat["train_crisis_id"]
            class_weight = weights_from_costs(class_costs, dat["train_y"])
            sample_weight = compute_sample_weight(class_weight, dat["train_y"])
            cv_hyper, cv_fold_vector = create_grouped_folds(
                dat['train_y'], dat["train_crisis_id"], nfolds=hyper_folds, reps=rep_cv
            )

        # rescale all variables according to the training set
        scaler = StandardScaler()
        dat['train_x_scaled'] = scaler.fit_transform(dat['train_x'])
        dat['test_x_scaled'] = scaler.transform(dat['test_x'])

        results["data"].append(dat)

        # Train and test prediction models
        data = {
            "trainx": dat['train_x_scaled'],
            "trainy": dat['train_y'],
            "testx": dat['test_x_scaled']
        }

        for algo in algo_names:

            out = globals()[algo](data,
                        config=config,
                        cv_hyper=cv_hyper,
                        group=group,
                        sample_weight=sample_weight,
                        do_cv = False, name = algo)

            append_results(results, out)

    return results

def append_results(results, add):
    """
    Appends the results obtained for a single fold to the previous results.
    """
    name = add["name"]
    ix_test = results["ix_test"][-1] # last element in list

    results['predictions'].loc[ix_test, name] = add["pred"]
    results['fits'][name].append(add["fit"])

    results["models"][name].append(add["model"])
    results["parameters"][name].append(add["hyper_params"])
    results["time"][name].append(add["time"])
    if add["shapley"] is not None:
        results["shapley"].loc[name, ix_test, :] = add["shapley"]

        if "shapley_inter" in add.keys():
            if not add["shapley_inter"] is None:
                results["shapley_inter"].loc[name, ix_test, :, :] = add["shapley_inter"]
