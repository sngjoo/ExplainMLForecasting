"""
This module contains the implementation of the `Run` class, 
which is used to run and evaluate machine learning experiments.

Classes:
    Run: A class to run and evaluate machine learning experiments.
"""

import os
import sys
import pickle
import math
import multiprocessing
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from sklearn.metrics import roc_curve

from ExplainMLForecasting.configure import Config
from ExplainMLForecasting.make_data import create_data
import ExplainMLForecasting.utils
from ExplainMLForecasting.train_and_test import train_and_test

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"


class Run:
    """
    A class to run and evaluate machine learning experiments.

    Attributes:
        iter (int): Number of iterations to run the cross-validation experiment.
        config (Config): Configuration object that specifies the data set and experimental setup.
        main_data (pd.DataFrame): The main data set used in the experiment.
        results (list): A list of dictionaries containing the results of the
                        cross-validation experiment.
    """

    def __init__(self, iter, exp_algos, data_predictors):

        self.iter = iter

        self.config = Config()

        self.config.exp_n_kernels = multiprocessing.cpu_count() - 1
        self.config.exp_algos = exp_algos
        self.config.data_predictors = data_predictors
        self.main_data = create_data(self.config)
        self.results = []

    def run_experiment(self):
        """
        Runs the cross-validation experiment multiple times and stores the results.
        """
        for _ in range(
            self.iter
        ):  # repeat the cross-validation experiment 'iter' times
            result = train_and_test(self.main_data, self.config)
            self.results.append(result)

        # pickle.dump(self.results, open("1000iter_extree.pickle", "wb"))
        # self.results = pickle.load(open("1000iter_extree.pickle", 'rb'))

    def evaluate(self):
        """
        Evaluate the performance of different algorithms across multiple iterations.
        This method calculates various performance metrics such as AUC, accuracy,
        balanced accuracy, true positive rate, false positive rate, true positives,
        true negatives, false positives, and false negatives for each algorithm
        over multiple iterations. The results are stored in an xarray DataArray
        and the mean and standard error of the performance metrics are computed.

        Returns:
            tuple: A tuple containing:
            - performance_across_mean (pd.DataFrame): The mean performance metrics
              across iterations for each algorithm.
            - performance_across_se (pd.DataFrame): The standard error of the
              performance metrics across iterations for each algorithm.
        """

        metrics = [
            "auc",
            "accuracy",
            "balanced",
            "tp_rate",
            "fp_rate",
            "tp",
            "tn",
            "fp",
            "fn",
        ]

        output_metric_across = xr.DataArray(
            np.zeros((len(self.results), len(self.config.exp_algos), len(metrics)))
            * float("nan"),
            [
                ("iterations", np.arange(len(self.results))),
                ("algorithms", self.config.exp_algos),
                ("metrics", metrics),
            ],
        )

        for r in range(self.iter):
            res_across_folds = (
                self.results[r]["predictions"]
                .apply(
                    lambda x: np.array(
                        [
                            utils.performance_results(self.main_data.crisis.values, x)[
                                z
                            ]
                            for z in metrics
                        ]
                    )
                )
                .T
            )
            res_across_folds.columns = metrics
            output_metric_across.loc[r, :, :] = res_across_folds

        performance_across_mean = output_metric_across.mean(axis=0).to_pandas()
        performance_across_se = output_metric_across.std(
            axis=0
        ).to_pandas() / math.sqrt(float(self.iter))

        return performance_across_mean, performance_across_se

    def make_roc(self):
        """
        Generate ROC curves and save them to Excel files.
        This method calculates the mean predictions across multiple iterations,
        computes the ROC curve for each algorithm, and saves the false positive
        rate (fpr) and true positive rate (tpr) values to separate Excel files.
        Steps:
        1. Compute the mean predictions for each algorithm across iterations.
        2. For each algorithm, calculate the ROC curve using the true labels and mean predictions.
        3. Save the fpr and tpr values to an Excel file for each algorithm.
        The Excel files are saved in the 'results' directory with filenames in the format:
        'roc_val_<algorithm>.xlsx'.
        Returns:
            None
        """

        # make roc with mean of each iterations
        pred_all_mean = [
            np.array(self.results[x]["predictions"][self.config.exp_algos])
            for x in range(self.iter)
        ]
        pred_all_mean = pd.DataFrame(np.stack(pred_all_mean).mean(0))
        pred_all_mean.columns = self.config.exp_algos

        roc_val = pd.DataFrame()
        for algo in pred_all_mean.columns:
            fpr, tpr, thres = roc_curve(
                self.main_data.crisis.values, pred_all_mean.loc[:, algo]
            )

            temp = pd.DataFrame([fpr, tpr]).T
            temp.columns = [f"fpr_{algo}", f"tpr_{algo}"]

            roc_val = pd.concat([roc_val, temp], axis=1)

            roc_val.to_excel(f"results/roc_val_{algo}.xlsx")

    def make_shapley(self):
        """
        Calculate and save the mean SHAP values for each model specified in the configuration.
        This method iterates over the models listed in `self.config.exp_algos`, computes the
        mean SHAP values for each model across all results, and saves the mean SHAP values
        to an Excel file. The SHAP values are extracted from the `self.results` attribute,
        which is expected to be a list of dictionaries containing SHAP values under the key
        "shapley". The mean SHAP values are computed using `numpy.nanmean` to handle any NaN values.
        The resulting mean SHAP values are saved to an Excel file named
        `results/shap_mean_{model_name}_1000.xlsx`.

        Returns:
            None
        """

        for model_name in self.config.exp_algos:

            shap_values = [
                np.array(x["shapley"].loc[model_name, :, :]) for x in self.results
            ]

            shap_values_mean = np.nanmean(np.dstack(shap_values), axis=2)
            df = pd.DataFrame(
                shap_values_mean, columns=self.results[0]["shapley"]["features"]
            )
            df.to_excel(f"results/shap_mean_{model_name}_1000.xlsx")
