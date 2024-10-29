"""
This module contains various machine learning functions and classes for training and evaluating 
different types of prediction models, including LightGBM, K-Nearest Neighbors, Decision Trees, 
XGBoost, Gaussian Process, Logistic Regression, ExtraTrees, Random Forest, Neural Networks, 
and Support Vector Machines. It also includes functionality for computing Shapley values 
using the SHAP library.

Classes:
    PredictionModel: A class for training and evaluating a prediction model,
                     computing Shapley values, and storing the results.
    NnetMultiObj: A class for training an ensemble of neural networks
                  using the specified resampling technique.
    SvmMultiObj: A class for training an ensemble of support vector machines
                 using the specified resampling technique.
"""

import time
from warnings import simplefilter
import shap
import numpy as np

from sklearn import svm as sk_svm
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import resample
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import ConvergenceWarning

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from ExplainMLForecasting.utils import hyperparam_search, create_grouped_folds, shapley_kernel_wrapper, upsample

simplefilter("ignore", category=ConvergenceWarning)

# Hyperparameter space for the support vector machines
svm_cspace = 2. ** np.linspace(-5., 10., 10)
svm_gammaspace = 2. ** np.linspace(-10., 3., 10)

class PredictionModel:
    """
    PredictionModel is a class for training and evaluating a prediction model,
    computing Shapley values, and storing the results.

    Attributes:
        model (object): Prediction model object in the standard sklearn format.
        name (str): Name given to the model.
        data (dict): Contains the training and test data.
        config (Config): Configuration object.

        best_hyper (dict or None): Best hyperparameters if available.
        shap_val (array-like or None): Shapley values.
        shap_val_inter (array-like or None): Shapley interaction values.
        output (dict): Dictionary containing model name, predictions, fitted values, 
                       model object, hyperparameters, Shapley values, 
                       Shapley interaction values, and training time.
    """

    def __init__(self, model, name, data, config, **kwargs):

        self.trainx = data["trainx"]
        self.trainy = data["trainy"]
        self.testx = data["testx"]
        self.config = config
        self.model = model
        self.name = name

        start_time = time.time()
        self._train() # train model
        stop_time = time.time()

        self.best_hyper = None
        if hasattr(self.model, "best_params_"):
            self.best_hyper = model.best_params_
        if hasattr(self.model, "best_estimator_"):
            self.model = self.model.best_estimator_

        self.shap_val, self.shap_val_inter = self._compute_shap() # compute Shapley values

        self.output = {
            "name": name,
            "pred": model.predict_proba(self.testx)[:, 1],
            "fit": model.predict_proba(self.trainx)[:, 1],
            "model": self.model,
            "hyper_params": self.best_hyper,
            "shapley": self.shap_val,
            "shapley_inter": self.shap_val_inter,
            "time": stop_time - start_time
        }

    def _train(self, **kwargs):
        """
        Train the prediction model and obtain predictions.
        """

        self.model.fit(self.trainx, self.trainy, **kwargs)

    def _compute_shap(self): # compute Shapley values
        """
        Compute Shapley values and optionally Shapley interaction values for the model.
        This method calculates the Shapley values using either the TreeExplainer or KernelExplainer
        from the SHAP library, depending on the model type. If the model is a tree-based model
        (e.g., "extree" or "forest"), it uses TreeExplainer. 
        For other models, it uses KernelExplainer.

        Returns:
            tuple: A tuple containing:
                - shap_val (numpy.ndarray or None): The computed Shapley values.
                - shap_val_inter (numpy.ndarray or None): The computed Shapley interaction values.
        """

        shap_val = None
        shap_val_inter = None

        if self.config.exp_do_shapley:

            if self.name in ["extree", "forest"]: # TreeExplainer
                explainer_tree = shap.TreeExplainer(self.model)
                shap_val = explainer_tree.shap_values(self.testx)[1]

                if self.config.exp_shapley_interaction: # compute Shapley interaction
                    shap_val_inter = explainer_tree.shap_interaction_values(self.testx)[1]

            else: # KernelExplainer
                shap_val = shapley_kernel_wrapper(self.model, self.trainx, self.testx, self.config)

        return (shap_val, shap_val_inter)

def lgbm(data, config, name):
    """
    Creates and trains a LightGBM classifier model.

    Args:
        data (pd.DataFrame): The dataset to be used for training the model.
        config (Config): Configuration parameters for the model.
        name (str): The name assigned to the model instance.
    Returns:
        dict: The output of the PredictionModel instance.
    """

    model = LGBMClassifier()
    model_instance = PredictionModel(model, name, data, config)
    return model_instance.output

def knn(data, config, name):
    """
    Creates and trains a K-Nearest Neighbors (KNN) model.

    Args:
        data (pd.DataFrame): The dataset to be used for training the model.
        config (Config): Configuration parameters for the model.
        name (str): The name assigned to the model instance.
    Returns:
        dict: The output of the PredictionModel instance.
    """

    model = KNeighborsClassifier()
    model_instance = PredictionModel(model, name, data, config)
    return model_instance.output

def dtree(data, config, name):
    """
    Creates and trains a Decision Tree classifier model.

    Args:
        data (pd.DataFrame): The dataset to be used for training the model.
        config (Config): Configuration parameters for the model.
        name (str): The name assigned to the model instance.
    Returns:
        dict: The output of the PredictionModel instance.
    """

    model = tree.DecisionTreeClassifier()
    model_instance = PredictionModel(model, name, data, config)
    return model_instance.output

def xgb(data, config, name):
    """
    Creates and trains a XGBoost classifier model.

    Args:
        data (pd.DataFrame): The dataset to be used for training the model.
        config (Config): Configuration parameters for the model.
        name (str): The name assigned to the model instance.
    Returns:
        dict: The output of the PredictionModel instance.
    """

    model = XGBClassifier()
    model_instance = PredictionModel(model, name, data, config)

    return model_instance.output

def gaussianprocess(data, config, name):
    """
    Creates and trains a Gaussian Process classifier model.

    Args:
        data (pd.DataFrame): The dataset to be used for training the model.
        config (Config): Configuration parameters for the model.
        name (str): The name assigned to the model instance.
    Returns:
        dict: The output of the PredictionModel instance.
    """

    model = GaussianProcessClassifier()
    model_instance = PredictionModel(model, name, data, config)
    return model_instance.output

def logreg(data, config, sample_weight, name):
    """
    Creates and trains a Logistic regression model.

    Args:
        data (pd.DataFrame): The dataset to be used for training the model.
        config (Config): Configuration parameters for the model.
        sample_weight (np.array): The sample weights for the training data.
        name (str): The name assigned to the model instance.
    Returns:
        dict: The output of the PredictionModel instance.
    """

    model = LogisticRegression(penalty="none", solver = "lbfgs")
    model_instance = PredictionModel(model, name, data, config, sample_weight = sample_weight)
    return model_instance.output

def extree(data, config, sample_weight, cv_hyper, do_cv, name):
    """
    Creates and trains a Extremely Randomized Trees (ExtraTrees) classifier model.

    Args:
        data (pd.DataFrame): The dataset to be used for training the model.
        config (Config): Configuration parameters for the model.
        sample_weight (array-like): Sample weights for training.
        cv_hyper (int): Number of cross-validation folds for hyperparameter tuning.
        do_cv (bool): Flag to indicate whether to perform hyperparameter tuning.
        name (str): The name assigned to the model instance.
    Returns:
        dict: The output of the PredictionModel instance.
    """
    if do_cv:
        hyperparameters = {
            'max_features': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
            'max_depth': [2, 3, 4, 5, 7, 10, 12, 15, 20]
        }
        model = hyperparam_search(ExtraTreesClassifier(n_estimators=1000,  n_jobs=1),
                                  hyperparameters,
                                  use=config.exp_search,
                                  n_jobs=config.exp_n_kernels, cv=cv_hyper,
                                  scoring=config.exp_optimization_metric,
                                  n_iter=config.exp_n_iter_rsearch,
                                  verbose=config.exp_verbose)
    else:
        model = ExtraTreesClassifier(n_estimators=1000, n_jobs=config.exp_n_kernels)

    model_instance = PredictionModel(model, name, data, config, sample_weight = sample_weight)

    return model_instance.output

def forest(data, config, sample_weight, cv_hyper, do_cv, name):
    """
    Creates and trains a Random Forest classifier model.

    Args:
        data (pd.DataFrame): The dataset to be used for training the model.
        config (Config): Configuration parameters for the model.
        sample_weight (array-like): Sample weights for training.
        cv_hyper (int): Number of cross-validation folds for hyperparameter tuning.
        do_cv (bool): Flag to indicate whether to perform hyperparameter tuning.
        name (str): The name assigned to the model instance.
    Returns:
        dict: The output of the PredictionModel instance.
    """

    if do_cv:
        hyperparameters = {'max_features': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                           'max_depth': [2, 3, 4, 5, 7, 10, 12, 15, 20]
                           }
        model = hyperparam_search(RandomForestClassifier(n_estimators=1000,  n_jobs=1),
                                    hyperparameters,
                                    use=config.exp_search,
                                    n_jobs=config.exp_n_kernels,
                                    cv=cv_hyper,
                                    scoring=config.exp_optimization_metric,
                                    n_iter=config.exp_n_iter_rsearch,
                                    verbose=config.exp_verbose)

    else:
        model = RandomForestClassifier(n_estimators=1000, n_jobs=config.exp_n_kernels)

    model_instance = PredictionModel(model, name, data, config, sample_weight = sample_weight)

    return model_instance.output

def nnet_single(data, cv_hyper, config, name):
    """
    Creates and trains a neural network model.

    Args:
        data (pd.DataFrame): The dataset to be used for training the model.
        cv_hyper (int): Number of cross-validation folds for hyperparameter tuning.
        config (Config): Configuration parameters for the model.
        name (str): The name assigned to the model instance.
    Returns:
        dict: The output of the PredictionModel instance.
    """

    n_features = data["trainx"].shape[1]
    hyperparameters = {'alpha': 10.0 ** np.linspace(-3.0, 3.0, 10),
                       'hidden_layer_sizes': list(
                               set([round(n_features / 3.0), round(n_features / 2.0), n_features,
                                    (n_features, n_features),
                                    (n_features, round(n_features / 2.0)),
                                    (n_features*2, n_features),
                                    (n_features*2, n_features*2)
                                    ])),
                        'activation': ['tanh', 'relu']}

    # Exclude single neuron or zero neuron network
    hyperparameters["hidden_layer_sizes"] = list(
        set(hyperparameters["hidden_layer_sizes"]).difference(set([0, (1, 0)]))
    )

    model = hyperparam_search(MLPClassifier(solver='lbfgs'),
                               hyperparameters,
                               use=config.exp_search,
                               n_jobs=config.exp_n_kernels, cv=cv_hyper,
                               scoring=config.exp_optimization_metric,
                               n_iter=config.exp_n_iter_rsearch,
                               verbose=config.exp_verbose)

    model_instance = PredictionModel(model, name, data, config)
    return model_instance.output

def nnet_multi(data, config, group, name):
    """
    Creates and trains a ensemble of neural network models, which is computationally intensive 
    and recommended to be run on a high-performance cluster. The ensemble searches for 
    hyperparameters for each of the 25 base models to increase the variance across models.

    Args:
        data (pd.DataFrame): The dataset to be used for training the model.
        config (Config): Configuration parameters for the model.
        group (str): The group identifier for the model.
        name (str): The name assigned to the model instance.
    Returns:
        dict: The output of the PredictionModel instance.
    """
    # resample_type is one of the following ["bootstrap", "copy", "upsample"]
    resample_type="bootstrap"

    n_features = data["trainx"].shape[1]
    hyperparameters = {'alpha': 10.0 ** np.linspace(-3.0, 3.0, 10),
                       'hidden_layer_sizes': list(
                               set([round(n_features / 3.0), round(n_features / 2.0), n_features,
                                    (n_features, n_features),
                                    (n_features, round(n_features / 2.0)),
                                    (n_features*2, n_features),
                                    (n_features*2, n_features*2)
                                    ])),
                        'activation': ['tanh', 'relu']}

    # Exclude single neuron or zero neuron network
    hyperparameters["hidden_layer_sizes"] = list(
        set(hyperparameters["hidden_layer_sizes"]).difference(set([0, (1, 0)]))
    )

    model = NnetMultiObj(
        resample_type=resample_type, config=config, hyperparameters=hyperparameters, group=group
    )
    model_instance = PredictionModel(model, name, data, config)
    return model_instance.output

class NnetMultiObj(BaseEstimator, ClassifierMixin):
    """
    This class trains an ensemble of neural networks using the specified resampling technique.

    Attributes:
        models (list): A list to store the trained models.
        n_models (int): The number of models in the ensemble.
        resample_type (str): The type of resampling technique to use ('bootstrap' or 'upsample').
        hyperparameters (dict): The hyperparameters for the neural network.
        config (Config): Configuration parameters for the model.
        group (array-like): The group identifier for the model.
    """

    def __init__(self, resample_type, config, hyperparameters, group):
        self.models = []
        self.n_models = 25
        self.resample_type = resample_type
        self.hyperparameters = hyperparameters
        self.config = config
        self.group = group

    def fit(self, x, y=None):
        """
        Fits multiple models to the provided data using 
        resampling techniques and hyperparameter search.

        Args:
            x (numpy.ndarray): The training data.
            y (numpy.ndarray): The target variable.
        Returns:
            object: The fitted model instance.
        """

        for _ in np.arange(self.n_models):

            if self.resample_type == "bootstrap":
                x_rs, y_rs, group_rs = resample(
                    x, y, self.group, replace=True
                    )
            elif self.resample_type == "upsample":
                x_rs, y_rs, group_rs = upsample(
                    x, y, group=self.group, costs={0: y.mean(), 1: 1 - y.mean()}
                )
            else:
                x_rs, y_rs, group_rs = x, y, self.group

            cv_hyper, cv_fold_vector = create_grouped_folds(y_rs, group_rs, nfolds=5, reps=1)

            m = hyperparam_search(MLPClassifier(solver='lbfgs'),
                                  self.hyperparameters,
                                  use=self.config.exp_search,
                                  n_jobs=self.config.exp_n_kernels, cv=cv_hyper,
                                  scoring=self.config.exp_optimization_metric,
                                  n_iter=self.config.exp_n_iter_rsearch,
                                  verbose=self.config.exp_verbose)
            m.fit(x_rs, y_rs)
            self.models.append(m)

        return self

    def predict_proba(self, x):
        """
        Predict class probabilities for the input data using an ensemble of models.

        Args:
            x (numpy.ndarray): Input features with shape (n_samples, n_features).
        Returns:
            numpy.ndarray: Averaged class probabilities with shape (n_samples, n_classes).
        """

        predm = np.zeros((x.shape[0], self.n_models, 2)) * np.nan
        for m in np.arange(len(self.models)):
            predm[:, m, :] = self.models[m].predict_proba(x)

        return predm.mean(axis=1)

def svm_single(data, cv_hyper, config, sample_weight, name):
    """
    Creates and trains a Support-vector machine model 
    with radial basis function kernel

    Args:
        data (pd.DataFrame): The dataset to be used for training the model.
        cv_hyper (int): Number of cross-validation folds for hyperparameter tuning.
        config (Config): Configuration parameters for the model.
        sample_weight (np.array): The sample weights for the training data.
        name (str): The name assigned to the model instance.
    Returns:
        dict: The output of the PredictionModel instance.
    """

    hyperparameters= {'C': svm_cspace, 'gamma': svm_gammaspace}
    model = hyperparam_search(sk_svm.SVC(kernel='rbf', probability=True),
                          hyperparameters,
                          use=config.exp_search,
                          n_jobs=config.exp_n_kernels,
                          cv=cv_hyper,
                          scoring=config.exp_optimization_metric,
                          n_iter=config.exp_n_iter_rsearch,
                          verbose=config.exp_verbose)
    model_instance = PredictionModel(model, name, data, config,
                                     sample_weight = sample_weight)
    return model_instance.output

def svm_multi(data, config, group, sample_weight, name):
    """
    Creates and trains a ensemble of Support-vector machine models, which is computationally 
    intensive and recommended to be run on a high-performance cluster. The ensemble searches for 
    hyperparameters for each of the 25 base models to increase the variance across models.

    Args:
        data (pd.DataFrame): The dataset to be used for training the model.
        config (Config): Configuration parameters for the model.
        group (str): The group identifier for the model.
        sample_weight (np.array): The sample weights for the training data.
        name (str): The name assigned to the model instance.
    Returns:
        dict: The output of the PredictionModel instance.
    """

    # resample_type is one of the following ["none", "bootstrap", "copy", "upsample"]
    resample_type = "upsample"

    if config.exp_do_upsample and (resample_type=="upsample"):
        raise ValueError(
           "The SVM ensemble upsamples the data already, It is not recommended to"
           "upsample another time using the the exp_do_upsample of the Config class."
           )

    hyperparameters = {'C': svm_cspace, "gamma": svm_gammaspace}
    model = SvmMultiObj(
        config=config,
        hyperparameters=hyperparameters,
        group=group,
        resample_type=resample_type,
        sample_weight=sample_weight
    )

    model_instance = PredictionModel(model, name, data, config)

    return model_instance.output

class SvmMultiObj(BaseEstimator, ClassifierMixin):
    """
    This class trains an ensemble of support vector machine 
    using the specified resampling technique.

    Attributes:
        models (list): A list to store the trained models.
        n_models (int): The number of models in the ensemble.
        hyperparameters (dict): The hyperparameters for the neural network.
        config (Config): Configuration parameters for the model.
        group (array-like): The group identifier for the model.
        resample_type (str): The type of resampling technique to use ('bootstrap' or 'upsample').
        sample_weight (array-like): The sample weights for the training data.
    """

    def __init__(self, config, hyperparameters, group, resample_type, sample_weight):
        self.models = []
        self.n_models = 25 # number of models in the ensemble
        self.hyperparameters = hyperparameters
        self.config = config
        self.group = group
        self.resample_type = resample_type
        self.sample_weight = sample_weight

    def fit(self, x, y=None):
        """
        Fits multiple models to the provided data using 
        resampling techniques and hyperparameter search.

        Args:
            x (numpy.ndarray): The training data.
            y (numpy.ndarray): The target variable.
        Returns:
            object: The fitted model instance.
        """

        for _ in np.arange(self.n_models):

            if self.resample_type == "bootstrap":
                x_rs, y_rs, group_rs = resample(x, y, self.group, replace=True)
            elif self.resample_type == "upsample":
                x_rs, y_rs, group_rs = upsample(
                    x, y, group=self.group, costs={0: y.mean(), 1: 1 - y.mean()}
                )
            else: x_rs, y_rs, group_rs = x, y, self.group

            cv_hyper, cv_fold_vector = create_grouped_folds(y_rs, group_rs, nfolds=5, reps=1)

            m = hyperparam_search(sk_svm.SVC(kernel='rbf', probability=True),
                              self.hyperparameters,
                              use=self.config.exp_search,
                              n_jobs=self.config.exp_n_kernels, cv=cv_hyper,
                              scoring=self.config.exp_optimization_metric,
                              n_iter=self.config.exp_n_iter_rsearch,
                              verbose=self.config.exp_verbose)
            if self.resample_type == "upsample":
                m.fit(x_rs, y_rs)
            else:
                m.fit(x_rs, y_rs, sample_weight=self.sample_weight)

            self.models.append(m)

        return self

    def predict_proba(self, x):
        """
        Predict class probabilities for the input data using an ensemble of models.

        Args:
            x (numpy.ndarray): Input features with shape (n_samples, n_features).
        Returns:
            numpy.ndarray: Averaged class probabilities with shape (n_samples, n_classes).
        """

        predm = np.zeros((x.shape[0], self.n_models, 2)) * np.nan
        for m in np.arange(len(self.models)):
            predm[:, m, :] = self.models[m].predict_proba(x)
        return predm.mean(axis=1)
