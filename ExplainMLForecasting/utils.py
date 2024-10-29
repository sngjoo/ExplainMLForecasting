"""
This script provides utility functions that are used in the experiments
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import shap

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression


def shapley_kernel_wrapper(model, trainx, testx, config):
    """
    This function is called by the prediction algorithms (ml_functions) to compute Shapley values.
    Note that the decision tree based models such as random forest do provide faster and exact
    (non-approximated) Shapley values with the TreeShapExplainer

    Args:
        model (sklearn.base.BaseEstimator): The machine learning model
                                            to compute Shapley values for.
        trainx (np.ndarray): The training data.
        testx (np.ndarray): The test data.
        config (object): Configuration object with the following attributes:
            - exp_shap_background (int): The number of background instances
                                         to use for the Shapley values.
    Returns:
        out (np.ndarray): An array of Shapley values for the test
    """

    if config.exp_shap_background >= trainx.shape[0]:
        background = trainx
    else:
        background = shap.kmeans(trainx, config.exp_shap_background)

        explainer = shap.KernelExplainer(model.predict_proba, background)
        # one background instance is enough if we use a linear model
        if isinstance(model, LogisticRegression):
            background = shap.kmeans(trainx, 1)
            backup_base = explainer.fnull
            explainer = shap.KernelExplainer(model.predict_proba, background)
            explainer.fnull = backup_base

    out = [
        explainer.shap_values(testx[i, :], l1_reg=0.0)[1] for i in np.arange(len(testx))
    ]
    return np.vstack(out)


def exclude_periods(data, config):
    """
    The function sets all cue values on the excluded periods to NA
    and returns an index of all the objects that should later be deleted.
    This way of processing ist best because all the preprocessing functions
    do not need consider cases where years are already missing

    Args:
        data (pd.DataFrame): The input dataset containing a 'year' column and other features.
        config (object): Configuration object with the following attributes:
            - data_exclude_extreme_period (bool): Flag to exclude extreme periods like the
                                                  Great Depression, World War 1, and World War 2.
            - data_period (str): Specifies the period to include in the data.
                                 It can be 'all', 'pre-ww2', or 'post-ww2'.
    Returns:
        tuple: A tuple containing:
            - data (pd.DataFrame): The modified dataset with excluded periods set to NA.
            - exclude_ix (np.ndarray): A boolean index array indicating the excluded periods.
    """

    exclude_ix = np.zeros(len(data)) > 1

    if config.data_exclude_extreme_period:
        # exclude great depression | but NOT the beginnig of this crisis
        exclude_ix = exclude_ix | (
            np.array(data["year"] > 1933) & np.array(data["year"] < 1939)
        )
        # exclude World War 1
        exclude_ix = exclude_ix | (
            np.array(data["year"] > 1913) & np.array(data["year"] < 1919)
        )
        # exclude World War 2
        exclude_ix = exclude_ix | (
            np.array(data["year"] > 1938) & np.array(data["year"] < 1946)
        )

    if config.data_period not in ["all", "pre-ww2", "post-ww2"]:
        raise ValueError("time split is either 'all', 'pre-ww2', or 'post-ww2'")

    if config.data_period == "pre-ww2":
        exclude_ix = exclude_ix | np.array(data["year"] > 1939)

    elif config.data_period == "post-ww2":
        exclude_ix = exclude_ix | np.array(data["year"] < 1946)

    feature_names = list(
        set(data.columns.values).difference(
            set(["year", "country", "iso", "crisis_id", "crisis"])
        )
    )

    # set all feature values to NA in the excluded periods
    data.loc[exclude_ix, feature_names] = np.nan

    return data, exclude_ix


def create_grouped_folds(y, y_group, y_group_2=None, nfolds=10, reps=1, balance=True):
    """
    Create folds such that all objects in the same y_group and in the same y_group_2
    (if not none) are assigned to the same fold

    Args:
        y (np.array): Binary outcome variable.
        y_group (np.array): Grouping variable, e.g., crisis indicator.
        y_group_2 (np.array, optional): Second grouping variable. Defaults to None.
        nfolds (int): Number of folds. Defaults to 10.
        reps (int): Number of replications of the n-fold cross-validation. Defaults to 1.
        balance (bool): If True, the outcome y is balanced as much as possible
                        (there are an equal number of positive observations in each fold).
                        Defaults to True.
    Returns:
        tuple: A tuple containing:
        - iterable (list): List of tuples, where each tuple contains the indices
                           for the training and test sets.
        - out (np.array): Array indicating the fold assignment for each observation.
    """

    no = y.size
    iterable = []
    for _ in np.arange(reps):
        placed = np.zeros(no, dtype=np.int64)
        out = np.zeros(no, dtype=np.int64) * np.nan
        pos_counter = np.zeros(nfolds, dtype=np.int64)
        neg_counter = np.zeros(nfolds, dtype=np.int64)

        # go through objects in random order
        oo = np.random.choice(np.arange(no), no, replace=False)
        for i in oo:
            if placed[i] == 0:
                if not y_group_2 is None:  # no verlap in year AND crisis_id
                    ix = np.where(
                        (y_group[i] == y_group) | (y_group_2[i] == y_group_2)
                    )[0]
                    for i in np.arange(25):
                        ix = np.where(
                            np.in1d(y_group, y_group[ix])
                            | np.in1d(y_group_2, y_group_2[ix])
                        )[0]
                else:  # no overlap in crisis_id
                    ix = np.where(y_group[i] == y_group)[0]

                placed[ix] = 1

                if balance:
                    if y[i] == 1:
                        rf = np.random.choice(
                            np.where(pos_counter == pos_counter.min())[0]
                        )
                        pos_counter[rf] += ix.size
                    else:
                        rf = np.random.choice(
                            np.where(neg_counter == neg_counter.min())[0]
                        )
                        neg_counter[rf] += ix.size
                else:
                    rf = int(np.random.randint(0, nfolds, 1))

                out[ix] = rf

        for f in np.arange(nfolds):
            ix_train = np.where(out != f)[0]
            ix_test = np.where(out == f)[0]
            # make sure that test set contains both classes
            if (not y[ix_test].mean() == 0) & (not y[ix_test].mean() == 1):
                iterable.append((ix_train, ix_test))

    if len(iterable) < nfolds * reps:
        print("Repeat folding, some test set had zero variance in criterion")
        return create_grouped_folds(
            y, y_group, y_group_2=y_group_2, nfolds=nfolds, reps=reps, balance=balance
        )
    else:
        return iterable, out


def create_forecasting_folds(y, year, min_crisis=20, temp_resolution=1):
    """
    Create folds for the forecasting experiment.
    This function generates training and testing indices for a forecasting experiment
    based on the provided binary outcome variable and corresponding years.
    It ensures that each training set contains a minimum number of crisis observations
    and that new models are trained at specified temporal resolutions.

    Args:
        y (np.array): Binary outcome variable.
        year (np.array): Time stamp for each observation.
        min_crisis (int): Minimum number of crisis observations in the training set. Default is 20.
        temp_resolution (int): Temporal resolution for training new models.
                               Default is 1, meaning a new model is trained for every year.
    Returns:
        tuple: A tuple containing:
            - iterable (list of tuples): Each tuple contains the training
                                         and testing indices for a fold.
            - last_train_year (list): List of the last training year for each fold.
    """

    iterable = []
    last_train_year = []
    uni_years = sorted(year.unique())
    del uni_years[-1]
    for i in np.arange(len(uni_years)):
        n_crisis = y[year <= uni_years[i]].sum()
        if (n_crisis >= min_crisis) & ((uni_years[i] % temp_resolution) == 0):
            ix_train = np.where(year <= uni_years[i - 1])[0]
            ix_test = np.where(year > uni_years[i - 1])[0]

            if (len(ix_train) > 0) & (len(ix_test) > 0):
                iterable.append((ix_train, ix_test))
                last_train_year.append(uni_years[i])
    return iterable, last_train_year


def hyperparam_search(
    model,
    parameters,
    use="grid",
    n_jobs=1,
    cv=None,
    scoring=None,
    n_iter=250,
    verbose=False,
):
    """
    Create a GridSearchCV or RandomizedSearchCV object for hyperparameter tuning.

    Args:
        model (sklearn.base.BaseEstimator): The machine learning model to be tuned.
        parameters (dict): Dictionary with parameters names (str) as keys and lists
                           of parameter settings to try as values.
        use (str, optional): The search method to use, either 'grid' for GridSearchCV or
                             'random' for RandomizedSearchCV. Defaults to 'grid'.
        n_jobs (int, optional): Number of jobs to run in parallel. Defaults to 1.
        cv (int or an iterable): Determines the cross-validation splitting strategy
        scoring (str, list/tuple or dict): A single string or a callable to evaluate
                                           the predictions on the test set. Defaults to None.
        n_iter (int): Number of parameter settings that are sampled in RandomizedSearchCV.
                      Ignored if use is 'grid'. Defaults to 250.
        verbose (bool): Controls the verbosity: the higher, the more messages. Defaults to False.
    Returns:
        model_out : GridSearchCV or RandomizedSearchCV object
                    The search object configured with the specified parameters.
    """

    if isinstance(cv, int):
        raise ValueError(
            "The argument cv should not be a number because the GridSearch algorithms"
            "in sklearn do always create the same folds even with differnt random seeds."
            "Rather you should pass folds that were created by function create_grouped_folds"
        )

    # do use gridsearch if less than n_iter
    if np.cumprod([len(parameters[x]) for x in parameters.keys()]).max() <= n_iter:
        use = "grid"  # combinations are tested

    if use == "grid":
        model_out = GridSearchCV(
            model, parameters, n_jobs=n_jobs, cv=cv, scoring=scoring, verbose=verbose
        )
    elif use == "random":
        model_out = RandomizedSearchCV(
            model,
            parameters,
            n_jobs=n_jobs,
            cv=cv,
            n_iter=n_iter,
            scoring=scoring,
            verbose=verbose,
        )
    else:
        raise ValueError("The argument 'use' should be either 'grid' or 'random'")

    return model_out


def write_file(
    data,
    file_name,
    path="../results/",
    round_digit=3,
    extension=".txt",
    short_name=6,
    append=False,
    shorten=True,
):
    """
    Writes a table as a text file to the specified path on the hard drive.

    Args:
        data (pd.DataFrame): The data to be written to the file.
        file_name (str): The name of the file to be written.
        path (str, optional): The directory path where the file will be saved.
                              Defaults to '../results/'.
        round_digit (int, optional): The number of decimal places to round the data.
                                     Defaults to 3.
        extension (str, optional): The file extension. Defaults to ".txt".
        short_name (int, optional): The maximum length of column and index names
                                    if shortened. Defaults to 6.
        append (bool, optional): If True, append to the file if it exists.
                                 If False, overwrite the file. Defaults to False.
        shorten (bool, optional): If True, shorten the column and index names.
                                  Defaults to True.
    """
    out = data.round(round_digit)
    if not os.path.exists(path):
        os.mkdir(path)
    if isinstance(data, pd.core.frame.DataFrame):
        if shorten:
            out.columns = [x.replace("_", "")[0:short_name] for x in out.columns.values]
            out.index = [
                str(x).replace("_", " ")[0:short_name] for x in out.index.values
            ]
        if append:
            out.to_csv(path + file_name + extension, sep="\t", mode="a", header=True)
        else:
            out.to_csv(path + file_name + extension, sep="\t", header=True)


def weights_from_costs(costs, y):
    """
    Weights observations according to the costs of the errors of the two classes.

    Args:
        costs (list or array-like): A list or array of two elements representing the
                                    costs associated with misclassifying each class.
        y (array-like): A binary array or list where each element is the true class
                        label (0 or 1) for each observation.
    Returns:
        dict: A dictionary with keys 0 and 1, where the values are the weights
              for class 0 and class 1, respectively.
    Example:
        If the cost vector is [0.5, 0.5] and class A is twice as prevalent as class B,
        objects in class B will get twice the weight as objects in class A.
    """

    p1 = y.mean()
    weights = {}
    weights[1] = costs[1] / (p1 * costs[1] + (1 - p1) * costs[0])
    weights[0] = costs[0] / (p1 * costs[1] + (1 - p1) * costs[0])
    return weights


def downsample(x, y, costs={0: 0.5, 1: 0.5}, group=None):
    """
    Downsample the majority class according to the costs of the errors.

    Args:
        x (numpy.ndarray): Feature matrix.
        y (numpy.ndarray): Target vector.
        costs (dict, optional): Dictionary specifying the cost of errors for each class.
                                Default is {0: 0.5, 1: 0.5}.
        group (numpy.ndarray, optional): Group labels for the samples. Default is None.
    Returns:
        tuple: A tuple containing the downsampled feature matrix, target vector, and group labels.
    """

    if group is None:
        group = np.arange(len(y))
    weights = weights_from_costs(costs, y)

    ix_pos = np.where(y == 1)[0]
    n_pos = ix_pos.size
    ix_neg = np.where(y == 0)[0]
    n_neg = ix_neg.size
    norm_w = min(weights.values())
    weights[0] = weights[0] / norm_w
    weights[1] = weights[1] / norm_w

    if weights[0] > weights[1]:
        ix_pos = np.random.choice(
            ix_pos, size=int(round(n_pos / weights[0])), replace=True
        )
    else:
        ix_neg = np.random.choice(
            ix_neg, size=int(round(n_neg / weights[1])), replace=True
        )
    ixses = np.concatenate((ix_pos, ix_neg))
    ixses = np.random.choice(ixses, size=ixses.size, replace=False)
    return x[ixses, :], y[ixses], group[ixses]


def upsample(x, y, group, costs):
    """
    Upsamples the minority class in the dataset.

    Args:
        x (numpy.ndarray): Feature matrix.
        y (numpy.ndarray): Target vector with binary class labels (0 and 1).
        group (numpy.ndarray): Group labels corresponding to each sample.
        costs (dict): Dictionary containing the costs associated with each class.
    Returns:
        tuple: A tuple containing the upsampled feature matrix, target vector, and group labels.
    """

    weights = weights_from_costs(costs, y)

    ix_pos = np.where(y == 1)[0]
    n_pos = ix_pos.size
    ix_neg = np.where(y == 0)[0]
    n_neg = ix_neg.size
    norm_w = min(weights.values())
    weights[0] = weights[0] / norm_w
    weights[1] = weights[1] / norm_w

    if weights[1] > weights[0]:
        ix_pos = np.random.choice(
            ix_pos, size=int(round(weights[1] * n_pos)), replace=True
        )
    else:
        ix_neg = np.random.choice(
            ix_neg, size=int(round(weights[0] * n_neg)), replace=True
        )

    ixses = np.concatenate((ix_pos, ix_neg))
    ixses = np.random.choice(ixses, size=ixses.size, replace=False)

    return x[ixses, :], y[ixses], group[ixses]


## UTILITIES FOR TRANSFORMING VARIABLES
def make_ratio(data_input, variables, denominator="gdp"):
    """
    Computes the ratio of specified variables to a denominator.

    Args:
        data_input (pd.DataFrame): The input data containing the variables.
        variables (str or list of str): The variable(s) to compute the ratio for.
        denominator (str): The denominator variable to use for the ratio. Default is "gdp".
    Returns:
        pd.DataFrame: A DataFrame with the computed ratio columns added.
        list of str: A list of the names of the new ratio columns.
    """

    names_out = []
    if isinstance(variables, str):
        variables = [variables]
    data = data_input.copy()
    for var in variables:
        varname = var + "_" + denominator
        data[varname] = data[var] / data[denominator]
        names_out.append(varname)
    return data, names_out


def make_shift(data_input, variables, shift_type, horizon=5):
    """
    Computes the change of a variable with respect to a certain horizon.

    Args:
        data_input (pd.DataFrame): Dataset. The transformed variable will be appended to this data.
        variables (list of str): Names of the variables in data_input that will be transformed.
        shift_type (str): Type of transformation.
                          Either "absolute" (change) or "percentage" (change).
        horizon (int, optional): The horizon over which the change is computed. Default is 5.
    Returns:
        pd.DataFrame: The input dataset with the transformed variables appended.
        list of str: Names of the newly created variables.
    """

    names_out = []
    data = data_input.copy()
    data_group = data.groupby("iso")
    if isinstance(variables, str):
        variables = [variables]

    for var in variables:
        if shift_type == "absolute":
            varname = var + "_rdiff" + str(horizon)
            data[varname] = data_group[var].diff(horizon)
        elif shift_type == "percentage":
            varname = var + "_pdiff" + str(horizon)
            # attention objects must be ordered by year and country as they are in the original data
            result = data_group[var].apply(lambda x: lag_pct_change(x, h=horizon))
            data[varname] = result.droplevel(0)
            # data[varname] = data_group[var].pct_change(horizon)

        names_out.append(varname)
    return data, names_out


def lag_pct_change(x, h):
    """
    Computes percentage changes

    Returns:
        float: The percentage change.
    """
    lag = np.array(pd.Series(x).shift(h))
    return (x - lag) / lag


def make_level_change(data_input, variables, change_type, horizon=10):
    """
    Computes the Hamilton filter or difference from moving average.

    Args:
        data_input (pd.DataFrame): Dataset. The transformed variable will be appended to this data.
        variables (list of str or str): Name(s) of the variables in data_input
                                        that will be transformed.
        change_type (str): Type of transformation. Either "ham" (Hamilton filter)
                           or "mad" (moving average difference).
        horizon (int, optional): The horizon for the moving average difference. Default is 10.
    Returns:
        tuple: A tuple containing the modified DataFrame and a list of names of the new variables.
    """

    names_out = []
    data = data_input.copy()
    data_grouped = data.groupby("iso")
    if isinstance(variables, str):
        variables = [variables]
    for var in variables:
        if change_type == "mad":
            varname = var + "_mad"
            data[varname] = np.nan
            data_mad = pd.DataFrame(
                data_grouped.apply(mov_ave_diff, var, horizon), columns=[varname]
            )
            for iso in data_mad.index.values:
                data.loc[data.iso == iso, varname] = data_mad.loc[iso, varname]

        if change_type == "ham":
            varname = var + "_ham"
            data[varname] = np.nan
            data_ham = pd.DataFrame(
                data_grouped.apply(hamilton_filter, var, 2, 4), columns=[varname]
            )
            for iso in data_ham.index.values:
                data.loc[data.iso == iso, varname] = data_ham.loc[iso, varname]
        names_out.append(varname)
    return data, names_out


def make_relative_change(data_input, variables, index="gdp", horizon=5):
    """
    Computes the change of a variable relative to the change of another variable.
    Args:
        data_input (pd.DataFrame): Dataset. The transformed variable will be appended to this data.
        variables (list of str): Names of the variables in data_input that will be transformed.
        index (str): Name of the variable to which the change is relative. Default is 'gdp'.
        horizon (int): The horizon over which the relative change is computed. Default is 5.
    Returns:
        tuple: A tuple containing the modified DataFrame and a list of names of the new variables.
    """

    names_out = []
    data = data_input.copy()
    data_grouped = data.groupby("iso")
    if isinstance(variables, str):
        variables = [variables]
    for var in variables:
        varname = var + "_idiff" + str(horizon)
        data[varname] = np.nan
        data_idiff = pd.DataFrame(
            data_grouped.apply(index_ratio_change, var, index, horizon),
            columns=[varname],
        )
        for iso in data_idiff.index.values:
            data.loc[data.iso == iso, varname] = data_idiff.loc[iso, varname]
        names_out.append(varname)
    return data, names_out


def mov_ave_diff(group, col, ma_len=10):
    """
    Computes the gap between a moving average (of length ma_len) and the
    observations on a grouped data set.

    Args:
        group (DataFrame): The grouped data set.
        col (str): The column name for which the moving average difference is computed.
        ma_len (int, optional): The length of the moving average window. Default is 10.
    Returns:
        numpy.ndarray: An array containing the differences between
                       the moving average and the observations.
    """

    values = group[col].values
    n = len(values)
    out = np.zeros(n) * np.nan
    if n >= ma_len:
        for i in range(n - ma_len + 1):
            out[i + ma_len - 1] = values[i + ma_len - 1] - np.mean(
                values[i - 1 : i + ma_len - 1]
            )
    return out


def index_ratio_change(group, ind1, ind2, l=5):
    """
    Calculate the relative change of ind1 to ind2 over a specified period for group values.
    Args:
        group (pd.DataFrame): The DataFrame containing the data.
        ind1 (str): The column name for the first index.
        ind2 (str): The column name for the second index.
        l (int, optional): The period over which to calculate the change. Default is 5.
    Returns:
        np.ndarray: An array containing the relative changes,
                    with NaN for positions where the period is not met.
    """

    val1 = group[ind1].values
    val2 = group[ind2].values
    n = len(val1)
    out = np.zeros(n) * np.nan

    if n >= l:
        for i in range(n - l):
            out[i + l] = (val1[i + l] / val1[i]) / (val2[i + l] / val2[i]) - 1

    return out


def hamilton_filter(group, col, h=2, p=4, output="cycle"):
    """Computes the Hamilton filter for a given time series.

    Args:
        group (pd.DataFrame): The dataframe containing the time series data.
        col (str): The column name of the time series data within the dataframe.
        h (int, optional): Look-ahead period. Default is 2.
        p (int, optional): Number of lagged variables. Default is 4.
        output (str, optional): The type of output to return. Options are "x", "trend",
                                "cycle", "random", or "all". Default is "cycle".
    Returns:
        np.ndarray or pd.DataFrame: Depending on the 'output' parameter, returns the original
                                    series, trend, cycle, random walk,
                                    or all of them in a dataframe.
        Notes:
        - The Hamilton filter is used to decompose a time series into trend and cycle components.
        - If the dataframe passed is full of NaNs, the function will return
          the original series for all components.
        - If 'output' is "all", a dataframe containing all components is returned and plotted.
        - If an invalid 'output' type is provided, an error message is printed.
    """

    x = group[col].values
    # note: Hamilton used 100 times x's logrithm in his employment data,
    # however, this is commented out because our data has negative values
    # x = 100*np.log(x)
    # Get the trend/predicted series
    trend = hamilton_filter_glm(x, h, p)
    if trend is not None:  # if dataframe passed is not full of nans
        # Get the cycle which is simple the original series substracted by the trend
        cycle = x - trend
        # Get the random walk series which is simply the difference between
        # the original series and the h look back
        df_x = pd.DataFrame(x)
        df_x_h = df_x.shift(h)
        random = df_x - df_x_h
    else:
        trend = x
        cycle = x
        random = x

    # Return required series in result, if all is selected then all results
    # are returned in a data frame
    if output == "x":
        return x
    elif output == "trend":
        return trend
    elif output == "cycle":
        return np.asarray(cycle)
    elif output == "random":
        return random
    elif output == "all":
        df = pd.DataFrame()
        df["x"] = x
        df["trend"] = trend
        df["cycle"] = cycle
        df["random"] = random
        df.plot()
        # pyplot.show()
        return df
    else:
        print("\nInvalid output type")


def hamilton_filter_glm(x, h=2, p=4):
    """
    Applies the Hamilton filter using a Generalized Linear Model (GLM) to a time series.
    Args:
        x (array-like): The input time series data.
        h (int): The number of periods to shift for the Hamilton filter. Default is 2.
        p (int): The number of lags to include in the model. Default is 4.
    Returns:
        array-like: The predicted values from the GLM, or the original time series
                    if the model cannot be fitted.
        Notes:
        - The function creates a DataFrame from the input time series
          and constructs lagged variables.
        - It then fits an Ordinary Least Squares (OLS) regression model to the data.
        - If the model cannot be fitted due to NaN values, the original time series is returned.
    """

    # Create dataframe for time series to be smoothed, the independent variable y
    df = pd.DataFrame(x)
    df.columns = ["yt8"]
    # Create matrix of dependent variables X which are the shifts of 8 period back
    # for 4 consecutive lags on current time t
    for lag in range(h, (h + p)):
        df["xt_" + str(lag - h + 1)] = df.yt8.shift(lag)
    # Getting the dependent varaibles X's index names
    x_columns = []
    for i in range(1, p + 1):
        new_s = "xt_" + str(i)
        x_columns.append(new_s)
    # y and X variables for regression
    y = df["yt8"]
    x = df[x_columns]

    xt_0 = pd.DataFrame(np.ones((df.shape[0], 1)))
    xt_0.columns = ["xt_0"]
    x = xt_0.join(x)
    # Build the OLS regression model and drop the NaN
    try:
        if sum(np.isnan(y)) != y.size:
            model = sm.OLS(y, x, missing="drop").fit()
            # Optional: print out the statistics
            model.summary()
            predictions = model.predict(x)
            return predictions
        return y
    except ValueError:
        pass


def all_same(items):
    """
    Check if all elements in a list are the same.

    Args:
        items (list): A list of elements to check.
    Returns:
        bool: True if all elements in the list are the same, False otherwise.
    """

    return all(x == items[0] for x in items)


def sigmoid(x):
    """
    Compute the sigmoid of x.

    Args:
        x (float): The input value or array of values.
    Returns:
        float: The sigmoid of the input value(s).
    """

    return 1 / (1 + np.exp(-x))


def sigmoidinv(x):
    """
    Compute the inverse of the sigmoid function.
    The sigmoid function is defined as 1 / (1 + exp(-x)). This function
    computes the inverse, which is log(x / (1 - x)).

    Args:
        x (float): The input value for which to compute the inverse sigmoid.
                Must be in the range (0, 1).
    Returns:
        float: The inverse of the sigmoid function evaluated at x.
    """

    return -np.log(1.0 / x - 1)


def normalize(data):
    """
    Normalize the given data by applying the normalize_value function to each element.

    Args:
        data (pandas.Series or pandas.DataFrame): The data to be normalized.
    Returns:
        pandas.Series or pandas.DataFrame: The normalized data.
    """

    return data.apply(normalize_value)


def normalize_value(x):
    """
    Normalize the input array to a range between 0 and 1.
    Args:
        x (numpy.ndarray): Input array to be normalized.
                           The array will be converted to float32 type.
    Returns:
        numpy.ndarray: Normalized array with values ranging from 0 to 1.
    """

    x = x.astype(dtype="float32")
    return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))


def performance_results(y_in, y_pred_in, threshold=0.5):
    """
    Computes performance metrics for binary classification.

    Args:
        y_in (np.array): True values (0 or 1) of the response variable.
        y_pred_in (np.array): Predicted values of the response variable (between 0 and 1).
        threshold (float, optional): Threshold for classifying the predicted values.
                                        If y_pred_in >= threshold, the predicted class is positive,
                                        otherwise negative. Default is 0.5.
    Returns:
        dict: A dictionary containing the following performance metrics:
            - 'tp' (int): True positives.
            - 'tn' (int): True negatives.
            - 'fp' (int): False positives.
            - 'fn' (int): False negatives.
            - 'accuracy' (float): Accuracy of the predictions.
            - 'tp_rate' (float): True positive rate (sensitivity or recall).
            - 'fp_rate' (float): False positive rate.
            - 'balanced' (float): Balanced accuracy.
            - 'auc' (float): Area under the ROC curve.
    """

    y_pred_in = np.array(y_pred_in, dtype=float)
    # types of Y and Y_pred are pd.Seres
    ix_miss = np.isnan(y_pred_in)

    y = y_in[~ix_miss].copy()
    y_pred = y_pred_in[~ix_miss].copy()

    n = y.size
    y = y * 1  # typecast boolean variables
    y_pred = y_pred * 1  # typecast boolean variables

    y_bool = np.array(y, dtype="bool")
    y_pred_bool = y_pred >= threshold
    # True positive (tp), ture negative (tn), false positive (fp), false negative (fn)
    tp = np.logical_and(y_bool, y_pred_bool).sum()
    tn = np.logical_and(~y_bool, ~y_pred_bool).sum()
    fp = np.logical_and(~y_bool, y_pred_bool).sum()
    fn = np.logical_and(y_bool, ~y_pred_bool).sum()

    out = {}
    if any(pd.isna(y_pred)):
        return dict(
            [
                ("accuracy", float("nan")),
                ("balanced", float("nan")),
                ("tp_rate", float("nan")),
                ("fp_rate", float("nan")),
                ("auc", float("nan")),
                ("tp", float("nan")),
                ("tn", float("nan")),
                ("fp", float("nan")),
                ("fn", float("nan")),
            ]
        )

    out["tp"] = tp
    out["tn"] = tn
    out["fp"] = fp
    out["fn"] = fn
    out["accuracy"] = float(tp + tn) / float(n)
    if tp + fn == 0:
        out["tp_rate"] = float("nan")
    else:
        out["tp_rate"] = float(tp) / float(tp + fn)

    if tn + fp == 0:
        out["fp_rate"] = float("nan")
    else:
        out["fp_rate"] = 1 - float(tn) / float(tn + fp)

    out["balanced"] = (out["tp_rate"] + (1 - out["fp_rate"])) / 2.0

    if tp + fn > 0 & tn + fp > 0:
        out["auc"] = roc_auc_score(y_bool, y_pred)
    else:
        out["auc"] = float("nan")
    return out


def remove_file(file_path):
    """
    removes file from hard drive
    """

    if os.path.exists(file_path):
        os.remove(file_path)
