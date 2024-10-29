"""
Module defines the `Config` class, which creates a configuration object specifying 
how data is processed and how experiments are run. The default values assigned here 
can be altered by the user in the experiment files.

Classes:
    Config: A class to hold configuration parameters for data processing and experimental setup.
"""


class Config:
    """
    A configuration class for setting up data processing
    and experimental parameters for forecasting models.

    Attributes:
        data_predictors (list): Names of the indicators used as predictors.
        data_horizon (int): Horizon of percentage and ratio changes (in years).
        data_period (str): The time frame to investigate.
                           Either 'all' observations, or 'pre-ww2' or 'post-ww2'.
        data_exclude_extreme_period (bool): Whether to exclude WW1, WW2, and the Great Depression.
        data_include_crisis_year (bool): Whether to exclude the actual crisis observation
                                         and only predict years ahead of a crisis.
        data_years_pre_crisis (int): Number of years before a crisis
                                     for which the outcome is set positive.
        data_post_crisis (int): How many observations (in years) after the crisis should be
                                deleted to avoid post-crisis bias.

        exp_n_kernels (int): The number of CPU kernels used in parallel.
        exp_nfolds (int): Number of folds in the cross-validation experiment.
        exp_algos (list): List of algorithms that are tested in the experiment.
        exp_year_split (int or None): If 'None', the cross-validation experiment is run.
                                      If a year 'y', all instances up to that year are used for
                                      training and the following observations for testing the model.
        exp_id (str): Specifies constraints for the cross-validation.

        exp_verbose (int): Determines how verbose the output of the hyperparameter search is.
        exp_hyper_folds (int): Number of folds in the cross-validation of the hyperparameters.
        exp_rep_cv (int): How often the cross-validation of the hyperparameters is repeated.
        exp_search (str): Either 'grid' search or 'random' search for hyperparameter optimization.
        exp_n_iter_rsearch (int): How many hyperparameter combinations are tested
                                  in the random search.
        exp_optimization_metric (str): Metric that is optimized in the hyperparameter search.

        exp_do_shapley (bool): Whether Shapley values are computed.
        exp_shap_background (int): Number of background samples used by Shapley Kernel explainer.
        exp_shapley_interaction (bool): Whether interactions of Shapley values are computed.
        exp_error_costs (str or dict): Cost associated with the false positive and
                                       false negative error.
        exp_do_upsample (bool): Whether the minority class is upsampled
                                according to the error costs.
        exp_bootstrap (str): Bootstrapping the training set with options 'no' (no bootstrapping),
                             'up' (upsampling), 'down' (downsampling).
        exp_bootstrap_replace (str): Whether to resample the minority class by replacement as well.
    """

    def __init__(self):

        # tail meaings - pdiff : percentage change
        #                rdiff : ratio change. Change of the variable relative to the change in GDP
        self.data_predictors = [
            "tloan_gdp_rdiff",
            "global_loan",
            "drate",
            "global_drate",
            "cpi_pdiff",
            "bmon_gdp_rdiff",
            "stock_pdiff",
            "cons_pdiff",
            "pdebt_gdp_rdiff",
            "inv_gdp_rdiff",
            "ca_gdp_rdiff",
            "tdbtserv_gdp_rdiff",
        ]

        self.data_horizon = 2
        self.data_period = "all"
        self.data_exclude_extreme_period = True
        self.data_include_crisis_year = False
        self.data_years_pre_crisis = 2
        self.data_post_crisis = 4

        # The following parameters determine experimental details

        self.exp_n_kernels = 1
        self.exp_nfolds = 5
        self.exp_algos = ["extree", "logreg"]
        self.exp_year_split = None

        # This variable specifies constraints for the cross-validation
        # 'no': no constraint used
        # 'crisis': the observation of a crisis (by default 1-2 years before crisis)
        #           are assigned to the same fold
        # 'year': all observations of a certain year are assigned to the same fold
        # 'year_and_crisis' combination of the two constraints above
        self.exp_id = "crisis"

        # Hyperparameter search
        self.exp_verbose = 0
        self.exp_hyper_folds = 5
        self.exp_rep_cv = 1
        self.exp_search = "grid"
        self.exp_n_iter_rsearch = 250
        self.exp_optimization_metric = "roc_auc"

        # Shapley
        self.exp_do_shapley = True
        self.exp_shap_background = 50
        self.exp_shapley_interaction = False

        # cost associated with the false positive and false negative error.
        # '0.5' : both errors are treated as equally important
        # 'balanced' : error of the minority classes is upweighted to the product of the
        #              error-weight and the proportion of objects in the class is equivalent
        # arbitrary values of dict form : e.g. {0: 0.1, 1: 0.9} means error in the positive
        #                                 class are 9 times more important than the error
        #                                 in the negative class
        self.exp_error_costs = "0.5"

        # whether the minority class is upsampled according to the error costs
        # If False, the objects are weighted according to the error costs.
        # Note that the weighting of objects is not supported by all algorithms.
        self.exp_do_upsample = False

        self.exp_bootstrap = "no"
        self.exp_bootstrap_replace = "no"

    def make_name(self, name_appx=""):
        """Creates a descriptive name according to the configuration.
        This name is used when saving the files in the results folder.
        It is based on some of the experiment's parameters, but the user can
        also add a suffix to the name with the `name_appx` argument.

        Args:
            name_appx (str, optional): A suffix to add to the generated name.
            efaults to an empty string.
        Returns:
            str: A descriptive name based on the configuration and provided suffix."""

        name_data = name_appx + str(self.data_horizon) + "_" + str(self.exp_id)

        if self.exp_year_split is None:
            exp_name = "CV"
        else:
            exp_name = "year" + str(int(self.exp_year_split))

        name = self.data_period + "_" + str(exp_name) + "_" + str(name_data)

        if self.data_include_crisis_year:
            name = name + "crsIncl_"

        if self.exp_do_shapley:
            name = name + "SHAP_"

        return name
