"""
This script runs an experiment using various algorithms and data predictors.
"""

from ExplainMLForecasting.experiment import Run

if __name__ == "__main__":

    ITERATIONS = 1000

    exp_algos = ["logreg", "dtree", "extree", "xgb", "forest", "nnet_single"]

    data_predictors = [
        "tloan_gdp_rdiff",
        "global_loan",
        "drate",
        "global_drate",
        "cpi_pdiff",
        "bmon_gdp_rdiff",
        "cons_pdiff",
        "pdebt_gdp_rdiff",
        "inv_gdp_rdiff",
        "ca_gdp_rdiff",
        "tdbtserv_gdp_rdiff",
        "hp_gdp_rdiff",
    ]

    run = Run(ITERATIONS, exp_algos, data_predictors)

    run.run_experiment()
    performance_across_mean, performance_across_se = run.evaluate()

    # print the mean performance metrics across iterations for each algorithm
    print(performance_across_mean)
    print(performance_across_se)
