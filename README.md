# ExplainMLForecasting
This library demonstrates (1) how to create financial crisis forecasting models and (2) how to explain predictions leveraing explainable AI (XAI) based on the forecast models in Bluwstein, Kristina, et al. (2023) and analyze the results for Korea's case.
This library is developed on top of [MachineLearningCrisisPrediction](https://github.com/bank-of-england/MachineLearningCrisisPrediction) package.

Specifically, this package includes
* Financial crisis forecasting models using machine learning techniques on the macrofinancial data of 19 countries over 1870-2020 time period
* Identifying economic drivers of the machine learning models using a novel framework based on Shapley values to uncover nonlinear relationships between the predictors and the risk of crisis
* Korea dataset added and preprocessed to be used with Jordà-Schularick-Taylor Macrohistory Database

Based on the analysis, machine learning models typically outperform logistic regression in out-of-sample prediction and forecasting experiments. Across all models the most important predictors are credit growth and the slope of the yield curve, both domestically and globally, which is the same conclusion as in Bluwstein, Kristina, et al. (2023).


## Prerequisites
- The code has been developed and used under `Python` 3.10.1

## Datasets
![dataset2](https://github.com/user-attachments/assets/4a0171e8-86fa-4dc4-8072-628bc9312c6c)

For the dataset, [Jordà-Schularick-Taylor Macrohistory Database](https://www.macrohistory.net/database/) is used. It is published under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. I accessed Version 6 from the dataset's website. This version is contained in the `data` folder of this repository.

In addition, I add 'KOREA' data by using [ECOS api](https://ecos.bok.or.kr/api/#/) which supports getting Korea economic datasets. I applied preprocessing on the dataset to change the format of the dataset based on the Jordà-Schularick-Taylor Macrohistory Database and the processed dataset is also attached in the `data` folder.

## Prediction Models
I trained and evaluated 8 models* and ROC curve of 6 models are shown below. Extra Tree model shows that financial crisis prediction for Korea in 2020 is not high. More models are available in `ml_funcions.py` and it has flexibility so users can add models by themselves.

*Models : Logistic Regression, Random Forest, ExtraTrees, Xgboost, LightGBM, Support Vector Machine, K-Nearest Neighbors, Neural Network

![ROC](https://github.com/user-attachments/assets/ce87e1dc-ded5-418b-9aee-784aa9c7ed5a)

## Interpreting the Predictions

Shapley value is used to interpret the predition models results.

![SHAP2](https://github.com/user-attachments/assets/29d5936e-5f6e-480c-844f-ef3036d2dde2)

And the diagram shows that while global interest rate differentials, global credit conditions, and domestic credit variables acted to increase the probability of a crisis, DSR (Debt Service Ratio), public debt, consumer prices, and housing indices acted to decrease the probability of a crisis.

In the December 2020 Financial Stability Report, the Bank of Korea (BOK) assessed that it is necessary to pay attention to the accelerating growth of private credit. This can be interpreted as the domestic credit variable, which showed the greatest contribution to crisis prediction in the optimal model, accurately reflecting the economic situation.

Furthermore, a comparison between the BOK's Financial Stability Index (FSI) and the results of the optimal model revealed similar trends, as shown in the below graph. 

![FSI](https://github.com/user-attachments/assets/739c11ab-9ce0-4cf0-9959-90137f765a09)

However, the optimal model demonstrated superior predictive power by anticipating the 2007 crisis earlier and maintaining low volatility after 2010.

## Getting Started
To get started with the `ExplainMLForecasting` package, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/ExplainMLForecasting.git
    cd ExplainMLForecasting
    ```

2. Install the package:
    ```sh
    python setup.py install
    ```

3. Run the example script:
    ```sh
    python example/example.py
    ```
## Reference
Bluwstein, Kristina, et al. "Credit growth, the yield curve and financial crisis prediction: Evidence from a machine learning approach." Journal of International Economics 145 (2023): 103773. [Github](https://github.com/bank-of-england/MachineLearningCrisisPrediction)