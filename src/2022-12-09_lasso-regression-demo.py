# # Regularized regression and random forests as tools for EDA
# 2022-12-09

# ## Overview

# Data: Ten baseline variables (age, sex, body mass index, average
# blood pressure, and six blood serum measurements) were obtained for each of
# n=n442 diabetes patients.

# Each of these 10 feature variables have been mean centered and scaled by the
# standard deviation times the square root of n_samples (i.e. the sum of squares of
# each column totals 1). This kind of standardization is required before using Lasso
# and ridge regression (and other regularization-based methods), and in general is a
# good idea when we want to determine predictor importance. [2][3][4]

# Choosing between ridge, lasso, and elastic net: ridge is a good default unless you
# suspect that only a few features are useful. Elastic net is preferred over lasso
# because lasso may behave erratically when p > n or several features are strongly
# correlated. [5]

# ## References

# 1. [scikit-learn doc on diabetes dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset)  # noqa
#
# 2. [When and Why to Standardize Your Data - Builtin.com](https://builtin.com/data-science/when-and-why-standardize-your-data)  # noqa
#
# 3. [When and why to standardize a variable - Listendata.com](https://www.listendata.com/2017/04/how-to-standardize-variable-in-regression.html)  # noqa
#
# 4. HOML book, p136
#
# 5. HOML book, p140
#


# ## Libraries
from typing import Type

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn.linear_model._base
from sklearn.datasets import load_diabetes
from sklearn.linear_model import ElasticNetCV, Lasso, LassoCV, LinearRegression, RidgeCV
from sklearn.model_selection import cross_val_score

# ## Data setup
diabetes = load_diabetes()

df_diabetes = pd.concat(
    [pd.DataFrame(diabetes["data"]), pd.DataFrame(diabetes["target"])], axis=1
)
num_x_cols = 10
x_col_names = [f"x_0{num}" for num in range(1, 10)] + [
    f"x_{num}" for num in range(10, 11)
]
df_diabetes.columns = x_col_names + ["y"]
df_diabetes.reset_index(drop=False, inplace=True)

df_diabetes.head()
df_diabetes.shape
df_diabetes.describe()
df_diabetes.info()

# ## Plots
df_melted = df_diabetes.melt(id_vars=["index"], value_vars=[col for col in df_diabetes])

reprint_plots = True
if reprint_plots:
    title = "Distribution of predictor variables"
    sns.displot(
        df_melted.query('variable != "y"'),
        x="value",
        hue="variable",
        kind="kde",
    )
    plt.title(title)
    plt.show()

    assert df_diabetes["x_02"].nunique() == 2

    title = "Boxplot of y vs. categorical variable, x_02"
    sns.boxplot(df_diabetes[["x_02", "y"]], x="x_02", y="y")
    plt.title(title)
    plt.show()

    df_diabetes.groupby("x_02")["y"].mean()

    title = "Distribution of response variable"
    sns.displot(
        df_melted.query('variable == "y"'),
        x="value",
        hue="variable",
        kind="kde",
    )
    plt.title(title)
    plt.show()

    title = "Response vs each predictor"
    fig = plt.figure(figsize=(9, 9))
    for idx, x_var in enumerate(df_diabetes[x_col_names]):
        ax = plt.subplot(5, 2, idx + 1)
        ax.plot(df_diabetes[x_var], df_diabetes["y"], "o", mfc="none")
        ax.set_ylabel("y")
        ax.set_xlabel(x_var)
        ax.grid(True)
    plt.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.show()

    # ## Analysis focusing on y, x_07, and x_02

    # - Assume that we want to interpret the coefficient of x_07
    # - Goal: show that regression allows us to adjust for x_02 and give a better
    #   estimate of the coefficient.

    x_var = "x_07"
    title = f"Response vs {x_var}, by levels of x_02"
    ax = sns.scatterplot(
        df_diabetes[["x_02", x_var, "y"]],
        x=x_var,
        y="y",
        hue="x_02",
        marker="$\circ$",  # noqa
        ec="face",
    )
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(bbox_to_anchor=(1, 0.5))
    ax.set_title(title)
    plt.show()


# ### Models

model_lm = LinearRegression()
model_ridge = RidgeCV(cv=5)
model_lasso = LassoCV(cv=5)
model_elastic_net = ElasticNetCV(cv=5)

models = {
    1: model_lm,
    2: model_ridge,
    3: model_lasso,
    4: model_elastic_net,
}
predictor_sets = ["all", "x_07", "x_07_x_02"]


def identify_predictors(predictor_str: str, df: pd.DataFrame) -> pd.DataFrame:
    if predictor_str == "all":
        return df.drop(columns=["index", "y"])
    elif predictor_str == "x_07":
        return df[["x_07"]]
    elif predictor_str == "x_07_x_02":
        return df[["x_07", "x_02"]]


def model_params_and_hyperparams(
    model: Type[sklearn.linear_model._base.LinearModel], df_X: pd.DataFrame
):
    if type(model) == LinearRegression:
        model.alpha_ = None
    alpha = model.alpha_

    coefs = model.coef_.squeeze().tolist()
    coefs = [coefs] if type(coefs) == float else coefs
    var_names = df_X.columns.to_list()
    coefs_named = [coef_name for coef_name in zip(var_names, coefs)]
    coefs_named = pd.DataFrame(coefs_named).rename(columns={0: "variable", 1: "coeff"})

    intercept = model.intercept_

    return alpha, coefs_named, intercept


results = {}
for model_id, model in models.items():
    for predictors in predictor_sets:
        df_X = identify_predictors(predictors, df_diabetes)
        X = df_X.to_numpy()
        y = df_diabetes[["y"]].to_numpy()

        model.fit(X, y)
        score = model.score(X, y)
        alpha, coefs_named, intercept = model_params_and_hyperparams(model, df_X)

        results[f"{model_id}-{model} with predictors: {predictors}"] = {
            "alpha": alpha,
            "score": score,
            "coeffs": coefs_named,
            "intercept": intercept,
        }

print(results)


# ## Tests

# Is the .score() method of e.g. `LassoCV` giving us a cross-validated r-squared?
# [See this](https://stats.stackexchange.com/questions/350484/why-is-r-squared-not-a-good-measure-for-regressions-fit-using-lasso)  # noqa

# Also see ISLR, p243.


def test_cross_validated_r_squared():
    assert type(models[3]) == sklearn.linear_model._coordinate_descent.LassoCV

    model = Lasso(alpha=models[3].alpha_)
    X = df_diabetes[["x_07", "x_02"]].to_numpy()  # todo: use iris data instead
    y = df_diabetes[["y"]].to_numpy()
    scores = cross_val_score(model, X, y, cv=5, scoring="r2")

    try:
        assert scores.mean() == models[3].score(X, y)
    except AssertionError as e:
        print(f"AssertionError: {e}")
        print(f"Difference in score = {scores.mean() - models[3].score(X, y)}")

    print("done")


test_cross_validated_r_squared()
