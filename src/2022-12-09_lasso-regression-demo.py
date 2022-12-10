# # Lasso regression and decision trees as tools for EDA
# 2022-12-09

# ## Overview

# Data: Ten baseline variables (age, sex, body mass index, average
# blood pressure, and six blood serum measurements) were obtained for each of
# n=n442 diabetes patients.

# Each of these 10 feature variables have been mean centered and scaled by the
# standard deviation times the square root of n_samples (i.e. the sum of squares of
# each column totals 1)

# ## References

# - [scikit-learn doc on diabetes dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset)  # noqa


# ## Libraries

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_diabetes

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
# - Goal: show that regression allows us to adjust for x_02 and give a better estimate
#   of the intercept.

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
