# # "Two-stage" regression to incorporate predictors into lowess

# ## Overview

# Adding an indicator for zero-fh-tbes: this is a general way of handling "outliers".
# We tell the model to treat these differently, and not force the overall time trend
# to absorb the effect of these outliers. This way, we sidestep the conversation of
# "Do we include outliers or not?", and use modeling to "automate away" the decision.
#
# A long time ago, we achieved something similar by using a single KM curve to unify
# censored and uncensored data points, instead of asking the user to select which of
# them to plot Similarly, by building a model that includes predictors of interest
# (tail_id, CMD, etc.), we are starting to "automate away" the process that happens
# after a univariate alert is thrown. Ideally, we should not need to follow up an
# alert with a lot of manual investigation - estimating the model gives us the
# results, with the additional benefit of considering interactions.

# todo:
#  - rename file
#  - clean up overview
#  - add interaction between x1 and x2
#  - show that slope of x1 in model m4 is equal to slope of x1 in model:
#      resid(y~x2+x3) ~ resid(x1~x2+x3)
#  - show graph of resid(y~x2+x3) vs resid(x1~x2+x3)
#  - try to fit lowess on resid(y~x2+x3) vs resid(x1~x2+x3). Think of these as
#      y_adjusted and x1_adjusted


# ## References:

# - [statsmodels docs on lowess](https://www.statsmodels.org/dev/generated/statsmodels.nonparametric.smoothers_lowess.lowess.html)  # noqa
# - [sklearn source for shuffling and splitting a dataset](https://github.com/scikit-learn/scikit-learn/blob/dc580a8ef5ee2a8aea80498388690e2213118efd/sklearn/model_selection/_split.py#L1721)  # noqa
# - [CV: how are partial regression slopes calculated in multiple regression?](https://stats.stackexchange.com/questions/77557/how-are-partial-regression-slopes-calculated-in-multiple-regression)  # noqa
#   - also see Agresti, p60.
# - [statsmodels github issue on lowess failing when too few unique values](https://github.com/statsmodels/statsmodels/issues/2449)  # noqa


import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score


@dataclass
class SimParams:
    num_rows: int = 100
    prop_zero_y: float = 0.65
    b_0: float = 50
    b_1: float = 0.4
    sigma: float = 10
    num_zero_y: float = math.ceil(prop_zero_y * num_rows)
    equality_thresh: float = 0.001


@dataclass
class ModelingParams:
    lowess_frac: float = 0.2


p = SimParams()
m = ModelingParams()


# ## Data setup

np.random.seed(2022)
x1 = np.random.uniform(low=0, high=100, size=p.num_rows)

# Set up y

y_no_noise_and_no_zero_values = p.b_0 + p.b_1 * x1
y_no_zero_values = y_no_noise_and_no_zero_values + np.random.normal(
    scale=p.sigma, size=p.num_rows
)
y = np.copy(y_no_zero_values)
idx_for_zero_y = np.random.permutation(np.arange(len(x1)))[: p.num_zero_y]
y[idx_for_zero_y] = 0

# Set up full X matrix, including indicator for zero_y outlier events, and interaction
# between x1 and x2:

x2 = np.zeros(p.num_rows)
x2[idx_for_zero_y] = 1
x3 = x1 * x2

X = np.stack((x1, x2, x3), axis=1)
assert X.shape == (100, 3)

# Putting it all in a df for display:

df_X = pd.DataFrame(X)
df_X.columns = ["x1", "x2", "x3"]
df_X.head(10)


# Plots showing how we built up the data

title = "Building up the example data"
subtitles = [
    "Deterministic relationship",
    "... noise added",
    "... including zero-y outliers",
]
fig = plt.figure(figsize=(10, 7))
for idx, y in enumerate([y_no_noise_and_no_zero_values, y_no_zero_values, y]):
    ax = plt.subplot(1, 3, idx + 1)
    ax.plot(x1, y, "o", mfc="none")
    ax.set_ylim(0, 120)
    ax.set_title(subtitles[idx])
    ax.set_xlabel("x1")
    ax.set_ylabel("y")
fig.suptitle(title)
fig.tight_layout(pad=2)
fig.show()


# ## Single-stage lowess for y~x1

lowess = sm.nonparametric.lowess
y_lowess = lowess(y, x1, frac=m.lowess_frac)
assert y_lowess.shape == (100, 2)


title = "Single-stage lowess: the zero-y outliers tend to bias the trend downward"
fig, ax = plt.subplots()
ax.plot(x1, y, "o", mfc="none", label="y")
ax.plot(
    y_lowess[:, 0],
    y_lowess[:, 1],
    "o",
    mfc="none",
    label="y_lowess_trend",
    color="darkorange",
)
ax.plot(y_lowess[:, 0], y_lowess[:, 1], color="darkorange")
ax.set_title(title)
ax.set_xlabel("x1")
ax.set_ylabel("y")
ax.legend()
fig.show()


# ## "Two-stage" Lowess

# ### First stage model 1: y~x2 (x2 is categorical indicator for outliers).

# The residuals from this model give us "the part of y that is not explained by x2".

X_categorical = X[:, 1].reshape(-1, 1)

lm_y_x2 = LinearRegression(fit_intercept=True)
lm_y_x2.fit(X_categorical, y)

(
    lm_y_x2.intercept_,
    lm_y_x2.coef_,
    lm_y_x2.score(X_categorical, y),
    cross_val_score(lm_y_x2, X_categorical, y, cv=5, scoring="r2").mean(),
)

# Time series cv might be more appropriate when trends are expected
tscv_r2 = []
tscv = TimeSeriesSplit(n_splits=5)
for train, test in tscv.split(X_categorical):
    m = LinearRegression(fit_intercept=True)
    m.fit(X_categorical[train], y[train])
    r2 = m.score(X_categorical[test], y[test])
    tscv_r2.append(r2)
np.array(tscv_r2).mean()


predictions_lm_y_x2 = lm_y_x2.predict(X_categorical)
residuals_lm_y_x2 = y - predictions_lm_y_x2

# Notice how much variability in y is explained by categorical x2:

print(
    f"Prop of variation in y explained by categorical x2: "
    f"{round(lm_y_x2.score(X_categorical, y), 3)}"
)


# ### First stage model 2: x1 ~ x2

# The residuals from this model give us "the part of x1 that is not explained by x2".

df_means = df_X.groupby("x2").mean().reset_index().rename(columns={"x1": "mean_x1"})
means_x1 = df_means["mean_x1"].to_numpy()

title = "Relationship between x1 and x2"
fig, ax = plt.subplots()
ax.plot(x2, x1, "o", mfc="none")
ax.plot(df_means["x2"].unique(), means_x1, label="trend")
ax.set_title(title)
ax.set_xlabel("x2")
ax.set_ylabel("x1")
ax.legend()
fig.show()

lm_x1_x2 = LinearRegression(fit_intercept=True)
lm_x1_x2.fit(X_categorical, x1)

(
    lm_x1_x2.intercept_,
    lm_x1_x2.coef_,
    lm_x1_x2.score(X_categorical, x1),
    cross_val_score(lm_x1_x2, X_categorical, x1, scoring="r2").mean(),
)

# Time series cv might be more appropriate when trends are expected
tscv_r2 = []
tscv = TimeSeriesSplit(n_splits=5)
for train, test in tscv.split(X_categorical):
    m = LinearRegression(fit_intercept=True)
    m.fit(X_categorical[train], x1[train])
    r2 = m.score(X_categorical[test], x1[test])
    tscv_r2.append(r2)
np.array(tscv_r2).mean()


predictions_lm_x1_x2 = lm_x1_x2.predict(X_categorical)
residuals_lm_x1_x2 = x1 - predictions_lm_x1_x2

# Putting it all in a df for display:

df_with_resids = pd.concat(
    [
        df_X,
        pd.Series(y),
        pd.Series(predictions_lm_y_x2),
        pd.Series(residuals_lm_y_x2),
        pd.Series(predictions_lm_x1_x2),
        pd.Series(residuals_lm_x1_x2),
    ],
    axis=1,
)
df_with_resids.columns = df_X.columns.tolist() + [
    "y",
    "predictions_lm_y_x2",
    "residuals_lm_y_x2",
    "predictions_lm_x1_x2",
    "residuals_lm_x1_x2",
]
df_with_resids.head(5)


# Plot the residuals from the two first-stage models, versus predictor x1:

title = "Residuals from y~x2, vs x1"
fig, ax = plt.subplots()
ax.plot(x1, residuals_lm_y_x2, "o", mfc="none")
ax.set_title(title)
ax.set_xlabel("x1")
ax.set_ylabel("residuals_lm_y_x2")
fig.show()

title = "Residuals from x1~x2, vs x2"
fig, ax = plt.subplots()
ax.plot(x2, residuals_lm_x1_x2, "o", mfc="none")
ax.set_title(title)
ax.set_xlabel("x2")
ax.set_ylabel("residuals_lm_x1_x2")
fig.show()


# ## Partial regression plot: residuals_lm_y_x2 vs residuals_lm_x1_x2

title = "Residuals from y~x2, vs residuals from x1~x2"
fig, ax = plt.subplots()
ax.plot(residuals_lm_x1_x2, residuals_lm_y_x2, "o", mfc="none")
ax.set_title(title)
ax.set_xlabel("residuals_lm_x1_x2")
ax.set_ylabel("residuals_lm_y_x2")
fig.show()


# ## Lowess on residuals, focusing on non-outlier points

# residuals_lm_x1_x2_jittered = (
#     residuals_lm_x1_x2 + np.random.normal(scale=100, size=p.num_rows)
# )
# residuals_lm_y_x2_jittered = (
#     residuals_lm_y_x2 + np.random.normal(scale=100, size=p.num_rows)
# )

# resid_lowess = lowess(
#     residuals_lm_x1_x2_jittered, residuals_lm_y_x2, frac=m.lowess_frac
# )
# assert resid_lowess.shape == (p.num_rows, 2)

# todo: explain these

# regress resid(y ~ x2) ~ resid(x1 ~ x2)
m = LinearRegression().fit(residuals_lm_x1_x2.reshape(-1, 1), residuals_lm_y_x2)
y2 = m.predict(residuals_lm_x1_x2.reshape(-1, 1))
m.coef_

# regress y ~ x1 + x2
m2 = LinearRegression().fit(X[:, :2], y)
m2.coef_

# regress y ~ x1
m3 = LinearRegression().fit(X[:, 0].reshape(-1, 1), y)
m3.coef_

assert m.coef_.tolist()[0] - m2.coef_[0] < p.equality_thresh


# regress y ~ x1 + x2 + x3
m4 = LinearRegression().fit(X, y)
predict_m4 = m4.predict(X)
m4.coef_, m4.intercept_

m4.score(X, y)
cross_val_score(m4, X, y, cv=5, scoring="r2").mean()

# Time series cv might be more appropriate when trends are expected
tscv_r2 = []
tscv = TimeSeriesSplit(n_splits=5)
for train, test in tscv.split(X):
    m = LinearRegression(fit_intercept=True)
    m.fit(X[train], y[train])
    r2 = m.score(X[test], y[test])
    tscv_r2.append(r2)
np.array(tscv_r2).mean()

title = r"$\bf{Using \ an \ interaction \ to \ fit \ the \ data \ better}$"
title += "\nThink of this as feature engineering and getting a single model that better"
title += (
    "\nfits the data, instead of asking the user to manually play around and try to "
)
title += "\nfind the best feature representation"
fig, ax = plt.subplots()
ax.plot(x1, y, "o", mfc="none", label="data")
ax.plot(x1, predict_m4, "x", mfc="none", label="fit from model y~x1:x2")
ax.plot(
    y_lowess[:, 0],
    y_lowess[:, 1],
    ".",
    mfc="none",
    label="fit from lowess",
    color="salmon",
)
ax.plot(
    y_lowess[:, 0],
    y_lowess[:, 1],
    color="salmon",
)
ax.set_title(title)
ax.title.set_size(8)
ax.set_xlabel("x1")
ax.set_ylabel("y")
ax.legend()
fig.show()


# todo: update this. show that slope of x1 in model m4 is equal to slope of x1 in
#   model: resid(y~x2+x3) ~ resid(x1~x2+x3)
title = "Second-stage lowess: running lowess on resids of lm(y~x2) vs "
title += "\nresids of lm(x1~x2)"
fig, ax = plt.subplots()
ax.plot(residuals_lm_x1_x2, residuals_lm_y_x2, "o", mfc="none", label="")
ax.plot(residuals_lm_x1_x2, y2, label="lm_fit")
ax.set_title(title)
ax.set_xlabel("residuals_lm_x1_x2")
ax.set_ylabel("residuals_lm_y_x2")
ax.legend()
fig.show()


title = "hi"  # todo:
fig, ax = plt.subplots()
ax.plot(x1, y, "o", mfc="none", label="y")
ax.plot(
    y_lowess[:, 0],
    y_lowess[:, 1],
    "o",
    mfc="none",
    label="y_lowess_trend",
    color="darkorange",
)
ax.plot(y_lowess[:, 0], y_lowess[:, 1], color="darkorange")
ax.plot(x1, m.coef_[0] * x1, color="navy")  # todo: fix this
ax.set_title(title)
ax.set_xlabel("x1")
ax.set_ylabel("y")
ax.legend()
fig.show()
