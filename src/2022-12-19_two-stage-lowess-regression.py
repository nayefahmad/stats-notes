# # Two-stage regression to incorporate predictors into lowess

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

# ## References:

# - [statsmodels docs on lowess](https://www.statsmodels.org/dev/generated/statsmodels.nonparametric.smoothers_lowess.lowess.html)  # noqa
# - [sklearn source for shuffling and splitting a dataset](https://github.com/scikit-learn/scikit-learn/blob/dc580a8ef5ee2a8aea80498388690e2213118efd/sklearn/model_selection/_split.py#L1721)  # noqa


import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression


@dataclass
class SimParams:
    num_rows: int = 100
    prop_zero_y: float = 0.35
    b_0: float = 50
    b_1: float = 0.4
    sigma: float = 10
    num_zero_y: float = math.ceil(prop_zero_y * num_rows)


p = SimParams()


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

# Set up full X matrix, including indicator for zero_y outlier events

x2 = np.zeros(p.num_rows)
x2[idx_for_zero_y] = 1
X = np.stack((x1, x2), axis=1)
assert X.shape == (100, 2)


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
    ax.set_ylim(0, 80)
    ax.set_title(subtitles[idx])
fig.suptitle(title)
fig.tight_layout(pad=2)
fig.show()


# ## Single-stage Lowess on y

lowess = sm.nonparametric.lowess
y_lowess = lowess(y, x1, frac=0.2)
assert y_lowess.shape == (100, 2)


title = "Single-stage lowess: the zero-y outliers tend to bias the trend downward"
fig, ax = plt.subplots()
ax.plot(x1, y, "o", mfc="none", label="y")
ax.plot(y_lowess[:, 0], y_lowess[:, 1], "o", mfc="none", label="y_lowess_trend")
ax.set_title(title)
ax.legend()
fig.show()


# ## Two-stage Lowess

# First stage model: y on categorical indicator for outlier

lm = LinearRegression()
lm.fit(X[:, 1].reshape(-1, 1), y)
effect_of_outlier = lm.coef_
predictions = lm.predict(X[:, 1].reshape(-1, 1))
residuals = y - predictions

df_display = pd.DataFrame(np.stack((x1, x2, predictions, residuals), axis=1))
df_display.columns = ["x1", "x2", "predictions", "residuals"]
df_display.head(10)


title = "Residuals from first-stage regression model"
fig, ax = plt.subplots()
ax.plot(x1, residuals, "o", mfc="none")
ax.set_title(title)
ax.legend()
fig.show()


# Lowess on residuals

resid_lowess = lowess(residuals, x1, frac=0.2)
assert resid_lowess.shape == (100, 2)


title = "Two-stage lowess: ___"
fig, ax = plt.subplots()
ax.plot(x1, residuals, "o", mfc="none", label="residuals_from_model_01")
ax.plot(
    resid_lowess[:, 0], resid_lowess[:, 1], "o", mfc="none", label="resid_lowess_trend"
)
ax.set_title(title)
ax.legend()
fig.show()


LinearRegression().fit(X[:, 0].reshape(-1, 1), residuals).coef_
LinearRegression().fit(X[:, 0].reshape(-1, 1), y).coef_
