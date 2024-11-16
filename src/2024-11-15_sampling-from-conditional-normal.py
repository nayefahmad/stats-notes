"""
This shows that in the presence of a trend, we have to model the trend in order to
simulate future data points correctly.

Compare this with an approach where we just take cdf of training data and predict
new data points (outside the training range) using that.
"""

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as sm
from scipy.stats import norm

# 1. Simulate data
np.random.seed(2024)
max_x_in_training_data = 10
n = 100
x = np.random.rand(n) * max_x_in_training_data
error = np.random.randn(n)
y = 2 * x + 1 + error

# 2. Fit regression model
data = {"x": x, "y": y}
model = sm.ols("y ~ x", data=data).fit()

# 3. Simulate data points outside training range
x_0 = max_x_in_training_data + np.abs(np.random.randn(10) * max_x_in_training_data)
y_pred = model.predict({"x": x_0})
resid_std_error = np.sqrt(model.scale)
simulated_y = norm.rvs(loc=y_pred, scale=resid_std_error)

# 4. Plot the data
plt.figure(figsize=(8, 6))

# Scatter plot of original data
plt.scatter(x, y, label="Original Data", alpha=0.7)

# Scatter plot of simulated data
plt.scatter(x_0, simulated_y, label="Simulated Data", color="red", marker="x", s=100)

# Plot the regression line
x_line = np.linspace(min(x), max(x_0), 100)
y_line = model.predict({"x": x_line})
plt.plot(x_line, y_line, label="Regression Line", color="green")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Original and Simulated Data with Regression Line")
plt.legend()
plt.grid(True)
plt.show()
