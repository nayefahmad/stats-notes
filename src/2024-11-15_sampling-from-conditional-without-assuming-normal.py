"""
This shows that in the presence of a trend, we have to model the trend in order to
simulate future data points correctly.

The added complication here is that we can't assume the conditional distribution is
normal. As an example, the simulated data here is conditionally Weibull-distributed.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as sm
from scipy.stats import norm, weibull_min
from sklearn.neighbors import KernelDensity

# 1. Simulate data with Weibull-distributed error
np.random.seed(2021)
max_x_in_training_data = 10
n = 100
x = np.random.rand(n) * max_x_in_training_data
shape_param = 0.7  # Shape parameter of the Weibull distribution
scale_param = 10  # Scale parameter of the Weibull distribution

# Generate Weibull-distributed errors
error = weibull_min(c=shape_param, scale=scale_param, loc=0).rvs(size=n)
y = 2 * x + error  # Generate dependent variable

# 2. Fit regression model
data = {"x": x, "y": y}
model = sm.ols("y ~ x", data=data).fit()

# 3. Nonparametric conditional distribution using KDE
x_0 = max_x_in_training_data + np.abs(np.random.randn(n) * max_x_in_training_data)
y_pred = model.predict({"x": x_0})

# # Assuming conditional distribution is normal :
resid_std_error = np.sqrt(model.scale)
simulated_y_normal = norm.rvs(loc=y_pred, scale=resid_std_error)

# # Accounting for non-normal data:
# Calculate residuals on training data
residuals = np.array(y - model.predict(data))

# Fit KDE to residuals
kde = KernelDensity(kernel="gaussian", bandwidth=1.0)
kde.fit(residuals.reshape(-1, 1))  # KDE requires 2D input

# Sample from KDE
residual_samples = kde.sample(n_samples=1000).flatten()

replot = True
if replot:
    txt = "Distribution of points sampled from model residuals"
    plt.hist(residual_samples)
    plt.title(txt)
    plt.show()

# Generate simulated y-values for x_0
simulated_y = [pred + np.random.choice(residual_samples) for pred in y_pred]

# 4. Plot
plt.figure(figsize=(8, 6))
plt.scatter(x, y, label="Original Data", alpha=0.7)
plt.scatter(
    x_0,
    simulated_y,
    label="Simulated Data (Nonparametric KDE)",
    color="red",
    marker="x",
    s=100,
)
plt.scatter(
    x_0,
    simulated_y_normal,
    label="Simulated Data (Assuming Gaussian)",
    facecolors="none",
    edgecolors="grey",
    marker="o",
    s=100,
)
x_line = np.linspace(min(x), max(x_0), 100)
y_line = model.predict({"x": x_line})
plt.plot(x_line, y_line, label="Regression Line", color="green")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Original and Simulated Data with Regression Line (Weibull Residuals)")
plt.legend()
plt.grid(True)

save_fig = False
if save_fig:
    plt.savefig(Path(__file__).parents[1] / "dst" / "images" / "2024-11-15.png")
else:
    plt.show()
