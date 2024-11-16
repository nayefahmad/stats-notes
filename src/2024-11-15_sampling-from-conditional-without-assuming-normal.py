"""
This shows that in the presence of a trend, we have to model the trend in order to
simulate future data points correctly.

The added complication here is that we can't assume the conditional distribution is
normal.
"""

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as sm
from sklearn.neighbors import KernelDensity

# 1. Simulate data with Weibull-distributed error
np.random.seed(2024)
max_x_in_training_data = 10
n = 100
x = np.random.rand(n) * max_x_in_training_data
shape_param = 0.7  # Shape parameter of the Weibull distribution
scale_param = 10  # Scale parameter of the Weibull distribution
error = (
    np.random.weibull(shape_param, n) * scale_param
)  # Generate Weibull-distributed errors
y = 2 * x + 1 + error  # Generate dependent variable

# 2. Fit regression model
data = {"x": x, "y": y}
model = sm.ols("y ~ x", data=data).fit()

# 3. Nonparametric conditional distribution using KDE
x_0 = max_x_in_training_data + np.abs(np.random.randn(100) * max_x_in_training_data)
y_pred = model.predict({"x": x_0})

# Calculate residuals
residuals = np.array(y - model.predict(data))

# Fit KDE to residuals
kde = KernelDensity(kernel="gaussian", bandwidth=1.0)
kde.fit(residuals.reshape(-1, 1))  # KDE requires 2D input

# Sample from KDE once
residual_samples = kde.sample(n_samples=1000).flatten()

# Generate simulated y-values for x_0
simulated_y = [pred + np.random.choice(residual_samples) for pred in y_pred]

# 4. Plot
plt.figure(figsize=(8, 6))
plt.scatter(x, y, label="Original Data", alpha=0.7)
plt.scatter(
    x_0,
    simulated_y,
    label="Simulated Data (Nonparametric)",
    color="red",
    marker="x",
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
plt.show()
