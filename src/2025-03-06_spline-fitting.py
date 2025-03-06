import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_lsq_spline

# Generate synthetic data
np.random.seed(42)
X = np.sort(np.random.uniform(0, 10, 30))  # Predictor variable
y = np.sin(X) + np.random.normal(0, 0.2, size=len(X))  # Response with noise

# Define knots (excluding boundary knots)
num_knots = 4  # Adjust as needed
knots = np.linspace(X.min(), X.max(), num_knots + 2)[1:-1]  # Interior knots only

# Ensure the full knot vector includes boundaries
t = np.concatenate(([X.min()] * 4, knots, [X.max()] * 4))  # Corrected knot vector

# Fit cubic spline
spline = make_lsq_spline(X, y, t, k=3)

# Predict values, including extrapolation
X_new = np.linspace(-5, 15, 200)  # Includes points beyond observed range
y_pred = spline(X_new)

# Enforce linearity beyond observed range
X_left, X_right = X.min(), X.max()
slope_left, slope_right = spline.derivative(1)(X_left), spline.derivative(1)(X_right)

# Apply linear extension beyond the observed range
y_pred[X_new < X_left] = spline(X_left) + slope_left * (X_new[X_new < X_left] - X_left)
y_pred[X_new > X_right] = spline(X_right) + slope_right * (X_new[X_new > X_right] - X_right)

# Plot results
plt.scatter(X, y, label="Observed data", color="black")
plt.plot(X_new, y_pred, label="Cubic Spline with Linear Extrapolation", color="blue")
plt.axvline(X.min(), linestyle="--", color="gray", label="Extrapolation starts")
plt.axvline(X.max(), linestyle="--", color="gray")
plt.legend()
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Cubic Spline Regression with Linear Extrapolation")
plt.show()
