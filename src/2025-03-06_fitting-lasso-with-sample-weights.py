import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import PolynomialFeatures

# Generate example data
np.random.seed(42)
X = np.sort(10 * np.random.rand(100, 1), axis=0)  # Random X values in range [0, 10]
y = np.sin(X).ravel() + 0.1 * np.random.randn(100)  # Noisy sine function

# Create polynomial features
degree = 5  # Adjust as needed
poly = PolynomialFeatures(degree, include_bias=False)
X_poly = poly.fit_transform(X)

# Define sample weights that emphasize points near max(X)
X_max = X.max()
weights = np.exp(5 * (X.squeeze() - X_max) / X_max)  # Exponential weighting

# Fit LassoCV with weighted samples
lasso_cv = LassoCV(cv=5, alphas=np.logspace(-4, 2, 100), max_iter=5000)
lasso_cv.fit(X_poly, y, sample_weight=weights)

# Predict and plot
X_test = np.linspace(0, 10, 200).reshape(-1, 1)
X_test_poly = poly.transform(X_test)
y_pred = lasso_cv.predict(X_test_poly)

plt.scatter(X, y, label="Data", color="gray", alpha=0.6)
plt.plot(X_test, y_pred, label="Lasso Fit (Weighted)", color="red")
plt.legend()
plt.xlabel("X")
plt.ylabel("y")
plt.title("Polynomial Fit with LassoCV and Weighted Prioritization")
plt.show()
