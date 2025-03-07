import numpy as np
import matplotlib.pyplot as plt

# If not installed, run: pip install pygam
from pygam import LinearGAM, s

# -------------------------------------------------------
# 1) Example Data
#    X must be 2D: shape (n_samples, 1)
#    Y is 1D: shape (n_samples,)
# -------------------------------------------------------
X = np.array([0, 1, 2, 3, 4, 5, 6]).reshape(-1, 1)
Y = np.array([0.0, 1.0, 1.5, 2.2, 3.1, 3.2, 3.25])

# -------------------------------------------------------
# 2) Fit a monotonic increasing spline with pyGAM
#    We can use gridsearch to automatically select the spline's smoothing
# -------------------------------------------------------
gam = LinearGAM(s(0, constraints='monotonic_inc')).gridsearch(X, Y)

# -------------------------------------------------------
# 3) Predict on a fine grid for a smooth curve
# -------------------------------------------------------
X_fit = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
Y_fit = gam.predict(X_fit)

# -------------------------------------------------------
# 4) Plot data vs. monotonic GAM fit
# -------------------------------------------------------
plt.figure(figsize=(6,4))

# Plot raw data
plt.plot(X, Y, 'o', label="Data")

# Plot fitted curve
plt.plot(X_fit, Y_fit, '-', label="Monotonic GAM Fit")

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Monotonic-Increasing Spline Fit using pyGAM")
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------------------------------------
# Optionally, view a summary of the model
# -------------------------------------------------------
gam.summary()
