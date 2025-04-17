import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution


# Ackley function
def ackley(X):
    x, y = X
    a, b, c = 20, 0.2, 2 * np.pi
    sum_sq_term = -a * np.exp(-b * np.sqrt((x**2 + y**2) / 2))
    cos_term = -np.exp((np.cos(c * x) + np.cos(c * y)) / 2)
    return sum_sq_term + cos_term + a + np.exp(1)


# Bounds of search space
bounds = [(-5, 5), (-5, 5)]

# Track candidate solutions from generations
sampled_points = {"initial": [], "midway": [], "final": []}

# Wrap the objective to capture evaluations
evaluated = []


def wrapped_ackley(x):
    evaluated.append(x)
    return ackley(x)


# Run DE
result = differential_evolution(
    wrapped_ackley,
    bounds,
    maxiter=40,
    popsize=15,
    seed=42,
    polish=False,
    updating="deferred",
)

# Split evaluated population history into snapshots
n_total = len(evaluated)
sampled_points["initial"] = np.array(evaluated[:30])
sampled_points["midway"] = np.array(evaluated[n_total // 2 : n_total // 2 + 30])  # noqa
sampled_points["final"] = np.array(evaluated[-30:])

# Create surface plot of Ackley function
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

x = np.linspace(-5, 5, 200)
y = np.linspace(-5, 5, 200)
X, Y = np.meshgrid(x, y)
Z = ackley([X, Y])

# Surface
ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.7, edgecolor="none")

# Overlay sampled populations
colors = ["red", "orange", "blue"]
labels = ["Initial", "Midway", "Final"]

for i, stage in enumerate(["initial", "midway", "final"]):
    pts = sampled_points[stage]
    zs = np.array([ackley(p) for p in pts])
    ax.scatter(pts[:, 0], pts[:, 1], zs, color=colors[i], label=labels[i], s=40)

ax.set_title("Differential Evolution on Ackley Function")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("f(x, y)")
ax.legend()
plt.tight_layout()
plt.show()
