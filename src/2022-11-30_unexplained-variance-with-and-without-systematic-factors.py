# # "Unexplained variance" with and without systematic factors

# ## Overview

# Is this difference explainable by chance alone? We think that a hypothesis test on
# its own answers this question. It doesn't. This sets up a decision tree with
# "chance" as an explanatory factor. What are the other branches? Be specific. Name
# variables. Explain whether those variables are measured and how.

# The thing is, in order to know whether effects are explainable by chance alone,
# we need to know how much chance there is. We cannot know this without including all
# important systematic effects - trends, confounders, etc.

# This example shows how different the "unexplained variance" can look when we do or
# don't incorporate all systematic factors. Remember that it is the unexplained variance
# in the sample that is used to infer the amount of variance/chance in any test
# statistic of interest

from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Params:
    num_rows: int
    alpha: float
    beta: List[float]
    sigma: float


p = Params(num_rows=200, alpha=2, beta=[5], sigma=0.5)

x_range = (0, 5)

x = np.linspace(x_range[0], x_range[1], p.num_rows)
underlying_relationship = p.alpha + p.beta * x
y = underlying_relationship + np.random.normal(scale=p.sigma, size=p.num_rows)

fig, ax = plt.subplots()
ax.plot(x, y, "x", label="observerd_data")
ax.plot(x, underlying_relationship, "x", label="underlying_relationship")
ax.set_title("Observed data and underlying relationship")
ax.set_ylim(-3, 40)
ax.legend()
fig.show()
