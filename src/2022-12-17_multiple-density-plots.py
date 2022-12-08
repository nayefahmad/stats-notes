# # Displaying multiple overlaid density plots

# ## Overview

# This is a simple example showing how overlaying multiple density plots can be used
# to visualize sampling variability when drawing from a single distribution.

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from plotnine import aes, geom_density, ggplot
from scipy.stats import weibull_min

num_samples = 50
num_iterations = 20
shape = 1.1

data = {}
for idx in range(num_iterations):
    iter_num = f"0{idx + 1}" if idx < 9 else idx + 1
    data[f"iter_{iter_num}"] = weibull_min.rvs(c=shape, size=num_samples)

df = pd.DataFrame(data)
df["id"] = range(num_samples)
df = df.melt(id_vars=["id"], value_vars=[col for col in data.keys()])
df.head()

# ## Plotting with seaborn

fig = sns.displot(df, x="value", hue="variable", kind="kde", color="blue")
plt.show()


# ## Plotting with plotnine

(ggplot(df, aes(x="value", group="variable")) + geom_density(color="blue"))
