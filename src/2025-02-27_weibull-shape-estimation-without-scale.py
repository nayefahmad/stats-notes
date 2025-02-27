"""
# Estimating weibull shape more accurately by avoiding the scale, shape parametrization

Reference: Meeker, Escobar, Pascual, p210
"""

import matplotlib.pyplot as plt
import numpy as np
from lifelines import KaplanMeierFitter
from reliability.Fitters import Fit_Weibull_2P
from scipy.optimize import minimize
from scipy.stats import weibull_min


def compute_stats(estimates, true_value):
    bias = np.nanmean(estimates - true_value)
    rmse = np.sqrt(np.nanmean((estimates - true_value) ** 2))
    return bias, rmse


def neg_log_lik_tp(params, t, d, p):
    """
    Custom t_p-based negative log-likelihood (with p as the quantile level)
    """
    t_p, beta = params
    if t_p <= 0 or beta <= 0:
        return np.inf
    const = -np.log(1 - p)
    # Log-PDF for uncensored observations
    log_f = (
        np.log(const)
        + np.log(beta)
        + (beta - 1) * np.log(t)
        - beta * np.log(t_p)
        - const * (t / t_p) ** beta
    )
    # Log-Survival for censored observations
    log_S = -const * (t / t_p) ** beta
    ll = np.sum(d * log_f + (1 - d) * log_S)
    return -ll


def fit_tp_method(t, d, p, init=(8.0, 1.5)):
    bounds = [(1e-9, None), (1e-9, None)]
    res = minimize(
        neg_log_lik_tp, init, args=(t, d, p), method="L-BFGS-B", bounds=bounds
    )
    return res.x  # returns [t_p, beta]


if __name__ == "__main__":
    # Simulation settings  # todo: use dataclass
    n_sims = 500  # number of simulation replications
    n = 100  # sample size per replication
    beta_true = 2.0  # true shape parameter
    lambda_true = 10.0  # true scale (for standard Weibull)
    p_val = 0.1  # our chosen quantile for t_p parameterization

    # Containers for estimated beta (shape) from each method:
    beta_est_tp = []
    beta_est_scipy = []
    beta_est_rel = []

    for sim in range(n_sims):
        t_true = weibull_min(c=beta_true, scale=lambda_true, loc=0).rvs(n)
        censor_time = weibull_min(c=beta_true, scale=lambda_true, loc=0).rvs(n)
        t_obs = np.minimum(t_true, censor_time)
        d = (t_true <= censor_time).astype(int)  # 1 if event observed, 0 if censored

        plot_km = False
        if plot_km:
            observed = np.where(d == 1, True, False)
            censored = np.where(d == 0, True, False)
            assert len(t_obs[observed]) + len(t_obs[censored]) == n

            kmf = KaplanMeierFitter()
            kmf.fit(t_obs, observed)

            fig, ax = plt.subplots()
            kmf.plot_survival_function(
                ax=ax, ci_show=False, label="Kaplan-Meier Estimate", linewidth=2
            )

            # Extract survival function values
            survival_times = kmf.survival_function_.index
            survival_probs = kmf.survival_function_["KM_estimate"]

            # Overlay censored points
            censored_times = t_obs[censored]
            censored_probs = [
                kmf.predict(t) for t in censored_times
            ]  # Get survival probability at censored times
            ax.scatter(
                censored_times,
                censored_probs,
                marker="+",
                color="red",
                label="Censored",
                s=100,
            )

            ax.set_xlabel("Time")
            ax.set_ylabel("Survival Probability")
            ax.set_title("Kaplan-Meier Survival Curve with Censored Points")
            ax.legend()
            ax.grid()
            plt.show()

        # --- Method 1: Custom t_p approach ---
        # Use the median of uncensored observations as an initial guess for t_p.
        if np.sum(d) > 0:
            t_p_init = np.median(t_obs[d == 1])
        else:
            t_p_init = np.median(t_obs)
        init_tp = (t_p_init, 1.0)  # initial guess for [t_p, beta]
        try:
            _, beta_tp = fit_tp_method(t_obs, d, p_val, init=init_tp)
        except Exception as e:
            beta_tp = np.nan
            print(f"Error: {e}")
        beta_est_tp.append(beta_tp)

        # --- Method 2: SciPy's weibull_min.fit ---
        try:
            # We fix location to 0 for a 2-parameter fit.
            params = weibull_min.fit(t_obs, floc=0)
            c, loc, scale = params
            beta_scipy = c  # shape parameter returned by SciPy
        except Exception as e:
            beta_scipy = np.nan
            print(f"Error: {e}")
        beta_est_scipy.append(beta_scipy)

        # --- Method 3: Reliability package fit ---
        try:
            fit = Fit_Weibull_2P(failures=t_obs, right_censored=(1 - d).astype(bool))
            beta_rel = fit.beta
        except Exception as e:
            beta_rel = np.nan
            print(f"Error: {e}")
        beta_est_rel.append(beta_rel)

    # Convert to NumPy arrays for analysis:
    beta_est_tp = np.array(beta_est_tp)
    beta_est_scipy = np.array(beta_est_scipy)
    beta_est_rel = np.array(beta_est_rel)

    bias_tp, rmse_tp = compute_stats(beta_est_tp, beta_true)
    bias_scipy, rmse_scipy = compute_stats(beta_est_scipy, beta_true)
    bias_rel, rmse_rel = compute_stats(beta_est_rel, beta_true)

    results = [beta_est_tp, beta_est_scipy, beta_est_rel]
    titles = ["t_p approach", "scipy approach", "reliability approach"]
    fig, ax = plt.subplots(3, 1)
    for idx, values in enumerate(results):
        ax[idx].hist(values, bins=30, edgecolor="black")  # Histogram for each dataset
        ax[idx].set_xlim(0, 3)  # Set x-axis limits
        ax[idx].set_title(titles[idx])
    plt.tight_layout()
    plt.show()

    print("Custom t_p approach: bias = {:.4f}, RMSE = {:.4f}".format(bias_tp, rmse_tp))
    print(
        "SciPy approach:      bias = {:.4f}, RMSE = {:.4f}".format(
            bias_scipy, rmse_scipy
        )
    )
    print("Reliability approach:", end=" ")
    print("bias = {:.4f}, RMSE = {:.4f}".format(bias_rel, rmse_rel))
