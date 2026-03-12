import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    explained_variance_score,
    median_absolute_error
)

from scipy import stats

sns.set_style("whitegrid")


# --------------------------------------------------
# GRID UTILITY
# --------------------------------------------------

def compute_grid(n_plots, base_rows, base_cols):

    capacity = base_rows * base_cols

    if n_plots <= capacity:
        return base_rows, base_cols

    n = math.ceil(math.sqrt(n_plots))
    return n, n


# --------------------------------------------------
# PREDICTED VS TRUE
# --------------------------------------------------

def plot_pred_vs_true(ax, y_true, y_pred):

    sns.scatterplot(x=y_true, y=y_pred, ax=ax)

    min_v = min(np.min(y_true), np.min(y_pred))
    max_v = max(np.max(y_true), np.max(y_pred))

    ax.plot([min_v, max_v], [min_v, max_v], "--", color="gray")

    ax.set_xlabel("True Values")
    ax.set_ylabel("Predicted Values")

    ax.set_title("Predicted vs True")


# --------------------------------------------------
# RESIDUALS VS PREDICTION
# --------------------------------------------------

def plot_residuals(ax, y_pred, residuals):

    sns.scatterplot(x=y_pred, y=residuals, ax=ax)

    ax.axhline(0, linestyle="--", color="gray")

    ax.set_xlabel("Predicted Value")
    ax.set_ylabel("Residual")

    ax.set_title("Residuals vs Predictions")


# --------------------------------------------------
# RESIDUAL DISTRIBUTION
# --------------------------------------------------

def plot_residual_distribution(ax, residuals):

    sns.histplot(residuals, bins=30, kde=True, ax=ax)

    ax.set_xlabel("Residual")
    ax.set_ylabel("Density")

    ax.set_title("Residual Distribution")


# --------------------------------------------------
# QQ PLOT
# --------------------------------------------------

def plot_qq(ax, residuals):

    stats.probplot(residuals, dist="norm", plot=ax)

    ax.set_title("Q-Q Plot (Residual Normality)")


# --------------------------------------------------
# ERROR VS TRUE
# --------------------------------------------------

def plot_error_vs_true(ax, y_true, residuals):

    sns.scatterplot(x=y_true, y=residuals, ax=ax)

    ax.axhline(0, linestyle="--", color="gray")

    ax.set_xlabel("True Value")
    ax.set_ylabel("Residual")

    ax.set_title("Residuals vs True Values")


# --------------------------------------------------
# ABSOLUTE ERROR DISTRIBUTION
# --------------------------------------------------

def plot_abs_error_distribution(ax, abs_error):

    sns.histplot(abs_error, bins=30, kde=True, ax=ax)

    ax.set_xlabel("Absolute Error")
    ax.set_ylabel("Density")

    ax.set_title("Absolute Error Distribution")


# --------------------------------------------------
# CUMULATIVE ERROR CURVE
# --------------------------------------------------

def plot_cumulative_error(ax, abs_error):

    sorted_err = np.sort(abs_error)

    perc = np.arange(1, len(abs_error)+1) / len(abs_error)

    ax.plot(sorted_err, perc)

    ax.set_xlabel("Absolute Error")
    ax.set_ylabel("Fraction of Samples")

    ax.set_title("Cumulative Error Curve")


# --------------------------------------------------
# MAIN FUNCTION
# --------------------------------------------------

def regression_report_with_plots(
        y_true,
        y_pred,
        grid=(2,2)
):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    residuals = y_true - y_pred
    abs_error = np.abs(residuals)

    # --------------------------------------------------
    # METRICS
    # --------------------------------------------------

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    metrics = {

        "mae": mean_absolute_error(y_true, y_pred),

        "mse": mse,

        "rmse": rmse,

        "r2": r2_score(y_true, y_pred),

        "explained_variance": explained_variance_score(y_true, y_pred),

        "median_absolute_error": median_absolute_error(y_true, y_pred)
    }

    # --------------------------------------------------
    # PLOTS
    # --------------------------------------------------

    plots = []

    plots.append(("pred_true", (y_true, y_pred)))
    plots.append(("residuals", (y_pred, residuals)))
    plots.append(("res_dist", residuals))
    plots.append(("qq", residuals))
    plots.append(("err_true", (y_true, residuals)))
    plots.append(("abs_err", abs_error))
    plots.append(("cum_err", abs_error))

    # --------------------------------------------------
    # SUBPLOTS
    # --------------------------------------------------

    rows, cols = compute_grid(len(plots), grid[0], grid[1])

    fig, axes = plt.subplots(rows, cols, figsize=(5*cols,5*rows))
    axes = axes.flatten()

    for i,(ptype,data) in enumerate(plots):

        ax = axes[i]

        if ptype=="pred_true":
            plot_pred_vs_true(ax,*data)

        elif ptype=="residuals":
            plot_residuals(ax,*data)

        elif ptype=="res_dist":
            plot_residual_distribution(ax,data)

        elif ptype=="qq":
            plot_qq(ax,data)

        elif ptype=="err_true":
            plot_error_vs_true(ax,*data)

        elif ptype=="abs_err":
            plot_abs_error_distribution(ax,data)

        elif ptype=="cum_err":
            plot_cumulative_error(ax,data)

    for j in range(len(plots),len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

    return pd.DataFrame(
        list(metrics.items()),
        columns=["metric","value"]
    )


# --------------------------------------------------
# EXAMPLE
# --------------------------------------------------

if __name__ == "__main__":

    y_true = [3, -0.5, 2, 7]
    y_pred = [2.5, 0.0, 2, 8]

    report = regression_report_with_plots(
        y_true,
        y_pred,
        grid=(2,2)
    )

    print(report)