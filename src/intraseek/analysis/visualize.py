from math import ceil
from typing import Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_dist(data: pd.DataFrame, target: str, title: str, color: str = "r") -> None:
    """Plot distribution charts.

    Description:
        Plot target distribution using box plot and histogram

    Args:
        data (pd.DataFrame): A dataset
        target (str): Target column name
        title (str): A title of the plot

    Returns:
        None
    """
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    data.boxplot(column=[target], ax=ax1, fontsize=14)
    ax1.set_title("box plot", fontsize=14)
    ax1.set_ylabel(target, fontsize=14)

    data[target].plot(
        kind="hist",
        bins=50,
        color=color,
        density=True,
        edgecolor="k",
        linewidth=0.8,
        fontsize=14,
    )
    data[target].plot(kind="kde", color="k", style="--", linewidth=0.8)
    ax2.set_title("histogram", fontsize=14)
    ax2.set_xlabel(target, fontsize=14)
    ax2.set_ylabel("density", fontsize=14)

    plt.suptitle(
        f"{title}\n\n"
        + f"skewness : {data[target].skew():.3f} , "
        + f"kurtosis : {data[target].kurt():.3f}",
        fontsize=16,
        y=1.10,
    )
    plt.show()


def plot_ts(
    data: pd.DataFrame,
    target: str,
    color: str = "r",
    vspan_ranges: Optional[List[Tuple[Any, Any]]] = None,
    line_style: str = "line",  # 라인 스타일 인자 추가
) -> None:
    """Plot univariate time series.

    Description:
        Plot a simple linechart with vspan_ranges

    Args:
        data (pd.DataFrame): A dataset
        target (str): Target column name
        color (str): A color of line
        vspan_ranges (List): Ranges of gray areas
        line_style (str): Line style ('line', 'marker', 'line-marker')  # 라인 스타일 설명 추가

    Returns:
        None
    """

    _, ax = plt.subplots(figsize=(15, 5))
    ax.margins(x=0)

    if line_style == "line":
        data[target].plot(ax=ax, color=color, fontsize=14, alpha=0.9, linewidth=1.5, label=target)
    elif line_style == "marker":
        data[target].plot(
            ax=ax,
            color=color,
            fontsize=14,
            style="o",
            alpha=0.5,
            markersize=6,
            label=target,
        )
    elif line_style == "line-marker":
        data[target].plot(
            ax=ax,
            color=color,
            fontsize=14,
            linestyle="-",
            style="o",
            alpha=0.5,
            markersize=6,
            linewidth=1.5,
            label=target,
        )

    ax.legend(facecolor="white", fontsize=13, frameon=True, edgecolor="black")

    if vspan_ranges:
        for start, end in vspan_ranges:
            ax.axvspan(start, end, color="gray", alpha=0.3)
    plt.show()


def plot_multiple_ts(
    data: pd.DataFrame,
    feature_cols: List[List[str]],
    target_col: str,
    feature_colors: List[List[str]],
    target_color: str = "r",
    line_style: Optional[List[str]] = ["line"],
    vspan_ranges: Optional[List[Tuple[Any, Any]]] = None,
) -> None:
    """Plot multivariate time series

    Description:
        Plot multiple linecharts with vspan_ranges

    Args:
        data (pd.DataFrame): A dataset
        feature_cols (List): Feature columns
        target (str): Target column name
        feature_colors (List): Colormaps for features
        target_color (str): Color for target
        line_style: (List) = plot style('line', 'marker', and 'line-marker'),
        vspan_ranges (List): Ranges of gray areas

    Returns:
        None
    """

    _, axes = plt.subplots(
        len(feature_cols) + 1,
        1,
        figsize=(15, 5 * len(feature_cols) + 3),
        gridspec_kw={"height_ratios": [*[5] * len(feature_cols), 3]},
        sharex=True,
    )

    plt.tight_layout()

    for i, _ in enumerate(feature_cols):  # draw column charts
        axes.ravel()[i].margins(x=0)

        if line_style[i] == "line":
            data[feature_cols[i]].plot(
                ax=axes.ravel()[i],
                color=feature_colors[i],
                fontsize=14,
                alpha=0.9,
                linewidth=1,
            )

        elif line_style[i] == "marker":
            data[feature_cols[i]].plot(
                ax=axes.ravel()[i],
                color=feature_colors[i],
                fontsize=14,
                style="o",
                alpha=0.6,
            )

        elif line_style[i] == "line-marker":
            data[feature_cols[i]].plot(
                ax=axes.ravel()[i],
                color=feature_colors[i],
                fontsize=14,
                linestyle="-",
                style="o",
                alpha=0.6,
                linewidth=1,
            )

        axes.ravel()[i].legend(
            loc="upper right",
            facecolor="white",
            fontsize=13,
            edgecolor="black",
            frameon=True,
        )

    axes.ravel()[len(feature_cols)].margins(x=0)

    data[target_col].plot(  # draw a target chart
        ax=axes.ravel()[len(feature_cols)],
        color=target_color,
        fontsize=14,
        alpha=0.9,
        linewidth=1.2,
        style="--",
    )

    axes.ravel()[len(feature_cols)].set_xlabel("")

    axes.ravel()[len(feature_cols)].legend(
        loc="upper right",
        facecolor="white",
        fontsize=13,
        edgecolor="black",
        frameon=True,
    )

    if vspan_ranges:  # draw gray areas
        for start, end in vspan_ranges:
            for i in range(len(feature_cols) + 1):
                axes.ravel()[i].axvspan(start, end, color="gray", alpha=0.3)

    plt.show()


def corr_heatmap(
    data: pd.DataFrame,
    target: str,
    tick_rot: Tuple[int] = (90, 0, 0, 0),
    title: str = "target",
    method: str = "pearson",
) -> None:
    """Correlation heatmap.

    Description:
        Plot correlation heatmap (among features, between features and target)

    Args:
        data (pd.DataFrame): A dataset
        target (str): Target column name
        tick_rot (Tuple): Tick rotation
        title (str): A title of the heatmap
        method (str): Correlation calculation method

    Returns:
        None
    """

    correlations = data.corr(method=method)

    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    _, axes = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [2, 1]})

    sns.heatmap(
        correlations,
        cmap=cmap,
        vmax=1.0,
        center=0,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        annot=True,
        annot_kws={"size": 10, "alpha": 0.8},
        cbar_kws={"shrink": 0.85},
        ax=axes[0],
    )

    corr = data.corrwith(data[target], method=method).reset_index()
    corr.columns = ["index", "Correlations"]
    corr = corr.set_index("index")
    corr = corr.sort_values(by=["Correlations"], ascending=False)

    sns.heatmap(
        corr,
        annot=True,
        annot_kws={"size": 11, "alpha": 0.8},
        fmt=".2f",
        cmap=cmap,
        cbar_kws={"shrink": 0.99},
        ax=axes[1],
    )

    axes[0].set_title("Heatmap", fontsize=14)
    axes[1].set_title(f"Correlation of Variables with {title}", fontsize=14)

    for i in range(2):
        axes[i].tick_params(axis="x", labelsize=12, rotation=tick_rot[2 * i])
        axes[i].tick_params(axis="y", labelsize=12, rotation=tick_rot[(2 * i) + 1])

    plt.subplots_adjust(wspace=0.5)
    plt.show()


def plot_result(
    result_df: pd.DataFrame,
    target_col: str,
    pred_col: str,
    target_color: Optional[str] = "r",
    pred_color: Optional[str] = "g",
    ax=None,
) -> None:
    """
    Plots the target and prediction columns of a DataFrame.

    Args:
        result_df (pd.DataFrame): The DataFrame containing the data to be plotted.
        target_col (str): The name of the target column in result_df.
        pred_col (str): The name of the prediction column in result_df.
        target_color (str, optional): The color for the target plot. Default is 'r' (red).
        pred_color (str, optional): The color for the prediction plot. Default is 'g' (green).
        ax (matplotlib.axes._axes.Axes, optional): The matplotlib axes object to plot on.
        If not provided, a new figure and axes object will be created.

    Returns:
        None
    """
    if ax == None:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.margins(x=0)

    result_df[target_col].plot(ax=ax, color=target_color, marker="o", alpha=0.7, label=target_col)

    result_df[pred_col].plot(
        ax=ax,
        color=pred_color,
        linewidth=0,
        marker="X",
        markersize=9,
        markeredgecolor="k",
        alpha=0.7,
        label=pred_col,
    )


def plot_evaluations(
    train_result: pd.DataFrame,
    test_result: pd.DataFrame,
    target_col: str,
    pred_col: str,
    target_color: Optional[str] = "r",
    train_pred_color: Optional[str] = "b",
    test_pred_color: Optional[str] = "g",
) -> None:
    """
    Plots the target and prediction columns of train and test DataFrames.

    Args:
        train_result (pd.DataFrame): The training DataFrame containing the data to be plotted.
        test_result (pd.DataFrame): The testing DataFrame containing the data to be plotted.
        target_col (str): The name of the target column in the DataFrames.
        pred_col (str): The name of the prediction column in the DataFrames.
        target_color (str, optional): The color for the target plot. Default is 'r' (red).
        train_pred_color (str, optional): The color for the training prediction plot. Default is 'b' (blue).
        test_pred_color (str, optional): The color for the testing prediction plot. Default is 'g' (green).

    Returns:
        None
    """

    train_result = train_result.set_index("datetime")
    test_result = test_result.set_index("datetime")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.margins(x=0)

    plot_result(train_result, target_col, pred_col, target_color, train_pred_color, ax=ax)

    plot_result(test_result, target_col, pred_col, target_color, test_pred_color, ax=ax)


def plot_residuals(
    result_df: pd.DataFrame,
    target_col: str,
    pred_col: str,
    residual_col: str,
    color: Optional[str] = "b",
) -> None:
    """
    Plots the target vs prediction and target vs residuals of a DataFrame.

    Args:
        result_df (pd.DataFrame): The DataFrame containing the data to be plotted.
        target_col (str): The name of the target column in result_df.
        pred_col (str): The name of the prediction column in result_df.
        residual_col (str): The name of the residual column in result_df.
       color (str, optional): The color for the scatter plot. Default is 'b' (blue).

    Returns:
        None
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    result_min = np.trunc(result_df[[target_col, pred_col]].min().min())
    sns.scatterplot(data=result_df, x=target_col, y=pred_col, color=color, alpha=0.5, ax=ax1)
    ax1.axline((result_min, result_min), slope=1, label="Perfect fit")
    ax1.set_xlabel(target_col)
    ax1.set_ylabel(pred_col)
    ax1.set_title("targets vs predict values")

    sns.scatterplot(data=result_df, x=target_col, y=residual_col, color=color, alpha=0.5, ax=ax2)
    ax2.axhline(y=0, color="k", linewidth=1, linestyle="--")
    ax2.set_xlabel(target_col)
    ax2.set_ylabel("Residuals(true - predicted)")
    ax2.set_title("targets vs residuals")

    plt.tight_layout()
    plt.show()


def plot_multi_step_result(
    result_df: pd.DataFrame,
    target_col: str,
    pred_cols: List[str],
    pred_colormap: List[str],
    target_color: Optional[str] = "r",
    ax=None,
) -> None:
    """
    Plots the target and multiple prediction columns of a DataFrame.

    Args:
        result_df (pd.DataFrame): The DataFrame containing the data to be plotted.
        target_col (str): The name of the target column in result_df.
        pred_cols (List[str]): A list of names of prediction columns in result_df.
        pred_colormap (List[str]): A list of colors for each prediction column.
        target_color (str, optional): The color for the target plot. Default is 'r' (red).
        ax (matplotlib.axes._axes.Axes, optional): The matplotlib axes object to plot on.
        If not provided, a new figure and axes object will be created.

    Returns:
        None
    """
    if ax == None:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.margins(x=0)

    result_df[target_col].plot(ax=ax, color=target_color, marker="o", alpha=0.7, label=target_col)

    for i, col in enumerate(pred_cols):
        result_df[col].plot(
            ax=ax,
            color=pred_colormap[i],
            linewidth=0,
            marker="X",
            markersize=9,
            markeredgecolor="k",
            alpha=0.7,
            label=col,
        )


def plot_multi_step_evaluations(
    train_result: pd.DataFrame,
    test_result: pd.DataFrame,
    target_col: str,
    pred_cols: List[str],
    pred_colormap: List[str],
    target_color: Optional[str] = "r",
    plot_unused_span: Optional[int] = None,
    plot_test_span: bool = True,
) -> None:
    """
    Plots the target and multiple prediction columns of train and test DataFrames.

    Args:
        train_result (pd.DataFrame): The training DataFrame containing the data to be plotted.
        test_result (pd.DataFrame): The testing DataFrame containing the data to be plotted.
        target_col (str): The name of the target column in the DataFrames.
        pred_cols (List[str]): A list of names of prediction columns in the DataFrames.
        pred_colormap (List[str]): A list of colors for each prediction column.
        target_color (str, optional): The color for the target plot. Default is 'r' (red).
        plot_test_span (bool, optional): Whether to plot a span for the test data. Default is True.

    Returns:
        None
    """

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.margins(x=0)

    plot_result(train_result, target_col, pred_cols, target_color, pred_colormap, ax=ax)
    plot_result(test_result, target_col, pred_cols, target_color, pred_colormap, ax=ax)

    if plot_unused_span:
        start = train_result.index.max() - pd.Timedelta(weeks=(plot_unused_span - 1))
        end = train_result.index.max()
        ax.axvspan(start, end, color="gray", alpha=0.1)

    if plot_test_span:
        start, end = test_result.index.min(), test_result.index.max()
        ax.axvspan(start, end, color="gray", alpha=0.3)

    ax.legend("")


def plot_feature_vs_target(data, features, target, marker_color="r"):
    """
    Plot scatter plots of multiple features against a target variable.

    Args:
        data (pd.DataFrame): The input DataFrame containing features and target.
        features (list): A list of feature names to plot.
        target (str): The name of the target variable.
        marker_color (str, optional): The color for the scatter plot markers. Defaults to 'r'.

    Returns:
        None
    """
    n_features = len(features)
    n_cols = min(n_features, 3)
    n_rows = ceil(n_features / 3)

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(12, 4 * n_rows))

    for i, feature in enumerate(features):
        row, col = i // n_cols, i % n_cols
        sns.regplot(
            x=feature,
            y=target,
            data=data,
            ax=axes[row, col],
            scatter_kws={
                "marker": "o",
                "color": marker_color,
                "edgecolor": "black",
                "alpha": 0.4,
            },  # 마커 투명도 설정
            line_kws={"color": "black", "linestyle": "--", "linewidth": 0.5, "alpha": 0.6},
        )
        axes[row, col].set_title(f"{feature} vs {target}", fontsize=10)

    plt.tight_layout()
    plt.show()


def get_tlcc(x, y, lag_range):
    """
    Calculate the cross-correlation for different lags.

    Args:
        x (pd.Series): First time series data.
        y (pd.Series): Second time series data.
        lag_range (int): Maximum lag to consider (both positive and negative).

    Returns:
        pd.DataFrame: Cross-correlation values for each lag.
    """
    lags = range(-lag_range, lag_range + 1)
    cross_correlations = [x.corr(y.shift(lag)) for lag in lags]

    return pd.DataFrame({"lag": lags, "cross_correlation": cross_correlations})


def plot_tlcc(x, y, lag_range, color="r"):
    """
    Calculate time lagged cross correlation and plot the result.

    Args:
        x (pd.Series): First time series data.
        y (pd.Series): Second time series data.
        lag_range (int): Maximum lag to consider (both positive and negative).
        colormap (list): List of colors for plotting.

    Returns:
        None
    """
    cross_corr_df = get_tlcc(x, y, lag_range)

    max_corr_index = cross_corr_df["cross_correlation"].idxmax()
    offset = cross_corr_df.loc[max_corr_index, "lag"]

    if offset > 0:
        lead_info = f"S1 (x) leads by {offset} frames"
    elif offset < 0:
        lead_info = f"S2 (y) leads by {-offset} frames"
    else:
        lead_info = "S1 and S2 are synchronous"

    fig, ax = plt.subplots(figsize=(15, 4))
    ax.margins(x=0)
    ax.plot(cross_corr_df["lag"], cross_corr_df["cross_correlation"], color=color, linewidth=2)
    ax.axvline(0, color="k", linestyle="--", label="Center")
    ax.axvline(offset, color="r", linestyle="--", label="Peak synchrony")
    ax.set(title=f"Offset = {offset} frames\n{lead_info}", xlabel="Offset", ylabel="Pearson r")
    plt.legend()
    plt.show()
