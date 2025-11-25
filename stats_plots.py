import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.diagnostic import normal_ad
import statsmodels.formula.api as smf
import scipy.stats as stats
import itertools
from scipy.stats import pearsonr, norm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statistics import stdev
import matplotlib.pyplot as plt
from scipy.stats import linregress
from typing import Tuple, Union, Optional

def qqplot(
    data: Union[np.ndarray, list, pd.Series],
    label: str = "Data",
    alpha: float = 0.05,
    show_plot: bool = True
) -> Tuple[float, float, float]:
    """
    Performs the Anderson-Darling Normality Test and generates a Normal Q–Q Plot
    with a 95% confidence interval, mimicking the Minitab approach for adjusted
    A^2* and p-value calculation.

    The null hypothesis (H0) is that the data follows a normal distribution.

    Args:
        data: The data array (e.g., paired differences) to be tested.
        label: The name of the data for plot title/labels (e.g., "Differences").
        alpha: The significance level for the test (default is 0.05).
        show_plot: If True, displays the Matplotlib Q–Q plot.

    Returns:
        A dictionary containing: A²_raw, A²_adj, p_value, mean, stdev
    """
    if len(data) < 3:
        print("Warning: Data size is too small for meaningful A-D test.")
        return np.nan, np.nan, np.nan

    data = np.asarray(data)
    n = len(data)
    mean = np.mean(data)
    sd = np.std(data, ddof=1)

    # Handle case where standard deviation is zero (constant data)
    if sd == 0:
        print("Warning: Standard deviation is zero (all values are identical).")
        return {
            "A2_raw": 0.0,
            "A2_adj": 0.0,
            "p_value": 1.0,
            "mean": mean,
            "stdev": sd
        }

    # === 1. Standardize and sort data ===
    z = (data - mean) / sd
    z_sorted = np.sort(z)
    cdf = stats.norm.cdf(z_sorted)

    # === 2. Compute raw A² statistic ===
    i = np.arange(1, n + 1)
    A2_raw = -n - np.mean((2 * i - 1) * (np.log(cdf) + np.log(1 - cdf[::-1])))

    # === 3. Adjust for estimated μ,σ (Stephens 1974 approximation) ===
    A2_adj = A2_raw * (1 + 0.75 / n + 2.25 / n**2)

    # === 4. Convert adjusted A²* to p-value ===
    if A2_adj < 0.2:
        p_value = 1 - np.exp(-13.436 + 101.14 * A2_adj - 223.73 * A2_adj**2)
    elif A2_adj < 0.34:
        p_value = 1 - np.exp(-8.318 + 42.796 * A2_adj - 59.938 * A2_adj**2)
    elif A2_adj < 0.6:
        p_value = np.exp(0.9177 - 4.279 * A2_adj - 1.38 * A2_adj**2)
    elif A2_adj < 10:
        p_value = np.exp(1.2937 - 5.709 * A2_adj + 0.0186 * A2_adj**2)
    else:
        p_value = 3e-5

    if show_plot:
        # === 5. Prepare Q–Q plot elements ===
        data_sorted = np.sort(data)
        q_theor = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)

        slope, intercept = np.polyfit(q_theor, data_sorted, 1)
        predicted = slope * q_theor + intercept

        residuals = data_sorted - predicted
        s_err = np.sqrt(np.sum(residuals**2) / (n - 2))
        t_val = stats.t.ppf(1 - alpha/2, df=n - 2)
        x_mean = np.mean(q_theor)
        ci = t_val * s_err * np.sqrt(1/n + (q_theor - x_mean)**2 /
                                     np.sum((q_theor - x_mean)**2))

        # === 6. Plot ===
        plt.figure(figsize=(7, 6))
        plt.scatter(q_theor, data_sorted, color="#2563EB", label="Data Points")
        plt.plot(q_theor, predicted, "r-", label="Fit Line")
        plt.fill_between(q_theor, predicted - ci, predicted + ci,
                         color="#F87171", alpha=0.2, label="95% CI")

        plt.title(f"Normal Q–Q Plot for {label}", fontsize=14)
        plt.xlabel("Theoretical Quantiles (Standard Normal)")
        plt.ylabel(f"Sample Quantiles ({label})")

        # === Annotation box (with mean and stdev added) ===
        textstr = (
            f"n = {n}\n"
            f"Mean = {mean:.4f}\n"
            f"Stdev = {sd:.4f}\n"
            f"A² (raw): {A2_raw:.4f}\n"
            f"A²* (adj.): {A2_adj:.4f}\n"
            f"p-value: {p_value:.4f}"
        )
        conclusion = "Data APPEARS Normal (p > α)" if p_value >= alpha else "Data is NOT Normal (p < α)"
        textstr += f"\nConclusion (α={alpha}): {conclusion}"

        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                           alpha=0.8, edgecolor="#9CA3AF"))

        plt.grid(alpha=0.3, linestyle='--')
        plt.tight_layout()

    # === Console output ===
    print(f"\nMean: {mean:.4f}")
    print(f"Stdev: {sd:.4f}")
    print(f"A² (raw): {A2_raw:.4f}")
    print(f"A²* (adj.): {A2_adj:.4f}")
    print(f"Approx. p-value: {p_value:.4f}")

    # === Return extended results ===
    return {
        "A2_raw": A2_raw,
        "A2_adj": A2_adj,
        "p_value": p_value,
        "mean": mean,
        "stdev": sd
    }


def imr_chart(data, title="I-MR Chart", E2=2.660, D4=3.267, tick_count=10):
    """
    This script defines a function 'imr_chart' to create
    Individuals & Moving Range (I-MR) control charts.
    It also includes an example of loading data from an Excel file
    and using the function to generate a plot.

    Generates and displays a combined Individuals (I) and Moving Range (MR) plot.

    Parameters:
        - data (array-like): A 1D numpy array, list, pandas Series, or single-column pandas DataFrame
        containing numerical measurements.
        - title (str): The base title for the charts (e.g., "Brake Caliper Torsion").
        - E2 (float): Control chart constant for Individuals Chart limits (default for n=2 is 2.660).
        - D4 (float): Control chart constant for Moving Range Chart UCL (default for n=2 is 3.267).
        - tick_count (int): The number of evenly spaced ticks to display on the x-axis.

        Returns:
        - fig (matplotlib.figure.Figure): The Figure object containing the plots.
        - (ax1, ax2) (tuple): A tuple containing the Axes objects for the I-chart and MR-chart.
    """
    # --- Input Data Handling ---
    series_data = None
    if isinstance(data, pd.DataFrame):
        if data.shape[1] == 1:
            series_data = data.iloc[:, 0]
        else:
            # Raise an error to be clear.
            raise ValueError(
                f"Input data is a DataFrame with {data.shape[1]} columns. "
                "Please provide a pandas Series or a single-column DataFrame."
            )
    elif isinstance(data, pd.Series):
        series_data = data
    else:
        # Convert list or numpy array to a pandas Series for unified handling
        series_data = pd.Series(data)

    # Now, process the Series to get clean numpy data
    # Drop NaNs and convert to float
    try:
        imr_data = series_data.dropna().astype(float).values
    except ValueError as e:
        raise ValueError(f"Input data contains non-numeric values that could not be converted to float. Error: {e}")

    if len(imr_data) < 2:
        raise ValueError("Input data must contain at least 2 non-NaN numeric values to calculate a moving range.")
        
    # --- 1. Moving Range Calculations ---
    # Calculate the Moving Ranges (MR)
    mr = np.abs(np.diff(imr_data))
    # Calculate the Average Moving Range (MR_bar or mr_mean) - Center Line for MR Chart
    mr_mean = np.mean(mr)

    # --- 2. Individuals (I) Chart Calculations ---
    # Center Line for I Chart
    mean = np.mean(imr_data)
    
    # UCL and LCL for I Chart
    ucl_i = mean + (E2 * mr_mean)
    lcl_i = mean - (E2 * mr_mean)
    
    # Optional: Adjust LCL_I to 0 if the measure cannot be negative
    # lcl_i = np.maximum(0, lcl_i) # Uncomment this line if measurements cannot be negative

    # --- 3. Moving Range (MR) Chart Calculations ---
    # UCL_MR = D4 * MR_bar
    ucl_mr = mr_mean * D4
    # LCL_MR is always 0 for n=2
    lcl_mr = 0

    # --- Create the Combined Figure ---
    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
    fig.suptitle(f"I-MR Control Chart: {title}", fontsize = 16)

    # Box style for text labels
    bbox_props = dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.6)

    # --- Individuals Chart (ax1) ---
    x1 = np.arange(len(imr_data))
    ax[0].plot(x1, imr_data, marker='o', linestyle='-', color='b', label='Measurement')
    ax[0].plot(x1, [mean] * len(x1), color='green', linestyle='-', label='Mean')
    ax[0].plot(x1, [ucl_i] * len(x1), color='red', linestyle='--', label='UCL/LCL')
    ax[0].plot(x1, [lcl_i] * len(x1), color='red', linestyle='--')
    # Add text labels for control lines
    transform_mix = ax[0].get_yaxis_transform()
    ax[0].text(0.99, ucl_i, r'$UCL = %.3f$' % ucl_i,
             color='red', va='bottom', ha='right', fontweight='bold', transform=transform_mix, bbox=bbox_props)
    ax[0].text(0.99, mean, r'$\overline{X} = %.3f$' % mean,
             color='green', va='bottom', ha='right', fontweight='bold', transform=transform_mix, bbox=bbox_props)
    ax[0].text(0.99, lcl_i, r'$LCL = %.3f$' % lcl_i,
             color='red', va='bottom', ha='right', fontweight='bold', transform=transform_mix, bbox=bbox_props)

    ax[0].set_title("Individuals (I) Chart")
    ax[0].set_xlabel("Observation")
    ax[0].set_ylabel("Measurement")
    ax[0].set_xlim(-0.5, len(imr_data) - 0.5)

    # Set x-axis ticks
    xticks_i = np.linspace(0, len(x1) - 1, tick_count, dtype=int)
    ax[0].set_xticks(xticks_i)
    ax[0].set_xticklabels([str(i + 1) for i in xticks_i])  # Label starting from 1

    # Extend y-axis range
    ymin, ymax = ax[0].get_ylim()
    y_range = ymax - ymin
    ax[0].set_ylim(ymin - 0.1 * y_range, ymax + 0.1 * y_range)
    #ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, shadow=True)

    # --- Moving Range Chart (ax2) ---
    x2 = np.arange(len(mr))
    ax[1].plot(x2, mr, marker='o', linestyle='-', color='b', label='MR')
    ax[1].plot(x2, [mr_mean] * len(x2), color='green', linestyle='-', label=r'$\overline{MR}$')
    ax[1].plot(x2, [ucl_mr] * len(x2), color='red', linestyle='--', label='UCL')
    ax[1].plot(x2, [lcl_mr] * len(x2), color='red', linestyle='--', label='LCL')

    # Add text labels for control lines
    transform_mix_mr = ax[1].get_yaxis_transform()
    ax[1].text(0.99, ucl_mr, r'$UCL = %.3f$' % ucl_mr,
             color='red', va='bottom', ha='right', fontweight='bold', transform=transform_mix_mr, bbox=bbox_props)
    ax[1].text(0.99, mr_mean, r'$\overline{MR} = %.3f$' % mr_mean,
             color='green', va='bottom', ha='right', fontweight='bold', transform=transform_mix_mr, bbox=bbox_props)
    ax[1].text(0.99, lcl_mr, r'$LCL = %.3f$' % lcl_mr,
             color='red', va='bottom', ha='right', fontweight='bold', transform=transform_mix_mr, bbox=bbox_props)

    ax[1].set_title("Moving Range (MR) Chart")
    ax[1].set_xlabel("Observation (starting at 2)")
    ax[1].set_ylabel("Moving Range (MR$_i$)")
    ax[1].set_xlim(-0.5, len(mr) - 0.5)

    # Set x-axis ticks
    xticks_mr = np.linspace(0, len(x2) - 1, tick_count, dtype=int)
    ax[1].set_xticks(xticks_mr)
    # MR starts from observation 2 -> label = index + 2
    ax[1].set_xticklabels([str(i + 2) for i in xticks_mr])

    # Extend y-axis range
    ymin_mr, ymax_mr = ax[1].get_ylim()
    y_range_mr = ymax_mr - ymin_mr
    # Ensure LCL at 0 is visible
    ax[1].set_ylim(min(ymin_mr - 0.1 * y_range_mr, -0.05 * ymax_mr), 
                 ymax_mr + 0.1 * y_range_mr)
    #ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, shadow=True)

    plt.tight_layout()
    plt.show()
    

def p_chart(defects, subgroup_size, time_stamp, title="P-Chart"):
    """
    Generates and displays a P (Proportion) control chart.

    Parameters:
    - defects (pd.Series or array-like): The number of defective units in each subgroup.
    - subgroup_size (pd.Series or array-like): The size of each subgroup (n).
    - time_stamp (pd.Series or array-like): Labels for the x-axis (e.g., month names).
    - title (str): The base title for the chart.

    Returns:
    - fig (matplotlib.figure.Figure): The Figure object containing the plot.
    - ax (matplotlib.axes.Axes): The Axes object for the P-chart.
    """
    
    # Convert inputs to pandas Series for consistent handling
    defects = pd.Series(defects).astype(float)
    subgroup_size = pd.Series(subgroup_size).astype(float)
    time_stamp = pd.Series(time_stamp)

    if len(defects) != len(subgroup_size) or len(defects) != len(time_stamp):
        raise ValueError("Input series (defects, subgroup_size, time_stamp) must all have the same length.")

    # 1. Calculate Proportion Defective (p_i)
    p_i = defects / subgroup_size
    
    # 2. Calculate Overall Average Proportion Defective (p_bar)
    total_defects = defects.sum()
    total_produced = subgroup_size.sum()
    p_bar = total_defects / total_produced

    # 3. Calculate UCL and LCL for each subgroup
    # UCL/LCL = p_bar +/- 3 * sqrt(p_bar * (1 - p_bar) / n)
    std_error = np.sqrt(p_bar * (1 - p_bar) / subgroup_size)
    
    UCL = p_bar + 3 * std_error
    LCL = p_bar - 3 * std_error
    
    # LCL cannot be negative
    LCL = LCL.clip(lower=0)
    
    # --- Create the Figure ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(f"P-Chart: {title}", fontsize=14)

    # Box style for text labels
    bbox_props = dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.6)
    
    # Plot proportion defective (p_i)
    ax.plot(time_stamp, p_i, marker='o', linestyle='-', color='blue', label=r'$p_i$')
    
    # Plot Center Line (p_bar)
    ax.plot(time_stamp, [p_bar] * len(time_stamp), color='green', linestyle='-', label=r'$\overline{p}$')
    
    # Plot UCL and LCL (using step to show changing limits)
    # Note: Using the original time_stamp as x is usually fine for P-charts with dates
    ax.step(time_stamp, UCL, where='mid', color='red', linestyle='-', label='UCL')
    ax.step(time_stamp, LCL, where='mid', color='red', linestyle='-', label='LCL')

    # --- Add control limit labels to the right side ---
    transform_mix = ax.get_yaxis_transform()
    text_x_position = 0.99
    
    # Display the last calculated UCL/LCL values (since they vary, we just show one point)
    ucl_val_to_display = UCL.iloc[-1]
    lcl_val_to_display = LCL.iloc[-1]
    offset = 0.002 # Small offset for clarity

    # Label for p_bar (Center Line)
    ax.text(text_x_position, p_bar, r'$\overline{p} = %.5f$' % p_bar,
            color='green', va='center', ha='right', fontweight='bold',
            transform=transform_mix,
            bbox=bbox_props)
    
    # Label for UCL
    ax.text(text_x_position, ucl_val_to_display + offset, f"UCL = {ucl_val_to_display:.5f}",
            color='red', va='bottom', ha='right', fontweight='bold',
            transform=transform_mix,
            bbox=bbox_props) 
    
    # Label for LCL
    ax.text(text_x_position, lcl_val_to_display - offset, f"LCL = {lcl_val_to_display:.5f}",
            color='red', va='top', ha='right', fontweight='bold',
            transform=transform_mix,
            bbox=bbox_props)

    # Final plot styling
    ax.set_xlabel("Subgroup")
    ax.set_ylabel("Proportion Defective (p)")
    plt.xticks(rotation=45)
    ax.margins(x=0.03) # Add horizontal margin
    
    # Extend y-axis range
    ymin, ymax = ax.get_ylim()
    y_range = ymax - ymin
    ax.set_ylim(ymin - 0.1 * y_range, ymax + 0.1 * y_range)

    #ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, shadow=True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show


def xbarr_chart(df: pd.DataFrame, subgroup_col: str, value_col: str):
    """
    Generate an X-bar and R control chart from subgrouped data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing subgrouped data.
    subgroup_col : str
        Column name representing subgroups.
    value_col : str
        Column name containing measured values.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object for the chart.
    ax : np.ndarray
        Array of matplotlib Axes for the two charts (X-bar, R).
    """

    # --- Clean and group data ---
    df = df[[subgroup_col, value_col]].dropna(how="all").copy()
    df.columns = df.columns.str.strip()

    group_stats = df.groupby(subgroup_col)[value_col].agg(["mean", "min", "max", "count"])
    group_stats["range"] = group_stats["max"] - group_stats["min"]
    group_stats.reset_index(inplace=True)

    # --- SPC constants (for subgroup sizes 2–10) ---
    spc_constants = {
        2: {"A2": 1.88, "D3": 0.0, "D4": 3.27},
        3: {"A2": 1.023, "D3": 0.0, "D4": 2.574},
        4: {"A2": 0.729, "D3": 0.0, "D4": 2.282},
        5: {"A2": 0.577, "D3": 0.0, "D4": 2.114},
        6: {"A2": 0.483, "D3": 0.0, "D4": 2.004},
        7: {"A2": 0.419, "D3": 0.076, "D4": 1.924},
        8: {"A2": 0.373, "D3": 0.136, "D4": 1.864},
        9: {"A2": 0.337, "D3": 0.184, "D4": 1.816},
        10: {"A2": 0.308, "D3": 0.223, "D4": 1.777}
    }

    group_stats["A2"] = group_stats["count"].map(lambda n: spc_constants.get(n, {"A2": np.nan})["A2"])
    group_stats["D3"] = group_stats["count"].map(lambda n: spc_constants.get(n, {"D3": np.nan})["D3"])
    group_stats["D4"] = group_stats["count"].map(lambda n: spc_constants.get(n, {"D4": np.nan})["D4"])

    # --- Compute control limits ---
    R_bar = group_stats["range"].mean()
    X_bar_bar = group_stats["mean"].mean()

    group_stats["UCL_R"] = group_stats["D4"] * R_bar
    group_stats["LCL_R"] = group_stats["D3"] * R_bar
    group_stats["UCL_X"] = X_bar_bar + group_stats["A2"] * R_bar
    group_stats["LCL_X"] = X_bar_bar - group_stats["A2"] * R_bar

    # --- Plot X-bar and R charts ---
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # Box style with no frame
    bbox_props = dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.5)
    x = group_stats[subgroup_col]

    # Offset for text labels
    offset = 0.1

    # ===== X-bar Chart =====
    # Plot subgroup means
    ax[0].plot(x, group_stats["mean"], marker="o", linestyle='-', label="Subgroup Mean")

    # Plot control lines spanning the x-range
    ax[0].plot(x, [X_bar_bar]*len(x), color="green", linestyle="-", label=r'$\overline{X}$')
    ax[0].plot(x, group_stats["UCL_X"], color="red", linestyle="-", label="UCL")
    ax[0].plot(x, group_stats["LCL_X"], color="red", linestyle="-", label="LCL")

    # Annotate control lines
    ax[0].text(0.99, X_bar_bar + offset, r'$\overline{X} = %.2f$' % X_bar_bar,
            color='green', va='bottom', ha='right', fontweight='bold',
            transform=ax[0].get_yaxis_transform(), bbox=bbox_props)
    ax[0].text(0.99, group_stats["UCL_X"].iloc[-1] + offset, r'UCL = %.2f' % group_stats["UCL_X"].iloc[-1],
            color='red', va='bottom', ha='right', fontweight='bold',
            transform=ax[0].get_yaxis_transform(), bbox=bbox_props)
    ax[0].text(0.99, group_stats["LCL_X"].iloc[-1] + offset, r'LCL = %.2f' % group_stats["LCL_X"].iloc[-1],
            color='red', va='bottom', ha='right', fontweight='bold',
            transform=ax[0].get_yaxis_transform(), bbox=bbox_props)

    # Extend y-axis range by 10% above and below
    ymin, ymax = ax[0].get_ylim()
    y_range = ymax - ymin
    ax[0].set_ylim(ymin - 0.1 * y_range, ymax + 0.1 * y_range)

    ax[0].set_title("X-bar Chart")
    ax[0].set_ylabel("Subgroup Mean")

    # ===== R Chart =====
    # Plot subgroup ranges
    ax[1].plot(x, group_stats["range"], marker="o", linestyle='-', label="Range")

    # Plot control lines spanning x-range
    ax[1].plot(x, [R_bar]*len(x), color="green", linestyle="-", label=r'$\overline{R}$')
    ax[1].plot(x, group_stats["UCL_R"], color="red", linestyle="-", label="UCL")
    ax[1].plot(x, group_stats["LCL_R"], color="red", linestyle="-", label="LCL")

    # Annotate control lines
    ax[1].text(0.99, R_bar + 1, r'$\overline{R} = %.2f$' % R_bar,
            color='green', va='bottom', ha='right', fontweight='bold',
            transform=ax[1].get_yaxis_transform(), bbox=bbox_props)
    ax[1].text(0.99, group_stats["UCL_R"].iloc[-1] + offset, r'UCL = %.2f' % group_stats["UCL_R"].iloc[-1],
            color='red', va='bottom', ha='right', fontweight='bold',
            transform=ax[1].get_yaxis_transform(), bbox=bbox_props)
    ax[1].text(0.99, group_stats["LCL_R"].iloc[-1] + offset, r'LCL = %.2f' % group_stats["LCL_R"].iloc[-1],
            color='red', va='bottom', ha='right', fontweight='bold',
            transform=ax[1].get_yaxis_transform(), bbox=bbox_props)

    # Extend y-axis range by 10% above and below
    ymin, ymax = ax[1].get_ylim()
    y_range = ymax - ymin
    ax[1].set_ylim(ymin - 0.1 * y_range, ymax + 0.1 * y_range)

    ax[1].set_title("R Chart")
    ax[1].set_xlabel("Subgroup")
    ax[1].set_ylabel("Range")

    plt.tight_layout()
    plt.show()


def u_chart(df, unit_col, defects_col, time_col, 
                 chart_title='U-Chart', xlabel='Time', ylabel='Defects per Unit'):
    """
    Generate a u-chart from defect data.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the defect data.
    unit_col : str
        Column name for number of units produced (e.g., 'Airbags Produced').
    defects_col : str
        Column name for number of defects (e.g., 'Defects Identified').
    time_col : str
        Column name for the time or subgroup (e.g., 'Month_1').
    chart_title : str
        Title of the chart.
    xlabel : str
        Label for the x-axis.
    ylabel : str
        Label for the y-axis.
    """
    df = df.copy()  # avoid modifying original DataFrame

    # Step 1: Calculate defects per unit
    df['u'] = df[defects_col] / df[unit_col]

    # Step 2: Calculate average u (u-bar)
    u_bar = df[defects_col].sum() / df[unit_col].sum()

    # Step 3: Calculate control limits
    df['UCL'] = u_bar + 3 * np.sqrt(u_bar / df[unit_col])
    df['LCL'] = u_bar - 3 * np.sqrt(u_bar / df[unit_col])
    df['LCL'] = df['LCL'].apply(lambda x: max(x, 0))  # LCL cannot be negative

    # Step 4: Plot the U-chart (style preserved)
    fig, ax = plt.subplots(figsize=(8, 4))

    ucl_val_to_display = df['UCL'].iloc[-1]
    lcl_val_to_display = df['LCL'].iloc[-1]
    x = df[time_col]

    ax.step(x, df['UCL'], where='mid', color='red', linestyle='-', label='UCL')
    ax.step(x, df['LCL'], where='mid', color='red', linestyle='-', label='LCL')
    ax.plot(x, df['u'], marker='o', linestyle='-', label='u (defects/unit)')
    ax.plot(x, [u_bar]*len(x), color='green', linestyle='-', label='u-bar')

    offset = 0.002
    ax.text(0.99, u_bar + offset, r'$\overline{u} = %.4f$' % u_bar,
            color='green', va='bottom', ha='right', fontweight='bold',
            transform=ax.get_yaxis_transform())
    ax.text(0.99, ucl_val_to_display + offset, r'$UCL = %.4f$' % ucl_val_to_display,
            color='red', va='bottom', ha='right', fontweight='bold',
            transform=ax.get_yaxis_transform())
    ax.text(0.99, lcl_val_to_display + offset, r'$LCL = %.4f$' % lcl_val_to_display,
            color='red', va='bottom', ha='right', fontweight='bold',
            transform=ax.get_yaxis_transform())

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(chart_title)
    ax.tick_params(axis='x', rotation=45)

    ax.margins(x=0.05)
    ax.margins(y=0.05)

    plt.tight_layout()
    plt.show()

    # Return processed DataFrame in case user wants it
    return df


def run_chart(df, value_col, chart_title="Run Chart", xlabel="Observation", ylabel=None):
    """
    Generate a run chart with trend line and mean line.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data.
    value_col : str
        Column name of the values to plot.
    chart_title : str
        Title of the chart.
    xlabel : str
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis. If None, defaults to value_col.
    
    Returns
    -------
    dict
        Dictionary with 'x', 'y', 'trend', 'mean', 'slope', 'p_value'.
    """
    df = df.dropna(subset=[value_col]).copy()

    # X-axis: observation number
    x = np.arange(1, len(df) + 1)
    y = df[value_col].values

    # --- Linear regression ---
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    trend = intercept + slope * x

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 4))

    # Run chart
    ax.plot(x, y, marker='o', linestyle='-', color='blue', label='RunCh')

    # Trend line
    ax.plot(x, trend, color='red', linestyle='--', 
            label=f'Trend (slope={slope:.4f}, p={p_value:.4f})')

    # Mean line
    mean_val = np.mean(y)
    ax.plot(x, [mean_val]*len(x), color='green', linestyle='--', 
            label=f'Mean = {mean_val:.4f}')

    # Labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel if ylabel else value_col)
    ax.set_title(chart_title)

    # Move legend below the plot
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        frameon=False
    )

    # X-ticks
    ax.set_xticks(x)
    ax.set_xticklabels(x)

    # Extend margins
    ax.margins(x=0.05)
    ax.margins(y=0.05)

    plt.tight_layout()
    plt.show()

    # Return useful info
    return {
        #"x": x,
        #"y": y,
        #"trend": trend,
        "mean": mean_val,
        "slope": slope,
        "p_value": p_value
    }


def last25_pp(data, title_prefix):
    """
    Plot last 25 observations and normal probability plot.
    Returns last25, mean_last25, r_squared.

    Parameters
    ----------
    data : array-like or pd.Series
        Data to be analyzed and plotted.
    title_prefix : str
        Prefix for the plot titles. Must be provided by the caller.
    
    Returns
    -------
    dict
        Dictionary containing last25, mean_last25, and R-squared of the normal probability plot fit.
    """
    if isinstance(data, np.ndarray):
        s = pd.Series(data, index=np.arange(1, len(data) + 1))
    else:
        s = data

    last25 = s.tail(25)
    mean_last25 = last25.mean()

    fig, axes = plt.subplots(2, 1, figsize=(8, 6))
    plt.subplots_adjust(hspace=0.4)

    # Last 25 observations
    ax1 = axes[0]
    ax1.plot(last25.index, last25.values, marker='o', linestyle='', label='Observation')
    ax1.axhline(mean_last25, color='green', linestyle='--', label=f'Mean = {mean_last25:.3f}')
    ax1.set_title(f'{title_prefix} - Last 25 Observations')
    ax1.set_xlabel('Original Index')
    ax1.set_ylabel('Value')

    if isinstance(last25.index, pd.DatetimeIndex):
        fig.autofmt_xdate()

    # Normal probability plot
    ax2 = axes[1]
    (osm, osr), (slope, intercept, r) = stats.probplot(s, dist="norm")
    r_squared = r**2
    ax2.plot(osm, slope * osm + intercept, color='red', linestyle='--', label='Fit Line')
    ax2.plot(osm, osr, 'o', color='blue', label='Data')
    ax2.set_xlim(-2.1, 2.1)
    ax2.set_title(f'{title_prefix} - Normal Probability Plot')
    ax2.set_xlabel('Theoretical Quantiles (Normal)')
    ax2.set_ylabel('Ordered Values')
    ax2.text(
        0.98, 0.05, f'$R^2$ = {r_squared:.4f}',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
        fontsize=12, ha='right', va='bottom', transform=ax2.transAxes
    )
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

    return {
        #"last25": last25,
        "mean_last25": mean_last25,
        "r_squared": r_squared
    }


def boxcox(data, USL, LSL, minitab_adjust=True, tol=0.25):
    """
    Perform Box-Cox transformation on data and transform specification limits.
    Suggests a human-readable transformation type based on lambda.

    Parameters
    ----------
    data : array-like
        Original data (must be positive).
    USL : float
        Upper specification limit.
    LSL : float
        Lower specification limit.
    minitab_adjust : bool
        If True, force log transform if |λ| < tol (Minitab style).
    tol : float
        Threshold for minitab adjustment.

    Returns
    -------
    dict
        Contains transformed data, lambda, transformed spec limits, mean, std dev,
        and suggested transformation type.
    """
    original_data = np.array(data)
    transformed_data, optimal_lambda = stats.boxcox(original_data)
    
    # Optional Minitab adjustment
    if minitab_adjust and abs(optimal_lambda) < tol:
        transformed_data = np.log(original_data)
        optimal_lambda = 0

    # Transform specification limits
    if optimal_lambda != 0:
        USL_prime = (USL**optimal_lambda - 1) / optimal_lambda
        LSL_prime = (LSL**optimal_lambda - 1) / optimal_lambda
    else:
        USL_prime = np.log(USL)
        LSL_prime = np.log(LSL)
    
    # Transformed statistics
    X_bar_prime = np.mean(transformed_data)
    s_prime = stdev(transformed_data)
    
    # Suggest human-readable transformation type
    def suggest_transform(lambda_val):
        if np.isclose(lambda_val, 0, atol=0.05):
            return "log transform"
        elif np.isclose(lambda_val, 0.5, atol=0.05):
            return "square root transform"
        elif np.isclose(lambda_val, -0.5, atol=0.05):
            return "reciprocal square root transform"
        elif np.isclose(lambda_val, 1, atol=0.05):
            return "no transform"
        elif np.isclose(lambda_val, -1, atol=0.05):
            return "reciprocal transform"
        elif np.isclose(lambda_val, 2, atol=0.05):
            return "square transform"
        elif np.isclose(lambda_val, -2, atol=0.05):
            return "reciprocal square transform"
        else:
            return f"power transform (λ={lambda_val:.4f})"
    
    suggested_transform = suggest_transform(optimal_lambda)

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    stats.probplot(original_data, dist="norm", plot=axes[0])
    axes[0].set_title('Original Data (Non-Normal)')
    
    stats.probplot(transformed_data, dist="norm", plot=axes[1])
    axes[1].set_title(f'Box-Cox Transformed Data (λ={optimal_lambda:.4f})')
    
    plt.tight_layout()
    plt.show()
    
    # Print results
    print(f"\nOptimal Lambda (λ): {optimal_lambda:.4f}")
    print(f"Suggested Transformation: {suggested_transform}")
    print(f"Transformed USL (USL'): {USL_prime:.4f}")
    print(f"Transformed LSL (LSL'): {LSL_prime:.4f}")
    print(f"Transformed Mean (X̄'): {X_bar_prime:.4f}")
    print(f"Transformed StDev (s'): {s_prime:.4f}")
    
    return {
        "transformed_data": transformed_data,
        "optimal_lambda": optimal_lambda,
        "suggested_transform": suggested_transform,
        "USL_prime": USL_prime,
        "LSL_prime": LSL_prime,
        "X_bar_prime": X_bar_prime,
        "s_prime": s_prime
    }


def pareto(df: pd.DataFrame, category_col: str, title: str = "Pareto Chart", figsize=(8, 5)):
    """
    Generate a Pareto chart showing frequency and cumulative percentage.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the categorical data.
    category_col : str
        Column name of the categorical variable (e.g., defect type).
    title : str, optional
        Title for the plot.
    figsize : tuple, optional
        Figure size in inches (default: (8, 5)).
    """

    # --- Count occurrences and cumulative percentage ---
    counts = df[category_col].value_counts().sort_values(ascending=False)
    cum_percent = counts.cumsum() / counts.sum() * 100

    # --- Create figure and axes ---
    fig, ax1 = plt.subplots(figsize=figsize)

    # Bar plot for counts
    ax1.bar(counts.index, counts.values, color='skyblue')
    ax1.set_xlabel(category_col)
    ax1.set_ylabel('Count', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    plt.xticks(rotation=45, ha='right')

    # Line plot for cumulative percentage
    ax2 = ax1.twinx()
    ax2.plot(counts.index, cum_percent.values, color='red', marker='o', linewidth=2)
    ax2.set_ylabel('Cumulative Percentage (%)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, 110)

    # Add percentage labels
    for i, val in enumerate(cum_percent.values):
        ax2.text(i, val + 2, f'{val:.1f}%', ha='center', color='red', fontsize=9)

    plt.title(title)
    plt.tight_layout()
    plt.show()


# --- 2️⃣ Diagnostics function ---
def ols_check(model, df_out=None, x_col=None, y_col=None):
    """
    Generate standard OLS diagnostic plots:
        1. Normal Q–Q Plot
        2. Residuals vs Fitted
        3. Histogram of Residuals
        4. Residuals vs Observation Order

    Parameters
    ----------
    model : statsmodels.regression.linear_model.RegressionResultsWrapper
        A fitted OLS model.
    df_out : pd.DataFrame, optional
        DataFrame returned by ols_regression() (needed for scatter plot of X vs Y with fitted line)
    x_col : str, optional
        Predictor column name (used for linear fit plot)
    y_col : str, optional
        Response column name (used for linear fit plot)
    """
    residuals = model.resid
    fitted = model.fittedvalues
    n = len(residuals)

    # --- Optional: Linear fit with CI & PI (for single predictor only) ---
    if df_out is not None and x_col is not None and y_col is not None:
        plt.figure(figsize=(10, 6))
        plt.scatter(df_out[x_col], df_out[y_col], color='blue', label='Data')
        plt.plot(df_out[x_col], df_out["Fitted"], color='red', label='Regression Line')
        plt.fill_between(df_out[x_col], df_out["CI_lower"], df_out["CI_upper"],
                         color='red', alpha=0.2, label='95% Confidence Interval')
        plt.fill_between(df_out[x_col], df_out["PI_lower"], df_out["PI_upper"],
                         color='orange', alpha=0.2, label='95% Prediction Interval')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title("Linear Regression with CI and PI")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid(True)  # ✅ add grid
        plt.tight_layout()
        plt.show()
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    # === 1️⃣ Normal Q–Q Plot of Residuals ===
    data_sorted = np.sort(residuals)
    q_theor = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)

    slope, intercept = np.polyfit(q_theor, data_sorted, 1)
    predicted = slope * q_theor + intercept

    # Anderson-Darling statistic for normality
    A2_raw, p_value = normal_ad(residuals)

    axes[0,0].scatter(q_theor, data_sorted, color='blue')
    axes[0,0].plot(q_theor, predicted, "r-")
    axes[0,0].set_title("Normal Q–Q Plot of Residuals")
    axes[0,0].set_xlabel("Theoretical Quantiles")
    axes[0,0].set_ylabel("Residuals")
    axes[0,0].grid(True)

    # Display n, AD statistic, p-value
    textstr = f"n = {n}\nA² (AD stat) = {A2_raw:.4f}\np-value ≈ {p_value:.4f}"
    axes[0,0].text(0.05, 0.95, textstr, transform=axes[0,0].transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8, edgecolor="#9CA3AF"))

    # === 2️⃣ Residuals vs Fitted ===
    axes[0,1].scatter(fitted, residuals, color='blue', alpha=0.7)
    axes[0,1].axhline(0, color='red', linestyle='--')
    axes[0,1].set_xlabel("Fitted Values")
    axes[0,1].set_ylabel("Residuals")
    axes[0,1].set_title("Residuals vs Fitted")
    axes[0,1].grid(True)

    # === 3️⃣ Histogram of Residuals ===
    sns.histplot(residuals, kde=True, bins=10, color='skyblue', ax=axes[1,0])
    axes[1,0].set_xlabel("Residuals")
    axes[1,0].set_title("Histogram of Residuals")
    axes[1,0].grid(True)

    # === 4️⃣ Residuals vs Observation Order ===
    axes[1,1].plot(range(1, n+1), residuals, color='blue', marker='o', linestyle='-', alpha=0.7)
    axes[1,1].axhline(0, color='red', linestyle='--')
    axes[1,1].set_xlabel("Observation Order")
    axes[1,1].set_ylabel("Residuals")
    axes[1,1].set_title("Residuals vs Observation")
    axes[1,1].grid(True)
    axes[1,1].set_xlim(1 - 0.05*n, n + 0.05*n)
    axes[1,1].margins(x=0)

    plt.tight_layout()
    plt.show()


# --- 1️⃣ Core OLS fitting function ---
def ols_reg(df: pd.DataFrame, x_cols, y_col, categorical=None, alpha=0.05):
    """
    Fit multiple regression with optional categorical predictors and compute VIF.

    Returns model, df_out, anova_table, equation, summary_stats, vif_data
    """
    if categorical is None:
        categorical = []

    # Ensure x_cols is a list
    if isinstance(x_cols, str):
        x_cols = [x_cols]

    # --- Build formula ---
    predictors = [f"C({col})" if col in categorical else col for col in x_cols]
    formula = f"{y_col} ~ " + " + ".join(predictors)

    # --- Fit model ---
    model = smf.ols(formula, data=df).fit()
    

    # --- Predictions with CI and PI ---
    pred = model.get_prediction(df[x_cols])
    pred_summary = pred.summary_frame(alpha=alpha)

    df_out = df.copy()
    df_out['Fitted'] = model.fittedvalues
    df_out['Residuals'] = model.resid
    df_out['CI_lower'] = pred_summary['mean_ci_lower']
    df_out['CI_upper'] = pred_summary['mean_ci_upper']
    df_out['PI_lower'] = pred_summary['obs_ci_lower']
    df_out['PI_upper'] = pred_summary['obs_ci_upper']

    anova_table = anova_lm(model, typ=2)

    # Mean square for predictors (individual)
    anova_table['mean_sq'] = anova_table['sum_sq'] / anova_table['df']

    # Add Regression row
    anova_table.loc['Regression', 'sum_sq'] = model.ess      # Explained sum of squares
    anova_table.loc['Regression', 'df'] = model.df_model     # Degrees of freedom of model
    anova_table.loc['Regression', 'mean_sq'] = model.ess / model.df_model
    anova_table.loc['Regression', 'F'] = model.fvalue
    anova_table.loc['Regression', 'PR(>F)'] = model.f_pvalue

    # Total row
    anova_table.loc['Total', ['sum_sq', 'df']] = [anova_table['sum_sq'].sum(), anova_table['df'].sum()]



    # --- Fitted equation ---
    params = model.params
    equation = f"{y_col} = " + " + ".join(
        [f"{params[i]:.4f}*{i}" if i != 'Intercept' else f"{params[i]:.4f}" for i in params.index]
    )

    # --- Summary stats ---
    summary_stats = {
        "Standard Error (S)": np.sqrt(model.scale),
        "R-squared": model.rsquared,
        "Adjusted R-squared": model.rsquared_adj
    }

    # --- VIF calculation from model design matrix ---
    X = model.model.exog
    X_names = model.model.exog_names

    vif_data = pd.DataFrame()
    vif_data["feature"] = X_names
    vif_data["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    vif_data = vif_data[vif_data["feature"] != "Intercept"]


    # --- Print key info ---
    print(model.summary())
    print(f"\nFitted Equation:\n{equation}")
    print("\nANOVA Table:")
    print(anova_table.round(4))
    print("\nSummary Stats:")
    for k, v in summary_stats.items():
        print(f"{k}: {v:.4f}")
    if vif_data is not None:
        print("\nVariance Inflation Factors (VIF):")
        print(vif_data.round(3))

    return {
    "model": model,
    "df_out": df_out,
    "anova_table": anova_table,
    "equation": equation,
    "summary_stats": summary_stats,
    "vif_data": vif_data
}



def pair_plot(df, alpha=0.05, plot_kind='scatter', diag_kind='hist', height=2.5):
    """
    Generate pairwise Pearson correlation table and annotated pairplot.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame (numeric columns will be selected automatically).
    alpha : float, optional
        Significance level for confidence intervals (default=0.05 → 95% CI).
    plot_kind : {'scatter', 'reg'}, optional
        Type of plot for pairplot (default='scatter').
    diag_kind : {'hist', 'kde'}, optional
        Kind of plot for diagonal subplots (default='hist').
    height : float, optional
        Height (in inches) of each subplot in the grid (default=2.5).

    Returns
    -------
    corr_results_df : pandas.DataFrame
        Table with Variable 1, Variable 2, Pearson r, p-value, CI lower, CI upper.
    """

    # --- Select numeric columns ---
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    numeric_df = df[numeric_cols].dropna()  # drop missing values if any

    # --- Define CI function ---
    def pearson_ci(r, n, alpha=alpha):
        if n <= 3 or abs(r) == 1:
            return (np.nan, np.nan)
        z = np.arctanh(r)
        se = 1 / np.sqrt(n - 3)
        z_crit = norm.ppf(1 - alpha / 2)
        lo_z, hi_z = z - z_crit * se, z + z_crit * se
        lo_r, hi_r = np.tanh([lo_z, hi_z])
        return lo_r, hi_r

    # --- Compute correlations ---
    results = []
    for col1, col2 in itertools.combinations(numeric_cols, 2):
        x = numeric_df[col1]
        y = numeric_df[col2]
        r, p = pearsonr(x, y)
        ci_low, ci_high = pearson_ci(r, len(x))
        results.append({
            "Variable 1": col1,
            "Variable 2": col2,
            "Pearson r": r,
            "p-value": p,
            "CI lower": ci_low,
            "CI upper": ci_high
        })

    corr_results_df = pd.DataFrame(results)

    # --- Round numeric values to 3 decimals ---
    corr_results_df = corr_results_df.round(3)

    # --- Pairplot ---
    sns.set(style="ticks", font_scale=1)
    pairplot = sns.pairplot(numeric_df, kind=plot_kind, diag_kind=diag_kind, height=height)

    # --- Annotate upper triangle with r and p ---
    for i, col1 in enumerate(numeric_cols):
        for j, col2 in enumerate(numeric_cols):
            if i < j:
                x = numeric_df[col1]
                y = numeric_df[col2]
                r, p = pearsonr(x, y)
                ax = pairplot.axes[j, i]
                ax.annotate(f"r={r:.2f}\np={p:.3f}",
                            xy=(0.5, 0.7), xycoords='axes fraction',
                            ha='center', fontsize=9, color='red')

    plt.suptitle("Pairwise Scatter Plots with Pearson Correlation", y=1.02)
    plt.show()

    return corr_results_df



def boxplot(data=None, groups=None, title="Comparison of Groups", palette=None, figsize=(8,6)):
    """
    Flexible Seaborn boxplot for multiple groups.

    Parameters
    ----------
    data : dict, DataFrame, or list of arrays, optional
        - dict: keys = group labels, values = numeric arrays/lists
        - DataFrame: each column is a group
        - list/tuple of arrays: groups must be provided
    groups : list of str, optional
        Names of groups (required if data is list/tuple)
    title : str
        Plot title
    palette : list or dict, optional
        Colors for each group
    figsize : tuple
        Figure size in inches
    """

    # --- Convert input into long-format DataFrame ---
    if isinstance(data, dict):
        all_values = np.concatenate(list(data.values()))
        all_labels = np.concatenate([[label]*len(vals) for label, vals in data.items()])
    elif isinstance(data, pd.DataFrame):
        all_values = data.to_numpy().flatten()
        all_labels = np.concatenate([[col]*len(data[col]) for col in data.columns])
    elif isinstance(data, (list, tuple)) and groups is not None:
        all_values = np.concatenate(data)
        all_labels = np.concatenate([[label]*len(vals) for label, vals in zip(groups, data)])
    else:
        raise ValueError("data must be a dict, DataFrame, or list/tuple with groups specified.")

    plot_df = pd.DataFrame({
        "Value": all_values,
        "Group": all_labels
    })

    if palette is None:
        palette = sns.color_palette("pastel", len(np.unique(all_labels)))

    # --- Plot ---
    plt.figure(figsize=figsize)
    sns.boxplot(
        x="Group",
        y="Value",
        hue="Group",
        data=plot_df,
        palette=palette,
        dodge=False
    )
    plt.legend([],[], frameon=False)  # hide redundant legend
    plt.title(title)
    plt.ylabel("Value")
    plt.show()




def compare_hist(data1, data2, label1='Before', label2='After', bins=15, alpha=0.5, 
                       figsize=(8,6), xlabel=None, ylabel='Frequency', title='Histogram Comparison'):
    """
    Plot overlapping histograms of two datasets for comparison.

    Parameters:
    - data1, data2: arrays or pandas Series of numeric data
    - label1, label2: labels for the two datasets
    - bins: number of bins in histogram
    - alpha: transparency for histograms
    - figsize: figure size
    - xlabel, ylabel, title: labels and title for the plot
    """
    plt.figure(figsize=figsize)
    sns.histplot(data1, bins=bins, color='blue', alpha=alpha, label=label1, kde=False)
    sns.histplot(data2, bins=bins, color='orange', alpha=alpha, label=label2, kde=False)
    
    if xlabel:
        plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()




def hist(data, bins=10, kde=True, title=None):
    """
    Plot a histogram of numerical data with mean and median lines.

    Parameters
    ----------
    data : array-like
        Numerical data to plot (list, NumPy array, or pandas Series).
    bins : int, default=10
        Number of histogram bins.
    kde : bool, default=True
        Whether to include kernel density estimate.
    title : str, optional
        Optional plot title. If None, no title will be added.
    """
    data = np.array(data)
    data = data[~np.isnan(data)]  # drop NaNs if present

    plt.figure(figsize=(8, 6))
    sns.histplot(data, bins=bins, kde=kde, color="skyblue", edgecolor="black")

    plt.axvline(np.mean(data), color="red", linestyle="--", linewidth=1.5, label=f"Mean = {np.mean(data):.3f}")
    plt.axvline(np.median(data), color="green", linestyle="-.", linewidth=1.5, label=f"Median = {np.median(data):.3f}")

    if title:
        plt.title(title, fontsize=14, weight="bold")
    plt.xlabel("Value", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()




def plot_posthoc(posthoc_table, method="games-howell", alpha=0.05, title=None):
    """
    Plot pairwise mean differences with 95% CI for Games-Howell or Tukey HSD.
    Automatically scales figure height and uses 'B - A' style y-axis labels
    matching the mean difference direction.

    Parameters
    ----------
    posthoc_table : pd.DataFrame
        Post-hoc table from pingouin (Games-Howell) or statsmodels (Tukey HSD).
    method : str
        "games-howell" or "tukey"
    alpha : float
        Significance level for CI (used only for Games-Howell)
    title : str
        Plot title
    """

    # --- Determine data and compute CI ---
    if method.lower() == "games-howell":
        # Mean differences already in 'diff' column
        diff = posthoc_table["diff"].values
        se = posthoc_table["se"].values
        df_vals = posthoc_table["df"].values


        t_vals = stats.t.ppf(1 - alpha/2, df=df_vals)
        ci_lower = diff - t_vals * se
        ci_upper = diff + t_vals * se

        # y-axis labels
        pairs = posthoc_table["A"] + " - " + posthoc_table["B"]

        if title is None:
            title = "Games-Howell Pairwise Mean Differences"

    elif method.lower() == "tukey":
        # Tukey HSD: mean differences in 'meandiff'
        diff = posthoc_table["meandiff"].values
        ci_lower = posthoc_table["lower"].values
        ci_upper = posthoc_table["upper"].values

        # y-axis labels
        pairs = posthoc_table["group2"] + " - " + posthoc_table["group1"]

        if title is None:
            title = "Tukey HSD Pairwise Mean Differences"

    else:
        raise ValueError("method must be 'games-howell' or 'tukey'")

    # --- Plot ---
    num_pairs = len(pairs)
    y_pos = np.arange(num_pairs)

    fig, ax = plt.subplots(figsize=(8, max(4, num_pairs*0.5)))

    # Horizontal lines for CI
    ax.hlines(y=y_pos, xmin=ci_lower, xmax=ci_upper, color='gray', lw=2)
    # Mean difference points
    ax.plot(diff, y_pos, 'o', color="#2563EB")
    # Vertical reference line at 0
    ax.axvline(0, color='red', linestyle='--')

    # Y-axis labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(pairs)

    # Slightly extend y-axis
    ax.set_ylim(-0.5, num_pairs - 0.5)

    ax.set_xlabel("Mean Difference")
    ax.set_ylabel("Pairwise Comparisons")
    ax.set_title(title)
    ax.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()






