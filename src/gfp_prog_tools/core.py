# -*- coding: utf-8 -*-
"""
General helper functions: file handling, data cleaning, resampling,
confidence intervals and graph formatting.

Merges General_functions.py (2021-12-13) and graph_parameters.py (2021-11-17).

@author: g.flood-page
"""
import os
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import dates as mdates
from scipy.signal import resample
from scipy.stats import t, norm

__all__ = [
    # files and data handling
    "Ensure_dir", "dateparse", "Eliminate_absurd_values", "Get_index_min_value", "In_with_tol",
    "Get_nb_files", "Get_indexes_compare_vectors", "Update_df",
    "Resample_data_frame_scipy", "Resample_data_frame_linear_regression",
    # confidence intervals
    "Calculate_student_t_confidence_interval", "Calculate_normal_confidence_interval",
    "Calculate_confidence_interval_from_sample",
    "Calculate_student_t_confidence_interval_curve_fit", "Calculate_normal_confidence_interval_curve_fit",
    # graphs
    "Set_graph_params", "Save_figure", "Limit_datetime_ticks",
    "markers", "linestyles", "Wrap_marker", "Wrap_linestyle",
]


# ---------------------------------------------------------------------------
# Files and data handling
# ---------------------------------------------------------------------------

def Ensure_dir(dir_path, dir_name):
    """
    Create a sub-directory if it does not already exist and return its path.

    Parameters
    ----------
    dir_path : str
        Parent directory.
    dir_name : str
        Name of the sub-directory to create inside dir_path.

    Returns
    -------
    str
        Full path of the sub-directory.
    """
    directory = os.path.join(dir_path, dir_name)  # full path of the sub-directory
    os.makedirs(directory, exist_ok=True)
    return directory


def dateparse(s):
    """
    Parse dates written day-first (e.g. 27/10/2026); unparseable values become NaT.

    Parameters
    ----------
    s : str, array-like or pandas.Series
        Date string(s) to parse.

    Returns
    -------
    pandas.Timestamp or pandas.DatetimeIndex
        Parsed date(s).
    """
    return pd.to_datetime(s, dayfirst=True, errors='coerce', cache=True)


def Eliminate_absurd_values(input_array, max_val, min_val, avg_val=11, cutoff_factor=1.5):
    """
    Replace values by NaN if they are below min_val, above max_val, or smaller than
    the local median divided by cutoff_factor. Function still needs to be tested.

    The local median is computed over a window of about avg_val neighbouring values;
    near the start and end of the array the window is shifted inwards.

    Parameters
    ----------
    input_array : array-like
        Values to clean.
    max_val : float
        Values above max_val are replaced by NaN.
    min_val : float
        Values below min_val are replaced by NaN.
    avg_val : int, optional
        Approximate number of values used for the local median (default 11).
    cutoff_factor : float, optional
        A value is rejected if it is cutoff_factor times smaller than the local median (default 1.5).

    Returns
    -------
    numpy.ndarray
        Copy of input_array with rejected values replaced by NaN.
    """
    input_array = np.asarray(input_array, dtype=float)  # works for lists and pandas Series alike
    avg_shift = avg_val//2  # half-width of the median window
    len_array = len(input_array)  # number of values
    res = np.zeros(len_array, float)  # cleaned output
    for i, val in enumerate(input_array):
        if i < avg_shift:  # median mostly to the right of the first values
            med = np.nanmedian(input_array[0:int(2*avg_shift-i)])  # local median
        elif i >= len_array-avg_shift-1:  # median mostly to the left of the last values
            med = np.nanmedian(input_array[int(2*len_array-2*avg_shift-i):len_array])
        else:  # centred window
            med = np.nanmedian(input_array[i-avg_shift:i+avg_shift])
        if val < med/cutoff_factor or val < min_val or val > max_val:
            res[i] = np.nan
        else:
            res[i] = val
    return res


def Get_index_min_value(input_array, ignore_inf=False):
    """
    Return the index of the (first) minimum of an array, ignoring NaN.

    Parameters
    ----------
    input_array : array-like
        Values to search.
    ignore_inf : bool, optional
        If True, -inf values are ignored when looking for the minimum (default False).

    Returns
    -------
    int or None
        Index of the minimum, or None if it could not be found (e.g. all NaN).
    """
    arr = np.asarray(input_array, dtype=float)  # array version of the input
    candidates = arr[arr != -np.inf] if ignore_inf else arr  # values eligible as minimum
    min_val = np.nanmin(candidates)  # minimum value
    idx = np.flatnonzero(arr == min_val)  # all indexes reaching the minimum
    if idx.size == 0:
        print("Get_index_min_value failed : could not retrieve index of array min")
        return None
    return int(idx[0])


def In_with_tol(value, array, tol):
    """
    Check whether value matches any element of array within a relative tolerance:
    |value - arval| < |arval|*tol.

    Parameters
    ----------
    value : float
        Value to look for.
    array : array-like
        Reference values.
    tol : float
        Relative tolerance (e.g. 0.05 for 5 %).

    Returns
    -------
    bool
        True if at least one element matches.
    """
    for arval in array:
        if abs(value-arval) < abs(arval)*tol:  # also correct for negative arval
            return True
    return False


def Get_nb_files(folder_path, keyword):
    """
    Count the files in a folder whose name contains keyword.

    Parameters
    ----------
    folder_path : str
        Folder to scan (not recursive).
    keyword : str
        Substring searched in each file name.

    Returns
    -------
    int
        Number of matching files.
    """
    return sum(keyword in filename for filename in os.listdir(folder_path))


def Get_indexes_compare_vectors(v1, v2):
    """
    For each value of v2, return the index of the closest value in v1.

    Parameters
    ----------
    v1 : array-like
        Reference vector, searched for the closest values.
    v2 : array-like
        Values to locate in v1.

    Returns
    -------
    numpy.ndarray of int
        Array of len(v2) indexes into v1.
    """
    dist = np.abs(np.subtract.outer(np.asarray(v2, float), np.asarray(v1, float)))  # |v2[j]-v1[i]|, shape (len(v2), len(v1))
    return np.argmin(dist, axis=1)


def Update_df(df, colname, values, position=-1):
    """
    Set a column of a DataFrame in place, creating it if it does not exist.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame modified in place.
    colname : str
        Column name.
    values : scalar or array-like
        Column values.
    position : int, optional
        Position of a new column; -1 (default) appends it as the last column.
        Ignored if the column already exists.
    """
    if colname in df:
        df.loc[:, colname] = values
    else:
        loc = len(df.columns) if position == -1 else position  # insertion position
        df.insert(loc=loc, column=colname, value=values)


def Resample_data_frame_scipy(dfin, factor=10):
    """
    Resample every column of a DataFrame with scipy.signal.resample (Fourier method).

    Parameters
    ----------
    dfin : pandas.DataFrame
        DataFrame to resample (all columns must be numeric).
    factor : float, optional
        Factor by which the number of rows is multiplied (default 10).

    Returns
    -------
    pandas.DataFrame
        Resampled DataFrame with int(len(dfin)*factor) rows.
    """
    resample_len = int(dfin.shape[0]*factor)  # number of rows after resampling
    return pd.DataFrame({col: resample(np.asarray(dfin[col], dtype=float), num=resample_len)
                         for col in dfin.columns})


def Resample_data_frame_linear_regression(dfin, factor=10):
    """
    Resample every column of a DataFrame by linear interpolation between consecutive rows.

    Parameters
    ----------
    dfin : pandas.DataFrame
        DataFrame to resample (all columns must be numeric).
    factor : int, optional
        Number of sub-intervals inserted between two consecutive rows (default 10).

    Returns
    -------
    pandas.DataFrame
        Resampled DataFrame with (len(dfin)-1)*factor+1 rows; original rows are kept.
    """
    n = dfin.shape[0]  # original number of rows
    resample_len = int((n-1)*factor+1)  # number of rows after resampling
    x_new = np.arange(resample_len)/factor  # new positions, in units of original row index
    return pd.DataFrame({col: np.interp(x_new, np.arange(n), np.asarray(dfin[col], dtype=float))
                         for col in dfin.columns})


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------

def Calculate_student_t_confidence_interval(mean, std, n, ci=0.95):
    """
    Confidence interval on a mean estimated from n samples, using Student's
    t-distribution (population standard deviation unknown, estimated from the sample).

    Parameters
    ----------
    mean : float or array-like
        Sample mean.
    std : float or array-like
        Sample standard deviation (computed with ddof=1).
    n : int or array-like
        Number of samples used to compute mean and std (n >= 2).
    ci : float, optional
        Confidence level (default 0.95).

    Returns
    -------
    d : float or numpy.ndarray
        Half-width of the interval: mean ± d.
    err : numpy.ndarray
        [lower, upper] bounds.
    """
    sem = std/np.sqrt(n)  # standard error of the mean
    # Student t interval with n-1 degrees of freedom: mean ± t_{(1+ci)/2, n-1} × s/√n
    # (e.g. ci=0.95, n=10 -> t_{0.975, 9} = 2.262)
    # scipy.stats.t.interval returns the bounds containing a probability ci of a
    # t distribution with df degrees of freedom, centred on loc and scaled by scale
    err = np.array(t.interval(ci, df=np.asarray(n)-1, loc=mean, scale=sem), dtype=float)  # [lower, upper]
    d = (err[1]-err[0])/2  # half-width: mean ± d
    return d, err


def Calculate_normal_confidence_interval(mean, std, n, ci=0.95):
    """
    Confidence interval on a mean estimated from n samples, using the normal
    distribution. Appropriate when std is the known population standard deviation,
    or when n is large (n > ~30), where it converges to the Student t result.

    Parameters
    ----------
    mean : float or array-like
        Sample mean.
    std : float or array-like
        Standard deviation of the population.
    n : int or array-like
        Number of samples used to compute the mean.
    ci : float, optional
        Confidence level (default 0.95).

    Returns
    -------
    d : float or numpy.ndarray
        Half-width of the interval: mean ± d.
    err : numpy.ndarray
        [lower, upper] bounds.
    """
    sem = std/np.sqrt(n)  # standard error of the mean
    # Normal interval: mean ± z_{(1+ci)/2} × σ/√n   (ci=0.95 -> z_{0.975} = 1.960)
    # scipy.stats.norm.interval returns the bounds containing a probability ci of a
    # normal distribution with mean loc and standard deviation scale
    err = np.array(norm.interval(ci, loc=mean, scale=sem), dtype=float)  # [lower, upper]
    d = (err[1]-err[0])/2  # half-width: mean ± d
    return d, err


def Calculate_confidence_interval_from_sample(data, ci=0.95, distribution="student"):
    """
    Mean and confidence interval of a 1-D sample (NaN values are ignored).

    Parameters
    ----------
    data : array-like
        Sample values.
    ci : float, optional
        Confidence level (default 0.95).
    distribution : {"student", "normal"}, optional
        Distribution used for the critical value (default "student").

    Returns
    -------
    mean : float
        Sample mean.
    d : float
        Half-width of the interval: mean ± d.
    err : numpy.ndarray
        [lower, upper] bounds.
    """
    x = np.asarray(data, dtype=float)  # sample as array
    x = x[~np.isnan(x)]  # drop missing values
    n = x.size  # sample size
    mean = x.mean()  # sample mean
    std = x.std(ddof=1)  # sample standard deviation (unbiased variance, n-1 denominator)
    if distribution == "student":
        d, err = Calculate_student_t_confidence_interval(mean, std, n, ci)
    elif distribution == "normal":
        d, err = Calculate_normal_confidence_interval(mean, std, n, ci)
    else:
        raise ValueError(f"distribution must be 'student' or 'normal', not {distribution!r}")
    return mean, d, err


def Calculate_student_t_confidence_interval_curve_fit(popt, pcov, n, ci=0.95):
    """
    Confidence intervals on parameters fitted with scipy.optimize.curve_fit,
    using Student's t-distribution.

    Use with curve_fit's default absolute_sigma=False: pcov is then scaled by the
    residual variance, which is itself estimated from the data, hence Student t
    with n - p degrees of freedom.

    Parameters
    ----------
    popt : array-like
        Fitted parameters returned by curve_fit (p values).
    pcov : array-like
        Covariance matrix returned by curve_fit (p × p).
    n : int
        Number of data points used in the fit (n > p).
    ci : float, optional
        Confidence level (default 0.95).

    Returns
    -------
    d : numpy.ndarray
        Half-width for each parameter: popt[i] ± d[i].
    err : numpy.ndarray
        Array of shape (p, 2): [lower, upper] for each parameter.
    """
    popt = np.asarray(popt, dtype=float)  # fitted parameters
    se = np.sqrt(np.diag(pcov))  # standard error of each parameter (already a standard error: no further /√n)
    dof = n-popt.size  # residual degrees of freedom: data points - fitted parameters
    # Student t interval on each parameter: θ_i ± t_{(1+ci)/2, n-p} × SE(θ_i)
    # scipy.stats.t.interval returns (lower, upper) arrays, one value per parameter
    err = np.column_stack(t.interval(ci, df=dof, loc=popt, scale=se))  # [lower, upper] per parameter
    d = (err[:, 1]-err[:, 0])/2  # half-width: popt[i] ± d[i]
    return d, err


def Calculate_normal_confidence_interval_curve_fit(popt, pcov, ci=0.95):
    """
    Confidence intervals on parameters fitted with scipy.optimize.curve_fit,
    using the normal distribution.

    Use with curve_fit(..., sigma=..., absolute_sigma=True), i.e. when the measurement
    uncertainties are known in absolute terms, or when the number of points is large.

    Parameters
    ----------
    popt : array-like
        Fitted parameters returned by curve_fit (p values).
    pcov : array-like
        Covariance matrix returned by curve_fit (p × p).
    ci : float, optional
        Confidence level (default 0.95).

    Returns
    -------
    d : numpy.ndarray
        Half-width for each parameter: popt[i] ± d[i].
    err : numpy.ndarray
        Array of shape (p, 2): [lower, upper] for each parameter.
    """
    popt = np.asarray(popt, dtype=float)  # fitted parameters
    se = np.sqrt(np.diag(pcov))  # standard error of each parameter
    # Normal interval on each parameter: θ_i ± z_{(1+ci)/2} × SE(θ_i)
    # scipy.stats.norm.interval returns (lower, upper) arrays, one value per parameter
    err = np.column_stack(norm.interval(ci, loc=popt, scale=se))  # [lower, upper] per parameter
    d = (err[:, 1]-err[:, 0])/2  # half-width: popt[i] ± d[i]
    return d, err


# ---------------------------------------------------------------------------
# Graphs
# ---------------------------------------------------------------------------

def Set_graph_params(font="Times New Roman", style="whitegrid"):
    """
    Apply the standard figure format: seaborn theme, colorblind palette, font and sizes.
    Call once before creating figures.

    Parameters
    ----------
    font : str, optional
        Font family used for all text, including math text (default "Times New Roman").
    style : str, optional
        Seaborn style: "whitegrid", "darkgrid", "white", "dark" or "ticks" (default "whitegrid").
    """
    custom_params = {
        'figure.figsize': [16, 10],
        'lines.linewidth': 3,  # width of lines
        'figure.dpi': 100,  # 200 e.g. is really fine, but slower
        'figure.titlesize': 22,
        'axes.titlesize': 20,
        'axes.labelsize': 18,
        'legend.fontsize': 18,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'lines.markersize': 16,
        'axes.xmargin': 0,
        'axes.ymargin': 0,
        # math text ($...$) in the same font as the rest of the figure
        'mathtext.fontset': 'custom',
        'mathtext.rm': font,
        'mathtext.it': f'{font}:italic',
        'mathtext.bf': f'{font}:bold',
        # embed TrueType fonts so text stays editable in pdf/eps files
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    }
    sns.set_theme(style=style, palette="colorblind", font=font, rc=custom_params)


def Save_figure(fig, folder, name, fmt="pdf", **kwargs):
    """
    Save a figure as folder/name.fmt, creating folder if needed.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    folder : str
        Output directory.
    name : str
        File name without extension.
    fmt : str, optional
        File format and extension, e.g. "pdf", "png", "svg" (default "pdf").
    **kwargs
        Passed to fig.savefig (e.g. dpi=300).

    Returns
    -------
    str
        Path of the saved file.
    """
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{name}.{fmt}")  # output file path
    fig.savefig(path, format=fmt, bbox_inches="tight", **kwargs)
    return path


def Limit_datetime_ticks(ax, max_ticks=10):
    """
    Limit the number of datetime ticks on the x axis and format them as dd/mm HH:MM.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes whose x axis holds datetimes.
    max_ticks : int, optional
        Maximum number of ticks (default 10).
    """
    locator = mdates.AutoDateLocator(minticks=3, maxticks=max_ticks)  # tick positions
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %H:%M"))
    ax.figure.autofmt_xdate()


# Line styles as (offset, (on, off, ...)) dash patterns, indexed 0-14
linestyles = [
    (0, ()),                    # 0  solid
    (0, (1, 2)),                # 1  dotted
    (0, (5, 3)),                # 2  dashed
    (0, (3, 3, 1, 3)),          # 3  dash-dot
    (0, (3, 3, 1, 3, 1, 3)),    # 4  dash-dot-dot
    (0, (1, 1)),                # 5  densely dotted
    (0, (5, 1)),                # 6  densely dashed
    (0, (3, 1, 1, 1)),          # 7  densely dash-dot
    (0, (3, 1, 1, 1, 1, 1)),    # 8  densely dash-dot-dot
    (0, (5, 1, 3, 1, 1, 1)),    # 9  long-short-dot
    (0, (3, 3, 1, 3, 1, 1)),    # 10 dash-dot-dot (tight end)
    (0, (1, 4)),                # 11 loosely dotted
    (0, (8, 3)),                # 12 long dashed
    (0, (8, 3, 1, 3)),          # 13 long dash-dot
    (0, (8, 3, 1, 3, 1, 3)),    # 14 long dash-dot-dot
]

# Markers, ordered so that consecutive series are easy to tell apart
markers = ["+", ".", "v", "1", "o", "^", "x", "2", "p", "3", "<", "4", "8", "s", ">", "P", "*", "h", "H", "X", "D", "d", "|", "_", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]


def Wrap_linestyle(index):
    """
    Return a line style from linestyles, cycling back to the start when needed.

    Parameters
    ----------
    index : int
        Series index.
    """
    return linestyles[index % len(linestyles)]


def Wrap_marker(index):
    """
    Return a marker from markers, cycling back to the start when needed.

    Parameters
    ----------
    index : int
        Series index.
    """
    return markers[index % len(markers)]
