# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 17:55:51 2021

@author: g.flood-page
"""
import os
import numpy as np
import math
import pandas as pd
from scipy.signal import resample
from scipy.stats import t, norm

def Ensure_dir(dir_path, dir_name):  # create directory and check if it already exists
    directory = dir_path+"\\"+dir_name  # os.path.join(dir_path, dir_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def dateparse(s):
    return pd.to_datetime(s, dayfirst=True, errors='coerce', cache=True)

def Eliminate_absurd_values(input_array,
                            max_val,
                            min_val,
                            avg_val=11,
                            cutoff_factor=1.5  # if value is x times smaller than median on avg_val values, replace by nan
                            ):
    """Function to replace values in an array by nan if below min_val, above max_val, or below cutoff_factor * the median of the avg_val nearest values - Function still needs to be tested"""
    avg_shift = avg_val//2  # nb of integers for which data sorting is not applied at start and end of file
    len_array = len(input_array)
    res = np.zeros(len_array, float)
    for i, val in enumerate(input_array):
        if i < avg_shift:  # median mostly to right of first values
            # Eliminating absurd values
            med = np.nanmedian(input_array[0:int(2*avg_shift-i)])
            if input_array[i] < med/cutoff_factor or input_array[i] < min_val or input_array[i] > max_val:
                res[i] = np.nan
            else:
                res[i] = input_array[i]
        elif i >= len_array-avg_shift-1:  # median mostly to left of last values
            # Eliminating absurd values
            med = np.nanmedian(input_array[int(2*len_array-2*avg_shift-i):len_array])
            if input_array[i] < med/cutoff_factor or input_array[i] < min_val or input_array[i] > max_val:
                res[i] = np.nan
            else:
                res[i] = input_array[i]
        else:
            # Eliminating absurd values
            med = np.nanmedian(input_array[i-avg_shift:i+avg_shift])
            if input_array[i] < med/cutoff_factor or input_array[i] < min_val or input_array[i] > max_val:
                res[i] = np.nan
            else:
                res[i] = input_array[i]
    return res


def Get_index_min_value(input_array, ignore_inf=False):
    """Returns the index corresponding to the min of the array"""
    if ignore_inf == True:  # want to ignore -infty
        min_val = np.nanmin(input_array[input_array != -np.inf])
    else:  # -infty is a possible minimum value
        min_val = np.nanmin(input_array)
    for i, val in enumerate(input_array):
        if val == min_val:
            return i
    print("Get_index_min_value failed : could not retrieve index of array min")


def In_with_tol(value, array, tol):
    """Check if value in an array with a relative tolerance of tol arval*(1-tol)<value<arval*(1+tol)"""
    for i, arval in enumerate(array):
        if arval*(1-tol) < value < arval*(1+tol):
            return True
    return False


def Get_nb_files(folder_path, keyword):
    """Returns the number of files in folder containing the keyword"""
    i = 0
    for filename in os.listdir(os.path.join(folder_path)):
        if keyword in filename:
            i += 1
    return i

def Get_indexes_compare_vectors(v1, v2):
    """compares two arrays a1 and a2 - returns indexes in v1 corresponding to values closest to those in v2 """
    temp = np.ones((len(v1), len(v2)), dtype = float)*v2
    return np.argmin(np.abs(temp.T-v1), axis = 1) 

def Update_df(df, colname, values, position = -1):
    """Checks if a column already exists in df before inserting new column - column is either the last one (-1) or given by position"""
    if colname in df:
        df.loc[:, colname]=values
    else:
        if position==-1:
            df.insert(loc=len(df.columns), column = colname, value = values)
        else:
            df.insert(loc=position, column = colname, value = values)

def Resample_data_frame_scipy(dfin, factor = 10):
    """
    Returns a resampled version of a dataframe resampled using the factor "factor"\n
    Parameters : \n
    - df : dataframe that needs resampling\n
    - factor : by which factor to increase the number of data points 
    """
    titles = list(dfin.columns.values)
    resample_len = int(dfin.shape[0]*factor)
    for j in range(dfin.shape[1]): #for every column
        res = resample(np.array(dfin.iloc[:, j], dtype = float), num = resample_len)
        if j==0: #first column
            dfout = pd.DataFrame({
                titles[0]:res
            })
        else:
            Update_df(dfout, titles[j], res)
    return dfout

def Resample_data_frame_linear_regression(dfin, factor = 10):
    """
    Returns a resampled version of a dataframe resampled using the factor "factor"\n
    Parameters : \n
    - df : dataframe that needs resampling\n
    - factor : by which factor to increase the number of data points 
    """
    titles = list(dfin.columns.values)
    resample_len = int((dfin.shape[0]-1)*factor+1)
    for j in range(dfin.shape[1]): #for every column
        new = np.zeros(resample_len, dtype = float)
        for k, vali in enumerate(dfin.iloc[:-1, j]):
            l=0
            slope = (dfin.iloc[k+1, j]-vali)/factor
            for l in range(factor+1):
                new[k*factor+l] = vali+slope*l
        if j==0: #first column
            dfout = pd.DataFrame({
                titles[0]:new
            })
        else:
            Update_df(dfout, titles[j], new)
    return dfout


def Calculate_student_t_confidence_interval_curve_fit(ci, popt, pcov, df):
    """
    Calculates confidence interval from curve_fit fitting process using Student's t-distribution and ci confidence level\n
    Returns [(max-min)/2], [min, max] for each parameter in popt
    """
    stds = np.sqrt(np.diag(pcov)) #array containing std on all parameters
    #df = dfinc.shape[0] #degrees of freedom : nb points -1
    err = [] #confidence intervals
    d = [] #error : (cimax-cimin)/2
    for i in range(popt.shape[0]):
        err.append(np.array(t.interval(ci, df=df, loc=popt[i], scale=stds[i]/np.sqrt(df)), dtype=float))
        d.append((err[i][1]-err[i][0])/2) 
    return  d, err

def Calculate_normal_confidence_interval_curve_fit(ci, popt, pcov, df):
    """
    Calculates confidence interval from curve_fit fitting process using normal distribution and ci confidence level\n
    Returns [(max-min)/2], [min, max] for each parameter in popt
    """
    stds = np.sqrt(np.diag(pcov)) #array containing std on all parameters
    #df = dfinc.shape[0] #degrees of freedom : nb points -1
    err = [] #confidence intervals
    d = [] #error : (cimax-cimin)/2
    for i in range(popt.shape[0]):
        err.append(np.array(norm.interval(ci,loc=popt[i], scale=stds[i]/np.sqrt(df)), dtype=float))
        d.append((err[i][1]-err[i][0])/2) 
    return  d, err

def Calculate_student_t_confidence_interval(ci, mean, std, df):
    """
    Calculates confidence interval from mean and std using Student's t-distribution and ci confidence level\n
    Returns [(max-min)/2], [min, max] for each parameter in popt
    """
    err = np.array(t.interval(ci, df=df, loc=mean, scale=std/np.sqrt(df)), dtype=float)
    d = (err[1]-err[0])/2 
    return  d, err

def Calculate_normal_confidence_interval(ci, mean, std, df):
    """
    Calculates confidence interval from mean and std using Normal-distribution and ci confidence level\n
    Returns [(max-min)/2], [min, max] for each parameter in popt
    """
    err = np.array(norm.interval(ci, loc=mean, scale=std/np.sqrt(df)), dtype=float)
    d = (err[1]-err[0])/2 
    return  d, err