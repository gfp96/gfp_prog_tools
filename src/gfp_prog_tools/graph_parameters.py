# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 18:44:27 2021

@author: g.flood-page
"""

import matplotlib.pyplot as plt
from matplotlib import dates as mdates
import seaborn as sns
from collections import OrderedDict

def _limit_datetime_ticks(ax, max_ticks=10):
        locator = mdates.AutoDateLocator(minticks=3, maxticks=max_ticks)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %H:%M"))
        ax.figure.autofmt_xdate()

def Set_graph_params():
    graph_font = {'family': 'DejaVu Sans',
                  'weight': 'normal',
                  'size': 18}
    plt.rcParams['figure.figsize'] = [16, 10]
    plt.rc('font', **graph_font)  # set font parameters
    plt.rcParams['lines.linewidth'] = 3  # set width of lines
    plt.rcParams['figure.dpi'] = 100  # 200 e.g. is really fine, but slower
    plt.rcParams['figure.titlesize'] = 22
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['legend.fontsize'] = 18
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.labelsize'] = 18
    #plt.rcParams["figure.autolayout"] = True
    #plt.rcParams['figure.constrained_layout.use'] = True
    plt.rcParams['lines.markersize'] = 10

    custom_params = {
        'figure.figsize': [16, 10],
        #'font.family': 'sans-serif',  # set font parameters
        'lines.linewidth':3,  # set width of lines
        'figure.dpi':100,  # 200 e.g. is really fine, but slower
        'figure.titlesize':22,
        'axes.titlesize':20,
        'axes.labelsize':18,
        'legend.fontsize':18,
        'xtick.labelsize':18,
        'ytick.labelsize':18,
        'lines.markersize':16,
        "axes.xmargin" : 0,
        "axes.ymargin" : 0
    }
    sns.set_theme(rc=custom_params)
    sns.set_style("whitegrid")


#list of linestyles
linestyles = OrderedDict(
    [(0,               (0, ())),
    (1,              (0, (1, 2))),
    (5,      (0, (1, 1))),
    (2,              (0, (5, 3))),
    (6,      (0, (5, 1))),
    (3,          (0, (3, 3, 1, 3))),
    (7,  (0, (3, 1, 1, 1))),
    (4,         (0, (3, 3, 1, 3, 1, 3))),
    (8, (0, (3, 1, 1, 1, 1, 1))),
    (9, (0, (5, 1, 3, 1, 1, 1))),
    (10, (0, (3, 3, 1, 3, 1, 1))),
    (11, (0, (3, 1, 1, 1, 1, 1))),
    (12, (0, (3, 1, 1, 1, 1, 1))),
    (13, (0, (3, 1, 1, 1, 1, 1))),
    (14, (0, (3, 1, 1, 1, 1, 1)))]
    )

def _wrap_linestyle(index):
    """Return a linestyle from the global list, cycling back to the start when needed."""
    return linestyles[index % len(linestyles)]

markers = ["+", ".","v","1","o","^","x","2","p","3","<","4","8","s",">","P","*","h","H","X","D","d","|","_",0,1,2,3,4,5,6,7,8,9,10,11]

def _wrap_marker(index):
    """Return a marker from the global list, cycling back to the start when needed."""
    return markers[index % len(markers)]
    # sns.set_theme(rc('font', **graph_font)
    # plt.rc('font', **graph_font)  # set font parameters
    # plt.rcParams['lines.linewidth'] = 3  # set width of lines
    # plt.rcParams['figure.dpi'] = 100  # 200 e.g. is really fine, but slower
    # plt.rcParams['figure.titlesize'] = 22
    # plt.rcParams['axes.titlesize'] = 20
    # plt.rcParams['axes.labelsize'] = 18
    # plt.rcParams['legend.fontsize'] = 18
    # plt.rcParams['xtick.labelsize'] = 18
    # plt.rcParams['ytick.labelsize'] = 18
    # plt.rcParams["figure.autolayout"] = True
    # #plt.rcParams['figure.constrained_layout.use'] = True
    # plt.rcParams['lines.markersize'] = 10
    # sns.set(rc={'figure.figsize':(11.7,8.27)})
    # #plt.rc('text', usetex=True)
