#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
A python module sparselab.util

This is a submodule of sparselab. This module saves some common functions,
variables, and data types in the sparselab module.
'''
__author__ = "Kazunori Akiyama"
__version__ = "1.0"
__date__ = "Jan 6 2017"

# a default augument for pandas.Dataframe.to_csv()
args_tocsv = {
    'float_format': r"%22.16e",
    'index': False,
    'index_label': False
}

def matplotlibrc(nrows=1,ncols=1,width=250,height=250):
    import matplotlib

    # Get this from LaTeX using \showthe\columnwidth
    fig_width_pt  = width*ncols
    fig_height_pt = height*nrows
    inches_per_pt = 1.0/72.27               # Convert pt to inch
    fig_width     = fig_width_pt*inches_per_pt  # width in inches
    fig_height    = fig_height_pt*inches_per_pt # height in inches
    fig_size      = [fig_width,fig_height]
    params = {'axes.labelsize': 13,
              'axes.titlesize': 13,
              'text.fontsize' : 15,
              'legend.fontsize': 14,
              'xtick.labelsize': 14,
              'ytick.labelsize': 14,
              'figure.figsize': fig_size,
              'figure.dpi'    : 300
    }
    matplotlib.rcParams.update(params)
