#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
This is a submodule of sparselab. This module is a wrapper of C library of
MFISTA in src/mfista
'''
# -------------------------------------------------------------------------
# Modules
# -------------------------------------------------------------------------
# standard modules
import ctypes
import os
import copy

# numerical packages
import numpy as np
import pandas as pd

# internal LoadLibrary
from sparselab import uvdata

#-------------------------------------------------------------------------
# Default Parameters
#-------------------------------------------------------------------------
mfistaprm = {}
mfistaprm["dftsign"]=1
mfistaprm["cinit"]=10000.0

#-------------------------------------------------------------------------
# CLASS
#-------------------------------------------------------------------------
class _MFISTA_RESULT(ctypes.Structure):
    '''
    This class is for loading structured variables for results
    output from MFISTA.
    '''
    _fields_ = [
        ("M",ctypes.c_int),
        ("N",ctypes.c_int),
        ("NX",ctypes.c_int),
        ("NY",ctypes.c_int),
        ("N_active",ctypes.c_int),
        ("maxiter",ctypes.c_int),
        ("ITER",ctypes.c_int),
        ("nonneg",ctypes.c_int),
        ("lambda_l1",ctypes.c_double),
        ("lambda_tv",ctypes.c_double),
        ("lambda_tsv",ctypes.c_double),
        ("sq_error",ctypes.c_double),
        ("mean_sq_error",ctypes.c_double),
        ("l1cost",ctypes.c_double),
        ("tvcost",ctypes.c_double),
        ("tsvcost",ctypes.c_double),
        ("looe",ctypes.c_double),
        ("Hessian_positive",ctypes.c_double),
        ("finalcost",ctypes.c_double)
    ]

    def __init__(self,M,N):
        self.M = ctypes.c_int(M)
        self.N = ctypes.c_int(N)
        self.NX = ctypes.c_int(0)
        self.NY = ctypes.c_int(0)
        self.N_active = ctypes.c_int(0)
        self.maxiter = ctypes.c_int(0)
        self.ITER = ctypes.c_int(0)
        self.nonneg = ctypes.c_int(0)
        self.lambda_l1 = ctypes.c_double(0)
        self.lambda_tv = ctypes.c_double(0)
        self.lambda_tsv = ctypes.c_double(0)
        self.sq_error = ctypes.c_double(0.0)
        self.mean_sq_error = ctypes.c_double(0.0)
        self.l1cost = ctypes.c_double(0.0)
        self.tvcost = ctypes.c_double(0.0)
        self.tsvcost = ctypes.c_double(0.0)
        self.looe = ctypes.c_double(0.0)
        self.Hessian_positive = ctypes.c_double(0.0)
        self.finalcost = ctypes.c_double(0.0)


#-------------------------------------------------------------------------
# Wrapping Function
#-------------------------------------------------------------------------
def mfista_imaging(
    initimage, vistable,
    lambl1=-1., lambtv=-1, lambtsv=-1,
    normlambda=True, nonneg=True,
    totalflux=None, fluxconst=False,
    istokes=0, ifreq=0):

    # Total Flux constraint: Sanity Check
    dofluxconst = False
    if ((totalflux is None) and (fluxconst is True)):
        print("Error: No total flux is specified, although you set fluxconst=True.")
        print("       You need to set the total flux constraint by totalflux.")
        return -1

    # LOOE flags
    calclooe=False
    if calclooe:
        looe_flag=1
    else:
        looe_flag=-1

    # Nonneg condition
    if nonneg:
        nonneg_flag=1
    else:
        nonneg_flag=-1

    # get initial images
    Iin = np.float64(initimage.data[istokes, ifreq])
    Iout = copy.deepcopy(Iin)

    # size of images
    Nx = np.int32(initimage.header["nx"])
    Ny = np.int32(initimage.header["ny"])
    Nyx = Nx * Ny

    # pixel coordinates
    x, y = initimage.get_xygrid(twodim=True, angunit="rad")
    x = np.float64(x)
    y = np.float64(y)

    # reshape image and coordinates
    Iin = Iin.reshape(Nyx)
    x = x.reshape(Nyx)
    y = y.reshape(Nyx)

    # Add zero-uv point for total flux constraints.
    if dofluxconst:
        print("Total Flux Constraint: set to %g" % (totalflux))
        totalfluxdata = {
            'u': [0.],
            'v': [0.],
            'amp': [totalflux],
            'phase': [0.],
            'sigma': [1.]
        }
        totalfluxdata = uvdata.VisTable(totalfluxdata)
        fcvtable = pd.concat([totalfluxdata, vistable], ignore_index=True)
    else:
        print("Total Flux Constraint: disabled.")
        fcvtable = vistable.copy()

    # Pick up data sets
    u = np.asarray(fcvtable["u"], dtype=np.float64)
    v = np.asarray(fcvtable["v"], dtype=np.float64)
    Vamp = np.asarray(fcvtable["amp"], dtype=np.float64)
    Vpha = np.deg2rad(np.asarray(fcvtable["phase"], dtype=np.float64))
    Verr = np.asarray(fcvtable["sigma"], dtype=np.float64)
    Vfcv = np.concatenate([Vamp*np.cos(Vpha)/Verr, Vamp*np.sin(Vpha)/Verr])
    M = Vfcv.size
    del Vamp, Vpha

    # scale lambda
    if normlambda:
        # Guess Total Flux
        if totalflux is None:
            fluxscale = vistable["amp"].max()
            print("Flux Scaling Factor for lambda: The expected total flux is not given.")
            print("                                The scaling factor will be %g" % (fluxscale))
        else:
            fluxscale = np.float64(totalflux)
            print("Flux Scaling Factor for lambda: The scaling factor will be %g" % (fluxscale))
        lambl1_sim = lambl1 * M / 2. / fluxscale / 2.
        lambtv_sim = lambtv * M / 2. / fluxscale / 4. / 2.
        lambtsv_sim = lambtsv * M / 2. / fluxscale**2 / 4.
    else:
        lambl1_sim = lambl1 / 2.
        lambtv_sim = lambtv / 2.
        lambtsv_sim = lambtsv / 2.

    # make an MFISTA_result object
    mfista_result = _MFISTA_RESULT(M,Nyx)
    mfista_result.lambda_l1 = lambl1_sim
    mfista_result.lambda_tv = lambtv_sim
    mfista_result.lambda_tsv = lambtsv_sim


    # get pointor to variables
    c_double_p = ctypes.POINTER(ctypes.c_double)
    Iin_p = Iin.ctypes.data_as(c_double_p)
    Iout_p = Iout.ctypes.data_as(c_double_p)
    x_p = x.ctypes.data_as(c_double_p)
    y_p = y.ctypes.data_as(c_double_p)
    u_p = u.ctypes.data_as(c_double_p)
    v_p = v.ctypes.data_as(c_double_p)
    Vfcv_p = Vfcv.ctypes.data_as(c_double_p)
    Verr_p = Verr.ctypes.data_as(c_double_p)
    mfista_result_p = ctypes.byref(mfista_result)

    # Load libmfista.so
    libmfistapath = os.path.dirname(os.path.abspath(__file__))
    libmfistapath = os.path.join(libmfistapath,"libmfista.so")
    libmfista = ctypes.cdll.LoadLibrary(libmfistapath)
    libmfista.mfista_imaging(
        # Images
        Iin_p, Iout_p, x_p, y_p, ctypes.c_int(Nx), ctypes.c_int(Ny),
        # UV coordinates and Full Complex Visibility
        u_p, v_p, Vfcv_p, Verr_p, ctypes.c_int(M), ctypes.c_int(mfistaprm["dftsign"]),
        # Imaging Parameters
        ctypes.c_double(lambl1_sim), ctypes.c_double(lambtv_sim),
        ctypes.c_double(lambtsv_sim),
        ctypes.c_int(nonneg_flag), ctypes.c_int(looe_flag),
        ctypes.c_double(mfistaprm["cinit"]),
        # Results
        mfista_result_p)

    # Get Results
    outimage = copy.deepcopy(initimage)
    outimage.data[istokes, ifreq] = Iout.reshape(Ny, Nx)
    outimage.update_fits()

    return outimage
