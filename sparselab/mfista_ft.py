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
import collections
import itertools

# matplotlib
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import NullFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

# numerical packages
import numpy as np
import pandas as pd

# internal LoadLibrary
from . import util

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
        ("looe_m",ctypes.c_double),
        ("looe_std",ctypes.c_double),
        ("Hessian_positive",ctypes.c_double),
        ("finalcost",ctypes.c_double),
        ("comp_time",ctypes.c_double),
        ("model",ctypes.POINTER(ctypes.c_double)),
        ("residual",ctypes.POINTER(ctypes.c_double))
    ]

    def __init__(self,M,N):
        c_double_p = ctypes.POINTER(ctypes.c_double)

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
        self.looe_m = ctypes.c_double(0.0)
        self.looe_std = ctypes.c_double(0.0)
        self.Hessian_positive = ctypes.c_double(0.0)
        self.finalcost = ctypes.c_double(0.0)
        self.residarr = np.zeros(M)
        self.residual = self.residarr.ctypes.data_as(c_double_p)
        self.modelarr = np.zeros(M)
        self.model = self.modelarr.ctypes.data_as(c_double_p)

#-------------------------------------------------------------------------
# Wrapping Function
#-------------------------------------------------------------------------
def mfista_ft(
    fdfin, ptable,
    lambl1=-1., lambtv=-1, lambtsv=-1,
    normlambda=True, looe=False,
    normfactor=None,
    istokes=0, ifreq=0):

    # LOOE flags
    if looe:
        looe_flag=1
    else:
        looe_flag=-1

    # get initial Faraday Dispersion Function
    N = len(fdfin)*2
    RM = np.asarray(fdfin["RM"],dtype=np.float64)
    dRM = np.asarray(fdfin["dRM"],dtype=np.float64)
    FDFQ = np.asarray(fdfin["Q"],dtype=np.float64)
    FDFU = np.asarray(fdfin["U"],dtype=np.float64)
    FDFPin = np.concatenate([FDFQ*dRM,FDFU*dRM])
    FDFPout = copy.deepcopy(FDFPin)

    # get ptable
    M = len(ptable)*2
    lambsq = np.asarray(ptable["lambsq"],dtype=np.float64)
    PQ = np.asarray(ptable["Q"],dtype=np.float64)
    PU = np.asarray(ptable["U"],dtype=np.float64)
    Perr = np.asarray(ptable["sigma"],dtype=np.float64)
    PP = np.concatenate([PQ/Perr,PU/Perr])
    PP *= 2/np.sqrt(M)
    print(np.square(PP).sum())

    # scale lambda
    if normlambda:
        # Guess Total Flux
        if normfactor is None:
            fluxscale = np.max(np.sqrt(PQ*PQ+PU*PU))
            print("Flux Scaling Factor for lambda: The expected total flux is not given.")
            print("                                The scaling factor will be %g" % (fluxscale))
        else:
            fluxscale = np.float64(normfactor)
            print("Flux Scaling Factor for lambda: The scaling factor will be %g" % (fluxscale))
        lambl1_sim = lambl1 / fluxscale
        lambtv_sim = lambtv / fluxscale / 4.
        lambtsv_sim = lambtsv / fluxscale**2 / 4.
    else:
        lambl1_sim = lambl1
        lambtv_sim = lambtv
        lambtsv_sim = lambtsv

    #print(lambl1_sim,lambtv_sim,lambtsv_sim)
    if lambl1_sim < 0: lambl1_sim = 0.
    if lambtv_sim < 0: lambtv_sim = 0.
    if lambtsv_sim < 0: lambtsv_sim = 0.
    #print(lambl1_sim,lambtv_sim,lambtsv_sim)

    # make an MFISTA_result object
    mfista_result = _MFISTA_RESULT(M,N)
    mfista_result.lambda_l1 = lambl1_sim
    mfista_result.lambda_tv = lambtv_sim
    mfista_result.lambda_tsv = lambtsv_sim

    # get pointor to variables
    c_double_p = ctypes.POINTER(ctypes.c_double)
    FDFPin_p = FDFPin.ctypes.data_as(c_double_p)
    FDFPout_p = FDFPout.ctypes.data_as(c_double_p)
    RM_p = RM.ctypes.data_as(c_double_p)
    PP_p = PP.ctypes.data_as(c_double_p)
    Perr_p = Perr.ctypes.data_as(c_double_p)
    lambsq_p = lambsq.ctypes.data_as(c_double_p)
    mfista_result_p = ctypes.byref(mfista_result)

    # Load libmfista.so
    libmfistapath = os.path.dirname(os.path.abspath(__file__))
    libmfistapath = os.path.join(libmfistapath,"libmfista_dft.so")
    libmfista = ctypes.cdll.LoadLibrary(libmfistapath)
    libmfista.mfista_ft(
        #Full Complex Visibility
        PP_p,
        # Array Size
        ctypes.c_int(M), ctypes.c_int(N),
        # UV coordinates and Errors
        lambsq_p, RM_p, Perr_p,
        # Imaging Parameters
        ctypes.c_double(lambl1_sim), ctypes.c_double(lambtv_sim),
        ctypes.c_double(lambtsv_sim),
        ctypes.c_double(mfistaprm["cinit"]), FDFPin_p, FDFPout_p,
        ctypes.c_int(0), ctypes.c_int(looe_flag),
        # Results
        mfista_result_p)

    # Get Results
    fdfout = copy.deepcopy(fdfin)
    fdfout.loc[:,"Q"] = FDFPout[0:N/2]/fdfout["dRM"]
    fdfout.loc[:,"U"] = FDFPout[N/2:]/fdfout["dRM"]
    fdfout.update()
    return fdfout

def mfista_stats(
    fdfin, ptable,
    lambl1=-1., lambtv=-1, lambtsv=-1,
    normlambda=True, looe=False,
    normfactor=None, fulloutput=False,
    istokes=0, ifreq=0):

    # LOOE flags
    if looe:
        looe_flag=1
    else:
        looe_flag=-1

    # get initial Faraday Dispersion Function
    N = len(fdfin)*2
    RM = np.asarray(fdfin["RM"],dtype=np.float64)
    dRM = np.asarray(fdfin["dRM"],dtype=np.float64)
    FDFQ = np.asarray(fdfin["Q"],dtype=np.float64)
    FDFU = np.asarray(fdfin["U"],dtype=np.float64)
    FDFPin = np.concatenate([FDFQ*dRM,FDFU*dRM])
    FDFPout = copy.deepcopy(FDFPin)

    # get ptable
    M = len(ptable)*2
    lambsq = np.asarray(ptable["lambsq"],dtype=np.float64)
    PQ = np.asarray(ptable["Q"],dtype=np.float64)
    PU = np.asarray(ptable["U"],dtype=np.float64)
    Perr = np.asarray(ptable["sigma"],dtype=np.float64)
    PP = np.concatenate([PQ/Perr,PU/Perr])
    PP *= 2/np.sqrt(M)
    print(np.square(PP).sum())

    # scale lambda
    if normlambda:
        # Guess Total Flux
        if normfactor is None:
            fluxscale = np.max(np.sqrt(PQ*PQ+PU*PU))
            print("Flux Scaling Factor for lambda: The expected total flux is not given.")
            print("                                The scaling factor will be %g" % (fluxscale))
        else:
            fluxscale = np.float64(normfactor)
            print("Flux Scaling Factor for lambda: The scaling factor will be %g" % (fluxscale))
        lambl1_sim = lambl1 / fluxscale
        lambtv_sim = lambtv / fluxscale / 4.
        lambtsv_sim = lambtsv / fluxscale**2 / 4.
    else:
        lambl1_sim = lambl1
        lambtv_sim = lambtv
        lambtsv_sim = lambtsv

    #print(lambl1_sim,lambtv_sim,lambtsv_sim)
    if lambl1_sim < 0: lambl1_sim = 0.
    if lambtv_sim < 0: lambtv_sim = 0.
    if lambtsv_sim < 0: lambtsv_sim = 0.
    #print(lambl1_sim,lambtv_sim,lambtsv_sim)

    # make an MFISTA_result object
    mfista_result = _MFISTA_RESULT(M,N)
    mfista_result.lambda_l1 = lambl1_sim
    mfista_result.lambda_tv = lambtv_sim
    mfista_result.lambda_tsv = lambtsv_sim

    # get pointor to variables
    c_double_p = ctypes.POINTER(ctypes.c_double)
    FDFPin_p = FDFPin.ctypes.data_as(c_double_p)
    RM_p = RM.ctypes.data_as(c_double_p)
    PP_p = PP.ctypes.data_as(c_double_p)
    Perr_p = Perr.ctypes.data_as(c_double_p)
    lambsq_p = lambsq.ctypes.data_as(c_double_p)
    mfista_result_p = ctypes.byref(mfista_result)

    # Load libmfista.so
    libmfistapath = os.path.dirname(os.path.abspath(__file__))
    libmfistapath = os.path.join(libmfistapath,"libmfista_dft.so")
    libmfista = ctypes.cdll.LoadLibrary(libmfistapath)
    libmfista.mfista_ft_results(
        #Full Complex Visibility
        PP_p,
        # Array Size
        ctypes.c_int(M), ctypes.c_int(N),
        # UV coordinates and Errors
        lambsq_p, RM_p, Perr_p,
        # Imaging Parameters
        ctypes.c_double(lambl1_sim), ctypes.c_double(lambtv_sim),
        ctypes.c_double(lambtsv_sim),
        ctypes.c_double(mfistaprm["cinit"]), FDFPin_p,
        ctypes.c_int(0), ctypes.c_int(looe_flag),
        # Results
        mfista_result_p)

    stats = collections.OrderedDict()
    # Cost and Chisquares
    stats["cost"] = mfista_result.finalcost
    stats["chisq"] = mfista_result.sq_error * M / 2 / 2
    stats["rchisq"] = mfista_result.sq_error / 2
    stats["looe_m"] = mfista_result.looe_m
    stats["looe_std"] = mfista_result.looe_std

    # Regularization functions
    if lambl1 > 0:
        stats["lambl1"] = lambl1
        stats["lambl1_sim"] = lambl1_sim
        stats["l1"] = mfista_result.l1cost
        stats["l1cost"] = mfista_result.l1cost*lambl1_sim
    else:
        stats["lambl1"] = 0.
        stats["lambl1_sim"] = 0.
        stats["l1"] = 0.
        stats["l1cost"] = 0.

    if lambtv > 0:
        stats["lambtv"] = lambtv
        stats["lambtv_sim"] = lambtv_sim
        stats["tv"] = mfista_result.tvcost
        stats["tvcost"] = mfista_result.tvcost*lambtv_sim
    else:
        stats["lambtv"] = 0.
        stats["lambtv_sim"] = 0.
        stats["tv"] = 0.
        stats["tvcost"] = 0.

    if lambtsv > 0:
        stats["lambtsv"] = lambtsv
        stats["lambtsv_sim"] = lambtsv_sim
        stats["tsv"] = mfista_result.tsvcost
        stats["tsvcost"] = mfista_result.tsvcost*lambtsv_sim
    else:
        stats["lambtsv"] = 0.
        stats["lambtsv_sim"] = 0.
        stats["tsv"] = 0.
        stats["tsvcost"] = 0.

    if fulloutput:
        # full complex visibilities
        model = mfista_result.modelarr
        resid = mfista_result.residarr
        rmod = model[0:M/2] * np.sqrt(M/4) * Perr
        imod = model[M/2:M] * np.sqrt(M/4) * Perr
        rred = resid[0:M/2] * np.sqrt(M/4)
        ired = resid[M/2:M] * np.sqrt(M/4)
        stats["Pmod"] = np.sqrt(Qmod*Qmod + Umod*Umod)
        stats["Qmod"] = rmod
        stats["Umod"] = imod
        stats["chimod"] = np.angle(rmod+1j*imod, deg=True)/2
        stats["Pred"] = np.sqrt(rred*rred + ired*ired)
        stats["Qred"] = rred
        stats["Ured"] = ired
        stats["chired"] = np.angle(rred+1j*ired, deg=True)/2
    return stats

def mfista_plots(fdfin, ptable, filename=None,
                 plotargs={'ms': 1., }):
    isinteractive = plt.isinteractive()
    backend = matplotlib.rcParams["backend"]

    if isinteractive:
        plt.ioff()
        matplotlib.use('Agg')

    nullfmt = NullFormatter()

    # Get model data
    modelptable = ptable.copy()
    modelptable.observe(fdfin)

    # Save fdf
    if filename is not None:
        util.matplotlibrc(nrows=4, ncols=2, width=400, height=150)
    else:
        matplotlib.rcdefaults()

    fig, axs = plt.subplots(nrows=4, ncols=2, sharex=False)
    fdfin.plot(axs=axs[:,0],color="red")
    ptable.plot(axs=axs[:,1],color="black", ploterror=True)
    modelptable.plot(axs=axs[:,1], color="red")
    if filename is not None:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()

    if isinteractive:
        plt.ion()
        matplotlib.use(backend)


def mfista_pipeline(
        fdfin,
        ptable,
        ftprm={},
        lambl1s=[-1.],
        lambtvs=[-1.],
        lambtsvs=[-1.],
        workdir="./",
        skip=False,
        sumtablefile="summary.csv",
        docv=False,
        seed=1,
        nfold=10,
        cvsumtablefile="summary.cv.csv"):
    '''
    A pipeline imaging function using static_dft_imaging and related fucntions.

    Args:
        initfdf (imdata.IMFITS object):
            initial fdf
        ftprm (dict-like; default={}):
            parameter sets for each imaging
        workdir (string; default = "./"):
            The directory where fdfs and summary files will be output.
        sumtablefile (string; default = "summary.csv"):
            The name of the output csv file that summerizes results.
        docv (boolean; default = False):
            Do cross validation
        seed (integer; default = 1):
            Random seed to make CV data sets.
        nfold (integer; default = 10):
            Number of folds in CV.
        cvsumtablefile (string; default = "cvsummary.csv"):
            The name of the output csv file that summerizes results of CV.

    Returns:
        sumtable:
            pd.DataFrame table summerising statistical quantities of each
            parameter set.
        cvsumtable (if docv=True):
            pd.DataFrame table summerising results of cross validation.
    '''
    if not os.path.isdir(workdir):
        os.makedirs(workdir)

    cvworkdir = os.path.join(workdir,"cv")
    if docv:
        if not os.path.isdir(cvworkdir):
            os.makedirs(cvworkdir)

    # Lambda Parameters
    lambl1s = -np.sort(-np.asarray(lambl1s))
    lambtvs = -np.sort(-np.asarray(lambtvs))
    lambtsvs = -np.sort(-np.asarray(lambtsvs))
    nl1 = len(lambl1s)
    ntv = len(lambtvs)
    ntsv = len(lambtsvs)

    # Summary Data
    sumtable = pd.DataFrame()
    if docv:
        cvsumtable = pd.DataFrame()
        ptables = ptable.gencvtables(nfold=nfold, seed=seed)

    # Start Imaging
    fdfold=fdfin
    for itsv, itv, il1 in itertools.product(np.arange(ntsv),
                                            np.arange(ntv),
                                            np.arange(nl1)):
        header = "tsv%02d.tv%02d.l1%02d" % (itsv, itv, il1)

        # output
        ftprm["lambl1"] = lambl1s[il1]
        ftprm["lambtv"] = lambtvs[itv]
        ftprm["lambtsv"] = lambtsvs[itsv]

        # Imaging and Plotting Results
        filename = header + ".fdf.csv"
        filename = os.path.join(workdir, filename)
        if (skip is False) or (os.path.isfile(filename) is False):
            fdfnew = mfista_ft(fdfold, ptable, **ftprm)
            fdfnew.to_csv(filename)
        else:
            fdfnew = ft.read_fdftable(filename)
        fdfold = fdfnew

        filename = header + ".model.csv"
        filename = os.path.join(workdir, filename)
        modelptable = ptable.copy()
        modelptable.observe(fdfnew)
        modelptable.to_csv(filename)


        filename = header + ".fit.pdf"
        filename = os.path.join(workdir, filename)
        mfista_plots(fdfnew, pdata, filename=filename)
        newstats = mfista_stats(fdfnew, pdata, fulloutput=False, **ftprm)

        # Make Summary
        tmpsum = collections.OrderedDict()
        tmpsum["itsv"] = itsv
        tmpsum["itv"] = itv
        tmpsum["il1"] = il1
        for key in newstats.keys():
            tmpsum[key] = newstats[key]

        # Cross Validation
        if docv:
            # Initialize Summary Table
            #    add keys
            tmpcvsum = pd.DataFrame()
            tmpcvsum["icv"] = np.arange(nfold)
            tmpcvsum["itsv"] = np.zeros(nfold, dtype=np.int32)
            tmpcvsum["itv"] = np.zeros(nfold, dtype=np.int32)
            tmpcvsum["il1"] = np.zeros(nfold, dtype=np.int32)
            tmpcvsum["lambtsv"] = np.zeros(nfold, dtype=np.float64)
            tmpcvsum["lambtv"] = np.zeros(nfold, dtype=np.float64)
            tmpcvsum["lambl1"] = np.zeros(nfold, dtype=np.float64)
            tmpcvsum["chisq"] = np.zeros(nfold, dtype=np.float64)
            tmpcvsum["rchisq"] = np.zeros(nfold, dtype=np.float64)
            tmpcvsum["tchisq"] = np.zeros(nfold, dtype=np.float64)
            tmpcvsum["trchisq"] = np.zeros(nfold, dtype=np.float64)
            tmpcvsum["vchisq"] = np.zeros(nfold, dtype=np.float64)
            tmpcvsum["vrchisq"] = np.zeros(nfold, dtype=np.float64)

            #    initialize some columns
            tmpcvsum.loc[:, "itsv"] = itsv
            tmpcvsum.loc[:, "itv"] = itv
            tmpcvsum.loc[:, "il1"] = il1
            tmpcvsum.loc[:, "lambtsv"] = lambtsvs[itsv]
            tmpcvsum.loc[:, "lambtv"] = lambtvs[itv]
            tmpcvsum.loc[:, "lambl1"] = lambl1s[il1]

            #   Imaging parameters
            cvftprm = copy.deepcopy(ftprm)
            cvftprm["looe"]=False

            #  N-fold CV
            for icv in np.arange(nfold):
                # Header of output files
                cvheader = header+".cv%02d" % (icv)

                # fdf Training Data
                filename = cvheader + ".t.csv"
                filename = os.path.join(cvworkdir, filename)
                if (skip is False) or (os.path.isfile(filename) is False):
                    cvfdfnew = mfista_ft(newfdf, ptables["t%d"%(icv)], **cvftprm)
                    cvfdfnew.to_csv(filename)
                else:
                    cvfdfnew = ft.read_fdftable(filename)

                # save model
                filename = header + ".t.model.csv"
                filename = os.path.join(workdir, filename)
                modelptable = ptables["t%d"%(icv)].copy()
                modelptable.observe(cvfdfnew)
                modelptable.to_csv(filename)

                # Make Plots
                filename = cvheader + ".t.fit.pdf"
                filename = os.path.join(cvworkdir, filename)
                mfista_plots(cvfdfnew, ptables["t%d"%(icv)], filename=filename)

                # Check Training data
                trainstats = mfista_stats(cvfdfnew, ptables["t%d"%(icv)])

                # Check validating data
                # save model
                filename = header + ".v.model.csv"
                filename = os.path.join(workdir, filename)
                modelptable = ptables["v%d"%(icv)].copy()
                modelptable.observe(cvfdfnew)
                modelptable.to_csv(filename)

                # Make Plots
                filename = cvheader + ".v.fit.pdf"
                filename = os.path.join(cvworkdir, filename)
                mfista_plots(cvfdfnew, ptables["v%d"%(icv)], filename=filename)

                #   Check Statistics
                validstats = mfista_stats(cvfdfnew, ptables["v%d"%(icv)])

                #   Save Results
                tmpcvsum.loc[icv, "tchisq"] = trainstats["chisq"]
                tmpcvsum.loc[icv, "trchisq"] = trainstats["rchisq"]

                tmpcvsum.loc[icv, "vchisq"] = validstats["chisq"]
                tmpcvsum.loc[icv, "vrchisq"] = validstats["rchisq"]

            # add current cv summary to the log file.
            cvsumtable = pd.concat([cvsumtable,tmpcvsum], ignore_index=True)
            cvsumtable.to_csv(os.path.join(workdir, cvsumtablefile))

            # Average Varidation Errors and memorized them
            tmpsum["tchisq"] = np.mean(tmpcvsum["tchisq"])
            tmpsum["trchisq"] = np.mean(tmpcvsum["trchisq"])
            tmpsum["vchisq"] = np.mean(tmpcvsum["vchisq"])
            tmpsum["vrchisq"] = np.mean(tmpcvsum["vrchisq"])

        # Output Summary Table
        tmptable = pd.DataFrame([tmpsum.values()], columns=tmpsum.keys())
        sumtable = pd.concat([sumtable, tmptable], ignore_index=True)
        sumtable.to_csv(os.path.join(workdir, sumtablefile))

    if docv:
        return sumtable, cvsumtable
    else:
        return sumtable
