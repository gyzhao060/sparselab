#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
This is a submodule of sparselab for imaging static images.
'''
__author__ = "Sparselab Developer Team"
# -------------------------------------------------------------------------
# Modules
# -------------------------------------------------------------------------
# standard modules
import os
import copy
import collections
import itertools

# numerical packages
import numpy as np
import pandas as pd

# matplotlib
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import NullFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

# internal modules
from . import util

class PTable(pd.DataFrame):
    '''
    '''
    @property
    def _constructor(self):
        return PTable

    @property
    def _constructor_sliced(self):
        return _PSeries

    def update(self, ref="lambsq"):
        '''
        ref=["lambsq", "lambda", "freq"]
        '''
        import astropy.constants
        c_si = astropy.constants.c.si.value

        if ref=="lambsq":
            self["lambda"] = np.sqrt(self["lambsq"])
            self["freq"] = c_si/self["lambda"]
        elif ref=="lambda":
            self["lambsq"] = np.square(self["lambda"])
            self["freq"] = c_si/self["lambda"]
        elif ref=="freq":
            self["lambda"] = c_si/self["freq"]
            self["lambsq"] = np.square(self["lambda"])
        else:
            raise(ValueError, "ref must be either of ['lambsq', 'lambda', 'freq']")
        if "Q" not in self.columns:
            self["Q"] = np.zeros(len(self))
        if "U" not in self.columns:
            self["U"] = np.zeros(len(self))
        self["P"] = np.sqrt(self["Q"]*self["Q"]+self["U"]*self["U"])
        self["chi"] = np.angle(self["Q"]+1j*self["U"], deg=True)/2
        if "sigma" not in self.columns:
            self["sigma"] = np.zeros(len(self))
        self = self[["lambsq", "Q", "U", "P", "chi", "sigma"]].reset_index(drop=True)

    def observe(self, fdftable, sigma=None):
        Ndata = len(self)
        Nrm = len(fdftable)
        RM = np.array(fdftable["RM"])
        P = np.array((fdftable["Q"] + 1j * fdftable["U"])*fdftable["dRM"])
        lambsq = np.array(self["lambsq"])
        dlambsq = np.abs(np.diff(lambsq).mean()/2.)
        #
        A = np.exp(1j*2*lambsq.reshape([Ndata,1]).dot(RM.reshape([1,Nrm])))
        P = A.dot(P)
        #
        self["Q"] = np.real(P)
        self["U"] = np.imag(P)
        self["sigma"] = np.zeros(Ndata)
        if sigma!=None:
            lambsq1 = lambsq - dlambsq/2.
            lambsq2 = lambsq + dlambsq/2.
            factor = 1/np.sqrt(1/np.sqrt(lambsq1)-1/np.sqrt(lambsq2))
            factor /= factor.max()
            self["Q"] += np.random.normal(0,sigma*factor,size=Ndata)
            self["U"] += np.random.normal(0,sigma*factor,size=Ndata)
            self.loc[:, "sigma"] = sigma*factor
        self.update()

    def plot(self,axs=None,ls="none",marker=".",xlim=[None,None],ploterror=False,**plotargs):
        if axs is None:
            fig, axs = plt.subplots(nrows=4,ncols=1,sharex=True)

        ax = axs[0]
        plt.sca(ax)
        if ploterror:
            plt.errorbar(self["lambsq"],self["P"],self["sigma"],
                         ls=ls,marker=marker,**plotargs)
        else:
            plt.plot(self["lambsq"],self["P"],ls=ls,marker=marker,**plotargs)
        plt.ylabel("$P$ (Jy)")
        plt.xlim(xlim)
        plt.ylim(0,)
        ymax = np.max(ax.get_ylim())
        #
        ax = axs[1]
        plt.sca(ax)
        if ploterror:
            plt.errorbar(self["lambsq"],self["chi"],np.rad2deg(self["sigma"]/self["P"]),
                         ls=ls,marker=marker,**plotargs)
        else:
            plt.plot(self["lambsq"],self["chi"],ls=ls,marker=marker,**plotargs)
        plt.ylabel("$\chi$ (deg)")
        plt.xlim(xlim)
        plt.ylim(-90,90)
        ax.set_yticks(np.arange(-90,91,10), minor=True)
        ax.set_yticks(np.arange(-90,91,30), minor=False)
        #
        ax = axs[2]
        plt.sca(ax)
        if ploterror:
            plt.errorbar(self["lambsq"],self["Q"],self["sigma"],
                         ls=ls,marker=marker,**plotargs)
        else:
            plt.plot(self["lambsq"],self["Q"],ls=ls,marker=marker,**plotargs)
        plt.axhline(0, ls="--", color="black")
        plt.ylabel("$Q$ (Jy)")
        plt.xlim(xlim)
        plt.ylim(-ymax,ymax)
        #
        ax = axs[3]
        plt.sca(ax)
        if ploterror:
            plt.errorbar(self["lambsq"],self["U"],self["sigma"],
                         ls=ls,marker=marker,**plotargs)
        else:
            plt.plot(self["lambsq"],self["U"],
                     ls=ls,marker=marker,**plotargs)
        plt.axhline(0, ls="--", color="black")
        plt.ylabel("$U$ (Jy)")
        plt.xlabel("$\lambda ^2$ (m$^{2}$)")
        plt.xlim(xlim)
        plt.ylim(-ymax,ymax)
        return axs

    def gencvtables(self, nfold=10, seed=0):
        '''
        This method generates self sets for N-fold cross varidations.

        Args:
            nfolds (int): the number of folds
            seed (int): the seed number of pseudo ramdam numbers

        Returns:

        '''
        Nself = self.shape[0]  # Number of self points
        Nval = Nself // nfold    # Number of Varidation self

        # Make shuffled self
        shuffled = self.sample(Nself,
                               replace=False,
                               random_state=np.int64(seed))

        # Name
        out = collections.OrderedDict()
        for icv in xrange(nfold):
            trainkeyname = "t%d" % (icv)
            validkeyname = "v%d" % (icv)
            if Nval * (icv + 1) == Nself:
                train = shuffled.loc[:Nval * icv, :]
            else:
                train = pd.concat([shuffled.loc[:Nval * icv, :],
                                   shuffled.loc[Nval * (icv + 1):, :]])
            valid = shuffled[Nval * icv:Nval * (icv + 1)]
            out[trainkeyname] = train
            out[validkeyname] = valid
        return out

    def rmsf_size(self):
        return 2.*np.sqrt(3)/(self["lambsq"].max()-self["lambsq"].min())

    def to_csv(self, filename, float_format=r"%22.16e", index=False,
               index_label=False, **args):
        '''
        Output table into csv files using pd.DataFrame.to_csv().
        Default parameter will be
          float_format=r"%22.16e"
          index=False
          index_label=False.
        see DocStrings for pd.DataFrame.to_csv().

        Args:
            filename (string or filehandle): output filename
            **args: other arguments of pd.DataFrame.to_csv()
        '''
        super(PTable, self).to_csv(filename,
                                   index=index, index_label=index_label, **args)

class FDFTable(pd.DataFrame):
    '''
    '''
    @property
    def _constructor(self):
        return FDFTable

    @property
    def _constructor_sliced(self):
        return _FDFSeries

    def blankFDF(self, dRM=1., refRM=0., Nrm=100):
        # Create Blank image
        RM = dRM*(np.arange(Nrm)-Nrm/2.) - refRM
        #
        Q = np.zeros(Nrm)
        U = np.zeros(Nrm)
        #
        self["RM"] = RM
        self["Q"] = Q
        self["U"] = U
        #
        self.update()

    def dirtybeam(self, ptable):
        outfdf = copy.deepcopy(self)

        Ndata = len(ptable)
        Nrm = len(self)
        RM = np.array(self["RM"])
        lambsq = np.array(ptable["lambsq"])
        P = np.arange(Ndata)
        P[:] = 1
        #
        A = np.exp(1j*2*RM.reshape([Nrm,1]).dot(lambsq.reshape([1,Ndata])))
        P = A.dot(P)
        #
        outfdf["Q"] = np.real(P)
        outfdf["U"] = np.imag(P)
        outfdf.update()
        outfdf.loc[:,["P","Q","U"]]/=outfdf["P"].max()
        return outfdf

    def dirtyfdf(self, ptable):
        outfdf = copy.deepcopy(self)

        Ndata = len(ptable)
        Nrm = len(self)
        RM = np.array(self["RM"])
        lambsq = np.array(ptable["lambsq"])
        P = np.array(ptable["Q"]+1j*ptable["U"])
        P2 = np.arange(Ndata)
        P2[:] = 1

        A = np.exp(-1j*2*RM.reshape([Nrm,1]).dot(lambsq.reshape([1,Ndata])))
        P = A.dot(P)/np.abs(A.dot(P2)).max()

        outfdf["Q"] = np.real(P)
        outfdf["U"] = np.imag(P)
        outfdf.update()
        return outfdf

    def add_point_source(self,RM=0,Pamp=0.,Pxi=0.):
        import copy
        outdata = copy.deepcopy(self)
        #
        if RM < np.min(outdata["RM"]) or RM > np.max(outdata["RM"]):
            print("WARNING: The specified position RM=%f is outside of the given image"%(RM))
            return outdata

        idx = np.argmin(np.abs(RM - np.asarray(outdata["RM"])))
        P = Pamp * np.exp(1j*(2*np.deg2rad(Pxi)))
        outdata.loc[idx,["Q"]] += np.real(P)
        outdata.loc[idx,["U"]] += np.imag(P)
        outdata.update()
        return outdata

    def add_rectlinear_source(self,RMrange=[0,0],Pamp=0.,Pxi=0.):
        import copy
        outdata = copy.deepcopy(self)
        #
        if np.max(RMrange) < np.min(self["RM"]) or np.min(RMrange) > np.max(self["RM"]):
            print("The specified rectlinear region is outside of the given image")

        idxs = (self["RM"] > np.min(RMrange)) * (self["RM"] < np.max(RMrange))
        P = Pamp * np.exp(1j*(2*np.deg2rad(Pxi)))
        outdata.loc[idxs,["Q"]] += np.real(P)
        outdata.loc[idxs,["U"]] += np.imag(P)
        outdata.update()
        return outdata

    def add_gaussian_source(self,RM=0,fwhm=20,Pamp=0.,Pxi=0.):
        import copy
        outdata = copy.deepcopy(self)

        sigma = fwhm/2./np.sqrt(2*np.log(2))
        P = Pamp * np.exp(1j*(2*np.deg2rad(Pxi))) * np.exp(-np.square(self["RM"]-RM)/2./sigma/sigma)
        outdata.loc[:,"Q"] += np.real(P)
        outdata.loc[:,"U"] += np.imag(P)
        outdata.update()
        return outdata

    def update(self):
        self["dRM"] = np.zeros(len(self))
        self.loc[:, "dRM"] = np.mean(np.diff(self["RM"]))
        self["P"] = np.sqrt(self["Q"]*self["Q"]+self["U"]*self["U"])
        self["chi"] = np.angle(self["Q"]+1j*self["U"], deg=True)/2
        self = self[["RM", "dRM", "Q", "U", "P", "chi"]].reset_index(drop=True)

    def plot(self,axs=None,ls="-",marker="None",xlim=[None,None],label="",**plotargs):
        if axs is None:
            fig, axs = plt.subplots(nrows=4,ncols=1,sharex=True)

        ax = axs[0]
        plt.sca(ax)
        plt.plot(self["RM"],self["P"],ls=ls,marker=marker,**plotargs)
        plt.ylabel("$P$ (Jy rad$^{-1}$ m$^{2}$)")
        plt.xlim(xlim)
        plt.ylim(0,)
        ymax = np.max(ax.get_ylim())

        ax = axs[1]
        plt.sca(ax)
        chi = self["chi"]
        chi[self["P"] < np.max(self["P"]) * 0.01]=0
        plt.plot(self["RM"],chi,ls=ls,marker=marker,**plotargs)
        plt.ylabel("$\chi$ (deg)")
        plt.xlim(xlim)
        plt.ylim(-90,90)
        ax.set_yticks(np.arange(-90,91,10), minor=True)
        ax.set_yticks(np.arange(-90,91,30), minor=False)

        ax = axs[2]
        plt.sca(ax)
        plt.plot(self["RM"],self["Q"],ls=ls,marker=marker,**plotargs)
        plt.axhline(0, ls="--", color="black")
        plt.ylabel("$Q$ (Jy rad$^{-1}$ m$^{2}$)")
        plt.xlim(xlim)
        plt.ylim(-ymax,ymax)
        #
        ax = axs[3]
        plt.sca(ax)
        plt.plot(self["RM"],self["U"],ls=ls,marker=marker,**plotargs)
        plt.axhline(0, ls="--", color="black")
        plt.ylabel("$U$ (Jy rad$^{-1}$ m$^{2}$)")
        plt.xlabel("$\phi$ (rad m$^{-2}$)")
        plt.xlim(xlim)
        plt.ylim(-ymax,ymax)
        return axs

    def to_csv(self, filename, float_format=r"%22.16e", index=False,
               index_label=False, **args):
        '''
        Output table into csv files using pd.DataFrame.to_csv().
        Default parameter will be
          float_format=r"%22.16e"
          index=False
          index_label=False.
        see DocStrings for pd.DataFrame.to_csv().

        Args:
            filename (string or filehandle): output filename
            **args: other arguments of pd.DataFrame.to_csv()
        '''
        super(FDFTable, self).to_csv(filename,
                                     index=index, index_label=index_label, **args)


def read_fdftable(filename, **args):
    '''
    This fuction loads FDF table from an input csv file using pd.read_csv().

    Args:
      filename:
        str, pathlib.Path, py._path.local.LocalPath or any object with a read()
        method (such as a file handle or StringIO)

    Returns:
      uvdata.VisTable object
    '''
    table = FDFTable(pd.read_csv(filename, **args))
    table.update()
    return table

def read_ptable(filename, **args):
    '''
    This fuction loads FDF table from an input csv file using pd.read_csv().

    Args:
      filename:
        str, pathlib.Path, py._path.local.LocalPath or any object with a read()
        method (such as a file handle or StringIO)

    Returns:
      uvdata.VisTable object
    '''
    table = PTable(pd.read_csv(filename, **args))
    table.update()
    return table

class _FDFSeries(pd.Series):

    @property
    def _constructor(self):
        return _FDFSeries

    @property
    def _constructor_expanddim(self):
        return _FDFTable

class _PSeries(pd.Series):

    @property
    def _constructor(self):
        return _PSeries

    @property
    def _constructor_expanddim(self):
        return _PTable
