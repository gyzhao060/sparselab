#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
A python module sparselab.uvdata

This is a submodule of sparselab handling various types of Visibility data sets.
'''
#-------------------------------------------------------------------------------
# Modules
#-------------------------------------------------------------------------------
# standard modules
import copy
import itertools


# numerical packages
import numpy as np
import scipy.special as ss
from scipy import optimize
from scipy import linalg
import pandas as pd
import xarray as xr
import astropy.constants as ac
import astropy.time as at
import astropy.io.fits as pyfits


# matplotlib
import matplotlib.pyplot as plt


# internal
import sparselab.imdata as imdata


#-------------------------------------------------------------------------------
# Classes for UVFITS FILE
#-------------------------------------------------------------------------------
class UVFITS():
    '''
    This is a class to load uvfits data and edit data sets before making tables 
    for imaging.
    '''
    def __init__(self, infile):
        '''
        Load an uvfits file. Currently, this function can read only 
        single-source uvfits file. The data will be uv-sorted.

        Args:
          infile (string): input uvfits file

        Returns:
          uvdata.UVFITS object
        '''
        self.read_uvfits(infile)
        self.uvsort()


    def read_uvfits(self, infile):
        '''
        Read the uvfits file. Currently, this function can read only 
        single-source uvfits file.

        Args:
          infile (string): input uvfits file
        '''


        #-----------------------------------------------------------------------
        # open uvfits file
        #-----------------------------------------------------------------------
        hdulist = pyfits.open(infile)
        print('CONTENTS OF INPUT FITS FILE:')
        hdulist.info()

        hduinfos = hdulist.info(output=False)
        for hduinfo in hduinfos:
            idx = hduinfo[0]
            if hduinfo[1] == "PRIMARY":
                grouphdu = hdulist[idx]
            elif hduinfo[1] == "AIPS NX":
                aipsnx = hdulist[idx]
            elif hduinfo[1] == "AIPS FQ":
                aipsfq = hdulist[idx]
            elif hduinfo[1] == "AIPS AN":
                aipsan = hdulist[idx]
        if not 'grouphdu' in locals():
            print("[Error]   %s does not contain the Primary HDU" % (infile))

        if not 'aipsfq' in locals():
            print("[Error]   %s does not contain AIPS FQ table" % (infile))
        else:
            self.aipsfq = aipsfq.copy()

        if not 'aipsan' in locals():
            print("[WARNING] %s does not have any AIPS AN tables" % (infile))
        else:
            self.aipsan = aipsan.copy()

        # Save Group HDU Header
        self.header = grouphdu.header.copy()


        #-----------------------------------------------------------------------
        # read random parameters
        #-----------------------------------------------------------------------
        pars = grouphdu.data.parnames
        firstdate = 0
        Npars = len(pars)
        for ipar in np.arange(Npars):
            par = pars[ipar]
            if par.find('UU') == 0:
                # FITS stores uvw coordinates in sec.
                usec = np.float64(grouphdu.data.par(ipar))
            elif par.find('VV') == 0:
                vsec = np.float64(grouphdu.data.par(ipar))
            elif par.find('WW') == 0:
                wsec = np.float64(grouphdu.data.par(ipar))
            elif par.find('BASELINE') == 0:
                bl = grouphdu.data.par(ipar)  # st1 * 256 + st2
                st1 = np.int64(bl // 256)
                st2 = np.int64(bl % 256)
            elif par.find('DATE') == 0:
                if firstdate == 0:
                    jd = np.float64(grouphdu.data.par(ipar))
                    firstdate += 1
                elif firstdate == 1:
                    jd += np.float64(grouphdu.data.par(ipar))
                    firstdate += 1
            elif par.find('INTTIM') == 0:
                integ = grouphdu.data.par(ipar)  # integration time
        if (not 'usec' in locals()) or (not 'vsec' in locals()) or (not 'wsec' in locals()) or \
           (not 'bl' in locals()) or (not 'jd' in locals()):
            print(
                "[Error] %s does not contain required random parameters in the Primary HDU" % (infile))


        #-----------------------------------------------------------------------
        # Make Coordinates
        #-----------------------------------------------------------------------
        coord = {}

        # UV Coordinate, etc
        uvsec = np.sqrt(usec * usec + vsec * vsec)
        coord["usec"] = ("data", np.asarray(
            usec, dtype=np.float64))  # U coordinate in sec
        coord["vsec"] = ("data", np.asarray(
            vsec, dtype=np.float64))  # V coordinate in sec
        coord["wsec"] = ("data", np.asarray(
            wsec, dtype=np.float64))  # W coordinate in sec
        coord["uvsec"] = ("data", np.asarray(
            uvsec, dtype=np.float64))  # UV Distance in sec
        coord["st1"] = ("data", np.asarray(
            st1, dtype=np.int64))  # Station ID 1
        coord["st2"] = ("data", np.asarray(
            st2, dtype=np.int64))  # Station ID 2
        # Original Baseline ID in FITS: st1 * 256 + st2
        coord["baseline"] = ("data", np.asarray(bl, dtype=np.int64))
        if "integ" in locals():
            coord["integ"] = ("data", np.float64(integ))  # integration time

        # Time Tag
        timeobj = at.Time(np.float64(jd), format='jd', scale='utc')
        datetime = timeobj.datetime
        gsthour = timeobj.sidereal_time(
            kind="mean", longitude="greenwich", model=None).hour
        coord["jd"] = ("data", np.asarray(jd, dtype=np.float64))
        coord["datetime"] = ("data", datetime)
        coord["gsthour"] = ("data", np.asarray(gsthour, dtype=np.float64))

        # Stokes parameter
        stokes = (np.arange(grouphdu.header['NAXIS3']) - grouphdu.header['CRPIX3'] +
                  1) * grouphdu.header['CDELT3'] + grouphdu.header['CRVAL3']
        coord["stokes"] = ("stokes", np.asarray(
            np.around(stokes), dtype=np.float64))

        # Frequency Parameter
        freqch = (np.arange(grouphdu.header['NAXIS4']) - grouphdu.header['CRPIX4'] +
                  1) * grouphdu.header['CDELT4'] + grouphdu.header['CRVAL4']
        #   Get IF Freq
        freqifdata = aipsfq.data['IF FREQ']
        if len(freqifdata.shape) == 1:
            freqif = aipsfq.data['IF FREQ']
        else:
            freqif = aipsfq.data['IF FREQ'][0]
        #   Get Obs Frequency and calc UVW
        freq = np.zeros([grouphdu.header['NAXIS5'], grouphdu.header['NAXIS4']])
        u = np.zeros([grouphdu.header['GCOUNT'],
                      grouphdu.header['NAXIS5'], grouphdu.header['NAXIS4']])
        v = np.zeros([grouphdu.header['GCOUNT'],
                      grouphdu.header['NAXIS5'], grouphdu.header['NAXIS4']])
        w = np.zeros([grouphdu.header['GCOUNT'],
                      grouphdu.header['NAXIS5'], grouphdu.header['NAXIS4']])
        uv = np.zeros([grouphdu.header['GCOUNT'],
                       grouphdu.header['NAXIS5'], grouphdu.header['NAXIS4']])
        for iif, ich in itertools.product(np.arange(grouphdu.header['NAXIS5']),
                                np.arange(grouphdu.header['NAXIS4'])):
            freq[iif, ich] = freqif[iif] + freqch[ich]
            u[:, iif, ich] = freq[iif, ich] * usec[:]
            v[:, iif, ich] = freq[iif, ich] * vsec[:]
            w[:, iif, ich] = freq[iif, ich] * wsec[:]
            uv[:, iif, ich] = freq[iif, ich] * uvsec[:]
        coord["freqch"] = ("freqch", np.asarray(freqch, dtype=np.float64))
        coord["freqif"] = ("freqif", np.asarray(freqif, dtype=np.float64))
        coord["freq"] = (("freqif", "freqch"),
                         np.asarray(freq, dtype=np.float64))
        coord["u"] = (("data", "freqif", "freqch"),
                      np.asarray(u, dtype=np.float64))
        coord["v"] = (("data", "freqif", "freqch"),
                      np.asarray(v, dtype=np.float64))
        coord["w"] = (("data", "freqif", "freqch"),
                      np.asarray(w, dtype=np.float64))
        coord["uv"] = (("data", "freqif", "freqch"),
                       np.asarray(uv, dtype=np.float64))

        # RA and Dec
        ra = (np.arange(grouphdu.header['NAXIS6']) - grouphdu.header['CRPIX6'] +
              1) * grouphdu.header['CDELT6'] + grouphdu.header['CRVAL6']
        dec = (np.arange(grouphdu.header['NAXIS7']) - grouphdu.header['CRPIX7'] +
               1) * grouphdu.header['CDELT7'] + grouphdu.header['CRVAL7']
        coord["ra"] = ("ra",  np.asarray(ra, dtype=np.float64))
        coord["dec"] = ("dec", np.asarray(dec, dtype=np.float64))

        # Reset Index
        self.data = xr.DataArray(grouphdu.data.data,
                                 coords=coord,
                                 dims=["data", "dec", "ra", "freqif", "freqch", "stokes", "complex"])

        # Close hdu
        hdulist.close()
        print("")

    def uvsort(self):
        '''
        Check station IDs of each visibility and switch its order if "st1" > "st2".
        Then, data will be TB-sorted.
        '''
        # check station IDs
        select = np.asarray(self.data["st1"] > self.data["st2"])
        if True in select:
            self.data.usec.loc[select] *= -1
            self.data.vsec.loc[select] *= -1
            self.data.wsec.loc[select] *= -1
            self.data.u.loc[select, :, :] *= -1
            self.data.v.loc[select, :, :] *= -1
            self.data.w.loc[select, :, :] *= -1
            self.data.baseline.loc[select] = 256 * \
                self.data.st2.loc[select] + self.data.st1.loc[select]
            dammy = self.data.st2.loc[select]
            self.data.st2.loc[select] = self.data.st1.loc[select]
            self.data.st1.loc[select] = dammy
            self.data.loc[select, :, :, :, :, :, 1] *= - \
                1  # frip imaginary part of visibilities

            Nselect = len(np.where(select)[0])
            print("Station IDs of %d data points are flipped due to their wrong orders (st1 > st2)." % (
                Nselect))
        else:
            print("Station IDs have correct orders (st1 < st2). ")

        # TB-sord data
        idx1 = np.argsort(self.data["baseline"])
        idx2 = np.argsort(self.data["jd"][idx1])
        idx = idx1[idx2]
        check = idx1 == np.arange(self.data.shape[1])
        if False in check:
            print("Data are not TB sorted. Sorting data....")
            self.data = self.data.loc[idx, :, :, :, :, :, :]
            print("Data sort was finished!")
        else:
            print("Data are TB sorted correctly.")
        print("")

    def make_vistable(self, flag=True):
        '''
        Convert visibility data to a two dimentional table.

        Args:
          flag (boolean): 
            if flag=True, data with weights <= 0 or sigma <=0 will be ignored.

        Returns:
          uvdata.VisTable object
        '''
        outdata = VisTable()

        # Get size of data
        Ndata, Ndec, Nra, Nif, Nch, Nstokes, Ncomp = self.data.shape

        # Get time
        # DOY, HH, MM, SS
        yday = at.Time(np.float64(self.data["jd"]), format='jd', scale='utc').yday
        year = np.zeros(Ndata, dtype=np.int32)
        doy = np.zeros(Ndata, dtype=np.int32)
        hour = np.zeros(Ndata, dtype=np.int32)
        minute = np.zeros(Ndata, dtype=np.int32)
        sec = np.zeros(Ndata, dtype=np.int32)
        for idata in np.arange(Ndata):
            time = yday[idata].split(":")
            year[idata] = np.int32(time[0])
            doy[idata] = np.int32(time[1])
            hour[idata] = np.int32(time[2])
            minute[idata] = np.int32(time[3])
            sec[idata] = np.int32(np.float64(time[4]))

        for idec, ira, iif, ich, istokes in itertools.product(np.arange(Ndec),
                                                    np.arange(Nra),
                                                    np.arange(Nif),
                                                    np.arange(Nch),
                                                    np.arange(Nstokes)):
            tmpdata = VisTable()

            # Time
            tmpdata["jd"] = np.float64(self.data["jd"])
            tmpdata["year"] = np.int32(year)
            tmpdata["doy"] = np.int32(doy)
            tmpdata["hour"] = np.int32(hour)
            tmpdata["min"] = np.int32(minute)
            tmpdata["sec"] = np.int32(sec)

            # Frequecny
            tmpdata["freq"] = np.zeros(Ndata, dtype=np.float32)
            tmpdata.loc[:, "freq"] = np.float64(self.data["freq"][iif, ich])

            # Stokes ID
            tmpdata["stokesid"] = np.zeros(Ndata, dtype=np.int32)
            if Nstokes == 1:
                tmpdata.loc[:, "stokesid"] = np.int32(self.data["stokes"])
            else:
                tmpdata.loc[:, "stokesid"] = np.int32(
                    self.data["stokes"][istokes])

            # band/if id, frequency
            tmpdata["bandid"] = np.zeros(Ndata, dtype=np.int32)
            tmpdata.loc[:, "bandid"] = np.int32(ich)
            tmpdata["ifid"] = np.zeros(Ndata, dtype=np.int32)
            tmpdata.loc[:, "ifid"] = np.int32(iif)
            tmpdata["ch"] = tmpdata["ifid"] + tmpdata["bandid"] * Nif

            # uvw
            tmpdata["u"] = np.float64(self.data["u"][:, iif, ich])
            tmpdata["v"] = np.float64(self.data["v"][:, iif, ich])
            tmpdata["w"] = np.float64(self.data["w"][:, iif, ich])
            tmpdata["uvdist"] = np.float64(self.data["uv"][:, iif, ich])

            # station number
            tmpdata["st1"] = np.int32(self.data["st1"])
            tmpdata["st2"] = np.int32(self.data["st2"])

            visreal = np.float64(
                self.data.data[:, idec, ira, iif, ich, istokes, 0])
            visimag = np.float64(
                self.data.data[:, idec, ira, iif, ich, istokes, 1])
            visweig = np.float64(
                self.data.data[:, idec, ira, iif, ich, istokes, 2])
            tmpdata["amp"] = np.sqrt(visreal * visreal + visimag * visimag)
            tmpdata["phase"] = np.rad2deg(np.arctan2(visimag, visreal))
            tmpdata["weight"] = visweig
            tmpdata["sigma"] = np.sqrt(1. / visweig)

            outdata = pd.concat([outdata, tmpdata])

        if flag:
            select = outdata["weight"] > 0
            select *= outdata["sigma"] > 0
            select *= np.isnan(outdata["weight"]) == False
            select *= np.isnan(outdata["sigma"]) == False
            select *= np.isinf(outdata["weight"]) == False
            select *= np.isinf(outdata["sigma"]) == False
            outdata = outdata.loc[select, :].reset_index(drop=True)

        return outdata

    #---------------------------------------------------------------------------
    # Edit UV fits files
    #---------------------------------------------------------------------------
    def select_stokes(self, stokes="I"):
        '''
        Pick up single polarization data

        Args: 
          stokes (string; default="I"):
            Output stokes parameters.
            Availables are ["I", "Q", "U", "V", "LL", "RR", "RL", "LR"].

        Output: uvdata.UVFITS object
        '''
        # get stokes data
        stokesids = np.asarray(self.data["stokes"], dtype=np.int64)

        # create output data
        outfits = copy.deepcopy(self)
        if stokes == "I":
            if (1 in stokesids):  # I <- I
                print("Stokes I data will be copied from the input data")
                outfits.data = self.data[:, :, :, :, :, stokesids == 1, :]
                outfits.header["CRPIX3"] = 1
                outfits.header["CRVAL3"] = 1
                outfits.header["CDELT3"] = 1
                outfits.header["CROTA3"] = 0
            elif (-1 in stokesids) and (-2 in stokesids):  # I <- (RR + LL)/2
                print("Stokes I data will be calculated from input RR and LL data")
                outfits.data = _bindstokes(
                    self.data, stokes=1, stokes1=-1, stokes2=-2, factr1=0.5, factr2=0.5)
                outfits.header["CRPIX3"] = 1
                outfits.header["CRVAL3"] = 1
                outfits.header["CDELT3"] = 1
                outfits.header["CROTA3"] = 0
            elif (-1 in stokesids):  # I <- RR
                print("Stokes I data will be copied from input RR data")
                outfits.data = self.data[:, :, :, :, :, stokesids == -1, :]
                outfits.header["CRPIX3"] = 1
                outfits.header["CRVAL3"] = -1
                outfits.header["CDELT3"] = -1
                outfits.header["CROTA3"] = 0
            elif (-2 in stokesids):  # I <- LL
                print("Stokes I data will be copied from input LL data")
                outfits.data = self.data[:, :, :, :, :, stokesids == -2, :]
                outfits.header["CRPIX3"] = 1
                outfits.header["CRVAL3"] = -2
                outfits.header["CDELT3"] = -1
                outfits.header["CROTA3"] = 0
            elif (-5 in stokesids) and (-6 in stokesids):  # I <- (XX + YY)/2
                print("Stokes I data will be calculated from input XX and YY data")
                outfits.data = _bindstokes(
                    self.data, stokes=1, stokes1=-5, stokes2=-6, factr1=0.5, factr2=0.5)
                outfits.data["stokes"] = 1
                outfits.header["CRPIX3"] = 1
                outfits.header["CRVAL3"] = 1
                outfits.header["CDELT3"] = 1
                outfits.header["CROTA3"] = 0
            elif (-5 in stokesids):  # I <- XX
                print("Stokes I data will be copied from input XX data")
                outfits.data = self.data[:, :, :, :, :, stokesids == -5, :]
                outfits.header["CRPIX3"] = 1
                outfits.header["CRVAL3"] = -5
                outfits.header["CDELT3"] = -1
                outfits.header["CROTA3"] = 0
            elif (-6 in stokesids):  # I <- YY
                print("Stokes I data will be copied from input YY data")
                outfits.data = self.data[:, :, :, :, :, stokesids == -6, :]
                outfits.header["CRPIX3"] = 1
                outfits.header["CRVAL3"] = -6
                outfits.header["CDELT3"] = -1
                outfits.header["CROTA3"] = 0
            else:
                print(
                    "[WARNING] No data are available to calculate Stokes %s" % (stokes))
        elif stokes == "Q":
            if (2 in stokesids):  # Q <- Q
                print("Stokes Q data will be copied from the input data")
                outfits.data = self.data[:, :, :, :, :, stokesids == 2, :]
                outfits.header["CRPIX3"] = 1
                outfits.header["CRVAL3"] = 2
                outfits.header["CDELT3"] = 1
                outfits.header["CROTA3"] = 0
            elif (-3 in stokesids) and (-4 in stokesids):  # Q <- (RL + LR)/2
                print("Stokes Q data will be calculated from input RL and LR data")
                outfits.data = _bindstokes(
                    self.data, stokes=2, stokes1=-3, stokes2=-4, factr1=0.5, factr2=0.5)
                outfits.data["stokes"] = 2
                outfits.header["CRPIX3"] = 1
                outfits.header["CRVAL3"] = 2
                outfits.header["CDELT3"] = 1
                outfits.header["CROTA3"] = 0
            elif (-5 in stokesids) and (-6 in stokesids):  # Q <- (XX - YY)/2
                print("Stokes Q data will be calculated from input XX and YY data")
                outfits.data = _bindstokes(
                    self.data, stokes=2, stokes1=-5, stokes2=-6, factr1=0.5, factr2=-0.5)
                outfits.data["stokes"] = 2
                outfits.header["CRPIX3"] = 1
                outfits.header["CRVAL3"] = 2
                outfits.header["CDELT3"] = 1
                outfits.header["CROTA3"] = 0
            else:
                print(
                    "[WARNING] No data are available to calculate Stokes %s" % (stokes))
        elif stokes == "U":
            if (3 in stokesids):  # V <- V
                print("Stokes U data will be copied from the input data")
                outfits.data = self.data[:, :, :, :, :, stokesids == 2, :]
                outfits.header["CRPIX3"] = 1
                outfits.header["CRVAL3"] = 3
                outfits.header["CDELT3"] = 1
                outfits.header["CROTA3"] = 0
            elif (-3 in stokesids) and (-4 in stokesids):  # U <- (RL - LR)/2i = (- RL + LR)i/2
                print("Stokes U data will be calculated from input RL and LR data")
                outfits.data = _bindstokes(
                    self.data, stokes=3, stokes1=-3, stokes2=-4, factr1=-0.5j, factr2=0.5j)
                outfits.data["stokes"] = 3
                outfits.header["CRPIX3"] = 1
                outfits.header["CRVAL3"] = 3
                outfits.header["CDELT3"] = 1
                outfits.header["CROTA3"] = 0
            elif (-7 in stokesids) and (-8 in stokesids):  # U <- (XY + YX)/2
                print("Stokes U data will be calculated from input XX and YY data")
                outfits.data = _bindstokes(
                    self.data, stokes=3, stokes1=-7, stokes2=-8, factr1=0.5, factr2=0.5)
                outfits.data["stokes"] = 3
                outfits.header["CRPIX3"] = 1
                outfits.header["CRVAL3"] = 3
                outfits.header["CDELT3"] = 1
                outfits.header["CROTA3"] = 0
            else:
                print(
                    "[WARNING] No data are available to calculate Stokes %s" % (stokes))
        elif stokes == "V":
            if (4 in stokesids):  # V <- V
                print("Stokes V data will be copied from the input data")
                outfits.data = self.data[:, :, :, :, :, stokesids == 4, :]
                outfits.header["CRPIX3"] = 1
                outfits.header["CRVAL3"] = 4
                outfits.header["CDELT3"] = 1
                outfits.header["CROTA3"] = 0
            elif (-1 in stokesids) and (-2 in stokesids):  # V <- (RR - LL)/2
                print("Stokes V data will be calculated from input RR and LL data")
                outfits.data = _bindstokes(
                    self.data, stokes=4, stokes1=-1, stokes2=-2, factr1=0.5, factr2=-0.5)
                outfits.data["stokes"] = 4
                outfits.header["CRPIX3"] = 1
                outfits.header["CRVAL3"] = 4
                outfits.header["CDELT3"] = 1
                outfits.header["CROTA3"] = 0
            elif (-7 in stokesids) and (-8 in stokesids):  # V <- (XY - YX)/2i = (-XY + YX)/2
                print("Stokes V data will be calculated from input XX and YY data")
                outfits.data = _bindstokes(
                    self.data, stokes=4, stokes1=-7, stokes2=-8, factr1=-0.5j, factr2=0.5j)
                outfits.data["stokes"] = 4
                outfits.header["CRPIX3"] = 1
                outfits.header["CRVAL3"] = 4
                outfits.header["CDELT3"] = 1
                outfits.header["CROTA3"] = 0
            else:
                print(
                    "[WARNING] No data are available to calculate Stokes %s" % (stokes))
        elif stokes == "RR":
            if (-1 in stokesids):  # V <- V
                print("Stokes RR data will be copied from the input data")
                outfits.data = self.data[:, :, :, :, :, stokesids == -1, :]
                outfits.header["CRPIX3"] = 1
                outfits.header["CRVAL3"] = -1
                outfits.header["CDELT3"] = 1
                outfits.header["CROTA3"] = 0
            else:
                print(
                    "[WARNING] No data are available to calculate Stokes %s" % (stokes))
        elif stokes == "LL":
            if (-2 in stokesids):  # V <- V
                print("Stokes LL data will be copied from the input data")
                outfits.data = self.data[:, :, :, :, :, stokesids == -2, :]
                outfits.header["CRPIX3"] = 1
                outfits.header["CRVAL3"] = -2
                outfits.header["CDELT3"] = 1
                outfits.header["CROTA3"] = 0
            else:
                print(
                    "[WARNING] No data are available to calculate Stokes %s" % (stokes))
        elif stokes == "RL":
            if (-3 in stokesids):  # V <- V
                print("Stokes RL data will be copied from the input data")
                outfits.data = self.data[:, :, :, :, :, stokesids == -3, :]
                outfits.header["CRPIX3"] = 1
                outfits.header["CRVAL3"] = -3
                outfits.header["CDELT3"] = 1
                outfits.header["CROTA3"] = 0
            else:
                print(
                    "[WARNING] No data are available to calculate Stokes %s" % (stokes))
        elif stokes == "LR":
            if (-4 in stokesids):  # V <- V
                print("Stokes LR data will be copied from the input data")
                outfits.data = self.data[:, :, :, :, :, stokesids == -4, :]
                outfits.header["CRPIX3"] = 1
                outfits.header["CRVAL3"] = -4
                outfits.header["CDELT3"] = 1
                outfits.header["CROTA3"] = 0
            else:
                print(
                    "[WARNING] No data are available to calculate Stokes %s" % (stokes))
        else:
            print(
                "[WARNING] Currently Stokes %s is not supported in this function." % (stokes))

        return outfits

    def weightcal(self, dofreq=0, solint=120., minpoint=2):
        '''
        This method will recalculate sigmas and weights of data from scatter
        in full complex visibilities over specified frequency and time segments.

        Arguments:
          self (uvarray.uvfits object):
            input uvfits data

          dofreq (int; default = 0):
            Parameter for multi-frequency data.
              dofreq = 0: calculate weights and sigmas over IFs and channels
              dofreq = 1: calculate weights and sigmas over channels at each IF
              dofreq = 2: calculate weights and sigmas at each IF and Channel

          solint (float; default = 120.):
            solution interval in sec

        Output: uvdata.UVFITS object
        '''
        # Default Averaging alldata
        doif = True
        doch = True
        if np.int64(dofreq) > 0:
            doif = False
        if np.int64(dofreq) > 1:
            doch = False

        # Save and Return re-weighted uv-data
        outfits = copy.deepcopy(self)

        # Get size of data
        Ndata, Ndec, Nra, Nif, Nch, Nstokes, Ncomp = self.data.shape

        # Get unix
        unix = at.Time(self.data["jd"], format="jd", scale="utc").unix
        baseline = np.asarray(self.data["baseline"], dtype=np.int64)

        if doif == True and doch == True:
            for idata in np.arange(Ndata):
                #if idata%1000 == 0: print("%d / %d"%(idata,Ndata))
                dataidx1 = np.where(np.abs(unix - unix[idata]) < solint)
                dataidx2 = np.where(baseline[dataidx1] == baseline[idata])
                seldata = self.data.data[dataidx1][dataidx2]
                for idec, ira, istokes in itertools.product(np.arange(Ndec),
                                                  np.arange(Nra),
                                                  np.arange(Nstokes)):
                    vreal = seldata[:, idec, ira, :, :, istokes, 0]
                    vimaj = seldata[:, idec, ira, :, :, istokes, 1]
                    vweig = seldata[:, idec, ira, :, :, istokes, 2]

                    Ntime = np.prod(vreal.shape)
                    vreal = vreal.reshape(Ntime)
                    vimaj = vimaj.reshape(Ntime)
                    vweig = vweig.reshape(Ntime)

                    select = vweig > 0
                    select *= (np.isnan(vweig) == False)
                    select *= (np.isinf(vweig) == False)
                    select = np.where(select)
                    vreal = vreal[select]
                    vimaj = vimaj[select]
                    vweig = vweig[select]
                    Nsel = len(vreal)

                    if Nsel < minpoint:
                        outfits.data[idata, idec, ira, :, :, istokes, 2] = 0.0
                        continue

                    vcomp = vreal + 1j * vimaj
                    vweig = 1. / np.var(vcomp)
                    outfits.data[idata, idec, ira, :, :, istokes, 2] = vweig
            select = self.data.data[:, :, :, :, :, :, 2] <= 0.0
            select += np.isnan(self.data.data[:, :, :, :, :, :, 2])
            select += np.isinf(self.data.data[:, :, :, :, :, :, 2])
            outfits.data.data[:, :, :, :, :, :, 2][np.where(select)] = 0.0
        elif doif == True:
            for idata in np.arange(Ndata):
                #if idata%1000 == 0: print("%d / %d"%(idata,Ndata))
                dataidx1 = np.where(np.abs(unix - unix[idata]) < solint)
                dataidx2 = np.where(baseline[dataidx1] == baseline[idata])
                seldata = self.data.data[dataidx1][dataidx2]
                for idec, ira, iif, istokes in itertools.product(np.arange(Ndec),
                                                       np.arange(Nra),
                                                       np.arange(Nif),
                                                       np.arange(Nstokes)):
                    vreal = seldata[:, idec, ira, iif, :, istokes, 0]
                    vimaj = seldata[:, idec, ira, iif, :, istokes, 1]
                    vweig = seldata[:, idec, ira, iif, :, istokes, 2]

                    Ntime = np.prod(vreal.shape)
                    vreal = vreal.reshape(Ntime)
                    vimaj = vimaj.reshape(Ntime)
                    vweig = vweig.reshape(Ntime)

                    select = vweig > 0
                    select *= (np.isnan(vweig) == False)
                    select *= (np.isinf(vweig) == False)
                    select = np.where(select)
                    vreal = vreal[select]
                    vimaj = vimaj[select]
                    vweig = vweig[select]
                    Nsel = len(vreal)

                    if Nsel < minpoint:
                        outfits.data[idata, idec, ira, iif, :, istokes, 2] = 0
                        continue

                    vcomp = vreal + 1j * vimaj
                    vweig = 1. / np.var(vcomp)
                    outfits.data[idata, idec, ira, :, :, istokes, 2] = vweig
            select = self.data.data[:, :, :, :, :, :, 2] <= 0.0
            select += np.isnan(self.data.data[:, :, :, :, :, :, 2])
            select += np.isinf(self.data.data[:, :, :, :, :, :, 2])
            outfits.data.data[:, :, :, :, :, :, 2][np.where(select)] = 0.0
        else:
            for idata in np.arange(Ndata):
                #if idata%1000 == 0: print("%d / %d"%(idata,Ndata))
                dataidx1 = np.where(np.abs(unix - unix[idata]) < solint)
                dataidx2 = np.where(baseline[dataidx1] == baseline[idata])
                seldata = self.data.data[dataidx1][dataidx2]
                for idec, ira, iif, ich, istokes in itertools.product(np.arange(Ndec),
                                                            np.arange(Nra),
                                                            np.arange(Nif),
                                                            np.arange(Nch),
                                                            np.arange(Nstokes)):
                    weight = outfits.data[idata, idec,
                                          ira, iif, ich, istokes, 2]
                    if weight <= 0.0 or np.isnan(weight) or np.isinf(weight):
                        outfits.data[idata, idec, ira,
                                     iif, ich, istokes, 2] = 0.0
                        continue

                    vreal = seldata[:, idec, ira, iif, ich, istokes, 0]
                    vimaj = seldata[:, idec, ira, iif, ich, istokes, 1]
                    vweig = seldata[:, idec, ira, iif, ich, istokes, 2]

                    Ntime = np.prod(vreal.shape)
                    vreal = vreal.reshape(Ntime)
                    vimaj = vimaj.reshape(Ntime)
                    vweig = vweig.reshape(Ntime)

                    select = vweig > 0
                    select *= (np.isnan(vweig) == False)
                    select *= (np.isinf(vweig) == False)
                    select = np.where(select)
                    vreal = vreal[select]
                    vimaj = vimaj[select]
                    vweig = vweig[select]
                    Nsel = len(vreal)

                    if Nsel < minpoint:
                        outfits.data[idata, idec, ira,
                                     iif, ich, istokes, 2] = 0
                        continue

                    vcomp = vreal + 1j * vimaj
                    outfits.data[idata, idec, ira, iif, ich,
                                 istokes, 2] = 1. / np.var(vcomp)
        return outfits


#-------------------------------------------------------------------------------
# Visibility Tables
#-------------------------------------------------------------------------------
class _UVTable(pd.DataFrame):
    '''
    This is a class describing common variables and methods of VisTable,
    BSTable and CATable.
    '''
    uvunit = "lambda"

    @property
    def _constructor(self):
        return _UVTable

    @property
    def _constructor_sliced(self):
        return _UVSeries

    def uvunitconv(self, unit1="l", unit2="l"):
        '''
        Derive a conversion factor of units for the baseline length from unit1
        to unit2. Available angular units are l[ambda], kl[ambda], ml[ambda], 
        gl[ambda], m[eter] and km[eter].

        Args:
          unit1 (str): the first unit
          unit2 (str): the second unit

        Returns:
          conversion factor from unit1 to unit2 in float.
        '''
        if unit1 == unit2:
            return 1.

        # Convert from unit1 to lambda
        if unit1.lower().find("l") == 0:
            conv = 1.
        elif unit1.lower().find("kl") == 0:
            conv = 1e3
        elif unit1.lower().find("ml") == 0:
            conv = 1e6
        elif unit1.lower().find("gl") == 0:
            conv = 1e9
        elif unit1.lower().find("m") == 0:
            conv = ac.c.si.value / self["freq"]
        elif unit1.lower().find("km") == 0:
            conv = ac.c.si.value / self["freq"] / 1e3
        else:
            print("Error: unit1=%s is not supported" % (unit1))
            return -1

        # Convert from lambda to unit2
        if unit2.lower().find("l") == 0:
            conv /= 1.
        elif unit2.lower().find("kl") == 0:
            conv /= 1e3
        elif unit2.lower().find("ml") == 0:
            conv /= 1e6
        elif unit2.lower().find("gl") == 0:
            conv /= 1e9
        elif unit2.lower().find("m") == 0:
            conv /= ac.c.si.value / self["freq"]
        elif unit2.lower().find("km") == 0:
            conv /= ac.c.si.value / self["freq"] / 1e3
        else:
            print("Error: unit2=%s is not supported" % (unit2))
            return -1

        return conv


    def get_unitlabel(self, uvunit=None):
        '''
        Get a unit label name for uvunits. 
        Available units are l[ambda], kl[ambda], ml[ambda], gl[ambda], m[eter]
        and km[eter].

        Args:
          uvunit (str, default is None):
            The input unit. If uvunit is None, it will use self.uvunit.

        Returns:
          The unit label name in string.
        '''
        if uvunit is None:
            uvunit = self.uvunit
        
        if uvunit.lower().find("l") == 0:
            unitlabel = r"$\lambda$"
        elif uvunit.lower().find("kl") == 0:
            unitlabel = r"$10^3 \lambda$"
        elif uvunit.lower().find("ml") == 0:
            unitlabel = r"$10^6 \lambda$"
        elif uvunit.lower().find("gl") == 0:
            unitlabel = r"$10^9 \lambda$"
        elif uvunit.lower().find("m") == 0:
            unitlabel = "m"
        elif uvunit.lower().find("km") == 0:
            unitlabel = "km"
        else:
            print("Error: uvunit=%s is not supported" % (unit2))
            return -1
        
        return unitlabel

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
        super(_UVTable, self).to_csv(filename,
                                     index=False, index_label=False, **args)


class VisTable(_UVTable):
    '''
    This class is for handling two dimentional tables of full complex visibilities
    and amplitudes. The class inherits pandas.DataFrame class, so you can use this
    class like pandas.DataFrame. The class also has additional methods to edit,
    visualize and convert data.
    '''
    @property
    def _constructor(self):
        return VisTable

    @property
    def _constructor_sliced(self):
        return _VisSeries

    def uvsort(self):
        '''
        Sort uvdata. First, it will check station IDs of each visibility 
        and switch its order if "st1" > "st2". Then, data will be TB-sorted.
        '''
        outdata = self.copy()

        # search where st1 > st2
        t = outdata["st1"] > outdata["st2"]
        dammy = outdata.loc[t, "st2"]
        outdata.loc[t, "st2"] = outdata.loc[t, "st1"]
        outdata.loc[t, "st1"] = dammy
        outdata.loc[t, "phase"] *= -1

        # sort with time, and stations
        outdata = outdata.sort_values(
            by=["year", "doy", "hour", "min", "sec", "st1", "st2", "ch"])

        return outdata


    def recalc_uvdist(self):
        '''
        Re-calculate the baseline length from self["u"] and self["v"].
        '''
        self["uvdist"] = np.sqrt(self["u"]*self["u"]+self["v"]*self["v"])


    def fit_beam(self, angunit="mas", errweight=0., ftsign=+1):
        '''
        This function estimates the synthesized beam size at natural weighting.

        keywords:
          angunit (string):
            Angular unit (uas, mas, asec or arcsec, amin or arcmin, degree)
          errweight (float; experimental):
            index for errer weighting
          ftsign (integer):
            a sign for fourier matrix
        '''
        # infer the parameters of clean beam
        parm0 = _calc_bparms(self)

        # generate a small image 4 times larger than the expected major axis size of the beam
        fitsdata = imdata.IMFITS(fov=[parm0[0],-parm0[0], -parm0[0], parm0[0]],
                                 nx=20, ny=20, angunit="deg")

        # create output fits
        dbfitsdata, dbflux = _calc_dbeam(
            fitsdata, self, errweight=errweight, ftsign=ftsign)

        X, Y = fitsdata.get_xygrid(angunit="deg", twodim=True)
        dbeam = dbfitsdata.data[0, 0]
        dbeam /= np.max(dbeam)

        parms = optimize.leastsq(_fit_chisq, parm0, args=(X, Y, dbeam))

        (maja, mina, PA) = parms[0]
        maja = np.abs(maja)
        mina = np.abs(mina)

        # adjust these parameters
        if maja < mina:
            maja, mina = mina, maja
            PA += 90
        while np.abs(PA) > 90:
            if PA > 90:
                PA -= 90
            elif PA < -90:
                PA += 90

        # return as parameters of gauss_convolve
        factor = fitsdata.angconv("deg", angunit)
        cb_parms = ({'majsize': maja * factor, 'minsize': mina *
                     factor, 'angunit': angunit, 'pa': PA})
        return cb_parms


    def fftshift(self, fitsdata, fgfov=1):
        '''
        Arguments: 
          vistable (pandas.Dataframe object):
            input visibility table

          fitsdata (imdata.IMFITS object):
            input imdata.IMFITS object

          fgfov (int)
            a number of gridded FOV/original FOV

        Output: pandas.Dataframe object
        '''
        # Copy vistable for edit
        vistable = copy.deepcopy(self)

        # Flip uv cordinates and phase for no gridded data
        if np.any(vistable.columns != "ugidx"):
            vistable.loc[vistable["u"] < 0, ("u", "v", "phase")] *= -1

        # Calculate du and dv
        Nupix = fitsdata.header["nx"] * fgfov
        Nvpix = fitsdata.header["ny"] * fgfov
        du = 1 / np.radians(np.abs(fitsdata.header["dx"]) * Nupix)
        dv = 1 / np.radians(fitsdata.header["dy"] * Nvpix)

        # Calculate index of uv
        if np.any(vistable.columns != "ugidx"):
            vistable["ugidx"] = np.float64(np.array(vistable["u"] / du))
            vistable["vgidx"] = np.float64(np.array(vistable["v"] / dv))

        # Shift vistable
        vistable.loc[vistable["vgidx"] < 0, "vgidx"] += Nvpix

        # Create new list for shift
        outlist = {
            "ugidx": [],
            "vgidx": [],
            "u": [],
            "v": [],
            "orgu": [],
            "orgv": [],
            "amp": [],
            "phase": [],
            "weight": [],
            "sigma": []
        }

        # Save shifted data
        outlist["ugidx"] = vistable["ugidx"]
        outlist["vgidx"] = vistable["vgidx"]
        outlist["u"] = vistable["ugidx"] * du
        outlist["v"] = vistable["vgidx"] * dv
        outlist["orgu"] = vistable["u"]
        outlist["orgv"] = vistable["v"]
        outlist["amp"] = vistable["amp"]
        outlist["phase"] = vistable["phase"]
        outlist["weight"] = vistable["weight"]
        outlist["sigma"] = vistable["sigma"]

        # Output as pandas.DataFrame
        outtable = pd.DataFrame(outlist, columns=[
            "ugidx", "vgidx", "u", "v", "orgu", "orgv", "amp", "phase", "weight", "sigma"])
        return outtable


    def make_bstable(self):
        '''
        This function calculates bi-spectra from full complex visibility data.
        It will output uvdata.BSTable object.

        Args: N/A

        Returns: uvdata.BSTable object
        '''
        # Get Number of Data
        Ndata = len(self["u"])

        # get list of timetags
        timetag = []
        for i in np.arange(Ndata):
            timetag.append("%04d-%03d-%02d-%02d-%5.2f_%d" % (self.loc[i, "year"],
                                                             self.loc[i,
                                                                      "doy"],
                                                             self.loc[i,
                                                                      "hour"],
                                                             self.loc[i,
                                                                      "min"],
                                                             self.loc[i,
                                                                      "sec"],
                                                             self.loc[i, "ch"]))
        timetag = np.asarray(timetag)
        timetagset = sorted(set(timetag))
        Ntt = len(timetagset)

        bstable = {}
        for column in BSTable.bstable_columns:
            if column in ["uvdistave", "uvdistmax", "uvdistmin",
                          "uvdist12", "uvdist23", "uvdist31"]:
                continue
            bstable[column] = []

        # calculate bi-spectrum for each timetag
        for itt in np.arange(Ntt):
            # get available station
            idx = timetag == timetagset[itt]
            sts = self.loc[idx, "st1"].tolist(
            ) + self.loc[idx, "st2"].tolist()
            sts = sorted(set(sts))

            # check if the number of stations exceed three
            Nsts = len(sts)
            if Nsts < 3:
                continue
            stsid = np.arange(Nsts)
            # the maximum number of closure phases
            Ntrimax = (Nsts - 1) * (Nsts - 2) / 2
            Nbl = Nsts * (Nsts - 1) / 2           # the number of baslines

            # calc bi-spectrum and output
            rank = 0
            Ntri = 0
            matrix = None
            for stid1, stid2, stid3 in itertools.combinations(stsid, 3):
                # if we already found the maximum number of triangles, skip the process
                if Ntri >= Ntrimax:
                    break

                # station number
                st1 = sts[stid1]
                st2 = sts[stid2]
                st3 = sts[stid3]

                # baseline ids
                blid1 = _getblid(stid1, stid2, Nsts)
                blid2 = _getblid(stid2, stid3, Nsts)
                blid3 = _getblid(stid1, stid3, Nsts)

                # calculate conversion matrix
                row = np.zeros(Nbl)
                row[blid1] = 1
                row[blid2] = 1
                row[blid3] = -1
                if matrix is None:
                    tmpmatrix = np.asarray([row])
                else:
                    tmpmatrix = np.append(matrix, row).reshape(Ntri + 1, Nbl)

                # Check if this triangle is redundant
                tmprank = np.linalg.matrix_rank(tmpmatrix)
                if rank == tmprank:
                    continue

                # Check if corresponding baseline data exist
                isbl1 = True
                isbl2 = True
                isbl3 = True
                bl1idx = idx & (self["st1"] == st1) & (self["st2"] == st2)
                bl2idx = idx & (self["st1"] == st2) & (self["st2"] == st3)
                bl3idx = idx & (self["st1"] == st1) & (self["st2"] == st3)
                if np.where(bl1idx)[0].shape[0] == 0:
                    isbl1 = False
                if np.where(bl2idx)[0].shape[0] == 0:
                    isbl2 = False
                if np.where(bl3idx)[0].shape[0] == 0:
                    isbl3 = False
                if False in [isbl1, isbl2, isbl3]:
                    continue

                # calculate bi-spectrum
                bl1data = self.loc[bl1idx, :].reset_index(drop=True).loc[0, :]
                bl2data = self.loc[bl2idx, :].reset_index(drop=True).loc[0, :]
                bl3data = self.loc[bl3idx, :].reset_index(drop=True).loc[0, :]

                amp = bl1data.loc["amp"] * \
                    bl2data.loc["amp"] * bl3data.loc["amp"]
                phase = bl1data.loc["phase"] + \
                    bl2data.loc["phase"] - bl3data.loc["phase"]
                ratio_12 = bl1data.loc["sigma"] / bl1data.loc["amp"]
                ratio_23 = bl2data.loc["sigma"] / bl2data.loc["amp"]
                ratio_13 = bl3data.loc["sigma"] / bl3data.loc["amp"]
                sigma = amp * np.sqrt((ratio_12)**2 +
                                      (ratio_23)**2 + (ratio_13)**2)

                bstable["jd"].append(bl1data.loc["jd"])
                bstable["year"].append(bl1data.loc["year"])
                bstable["doy"].append(bl1data.loc["doy"])
                bstable["hour"].append(bl1data.loc["hour"])
                bstable["min"].append(bl1data.loc["min"])
                bstable["sec"].append(bl1data.loc["sec"])
                bstable["u12"].append(bl1data.loc["u"])
                bstable["v12"].append(bl1data.loc["v"])
                bstable["w12"].append(bl1data.loc["w"])
                bstable["u23"].append(bl2data.loc["u"])
                bstable["v23"].append(bl2data.loc["v"])
                bstable["w23"].append(bl2data.loc["w"])
                bstable["u31"].append(-bl3data.loc["u"])
                bstable["v31"].append(-bl3data.loc["v"])
                bstable["w31"].append(-bl3data.loc["w"])
                bstable["st1"].append(st1)
                bstable["st2"].append(st2)
                bstable["st3"].append(st3)
                bstable["ch"].append(bl1data.loc["ch"])
                bstable["amp"].append(amp)
                bstable["phase"].append(phase)
                bstable["sigma"].append(sigma)
                # update rank and matrix
                rank += 1
                Ntri += 1
                matrix = tmpmatrix.copy()

        # form pandas dataframe
        bstable = BSTable(bstable)
        bstable["uvdist12"] = np.sqrt(
            np.square(bstable["u12"]) + np.square(bstable["v12"]))
        bstable["uvdist23"] = np.sqrt(
            np.square(bstable["u23"]) + np.square(bstable["v23"]))
        bstable["uvdist31"] = np.sqrt(
            np.square(bstable["u31"]) + np.square(bstable["v31"]))
        bstable["uvdistave"] = bstable["uvdist12"]
        bstable["uvdistmin"] = bstable["uvdist12"]
        bstable["uvdistmax"] = bstable["uvdist12"]

        for i in np.arange(len(bstable["uvdist12"])):
            uvdists = bstable.loc[i, ["uvdist12", "uvdist23", "uvdist31"]]
            bstable.loc[i, "uvdistave"] = np.mean(uvdists)
            bstable.loc[i, "uvdistmax"] = np.max(uvdists)
            bstable.loc[i, "uvdistmin"] = np.min(uvdists)
        bstable = bstable[BSTable.bstable_columns]

        for i in np.arange(len(BSTable.bstable_columns)):
            column = BSTable.bstable_columns[i]
            bstable[column] = BSTable.bstable_types[i](bstable[column])

        # Adjust Phase
        bstable["phase"] *= np.pi / 180.
        bstable["phase"] = np.arctan2(
            np.sin(bstable["phase"]), np.cos(bstable["phase"])) * 180. / np.pi

        return bstable

    def make_catable(self):
        '''
        This function calculates closure amplitudes from full complex visibility data.
        It will output uvdata.CATable object.

        Args: N/A

        Returns: uvdata.CATable object
        '''

        # Get Number of Data
        Ndata = len(self["u"])

        # get list of timetags
        timetag = []
        for i in np.arange(Ndata):
            timetag.append("%04d-%03d-%02d-%02d-%5.2f_%d" % (self.loc[i, "year"],
                                                             self.loc[i,
                                                                      "doy"],
                                                             self.loc[i,
                                                                      "hour"],
                                                             self.loc[i,
                                                                      "min"],
                                                             self.loc[i,
                                                                      "sec"],
                                                             self.loc[i, "ch"]))
        timetag = np.asarray(timetag)
        timetagset = sorted(set(timetag))
        Ntt = len(timetagset)

        catable = {}
        for column in CATable.catable_columns:
            if column in ["uvdistave", "uvdistmax", "uvdistmin",
                          "uvdist1", "uvdist2", "uvdist3", "uvdist4"]:
                continue
            catable[column] = []

        # calculate bi-spectrum for each timetag
        for itt in np.arange(Ntt):
            # get available station
            idx = timetag == timetagset[itt]
            sts = self.loc[idx, "st1"].tolist(
            ) + self.loc[idx, "st2"].tolist()
            sts = sorted(set(sts))

            # check if the number of stations exceed three
            Nsts = len(sts)
            if Nsts < 4:
                continue
            stsid = np.arange(Nsts)
            # the maximum number of closure phases
            Ncamax = Nsts * (Nsts - 3) / 2
            Nbl = Nsts * (Nsts - 1) / 2           # the number of baslines

            # calc bi-spectrum and output
            rank = 0
            Nca = 0
            matrix = None
            for stid1, stid2, stid3, stid4 in itertools.combinations(stsid, 4):
                output = _calc_caamp(self, catable, idx,
                                     stid1, stid2, stid3, stid4, sts, Nsts, Nbl,
                                     matrix, rank, Nca, Ncamax)
                if output is not None:
                    catable, rank, Nca, matrix = output
                output = _calc_caamp(self, catable, idx,
                                     stid1, stid3, stid4, stid2, sts, Nsts, Nbl,
                                     matrix, rank, Nca, Ncamax)
                if output is not None:
                    catable, rank, Nca, matrix = output
                output = _calc_caamp(self, catable, idx,
                                     stid1, stid2, stid4, stid3, sts, Nsts, Nbl,
                                     matrix, rank, Nca, Ncamax)
                if output is not None:
                    catable, rank, Nca, matrix = output

        # form pandas dataframe
        catable = CATable(catable)
        catable["uvdist1"] = np.sqrt(
            np.square(catable["u1"]) + np.square(catable["v1"]))
        catable["uvdist2"] = np.sqrt(
            np.square(catable["u2"]) + np.square(catable["v2"]))
        catable["uvdist3"] = np.sqrt(
            np.square(catable["u3"]) + np.square(catable["v3"]))
        catable["uvdist4"] = np.sqrt(
            np.square(catable["u4"]) + np.square(catable["v4"]))
        catable["uvdistave"] = catable["uvdist1"]
        catable["uvdistmin"] = catable["uvdist1"]
        catable["uvdistmax"] = catable["uvdist1"]
        for i in np.arange(len(catable["uvdist1"])):
            uvdists = catable.loc[i, ["uvdist1",
                                      "uvdist2", "uvdist3", "uvdist4"]]
            catable.loc[i, "uvdistave"] = np.mean(uvdists)
            catable.loc[i, "uvdistmax"] = np.max(uvdists)
            catable.loc[i, "uvdistmin"] = np.min(uvdists)

        catable = catable[CATable.catable_columns]
        for i in np.arange(len(CATable.catable_columns)):
            column = CATable.catable_columns[i]
            catable[column] = CATable.catable_types[i](catable[column])

        return catable

    def _make_gradvistable(self, normalize=True):
        gradself1 = self.copy()
        gradself2 = self.copy()

        # scale visibility
        gradself1.loc[:, "amp"] *= np.abs(gradself1["u"])
        gradself1.loc[:, "sigma"] *= np.abs(gradself1["u"])
        gradself1.loc[:, "phase"] -= 90
        gradself2.loc[:, "amp"] *= np.abs(gradself2["v"])
        gradself2.loc[:, "sigma"] *= np.abs(gradself2["v"])
        gradself2.loc[:, "phase"] -= 90

        # frip phase
        idx1 = gradself1["u"] < 0
        gradself1.loc[idx1, "phase"] += 180

        idx1 = gradself2["v"] < 0
        gradself2.loc[idx1, "phase"] += 180

        if normalize:
            maxamp = np.max([gradself1["amp"].max(),
                             gradself2["amp"].max()])
            gradself1.loc[:, "amp"] /= maxamp
            gradself1.loc[:, "sigma"] /= maxamp
            gradself2.loc[:, "amp"] /= maxamp
            gradself2.loc[:, "sigma"] /= maxamp
        return gradself1, gradself2

    def gridding(self, fitsdata, fgfov, mu=1, mv=1, c=1):
        '''
        Args: 
          vistable (pandas.Dataframe object):
            input visibility table

          fitsdata (imdata.IMFITS object):
            input imdata.IMFITS object

          fgfov (int)
            a number of gridded FOV/original FOV

          mu (int; default = 1):
          mv (int; default = 1):
            parameter for spheroidal angular function
            a number of cells for gridding

          c (int; default = 1):
            parameter for spheroidal angular fuction
            this parameter decides steepness of the function

        Returns: 
          uvdata.VisTable object
        '''
        # Copy vistable for edit
        vistable = copy.deepcopy(self)

        # Flip uv cordinates and phase, where u < 0 for avoiding redundant grids
        vistable.loc[vistable["u"] < 0, ("u", "v", "phase")] *= -1

        # Create column of full-comp visibility
        vistable["comp"] = vistable["amp"] * \
            np.exp(1j * np.radians(vistable["phase"]))
        Ntable = len(vistable)

        # Calculate du and dv
        du = 1 / \
            np.radians(np.abs(fitsdata.header["dx"])
                       * fitsdata.header["nx"] * fgfov)
        dv = 1 / np.radians(fitsdata.header["dy"]
                            * fitsdata.header["ny"] * fgfov)

        # Calculate index of uv for gridding
        vistable["ugidx"] = np.int64(np.around(np.array(vistable["u"] / du)))
        vistable["vgidx"] = np.int64(np.around(np.array(vistable["v"] / dv)))

        # Flag for skipping already averaged data
        vistable["skip"] = np.zeros(Ntable, dtype=np.int64)
        vistable.loc[:, ("skip")] = -1

        # Create new list for gridded data
        outlist = {
            "ugidx": [],
            "vgidx": [],
            "u": [],
            "v": [],
            "uvdist": [],
            "amp": [],
            "phase": [],
            "weight": [],
            "sigma": []
        }

        # Convolutional gridding
        for itable in np.arange(Ntable):
            if vistable["skip"][itable] > 0:
                continue

            # Get the grid index for the current data
            ugidx = vistable["ugidx"][itable]
            vgidx = vistable["vgidx"][itable]

            # Data index for visibilities on the same grid
            gidxs = (vistable["ugidx"] == ugidx) & (vistable["vgidx"] == vgidx)

            # Flip flags
            vistable.loc[gidxs, "skip"] = 1

            # Get data on the same grid
            U = np.array(vistable.loc[gidxs, "u"])
            V = np.array(vistable.loc[gidxs, "v"])
            Vcomps = np.array(vistable.loc[gidxs, "comp"])
            sigmas = np.array(vistable.loc[gidxs, "sigma"])
            weight = 1 / sigmas**2

            ugrid = ugidx * du
            vgrid = vgidx * dv

            # Calculate spheroidal angular function
            U = 2 * (ugrid - U) / (mu * du)
            V = 2 * (vgrid - V) / (mv * dv)
            uSAF = ss.pro_ang1(0, 0, c, U)[0]
            vSAF = ss.pro_ang1(0, 0, c, V)[0]

            # Convolutional gridding
            Vcomp_ave = np.sum(Vcomps * uSAF * vSAF) / np.sum(uSAF * vSAF)
            weight_ave = np.sum(weight * uSAF * vSAF) / np.sum(uSAF * vSAF)
            sigma_ave = 1 / np.sqrt(weight_ave)

            # Save gridded data on the grid
            outlist["ugidx"] += [ugidx]
            outlist["vgidx"] += [vgidx]
            outlist["u"] += [ugrid]
            outlist["v"] += [vgrid]
            outlist["uvdist"] += [np.sqrt(ugrid**2 + vgrid**2)]
            outlist["amp"] += [np.abs(Vcomp_ave)]
            outlist["phase"] += [np.angle(Vcomp_ave, deg=True)]
            outlist["weight"] += [weight_ave]
            outlist["sigma"] += [sigma_ave]

        # Output as pandas.DataFrame
        outtable = pd.DataFrame(outlist, columns=[
            "ugidx", "vgidx", "u", "v", "uvdist", "amp", "phase", "weight", "sigma"])
        return outtable


    #---------------------------------------------------------------------------
    # Plot Functions
    #---------------------------------------------------------------------------
    def uvplot(self, uvunit=None, conj=True,
               ls="none", marker=".", **plotargs):
        '''
        Plot uv-plot on the current axes.
        This method uses matplotlib.pyplot.plot().
        
        Args:
          uvunit (str, default = None):
            The unit of the baseline length. if uvunit is None, it will use
            self.uvunit.
        
          conj (boolean, default = True):
            if conj=True, it will plot complex conjugate components (i.e. (-u, -v)).
        
          **plotargs:
            You can set parameters of matplotlib.pyplot.plot.
            Defaults are {'ls': "none", 'marker': "."}
        '''
        # Set Unit
        if uvunit is None:
            uvunit = self.uvunit

        # Conversion Factor
        conv = self.uvunitconv(unit1="lambda", unit2=uvunit)

        # Label
        unitlabel = self.get_unitlabel(uvunit)

        # plotting
        plt.plot(self["u"] * conv, self["v"] * conv,
                 ls=ls, marker=marker, **plotargs)
        if conj:
            plotargs2 = copy.deepcopy(plotargs)
            plotargs2["label"] = ""
            plt.plot(-self["u"] * conv, -self["v"] * conv,
                     ls=ls, marker=marker, **plotargs)
        plt.xlabel(r"Baseline Length $u$ (%s)" % (unitlabel))
        plt.ylabel(r"Baseline Length $v$ (%s)" % (unitlabel))

        ax = plt.gca()
        ax.set_aspect("equal")
        xlim = np.asarray(ax.get_xlim())
        ylim = np.asarray(ax.get_ylim())
        ax.set_xlim(-np.sort(-xlim))
        ax.set_ylim(np.sort(ylim))


    def radplot_amp(self, uvunit=None, errorbar=True, model=None, modeltype="amp",
                    ls="none", marker=".", **plotargs):
        '''
        Plot visibility amplitudes as a function of baseline lengths
        on the current axes. This method uses matplotlib.pyplot.plot() or 
        matplotlib.pyplot.errorbar().
        
        Args:
          uvunit (str, default = None):
            The unit of the baseline length. if uvunit is None, it will use
            self.uvunit.
            
          errorbar (boolean, default = True):
            If errorbar is True, it will plot data with errorbars using 
            matplotlib.pyplot.errorbar(). Otherwise, it will plot data without
            errorbars using matplotlib.pyplot.plot(). 
            
            If you plot model closure phases (i.e. model is not None), 
            it will plot without errobars regardless of this parameter.
            
          model (dict-like such as pd.DataFrame, pd.Series, default is None):
            Model data sets. Model amplitudes must be given by model["fcvampmod"]
            for full complex visibilities (modeltype="fcv") or model["ampmod"]
            for visibility amplitudes (modeltype="amp").
            Otherwise, it will plot amplitudes in the table (i.e. self["amp"]).
            
          modeltype (string, default = "amp"):
            The type of models. If you would plot model amplitudes, set modeltype="amp".
            Else if you would plot model full complex visibilities, set modeltype="fcv".
            
          **plotargs:
            You can set parameters of matplotlib.pyplot.plot() or 
            matplotlib.pyplot.errorbars().
            Defaults are {'ls': "none", 'marker': "."}.
        '''
        # Set Unit
        if uvunit is None:
            uvunit = self.uvunit

        # Conversion Factor
        conv = self.uvunitconv(unit1="lambda", unit2=uvunit)

        # Label
        unitlabel = self.get_unitlabel(uvunit)

        # plotting data
        if model is not None:
            if modeltype.lower().find("amp") == 0:
                plt.plot(self["uvdist"] * conv, model["ampmod"],
                         ls=ls, marker=marker, **plotargs)
            elif modeltype.lower().find("fcv") == 0:
                plt.plot(self["uvdist"] * conv, model["fcvampmod"],
                         ls=ls, marker=marker, **plotargs)
        elif errorbar:
            plt.errorbar(self["uvdist"] * conv, self["amp"], self["sigma"],
                         ls=ls, marker=marker, **plotargs)
        else:
            plt.plot(self["uvdist"] * conv, self["amp"],
                     ls=ls, marker=marker, **plotargs)
        plt.xlabel(r"Baseline Length (%s)" % (unitlabel))
        plt.ylabel(r"Visibility Amplitude (Jy)")
        plt.xlim(0,)
        plt.ylim(0,)

    def radplot_phase(self, uvunit=None, errorbar=True, model=None,
                      ls="none", marker=".", **plotargs):
        '''
        Plot visibility phases as a function of baseline lengths
        on the current axes. This method uses matplotlib.pyplot.plot() or 
        matplotlib.pyplot.errorbar().
        
        Args:
          uvunit (str, default = None):
            The unit of the baseline length. if uvunit is None, it will use
            self.uvunit.
            
          errorbar (boolean, default = True):
            If errorbar is True, it will plot data with errorbars using 
            matplotlib.pyplot.errorbar(). Otherwise, it will plot data without
            errorbars using matplotlib.pyplot.plot(). 
            
            If you plot model closure phases (i.e. model is not None), 
            it will plot without errobars regardless of this parameter.
            
          model (dict-like such as pd.DataFrame, pd.Series, default is None):
            Model data sets. Model phases must be given by model["fcvphamod"].
            Otherwise, it will plot amplitudes in the table (i.e. self["phase"]).
            
          **plotargs:
            You can set parameters of matplotlib.pyplot.plot() or 
            matplotlib.pyplot.errorbars().
            Defaults are {'ls': "none", 'marker': "."}.
        '''
        # Set Unit
        if uvunit is None:
            uvunit = self.uvunit

        # Conversion Factor
        conv = self.uvunitconv(unit1="lambda", unit2=uvunit)

        # Label
        unitlabel = self.get_unitlabel(uvunit)

        # plotting data
        if model is not None:
            plt.plot(self["uvdist"] * conv, model["fcvphamod"],
                     ls=ls, marker=marker, **plotargs)
        elif errorbar:
            pherr = np.rad2deg(self["sigma"] / self["amp"])
            plt.errorbar(self["uvdist"] * conv, self["phase"], pherr,
                         ls=ls, marker=marker, **plotargs)
        else:
            plt.plot(self["uvdist"] * conv, self["phase"],
                     ls=ls, marker=marker, **plotargs)
        plt.xlabel(r"Baseline Length (%s)" % (unitlabel))
        plt.ylabel(r"Visibility Phase ($^\circ$)")
        plt.xlim(0,)
        plt.ylim(-180, 180)


class BSTable(_UVTable):
    '''
    This class is for handling two dimentional tables of Bi-spectrua of
    visibilities. The class inherits pandas.DataFrame class, so you can use this
    class like pandas.DataFrame. The class also has additional methods to edit,
    visualize and convert data.
    '''
    uvunit = "lambda"

    bstable_columns = ["jd", "year", "doy", "hour", "min", "sec",
                       "u12", "v12", "w12", "uvdist12",
                       "u23", "v23", "w23", "uvdist23",
                       "u31", "v31", "w31", "uvdist31",
                       "uvdistmin", "uvdistmax", "uvdistave",
                       "st1", "st2", "st3", "ch", "amp", "phase", "sigma"]
    bstable_types = [np.float64, np.int32, np.int32, np.int32, np.int32, np.float64,
                     np.float64, np.float64, np.float64, np.float64,
                     np.float64, np.float64, np.float64, np.float64,
                     np.float64, np.float64, np.float64, np.float64,
                     np.float64, np.float64, np.float64,
                     np.int32, np.int32, np.int32, np.int32,
                     np.float64, np.float64, np.float64]

    @property
    def _constructor(self):
        return BSTable

    @property
    def _constructor_sliced(self):
        return _BSSeries

    #---------------------------------------------------------------------------
    # Plot Functions
    #---------------------------------------------------------------------------
    def uvplot(self, uvunit=None, conj=True,
               ls="none", marker=".", **plotargs):
        '''
        Plot uv-plot on the current axes.
        This method uses matplotlib.pyplot.plot().
        
        Args:
          uvunit (str, default = None):
            The unit of the baseline length. if uvunit is None, it will use
            self.uvunit.
          conj (boolean, default = True):
            if conj=True, it will plot complex conjugate components (i.e. (-u, -v)).
          **plotargs:
            You can set parameters of matplotlib.pyplot.plot.
            Defaults are {'ls': "none", 'marker': "."}
        '''
        # Set Unit
        if uvunit is None:
            uvunit = self.uvunit

        # Conversion Factor
        conv = self.uvunitconv(unit1="lambda", unit2=uvunit)

        # Label
        unitlabel = self.get_unitlabel(uvunit)
        
        plotargs2 = copy.deepcopy(plotargs)
        plotargs2["label"] = ""

        # plotting
        plt.plot(self["u12"] * conv, self["v12"] *
                 conv, ls=ls, marker=marker, **plotargs)
        plt.plot(self["u23"] * conv, self["v23"] * conv,
                 ls=ls, marker=marker, **plotargs2)
        plt.plot(self["u31"] * conv, self["v31"] * conv,
                 ls=ls, marker=marker, **plotargs2)
        if conj:
            plt.plot(-self["u12"] * conv, -self["v12"] *
                     conv, ls=ls, marker=marker, **plotargs2)
            plt.plot(-self["u23"] * conv, -self["v23"] *
                     conv, ls=ls, marker=marker, **plotargs2)
            plt.plot(-self["u31"] * conv, -self["v31"] *
                     conv, ls=ls, marker=marker, **plotargs2)
        plt.xlabel(r"Baseline Length $u$ (%s)" % (unitlabel))
        plt.ylabel(r"Baseline Length $v$ (%s)" % (unitlabel))

        ax = plt.gca()
        ax.set_aspect("equal")
        xlim = np.asarray(ax.get_xlim())
        ylim = np.asarray(ax.get_ylim())
        ax.set_xlim(-np.sort(-xlim))
        ax.set_ylim(np.sort(ylim))

    def radplot(self, uvdtype="ave", uvunit=None, errorbar=True, model=None,
                ls="none", marker=".", **plotargs):
        '''
        Plot closure phases as a function of baseline lengths on the current axes.
        This method uses matplotlib.pyplot.plot() or matplotlib.pyplot.errorbar().
        
        Args:
          uvdtype (str, default = "ave"):
            The type of the baseline length plotted along the horizontal axis.
              "max": maximum of three baselines (=self["uvdistmax"])
              "min": minimum of three baselines (=self["uvdistmin"])
              "ave": average of three baselines (=self["uvdistave"])
          uvunit (str, default = None):
            The unit of the baseline length. if uvunit is None, it will use
            self.uvunit.
          errorbar (boolean, default = True):
            If errorbar is True, it will plot data with errorbars using 
            matplotlib.pyplot.errorbar(). Otherwise, it will plot data without
            errorbars using matplotlib.pyplot.plot(). 
            
            If you plot model closure phases (i.e. model is not None), 
            it will plot without errobars regardless of this parameter.
          model (dict-like such as pd.DataFrame, pd.Series, default is None):
            Model data sets. Model closure phases must be given by model["cpmod"].
            Otherwise, it will plot closure phases in the table (i.e. self["phase"]).
          **plotargs:
            You can set parameters of matplotlib.pyplot.plot() or 
            matplotlib.pyplot.errorbars().
            Defaults are {'ls': "none", 'marker': "."}.
        '''
        # Set Unit
        if uvunit is None:
            uvunit = self.uvunit

        # Conversion Factor
        conv = self.uvunitconv(unit1="lambda", unit2=uvunit)

        # uvdistance
        if uvdtype.lower().find("ave") * uvdtype.lower().find("mean") == 0:
            uvdist = self["uvdistave"] * conv
        elif uvdtype.lower().find("min") == 0:
            uvdist = self["uvdistmin"] * conv
        elif uvdtype.lower().find("max") == 0:
            uvdist = self["uvdistmax"] * conv
        else:
            print("[Error] uvdtype=%s is not available." % (uvdtype))
            return -1


        # Label
        unitlabel = self.get_unitlabel(uvunit)


        # plotting data
        if model is not None:
            plt.plot(uvdist, model["cpmod"],
                     ls=ls, marker=marker, **plotargs)
        elif errorbar:
            pherr = np.rad2deg(self["sigma"] / self["amp"])
            plt.errorbar(uvdist, self["phase"], pherr,
                         ls=ls, marker=marker, **plotargs)
        else:
            plt.plot(uvdist, self["phase"],
                     ls=ls, marker=marker, **plotargs)
        plt.xlabel(r"Baseline Length (%s)" % (unitlabel))
        plt.ylabel(r"Closure Phase ($^\circ$)")
        plt.xlim(0,)
        plt.ylim(-180, 180)


class CATable(_UVTable):
    uvunit = "lambda"

    catable_columns = ["jd", "year", "doy", "hour", "min", "sec",
                       "u1", "v1", "w1", "uvdist1",
                       "u2", "v2", "w2", "uvdist2",
                       "u3", "v3", "w3", "uvdist3",
                       "u4", "v4", "w4", "uvdist4",
                       "uvdistmin", "uvdistmax", "uvdistave",
                       "st1", "st2", "st3", "st4", "ch", "amp", "sigma", "logamp", "logsigma"]
    catable_types = [np.float64, np.int32, np.int32, np.int32, np.int32, np.float64,
                     np.float64, np.float64, np.float64, np.float64,
                     np.float64, np.float64, np.float64, np.float64,
                     np.float64, np.float64, np.float64, np.float64,
                     np.float64, np.float64, np.float64, np.float64,
                     np.float64, np.float64, np.float64,
                     np.int32, np.int32, np.int32, np.int32, np.int32,
                     np.float64, np.float64, np.float64, np.float64]

    @property
    def _constructor(self):
        return CATable

    @property
    def _constructor_sliced(self):
        return _CASeries


    #---------------------------------------------------------------------------
    # Plot Functions
    #---------------------------------------------------------------------------
    def uvplot(self, uvunit=None, conj=True,
               ls="none", marker=".", **plotargs):
        '''
        Plot uv-plot on the current axes.
        This method uses matplotlib.pyplot.plot().
        
        Args:
          uvunit (str, default = None):
            The unit of the baseline length. if uvunit is None, it will use
            self.uvunit.
          conj (boolean, default = True):
            if conj=True, it will plot complex conjugate components (i.e. (-u, -v)).
          **plotargs:
            You can set parameters of matplotlib.pyplot.plot.
            Defaults are {'ls': "none", 'marker': "."}
        '''
        
        # Set Unit
        if uvunit is None:
            uvunit = self.uvunit

        # Conversion Factor
        conv = self.uvunitconv(unit1="lambda", unit2=uvunit)

        # Label
        unitlabel = self.get_unitlabel(uvunit)

        plotargs2 = copy.deepcopy(plotargs)
        plotargs2["label"] = ""

        # plotting
        plt.plot(self["u1"] * conv, self["v1"] * conv,
                 ls=ls, marker=marker, **plotargs)
        plt.plot(self["u2"] * conv, self["v2"] * conv,
                 ls=ls, marker=marker, **plotargs2)
        plt.plot(self["u3"] * conv, self["v3"] * conv,
                 ls=ls, marker=marker, **plotargs2)
        plt.plot(self["u4"] * conv, self["v4"] * conv,
                 ls=ls, marker=marker, **plotargs2)
        if conj:
            plt.plot(-self["u1"] * conv, -self["v1"] * conv,
                     ls=ls, marker=marker, **plotargs2)
            plt.plot(-self["u2"] * conv, -self["v2"] * conv,
                     ls=ls, marker=marker, **plotargs2)
            plt.plot(-self["u3"] * conv, -self["v3"] * conv,
                     ls=ls, marker=marker, **plotargs2)
            plt.plot(-self["u4"] * conv, -self["v4"] * conv,
                     ls=ls, marker=marker, **plotargs2)
        plt.xlabel(r"Baseline Length $u$ (%s)" % (unitlabel))
        plt.ylabel(r"Baseline Length $v$ (%s)" % (unitlabel))

        ax = plt.gca()
        ax.set_aspect("equal")
        xlim = np.asarray(ax.get_xlim())
        ylim = np.asarray(ax.get_ylim())
        ax.set_xlim(-np.sort(-xlim))
        ax.set_ylim(np.sort(ylim))


    def radplot(self, uvdtype="ave", uvunit=None, errorbar=True, model=None,
                ls="none", marker=".", **plotargs):
        '''
        Plot log(closure amplitudes) as a function of baseline lengths
        on the current axes. This method uses matplotlib.pyplot.plot() or 
        matplotlib.pyplot.errorbar().
        
        Args:
          uvdtype (str, default = "ave"):
            The type of the baseline length plotted along the horizontal axis.
              "max": maximum of four baselines (=self["uvdistmax"])
              "min": minimum of four baselines (=self["uvdistmin"])
              "ave": average of four baselines (=self["uvdistave"])
          uvunit (str, default = None):
            The unit of the baseline length. if uvunit is None, it will use
            self.uvunit.
          errorbar (boolean, default = True):
            If errorbar is True, it will plot data with errorbars using 
            matplotlib.pyplot.errorbar(). Otherwise, it will plot data without
            errorbars using matplotlib.pyplot.plot(). 
            
            If you plot model closure phases (i.e. model is not None), 
            it will plot without errobars regardless of this parameter.
          model (dict-like such as pd.DataFrame, pd.Series, default is None):
            Model data sets. Model closure amplitudes must be given by model["camod"].
            Otherwise, it will plot closure amplitudes in the table (i.e. self["logamp"]).
          **plotargs:
            You can set parameters of matplotlib.pyplot.plot() or 
            matplotlib.pyplot.errorbars().
            Defaults are {'ls': "none", 'marker': "."}.
        '''
        # Set Unit
        if uvunit is None:
            uvunit = self.uvunit

        # Conversion Factor
        conv = self.uvunitconv(unit1="lambda", unit2=uvunit)

        # uvdistance
        if uvdtype.lower().find("ave") * uvdtype.lower().find("mean") == 0:
            uvdist = self["uvdistave"] * conv
        elif uvdtype.lower().find("min") == 0:
            uvdist = self["uvdistmin"] * conv
        elif uvdtype.lower().find("max") == 0:
            uvdist = self["uvdistmax"] * conv
        else:
            print("[Error] uvdtype=%s is not available." % (uvdtype))
            return -1

        # Label
        unitlabel = self.get_unitlabel(uvunit)

        # plotting data
        if model is not None:
            plt.plot(uvdist, model["camod"], ls=ls, marker=marker, **plotargs)
        elif errorbar:
            plt.errorbar(uvdist, self["logamp"], self["logsigma"],
                         ls=ls, marker=marker, **plotargs)
        else:
            plt.plot(uvdist, self["logamp"], ls=ls, marker=marker, **plotargs)
        plt.xlabel(r"Baseline Length (%s)" % (unitlabel))
        plt.ylabel(r"Log Closure Amplitude")
        plt.xlim(0,)


class _UVSeries(pd.Series):
    @property
    def _constructor(self):
        return _UVSeries

    @property
    def _constructor_expanddim(self):
        return _UVTable


class _VisSeries(_UVSeries):
    @property
    def _constructor(self):
        return _VisSeries

    @property
    def _constructor_expanddim(self):
        return VisTable


class _BSSeries(_UVSeries):

    @property
    def _constructor(self):
        return _BSSeries

    @property
    def _constructor_expanddim(self):
        return BSTable


class _CASeries(_UVSeries):

    @property
    def _constructor(self):
        return _CASeries

    @property
    def _constructor_expanddim(self):
        return CATable

#-------------------------------------------------------------------------------
#  Read CSV table files
#-------------------------------------------------------------------------------
def read_vistable(filename, uvunit=None, **args):
    '''
    This fuction loads uvdata.VisTable from an input csv file using pd.read_csv().

    Args:
      filename:
        str, pathlib.Path, py._path.local.LocalPath or any object with a read()
        method (such as a file handle or StringIO)
      uvunit (str, default is None): 
        units of uvdistance for plotting. If uvunit is None, uvunit will be
        inferred from the maximum baseline length. Availables are ["l[ambda]", 
        "kl[ambda]", "ml[ambda]", "gl[ambda]", "m", "km"].

    Returns:
      uvdata.VisTable object
    '''
    table = VisTable(pd.read_csv(filename, **args))

    maxuvd = np.max(table["uvdist"])

    if uvunit is None:
        if maxuvd < 1e3:
            table.uvunit = "lambda"
        elif maxuvd < 1e6:
            table.uvunit = "klambda"
        elif maxuvd < 1e9:
            table.uvunit = "mlambda"
        else:
            table.uvunit = "glambda"
    else:
        table.uvunit = uvunit

    return table


def read_bstable(filename, uvunit=None, **args):
    '''
    This fuction loads uvdata.BSTable from an input csv file using pd.read_csv().

    Args:
      filename:
        str, pathlib.Path, py._path.local.LocalPath or any object with a read()
        method (such as a file handle or StringIO)
      uvunit (str, default is None): 
        units of uvdistance for plotting. If uvunit is None, uvunit will be
        inferred from the maximum baseline length. Availables are ["l[ambda]", 
        "kl[ambda]", "ml[ambda]", "gl[ambda]", "m", "km"].

    Returns:
      uvdata.BSTable object
    '''
    table = BSTable(pd.read_csv(filename, **args))

    maxuvd = np.max(table["uvdistmax"])

    if uvunit is None:
        if maxuvd < 1e3:
            table.uvunit = "lambda"
        elif maxuvd < 1e6:
            table.uvunit = "klambda"
        elif maxuvd < 1e9:
            table.uvunit = "mlambda"
        else:
            table.uvunit = "glambda"
    else:
        table.uvunit = uvunit

    return table


def read_catable(filename, uvunit=None, **args):
    '''
    This fuction loads uvdata.CATable from an input csv file using pd.read_csv().

    Args:
      filename:
        str, pathlib.Path, py._path.local.LocalPath or any object with a read()
        method (such as a file handle or StringIO)
      uvunit (str, default is None): 
        units of uvdistance for plotting. If uvunit is None, uvunit will be
        inferred from the maximum baseline length. Availables are ["l[ambda]", 
        "kl[ambda]", "ml[ambda]", "gl[ambda]", "m", "km"].

    Returns:
      uvdata.CATable object
    '''
    table = CATable(pd.read_csv(filename, **args))

    maxuvd = np.max(table["uvdistmax"])

    if uvunit is None:
        if maxuvd < 1e3:
            table.uvunit = "lambda"
        elif maxuvd < 1e6:
            table.uvunit = "klambda"
        elif maxuvd < 1e9:
            table.uvunit = "mlambda"
        else:
            table.uvunit = "glambda"
    else:
        table.uvunit = uvunit

    return table


#-------------------------------------------------------------------------------
# Subfunctions for UVFITS
#-------------------------------------------------------------------------------
def _bindstokes(dataarray, stokes, stokes1, stokes2, factr1, factr2):
    '''
    This is a subfunction for uvdata.UVFITS.
    '''
    stokesids = np.asarray(dataarray["stokes"], dtype=np.int64)
    istokes1 = np.where(stokesids == stokes1)[0][0]
    istokes2 = np.where(stokesids == stokes2)[0][0]
    coords = dataarray.coords
    coords["stokes"] = np.asarray([stokes], dtype=np.float64)
    outdata = xr.DataArray(dataarray.data[:, :, :, :, :, istokes1:istokes1 + 1, :],
                           coords=coords,
                           dims=dataarray.dims)

    vcomp1 = dataarray.data[:, :, :, :, :, istokes1, 0] + \
        1j * dataarray.data[:, :, :, :, :, istokes1, 1]
    vweig1 = dataarray.data[:, :, :, :, :, istokes1, 2]
    vcomp2 = dataarray.data[:, :, :, :, :, istokes2, 0] + \
        1j * dataarray.data[:, :, :, :, :, istokes2, 1]
    vweig2 = dataarray.data[:, :, :, :, :, istokes2, 2]

    vcomp = factr1 * vcomp1 + factr2 * vcomp2
    vweig = np.power(np.abs(factr1)**2 / vweig1 +
                     np.abs(factr2)**2 / vweig2, -1)

    select = vweig1 <= 0
    select += vweig2 <= 0
    select += vweig <= 0
    select += np.isnan(vweig1)
    select += np.isnan(vweig2)
    select += np.isnan(vweig)
    select += np.isinf(vweig1)
    select += np.isinf(vweig2)
    select += np.isinf(vweig)
    vweig[np.where(select)] = 0.0

    outdata.data[:, :, :, :, :, 0, 0] = np.real(vcomp)
    outdata.data[:, :, :, :, :, 0, 1] = np.imag(vcomp)
    outdata.data[:, :, :, :, :, 0, 2] = vweig

    return outdata


#-------------------------------------------------------------------------------
# Subfunctions for VisTable
#-------------------------------------------------------------------------------
def _getblid(st1, st2, Nst):
    '''
    This function is a subfunction for uvdata.VisTable.
    It calculates an id number of the baseline from a given set of
    station numbers and the total number of stations.

    Arguments: 
      st1 (int): the first station ID number
      st2 (int): the second station ID number
      Nst (int): the total number of stations

    Return (int): the baseline ID number
    '''
    stmin = np.min([st1, st2])
    stmax = np.max([st1, st2])

    return stmin * Nst - stmin * (stmin + 1) / 2 + stmax - stmin - 1


def _calc_caamp(vistable, catable, idx,
                stid1, stid2, stid3, stid4, sts, Nsts, Nbl,
                matrix, rank, Nca, Ncamax):
    '''
    This is a sub function for uvdata.VisTable.make_catable, which calculates
    a closure amplitude on a given combination of stations at a given time
    '''
    # if we already found the maximum number of triangles, skip the process
    if Nca >= Ncamax:
        return None

    # station number
    st1 = sts[stid1]
    st2 = sts[stid2]
    st3 = sts[stid3]
    st4 = sts[stid4]

    # baseline ids
    blid1 = _getblid(stid1, stid2, Nsts)
    blid2 = _getblid(stid3, stid4, Nsts)
    blid3 = _getblid(stid1, stid3, Nsts)
    blid4 = _getblid(stid2, stid4, Nsts)

    # calculate conversion matrix
    row = np.zeros(Nbl)
    row[blid1] = 1
    row[blid2] = 1
    row[blid3] = -1
    row[blid4] = -1
    if matrix is None:
        tmpmatrix = np.asarray([row])
    else:
        tmpmatrix = np.append(matrix, row).reshape(Nca + 1, Nbl)

    # Check if this triangle is redundant
    tmprank = np.linalg.matrix_rank(tmpmatrix)
    if rank == tmprank:
        return None

    # Check if corresponding baseline data exist
    isbl1 = True
    isbl2 = True
    isbl3 = True
    isbl4 = True
    bl1idx = idx & (vistable["st1"] == np.min([st1, st2])) & (
        vistable["st2"] == np.max([st1, st2]))
    bl2idx = idx & (vistable["st1"] == np.min([st3, st4])) & (
        vistable["st2"] == np.max([st3, st4]))
    bl3idx = idx & (vistable["st1"] == np.min([st1, st3])) & (
        vistable["st2"] == np.max([st1, st3]))
    bl4idx = idx & (vistable["st1"] == np.min([st2, st4])) & (
        vistable["st2"] == np.max([st2, st4]))
    if np.where(bl1idx)[0].shape[0] == 0:
        isbl1 = False
    if np.where(bl2idx)[0].shape[0] == 0:
        isbl2 = False
    if np.where(bl3idx)[0].shape[0] == 0:
        isbl3 = False
    if np.where(bl4idx)[0].shape[0] == 0:
        isbl4 = False
    if False in [isbl1, isbl2, isbl3, isbl4]:
        return None

    # calculate bi-spectrum
    bl1data = vistable.loc[bl1idx, :].reset_index(drop=True).loc[0, :]
    bl2data = vistable.loc[bl2idx, :].reset_index(drop=True).loc[0, :]
    bl3data = vistable.loc[bl3idx, :].reset_index(drop=True).loc[0, :]
    bl4data = vistable.loc[bl4idx, :].reset_index(drop=True).loc[0, :]

    ratio_1 = bl1data.loc["sigma"] / bl1data.loc["amp"]
    ratio_2 = bl2data.loc["sigma"] / bl2data.loc["amp"]
    ratio_3 = bl3data.loc["sigma"] / bl3data.loc["amp"]
    ratio_4 = bl4data.loc["sigma"] / bl4data.loc["amp"]
    amp = bl1data.loc["amp"] * bl2data.loc["amp"] / \
        bl3data.loc["amp"] / bl4data.loc["amp"]
    logamp = np.log(amp)
    logsigma = np.sqrt((ratio_1)**2 + (ratio_2)**2 +
                       (ratio_3)**2 + (ratio_4)**2)
    sigma = amp * logsigma

    catable["jd"].append(bl1data.loc["jd"])
    catable["year"].append(bl1data.loc["year"])
    catable["doy"].append(bl1data.loc["doy"])
    catable["hour"].append(bl1data.loc["hour"])
    catable["min"].append(bl1data.loc["min"])
    catable["sec"].append(bl1data.loc["sec"])
    catable["u1"].append(bl1data.loc["u"])
    catable["v1"].append(bl1data.loc["v"])
    catable["w1"].append(bl1data.loc["w"])
    catable["u2"].append(bl2data.loc["u"])
    catable["v2"].append(bl2data.loc["v"])
    catable["w2"].append(bl2data.loc["w"])
    catable["u3"].append(bl3data.loc["u"])
    catable["v3"].append(bl3data.loc["v"])
    catable["w3"].append(bl3data.loc["w"])
    catable["u4"].append(bl4data.loc["u"])
    catable["v4"].append(bl4data.loc["v"])
    catable["w4"].append(bl4data.loc["w"])
    catable["st1"].append(st1)
    catable["st2"].append(st2)
    catable["st3"].append(st3)
    catable["st4"].append(st4)
    catable["ch"].append(bl1data.loc["ch"])
    catable["amp"].append(amp)
    catable["logamp"].append(logamp)
    catable["sigma"].append(sigma)
    catable["logsigma"].append(logsigma)

    # update rank and matrix
    rank += 1
    Nca += 1
    matrix = tmpmatrix.copy()

    return catable, rank, Nca, matrix

def _calc_dbeam(fitsdata, vistable, errweight=0, ftsign=+1):
    '''
    Calculate an array and total flux of dirty beam from the input visibility data

    keywords:
      fitsdata:
        input imdata.IMFITS object
      vistable:
        input visibility data
      errweight (float):
        index for errer weighting
      ftsign (integer):
        a sign for fourier matrix
    '''
    # create output fits
    outfitsdata = copy.deepcopy(fitsdata)

    # read uv information
    M = len(vistable)
    U = np.float64(vistable["u"])
    V = np.float64(vistable["v"])

    # create visibilies and error weighting
    Vis_point = np.ones(len(vistable), dtype=np.complex128)
    if errweight != 0:
        sigma = np.float64(vistable["sigma"])
        weight = np.power(sigma, errweight)
        Vis_point *= weight / np.sum(weight)

    # create matrix of X and Y
    Npix = outfitsdata.header["nx"] * outfitsdata.header["ny"]
    X, Y = outfitsdata.get_xygrid(angunit="deg", twodim=True)
    X = np.radians(X)
    Y = np.radians(Y)
    X = X.reshape(Npix)
    Y = Y.reshape(Npix)

    # create matrix of A
    if ftsign > 0:
        factor = 2 * np.pi
    elif ftsign < 0:
        factor = -2 * np.pi
    A = linalg.blas.dger(factor, X, U)
    A += linalg.blas.dger(factor, Y, V)
    A = np.exp(1j * A) / M

    # calculate synthesized beam
    dbeam = np.real(A.dot(Vis_point))
    dbtotalflux = np.sum(dbeam)
    dbeam /= dbtotalflux

    # save as fitsdata
    dbeam = dbeam.reshape((outfitsdata.header["ny"], outfitsdata.header["nx"]))
    for idxs in np.arange(outfitsdata.header["ns"]):
        for idxf in np.arange(outfitsdata.header["nf"]):
            outfitsdata.data[idxs, idxf] = dbeam[:]

    outfitsdata.update_fits()
    return outfitsdata, dbtotalflux


def _calc_bparms(vistable):
    '''
    Infer beam parameters (major size, minor size, position angle)

    keywords:
      vistable: input visibility data
    '''
    # read uv information
    U = np.float64(vistable["u"])
    V = np.float64(vistable["v"])

    # calculate minor size of the beam
    uvdist = np.sqrt(U * U + V * V)
    maxuvdist = np.max(uvdist)
    mina = np.rad2deg(1 / maxuvdist) * 0.6

    # calculate PA
    index = np.argmax(uvdist)
    angle = np.rad2deg(np.arctan2(U[index], V[index]))

    # rotate uv coverage for calculating major size
    PA = angle + 90
    cosPA = np.cos(np.radians(PA))
    sinPA = np.sin(np.radians(PA))
    newU = U * cosPA - V * sinPA
    newV = U * sinPA + V * cosPA

    # calculate major size of the beam
    maxV = np.max(np.abs(newV))
    maja = np.rad2deg(1 / maxV) * 0.6

    return maja, mina, PA


def _gauss_func(X, Y, maja, mina, PA, x0=0., y0=0., scale=1.):
    '''
    Calculate 2-D gauss function

    keywords:
      X: 2-D array of x-axis
      Y: 2-D array of y-axis
      maja (float): major size of the gauss
      mina (float): minor size
      PA (float): position angle
      x0 (float): value of x-position at the center of the gauss
      y0 (float): value of y-position at the center of the gauss
      scale (float): scaling factor
    '''
    # scaling
    maja *= scale
    mina *= scale

    # calculate gauss function
    cosPA = np.cos(np.radians(PA))
    sinPA = np.sin(np.radians(PA))
    L = ((X * sinPA + Y * cosPA)**2) / (maja**2) + \
        ((X * cosPA - Y * sinPA)**2) / (mina**2)
    return np.exp(-L * 4 * np.log(2))


def _fit_chisq(parms, X, Y, dbeam):
    '''
    Calculate residuals of two 2-D array

    keywords:
      parms: information of clean beam
      X: 2-D array of x-axis
      Y: 2-D array of y-axis
      dbeam: an array of dirty beam
    '''
    # get parameters of clean beam
    (maja, mina, angle) = parms

    # calculate clean beam and residuals
    cbeam = _gauss_func(X, Y, maja, mina, angle)
    cbeam /= np.max(cbeam)
    if cbeam.size == dbeam.size:
        return (dbeam - cbeam).reshape(dbeam.size)
    else:
        print("not equal the size of two beam array")