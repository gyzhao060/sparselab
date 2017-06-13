#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
A python module sparselab.imaging

This is a submodule of sparselab for imaging static images.
'''
__author__ = "Kazunori Akiyama and Kazuki kuramochi"
__version__ = "1.0"
__maintainer__ = "Kazunori Akiyama"
__date__ = "Jan 6 2017"

#-------------------------------------------------------------------------------
# Default Parameters
#-------------------------------------------------------------------------------
lbfgsbprms = {
    "m": 10,
    "factr": 1e1,
    "pgtol": 0.
}

#-------------------------------------------------------------------------------
# Reconstract static imaging
#-------------------------------------------------------------------------------
def static_dft_imaging(
        initimage, imagewin=None,
        vistable=None, amptable=None, bstable=None, catable=None,
        lambl1=-1., lambtv=-1, lambtsv=-1, normlambda=True, nonneg=True, niter=1000,
        totalflux=None, fluxconst=False,
        istokes=0, ifreq=0):
    '''

    '''
    import numpy as np
    import pandas as pd
    import fortlib
    import copy

    # Check Arguments
    if ((vistable is None) and (amptable is None) and
            (bstable is None) and (catable is None)):
        print("Error: No data are input.")
        return -1

    # Total Flux constraint: Sanity Check
    dofluxconst = False
    if ((vistable is None) and (amptable is None) and (totalflux is None)):
        print("Error: No absolute amplitude information in the input data.")
        print("       You need to set the total flux constraint by totalflux.")
        return -1
    elif ((totalflux is None) and (fluxconst is True)):
        print("Error: No total flux is specified, although you set fluxconst=True.")
        print("       You need to set the total flux constraint by totalflux.")
        return -1
    elif ((vistable is None) and (amptable is None) and
          (totalflux is not None) and (fluxconst is False)):
        print("Warning: No absolute amplitude information in the input data.")
        print("         The total flux will be constrained, although you do not set fluxconst=True.")
        dofluxconst = True
    elif fluxconst is True:
        dofluxconst = True

    # get initial images
    Iin = np.float64(initimage.data[istokes, ifreq])

    # size of images
    Nx = np.int32(initimage.header["nx"])
    Ny = np.int32(initimage.header["ny"])
    Nyx = Nx * Ny

    # pixel coordinates
    x, y = initimage.get_xygrid(twodim=True, angunit="rad")
    x = np.float64(x)
    y = np.float64(y)
    xidx = np.int32(np.arange(Nx) + 1)
    yidx = np.int32(np.arange(Ny) + 1)
    xidx, yidx = np.meshgrid(xidx, yidx)

    # apply the imaging area
    if imagewin is None:
        print("Imaging Window: Not Specified. We solve the image on all the pixels.")
        Iin = Iin.reshape(Nyx)
        x = x.reshape(Nyx)
        y = y.reshape(Nyx)
        xidx = xidx.reshape(Nyx)
        yidx = yidx.reshape(Nyx)
    else:
        print("Imaging Window: Specified. Images will be solved on specified pixels.")
        idx = np.where(imagewin)
        Iin = Iin[idx]
        x = x[idx]
        y = y[idx]
        xidx = xidx[idx]
        yidx = yidx[idx]

    # dammy array
    dammyreal = np.zeros(1, dtype=np.float64)

    # Full Complex Visibility
    Ndata = 0
    if dofluxconst:
        print("Total Flux Constraint: set to %g" % (totalflux))
        totalfluxdata = {
            'u': [0.],
            'v': [0.],
            'amp': [totalflux],
            'phase': [0.],
            'sigma': [1.]
        }
        totalfluxdata = pd.DataFrame(totalfluxdata)
        fcvtable = pd.concat([totalfluxdata, vistable], ignore_index=True)
    else:
        print("Total Flux Constraint: disabled.")
        if vistable is None:
            fcvtable = None
        else:
            fcvtable = vistable.copy()

    if fcvtable is None:
        isfcv = False
        vfcvr = dammyreal
        vfcvi = dammyreal
        varfcv = dammyreal
    else:
        isfcv = True
        phase = np.deg2rad(np.array(fcvtable["phase"], dtype=np.float64))
        amp = np.array(fcvtable["amp"], dtype=np.float64)
        vfcvr = amp * np.cos(phase)
        vfcvi = amp * np.sin(phase)
        varfcv = np.square(np.array(fcvtable["sigma"], dtype=np.float64))
        Ndata += len(vfcvr)
        del phase, amp

    # Visibility Amplitude
    if amptable is None:
        isamp = False
        vamp = dammyreal
        varamp = dammyreal
    else:
        isamp = True
        vamp = np.array(amptable["amp"], dtype=np.float64)
        varamp = np.square(np.array(amptable["sigma"], dtype=np.float64))
        Ndata += len(vamp)

    # Closure Phase
    if bstable is None:
        iscp = False
        cp = dammyreal
        varcp = dammyreal
    else:
        iscp = True
        cp = np.deg2rad(np.array(bstable["phase"], dtype=np.float64))
        varcp = np.square(
            np.array(bstable["sigma"] / bstable["amp"], dtype=np.float64))
        Ndata += len(cp)

    # Closure Amplitude
    if catable is None:
        isca = False
        ca = dammyreal
        varca = dammyreal
    else:
        isca = True
        ca = np.array(catable["logamp"], dtype=np.float64)
        varca = np.square(np.array(catable["logsigma"], dtype=np.float64))
        Ndata += len(ca)

    # Sigma for the total flux
    if dofluxconst:
        varfcv[0] = np.square(fcvtable.loc[0, "amp"] / (Ndata - 1.))

    # Normalize Lambda
    if normlambda:
        # Guess Total Flux
        if totalflux is None:
            fluxscale = []
            if vistable is not None:
                fluxscale.append(vistable["amp"].max())
            if amptable is not None:
                fluxscale.append(amptable["amp"].max())
            fluxscale = np.max(fluxscale)
            print("Flux Scaling Factor for lambda: The expected total flux is not given.")
            print(
                "                                The scaling factor will be %g" % (fluxscale))
        else:
            fluxscale = np.float64(totalflux)
            print(
                "Flux Scaling Factor for lambda: The scaling factor will be %g" % (fluxscale))
        lambl1_sim = lambl1 / fluxscale
        lambtv_sim = lambtv / fluxscale / 4.
        lambtsv_sim = lambtsv / fluxscale**2 / 4.
    else:
        lambl1_sim = lambl1
        lambtv_sim = lambtv
        lambtsv_sim = lambtsv

    # get uv coordinates and uv indice
    u, v, uvidxfcv, uvidxamp, uvidxcp, uvidxca = get_uvlist(
        fcvtable=fcvtable, amptable=amptable, bstable=bstable, catable=catable
    )

    # run imaging
    Iout = fortlib.static_imaging_dft(
        # Images
        iin=Iin, x=x, y=y, xidx=xidx, yidx=yidx, nx=Nx, ny=Ny,
        # UV coordinates,
        u=u, v=v,
        # Imaging Parameters
        lambl1=lambl1_sim, lambtv=lambtv_sim, lambtsv=lambtsv_sim,
        nonneg=nonneg, niter=niter,
        # Full Complex Visibilities
        isfcv=isfcv, uvidxfcv=uvidxfcv, vfcvr=vfcvr, vfcvi=vfcvi, varfcv=varfcv,
        # Visibility Ampltiudes
        isamp=isamp, uvidxamp=uvidxamp, vamp=vamp, varamp=varamp,
        # Closure Phase
        iscp=iscp, uvidxcp=uvidxcp, cp=cp, varcp=varcp,
        # Closure Amplituds
        isca=isca, uvidxca=uvidxca, ca=ca, varca=varca,
        # Following 3 parameters are for L-BFGS-B
        m=np.int32(lbfgsbprms["m"]), factr=np.float64(lbfgsbprms["factr"]),
        pgtol=np.float64(lbfgsbprms["pgtol"])
    )

    outimage = copy.deepcopy(initimage)
    outimage.data[istokes, ifreq] = 0.
    for i in np.arange(len(xidx)):
        outimage.data[istokes, ifreq, yidx[i] - 1, xidx[i] - 1] = Iout[i]
    outimage.update_fits()
    return outimage


def static_dft_stats(
        initimage, imagewin=None,
        vistable=None, amptable=None, bstable=None, catable=None,
        lambl1=1., lambtv=-1, lambtsv=1, normlambda=True,
        totalflux=None, fluxconst=False,
        istokes=0, ifreq=0, fulloutput=True, **args):
    '''

    '''
    import numpy as np
    import pandas as pd
    import fortlib
    import copy
    import collections

    # Check Arguments
    if ((vistable is None) and (amptable is None) and
            (bstable is None) and (catable is None)):
        print("Error: No data are input.")
        return -1

    # Total Flux constraint: Sanity Check
    dofluxconst = False
    if ((vistable is None) and (amptable is None) and (totalflux is None)):
        print("Error: No absolute amplitude information in the input data.")
        print("       You need to set the total flux constraint by totalflux.")
        return -1
    elif ((totalflux is None) and (fluxconst is True)):
        print("Error: No total flux is specified, although you set fluxconst=True.")
        print("       You need to set the total flux constraint by totalflux.")
        return -1
    elif ((vistable is None) and (amptable is None) and
          (totalflux is not None) and (fluxconst is False)):
        print("Warning: No absolute amplitude information in the input data.")
        print("         The total flux will be constrained, although you do not set fluxconst=True.")
        dofluxconst = True
    elif fluxconst is True:
        dofluxconst = True

    # get initial images
    Iin = np.float64(initimage.data[istokes, ifreq])

    # size of images
    Nx = np.int32(initimage.header["nx"])
    Ny = np.int32(initimage.header["ny"])
    Nyx = Nx * Ny

    # pixel coordinates
    x, y = initimage.get_xygrid(twodim=True, angunit="rad")
    x = np.float64(x)
    y = np.float64(y)
    xidx = np.int32(np.arange(Nx) + 1)
    yidx = np.int32(np.arange(Ny) + 1)
    xidx, yidx = np.meshgrid(xidx, yidx)

    # apply the imaging area
    if imagewin is None:
        print("Imaging Window: Not Specified. We solve the image on all the pixels.")
        Iin = Iin.reshape(Nyx)
        x = x.reshape(Nyx)
        y = y.reshape(Nyx)
        xidx = xidx.reshape(Nyx)
        yidx = yidx.reshape(Nyx)
    else:
        print("Imaging Window: Specified. Images will be solved on specified pixels.")
        idx = np.where(imagewin)
        Iin = Iin[idx]
        x = x[idx]
        y = y[idx]
        xidx = xidx[idx]
        yidx = yidx[idx]

    # dammy array
    dammyreal = np.zeros(1, dtype=np.float64)

    # Full Complex Visibility
    Ndata = 0
    if vistable is None:
        fcvtable = None
    else:
        fcvtable = vistable.copy()

    if fcvtable is None:
        isfcv = False
        vfcvr = dammyreal
        vfcvi = dammyreal
        varfcv = dammyreal
    else:
        isfcv = True
        phase = np.deg2rad(np.array(fcvtable["phase"], dtype=np.float64))
        amp = np.array(fcvtable["amp"], dtype=np.float64)
        vfcvr = amp * np.cos(phase)
        vfcvi = amp * np.sin(phase)
        varfcv = np.square(np.array(fcvtable["sigma"], dtype=np.float64))
        Ndata += len(vfcvr)
        del phase, amp

    # Visibility Amplitude
    if amptable is None:
        isamp = False
        vamp = dammyreal
        varamp = dammyreal
    else:
        isamp = True
        vamp = np.array(amptable["amp"], dtype=np.float64)
        varamp = np.square(np.array(amptable["sigma"], dtype=np.float64))
        Ndata += len(vamp)

    # Closure Phase
    if bstable is None:
        iscp = False
        cp = dammyreal
        varcp = dammyreal
    else:
        iscp = True
        cp = np.deg2rad(np.array(bstable["phase"], dtype=np.float64))
        varcp = np.square(
            np.array(bstable["sigma"] / bstable["amp"], dtype=np.float64))
        Ndata += len(cp)

    # Closure Amplitude
    if catable is None:
        isca = False
        ca = dammyreal
        varca = dammyreal
    else:
        isca = True
        ca = np.array(catable["logamp"], dtype=np.float64)
        varca = np.square(np.array(catable["logsigma"], dtype=np.float64))
        Ndata += len(ca)

    # Sigma for the total flux
    # if dofluxconst:
    #    varfcv[0] = np.square(fcvtable.loc[0,"amp"]/(Ndata-1.))

    # Normalize Lambda
    if normlambda:
        # Guess Total Flux
        if totalflux is None:
            fluxscale = []
            if vistable is not None:
                fluxscale.append(vistable["amp"].max())
            if amptable is not None:
                fluxscale.append(amptable["amp"].max())
            fluxscale = np.max(fluxscale)
            print("Flux Scaling Factor for lambda: The expected total flux is not given.")
            print(
                "                                The scaling factor will be %g" % (fluxscale))
        else:
            fluxscale = np.float64(totalflux)
            print(
                "Flux Scaling Factor for lambda: The scaling factor will be %g" % (fluxscale))
        lambl1_sim = lambl1 / fluxscale
        lambtv_sim = lambtv / fluxscale / 4.
        lambtsv_sim = lambtsv / fluxscale**2 / 4.
    else:
        lambl1_sim = lambl1
        lambtv_sim = lambtv
        lambtsv_sim = lambtsv

    # get uv coordinates and uv indice
    u, v, uvidxfcv, uvidxamp, uvidxcp, uvidxca = get_uvlist(
        fcvtable=fcvtable, amptable=amptable, bstable=bstable, catable=catable
    )

    # calculate all
    out = fortlib.static_imaging_dft.statistics(
        # Images
        iin=Iin, x=x, y=y, xidx=xidx, yidx=yidx, nx=Nx, ny=Ny,
        # UV coordinates,
        u=u, v=v,
        # Imaging Parameters
        lambl1=lambl1_sim, lambtv=lambtv_sim, lambtsv=lambtsv_sim,
        # Full Complex Visibilities
        isfcv=isfcv, uvidxfcv=uvidxfcv, vfcvr=vfcvr, vfcvi=vfcvi, varfcv=varfcv,
        # Visibility Ampltiudes
        isamp=isamp, uvidxamp=uvidxamp, vamp=vamp, varamp=varamp,
        # Closure Phase
        iscp=iscp, uvidxcp=uvidxcp, cp=cp, varcp=varcp,
        # Closure Amplituds
        isca=isca, uvidxca=uvidxca, ca=ca, varca=varca
    )
    stats = collections.OrderedDict()
    # Cost and Chisquares
    stats["cost"] = out[0]
    stats["chisq"] = out[2]
    stats["isfcv"] = isfcv
    stats["isamp"] = isamp
    stats["iscp"] = iscp
    stats["isca"] = isca
    stats["chisqfcv"] = out[3] * len(vfcvr)
    stats["chisqamp"] = out[4] * len(vamp)
    stats["chisqcp"] = out[5] * len(cp)
    stats["chisqca"] = out[6] * len(ca)
    stats["rchisqfcv"] = out[3]
    stats["rchisqamp"] = out[4]
    stats["rchisqcp"] = out[5]
    stats["rchisqca"] = out[6]

    # Regularization functions
    if lambl1 > 0:
        stats["lambl1"] = lambl1
        stats["lambl1_sim"] = lambl1_sim
        stats["l1"] = out[7]
        stats["l1cost"] = out[7] * lambl1_sim
    else:
        stats["lambl1"] = 0.
        stats["lambl1_sim"] = 0.
        stats["l1"] = out[7]
        stats["l1cost"] = 0.

    if lambtv > 0:
        stats["lambtv"] = lambtv
        stats["lambtv_sim"] = lambtv_sim
        stats["tv"] = out[8]
        stats["tvcost"] = out[8] * lambtv_sim
    else:
        stats["lambtv"] = 0.
        stats["lambtv_sim"] = 0.
        stats["tv"] = out[8]
        stats["tvcost"] = 0.

    if lambtsv > 0:
        stats["lambtsv"] = lambtsv
        stats["lambtsv_sim"] = lambtsv_sim
        stats["tsv"] = out[9]
        stats["tsvcost"] = out[9] * lambtsv_sim
    else:
        stats["lambtsv"] = 0.
        stats["lambtsv_sim"] = 0.
        stats["tsv"] = out[9]
        stats["tsvcost"] = 0.

    if fulloutput:
        # gradcost
        gradcostimage = initimage.data[istokes, ifreq, :, :].copy()
        gradcostimage[:, :] = 0.
        for i in np.arange(len(xidx)):
            gradcostimage[yidx[i] - 1, xidx[i] - 1] = out[1][i]
        stats["gradcost"] = gradcostimage
        del gradcostimage

        if isfcv:
            stats["fcvampmod"] = np.sqrt(out[10] * out[10] + out[11] * out[11])
            stats["fcvphamod"] = np.angle(out[10] + 1j * out[11], deg=True)
            stats["fcvrmod"] = out[10]
            stats["fcvimod"] = out[11]
            stats["fcvres"] = out[12]
        else:
            stats["fcvampmod"] = None
            stats["fcvphamod"] = None
            stats["fcvres"] = None

        if isamp:
            stats["ampmod"] = out[13]
            stats["ampres"] = out[14]
        else:
            stats["ampmod"] = None
            stats["ampres"] = None

        if iscp:
            stats["cpmod"] = np.rad2deg(out[15])
            stats["cpres"] = np.rad2deg(out[16])
        else:
            stats["cpmod"] = None
            stats["cpres"] = None

        if isca:
            stats["camod"] = out[17]
            stats["cares"] = out[18]
        else:
            stats["camod"] = None
            stats["cares"] = None

    return stats


def iterative_imaging(initimage, imageprm, Niter=10,
                      dothres=True, threstype="hard", threshold=0.3, 
                      doshift=True, shifttype="com",
                      doconv=True, convprm={}, 
                      save_totalflux=False):
    import numpy as np
    
    outimage = static_dft_imaging(initimage,**imageprm)
    oldcost = static_dft_stats(outimage, fulloutput=False, **imageprm)["cost"]
    for i in np.arange(Niter-1):
        # Edit Images
        if dothres:
            if threstype == "soft":
                outimage = outimage.soft_threshold(threshold=threshold, 
                                                   save_totalflux=save_totalflux)
            else:
                outimage = outimage.hard_threshold(threshold=threshold, 
                                                   save_totalflux=save_totalflux)
        if doshift:
            if shifttype == "peak":
                outimage = outimage.peakshift(save_totalflux=save_totalflux)
            else:
                outimage = outimage.comshift(save_totalflux=save_totalflux)
        if doconv:
            outimage = outimage.gauss_convolve(save_totalflux=save_totalflux, **convprm)
        
        # Imaging Again
        newimage = static_dft_imaging(outimage,**imageprm)
        newcost = static_dft_stats(newimage, fulloutput=False, **imageprm)["cost"]
        
        if oldcost < newcost:
            print("No improvement in cost fucntions. Don't update image.")
        else:
            oldcost = newcost
            outimage = newimage
    return outimage


def static_dft_pipeline(
        initimage,
        imagefunc=iterative_imaging,
        imageprm={},
        imagefargs={},
        angunit="uas",
        uvunit="gl",
        workdir="./",
        sumtablefile="summary.csv",
        lambl1s=[-1.],
        lambtvs=[-1.],
        lambtsvs=[-1.]):
    import itertools
    import numpy as np
    import pandas as pd
    import os
    import sparselab.util as util
    
    if not os.path.isdir(workdir):
        os.makedirs(workdir)
    
    # Lambda Parameters
    lambl1s = -np.sort(-np.asarray(lambl1s))
    lambtvs = -np.sort(-np.asarray(lambtvs))
    lambtsvs = -np.sort(-np.asarray(lambtsvs))
    nl1 = len(lambl1s)
    ntv = len(lambtvs)
    ntsv= len(lambtsvs)
    
    # Summary Data
    sumtable = pd.DataFrame()
    
    # Start Imaging
    for itsv,itv,il1 in itertools.product(np.arange(ntsv),
                                          np.arange(ntv),
                                          np.arange(nl1)):
        header = "tsv%02d.tv%02d.l1%02d"%(itsv,itv,il1)
        # output
        imageprm["lambl1"]=lambl1s[il1]
        imageprm["lambtv"]=lambtvs[itv]
        imageprm["lambtsv"]=lambtsvs[itsv]
        
        # Imaging
        outimage = imagefunc(initimage, imageprm=imageprm, **imagefargs)
        filename = header+".fits"
        outimage.save_fits(os.path.join(workdir,filename))
        
        # Statistics
        outstats = static_dft_stats(outimage, fulloutput=False, **imageprm)
        tmptable = pd.DataFrame([outstats.values()], columns=outstats.keys())
        tmptable["itsv"] = itsv
        tmptable["itv"] = itv
        tmptable["il1"] = il1
        sumtable = pd.concat([sumtable,tmptable], ignore_index=True)
        sumtable.to_csv(os.path.join(workdir,sumtablefile),**util.args_tocsv)
        
        # Make Plots
        filename = header+".summary.pdf"
        filename = os.path.join(workdir,filename)
        outstats = static_dft_plots(outimage, imageprm, filename=filename, 
                   angunit=angunit, uvunit=uvunit)
    return sumtable


def static_dft_plots(outimage, imageprm={}, filename=None, 
                     angunit="mas", uvunit="ml", plotargs={'ms':1.,}):
    import matplotlib.pyplot as plt
    import matplotlib
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.ticker import NullFormatter
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    import numpy as np
    
    from sparselab.util import matplotlibrc
    from sparselab.uvdata import VisTable, BSTable, CATable
    
    isinteractive = plt.isinteractive()
    backend = matplotlib.rcParams["backend"]
    
    if isinteractive:
        plt.ioff()
        matplotlib.use('Agg')
    
    nullfmt = NullFormatter()
    
    # Label
    if uvunit.lower().find("l")==0:
        unitlabel = r"$\lambda$"
    elif uvunit.lower().find("kl")==0:
        unitlabel = r"$10^3 \lambda$"
    elif uvunit.lower().find("ml")==0:
        unitlabel = r"$10^6 \lambda$"
    elif uvunit.lower().find("gl")==0:
        unitlabel = r"$10^9 \lambda$"
    elif uvunit.lower().find("m")==0:
        unitlabel = "m"
    elif uvunit.lower().find("km")==0:
        unitlabel = "km"
    else:
        print("Error: uvunit=%s is not supported" % (unit2))
        return -1
    
    # Get model data
    stats = static_dft_stats(outimage, fulloutput=True, **imageprm)
    
    # Open File
    if filename is not None:
        pdf = PdfPages(filename)
    
    # Save Image
    if filename is not None:
        matplotlibrc(nrows=1,ncols=1,width=600,height=600)
    else:
        matplotlib.rcdefaults()

    plt.figure()
    outimage.imshow(angunit=angunit)
    if filename is not None:
        pdf.savefig()
        plt.close()
    
    
    # Amplitude
    if stats["isfcv"]==True:
        table = VisTable(imageprm["vistable"])
        table["comp"] = table["amp"] * np.exp(1j*np.deg2rad(table["phase"]))
        table["real"] = np.real(table["comp"])
        table["imag"] = np.imag(table["comp"])
        
        normresidr = (stats["fcvrmod"]-table["real"])/table["sigma"]
        normresidi = (stats["fcvimod"]-table["imag"])/table["sigma"]
        normresid = np.concatenate([normresidr,normresidi])
        N = len(normresid)
        
        if filename is not None:
            matplotlibrc(nrows=3,ncols=1,width=600,height=200)
        else:
            matplotlib.rcdefaults()
        
        fig, axs = plt.subplots(nrows=3,ncols=1,sharex=True)
        plt.subplots_adjust(hspace=0)
        
        ax = axs[0]
        plt.sca(ax)
        table.radplot_amp(uvunit=uvunit, color="black", **plotargs)
        table.radplot_amp(uvunit=uvunit, model=stats, modeltype="fcv", color="red", **plotargs)
        plt.xlabel("")
        
        ax = axs[1]
        plt.sca(ax)
        table.radplot_phase(uvunit=uvunit, color="black", **plotargs)
        table.radplot_phase(uvunit=uvunit, model=stats, color="red", **plotargs)
        plt.xlabel("")
        
        ax = axs[2] 
        plt.sca(ax)
        plt.plot(table["uvdist"]*table.uvunitconv("lambda", uvunit),
                 normresidr, ls="none", marker=".", color="blue", label="real", **plotargs)
        plt.plot(table["uvdist"]*table.uvunitconv("lambda", uvunit),
                 normresidi, ls="none", marker=".", color="red", label="imag", **plotargs)
        plt.axhline(0,color="black",ls="--")
        plt.ylabel("Normalized Residuals")
        plt.xlabel(r"Baseline Length (%s)"%(unitlabel))
        plt.legend(ncol=2)
        
        divider = make_axes_locatable(ax) # Histgram
        cax = divider.append_axes("right", size="10%", pad=0.05)
        ymin,ymax = ax.get_ylim()
        xmin = np.min(normresid)
        xmax = np.max(normresid)
        y = np.linspace(ymin,ymax,1000)
        x = 1/np.sqrt(2*np.pi) * np.exp(-y*y/2.)
        cax.hist(normresid, bins=np.int(np.sqrt(N)), normed=True, orientation='horizontal')
        cax.plot(x,y,color="red")
        cax.set_ylim(ax.get_ylim())
        cax.axhline(0,color="black",ls="--")
        cax.yaxis.set_major_formatter(nullfmt)
        cax.xaxis.set_major_formatter(nullfmt)
        if filename is not None:
            pdf.savefig()
            plt.close()
    
    if stats["isamp"]==True:
        table = VisTable(imageprm["amptable"])
        normresid = stats["ampres"]/table["sigma"]
        N = len(normresid)
        
        if filename is not None:
            matplotlibrc(nrows=2,ncols=1,width=600,height=300)
        else:
            matplotlib.rcdefaults()
        
        fig, axs = plt.subplots(nrows=2,ncols=1,sharex=True)
        plt.subplots_adjust(hspace=0)
        
        ax = axs[0]
        plt.sca(ax)
        table.radplot_amp(uvunit=uvunit, color="black", **plotargs)
        table.radplot_amp(uvunit=uvunit, model=stats, modeltype="amp", color="red", **plotargs)
        plt.xlabel("")
        
        ax = axs[1] 
        plt.sca(ax)
        plt.plot(table["uvdist"]*table.uvunitconv("lambda", uvunit),
                 normresid, ls="none", marker=".", color="black", **plotargs)
        plt.axhline(0,color="black",ls="--")
        plt.ylabel("Normalized Residuals")
        plt.xlabel(r"Baseline Length (%s)"%(unitlabel))
        
        divider = make_axes_locatable(ax) # Histgram
        cax = divider.append_axes("right", size="10%", pad=0.05)
        ymin,ymax = ax.get_ylim()
        xmin = np.min(normresid)
        xmax = np.max(normresid)
        y = np.linspace(ymin,ymax,1000)
        x = 1/np.sqrt(2*np.pi) * np.exp(-y*y/2.)
        cax.hist(normresid, bins=np.int(np.sqrt(N)), normed=True, orientation='horizontal')
        cax.plot(x,y,color="red")
        cax.set_ylim(ax.get_ylim())
        cax.axhline(0,color="black",ls="--")
        cax.yaxis.set_major_formatter(nullfmt)
        cax.xaxis.set_major_formatter(nullfmt)
        if filename is not None:
            pdf.savefig()
            plt.close()
    
    # Closure Amplitude
    if stats["isca"]==True:
        table = CATable(imageprm["catable"])
        # Amplitudes
        normresid = stats["cares"]/table["logsigma"]
        N = len(normresid)
        
        if filename is not None:
            matplotlibrc(nrows=2,ncols=1,width=600,height=300)
        else:
            matplotlib.rcdefaults()
        
        fig, axs = plt.subplots(nrows=2,ncols=1,sharex=True)
        plt.subplots_adjust(hspace=0)
        
        ax = axs[0]
        plt.sca(ax)
        table.radplot(uvunit=uvunit, uvdtype="ave", color="black", **plotargs)
        table.radplot(uvunit=uvunit, uvdtype="ave", model=stats, color="red", **plotargs)
        plt.xlabel("")
        
        ax = axs[1] 
        plt.sca(ax)
        plt.plot(table["uvdistave"]*table.uvunitconv("lambda", uvunit),
                 normresid, ls="none", marker=".", color="black", **plotargs)
        plt.axhline(0,color="black",ls="--")
        plt.ylabel("Normalized Residuals")
        plt.xlabel(r"Baseline Length (%s)"%(unitlabel))
        
        divider = make_axes_locatable(ax) # Histgram
        cax = divider.append_axes("right", size="10%", pad=0.05)
        ymin,ymax = ax.get_ylim()
        xmin = np.min(normresid)
        xmax = np.max(normresid)
        y = np.linspace(ymin,ymax,1000)
        x = 1/np.sqrt(2*np.pi) * np.exp(-y*y/2.)
        cax.hist(normresid, bins=np.int(np.sqrt(N)), normed=True, orientation='horizontal')
        cax.plot(x,y,color="red")
        cax.set_ylim(ax.get_ylim())
        cax.axhline(0,color="black",ls="--")
        cax.yaxis.set_major_formatter(nullfmt)
        cax.xaxis.set_major_formatter(nullfmt)
        if filename is not None:
            pdf.savefig()
            plt.close()
    
    # Closure Phase
    if stats["iscp"]==True:
        table = BSTable(imageprm["bstable"])
        # Amplitudes
        normresid = stats["cpres"]/np.rad2deg(table["sigma"]/table["amp"])
        
        if filename is not None:
            matplotlibrc(nrows=2,ncols=1,width=600,height=300)
        else:
            matplotlib.rcdefaults()
        
        fig, axs = plt.subplots(nrows=2,ncols=1,sharex=True)
        plt.subplots_adjust(hspace=0)
        
        ax = axs[0]
        plt.sca(ax)
        table.radplot(uvunit=uvunit, uvdtype="ave", color="black", **plotargs)
        table.radplot(uvunit=uvunit, uvdtype="ave", model=stats, color="red", **plotargs)
        plt.xlabel("")
        
        ax = axs[1] 
        plt.sca(ax)
        plt.plot(table["uvdistave"]*table.uvunitconv("lambda", uvunit),
                 normresid, ls="none", marker=".", color="black", **plotargs)
        plt.axhline(0,color="black",ls="--")
        plt.ylabel("Normalized Residuals")
        plt.xlabel(r"Baseline Length (%s)"%(unitlabel))
        
        divider = make_axes_locatable(ax) # Histgram
        cax = divider.append_axes("right", size="10%", pad=0.05)
        ymin,ymax = ax.get_ylim()
        xmin = np.min(normresid)
        xmax = np.max(normresid)
        y = np.linspace(ymin,ymax,1000)
        x = 1/np.sqrt(2*np.pi) * np.exp(-y*y/2.)
        cax.hist(normresid, bins=np.int(np.sqrt(N)), normed=True, orientation='horizontal')
        cax.plot(x,y,color="red")
        cax.set_ylim(ax.get_ylim())
        cax.axhline(0,color="black",ls="--")
        cax.yaxis.set_major_formatter(nullfmt)
        cax.xaxis.set_major_formatter(nullfmt)
        if filename is not None:
            pdf.savefig()
            plt.close()
    
    # Close File
    if filename is not None:
        pdf.close()
    else:
        plt.show()
    
    if isinteractive:
        plt.ion()
        matplotlib.use(backend)

def get_uvlist(fcvtable=None, amptable=None, bstable=None, catable=None, thres=1e-2):
    '''

    '''
    import numpy as np

    if ((fcvtable is None) and (amptable is None) and
            (bstable is None) and (catable is None)):
        print("Error: No data are input.")
        return -1

    # Stack uv coordinates
    ustack = None
    vstack = None
    if fcvtable is not None:
        ustack = np.array(fcvtable["u"], dtype=np.float64)
        vstack = np.array(fcvtable["v"], dtype=np.float64)
        Nfcv = len(ustack)
    else:
        Nfcv = 0

    if amptable is not None:
        utmp = np.array(amptable["u"], dtype=np.float64)
        vtmp = np.array(amptable["v"], dtype=np.float64)
        Namp = len(utmp)
        if ustack is None:
            ustack = utmp
            vstack = vtmp
        else:
            ustack = np.concatenate((ustack, utmp))
            vstack = np.concatenate((vstack, vtmp))
    else:
        Namp = 0

    if bstable is not None:
        utmp1 = np.array(bstable["u12"], dtype=np.float64)
        vtmp1 = np.array(bstable["v12"], dtype=np.float64)
        utmp2 = np.array(bstable["u23"], dtype=np.float64)
        vtmp2 = np.array(bstable["v23"], dtype=np.float64)
        utmp3 = np.array(bstable["u31"], dtype=np.float64)
        vtmp3 = np.array(bstable["v31"], dtype=np.float64)
        Ncp = len(utmp1)
        if ustack is None:
            ustack = np.concatenate((utmp1, utmp2, utmp3))
            vstack = np.concatenate((vtmp1, vtmp2, vtmp3))
        else:
            ustack = np.concatenate((ustack, utmp1, utmp2, utmp3))
            vstack = np.concatenate((vstack, vtmp1, vtmp2, vtmp3))
    else:
        Ncp = 0

    if catable is not None:
        utmp1 = np.array(catable["u1"], dtype=np.float64)
        vtmp1 = np.array(catable["v1"], dtype=np.float64)
        utmp2 = np.array(catable["u2"], dtype=np.float64)
        vtmp2 = np.array(catable["v2"], dtype=np.float64)
        utmp3 = np.array(catable["u3"], dtype=np.float64)
        vtmp3 = np.array(catable["v3"], dtype=np.float64)
        utmp4 = np.array(catable["u4"], dtype=np.float64)
        vtmp4 = np.array(catable["v4"], dtype=np.float64)
        Nca = len(utmp1)
        if ustack is None:
            ustack = np.concatenate((utmp1, utmp2, utmp3, utmp4))
            vstack = np.concatenate((vtmp1, vtmp2, vtmp3, vtmp4))
        else:
            ustack = np.concatenate((ustack, utmp1, utmp2, utmp3, utmp4))
            vstack = np.concatenate((vstack, vtmp1, vtmp2, vtmp3, vtmp4))
    else:
        Nca = 0

    # make non-redundant u,v lists and index arrays for uv coordinates.
    Nstack = Nfcv + Namp + 3 * Ncp + 4 * Nca
    uvidx = np.zeros(Nstack, dtype=np.int32)
    maxidx = 1
    u = []
    v = []
    uvstack = np.sqrt(np.square(ustack) + np.square(vstack))
    uvthres = np.max(uvstack) * thres
    for i in np.arange(Nstack):
        if uvidx[i] == 0:
            dist1 = np.sqrt(
                np.square(ustack - ustack[i]) + np.square(vstack - vstack[i]))
            dist2 = np.sqrt(
                np.square(ustack + ustack[i]) + np.square(vstack + vstack[i]))
            #uvdist = np.sqrt(np.square(ustack[i])+np.square(vstack[i]))

            #t = np.where(dist1<uvthres)
            t = np.where(dist1 < thres * (uvstack[i] + 1))
            uvidx[t] = maxidx
            #t = np.where(dist2<uvthres)
            t = np.where(dist2 < thres * (uvstack[i] + 1))
            uvidx[t] = -maxidx
            u.append(ustack[i])
            v.append(vstack[i])
            maxidx += 1
    u = np.asarray(u)  # Non redundant u coordinates
    v = np.asarray(v)  # Non redundant v coordinates

    # distribute index information into each data
    if fcvtable is None:
        uvidxfcv = np.zeros(1, dtype=np.int32)
    else:
        uvidxfcv = uvidx[0:Nfcv]

    if amptable is None:
        uvidxamp = np.zeros(1, dtype=np.int32)
    else:
        uvidxamp = uvidx[Nfcv:Nfcv + Namp]

    if bstable is None:
        uvidxcp = np.zeros([3, 1], dtype=np.int32, order="F")
    else:
        uvidxcp = uvidx[Nfcv + Namp:Nfcv + Namp + 3 *
                        Ncp].reshape([Ncp, 3], order="F").transpose()

    if catable is None:
        uvidxca = np.zeros([4, 1], dtype=np.int32, order="F")
    else:
        uvidxca = uvidx[Nfcv + Namp + 3 * Ncp:Nfcv + Namp + 3 *
                        Ncp + 4 * Nca].reshape([Nca, 4], order="F").transpose()
    return (u, v, uvidxfcv, uvidxamp, uvidxcp, uvidxca)


#-------------------------------------------------------------------------
# Infer Beam Size
#-------------------------------------------------------------------------
def calc_dbeam(fitsdata, visdata, errweight=0, ftsign=+1):
    '''
    Calculate an array and total flux of dirty beam from the input visibility data

    keywords:
      fitsdata:
        input imdata.IMFITS object
      visdata:
        input visibility data
      errweight (float):
        index for errer weighting
      ftsign (integer):
        a sign for fourier matrix
    '''
    import numpy as np
    import pandas as pd
    import copy
    from scipy import linalg

    # create output fits
    outfitsdata = copy.deepcopy(fitsdata)

    # read uv information
    M = len(visdata)
    U = np.float64(visdata["u"])
    V = np.float64(visdata["v"])

    # create visibilies and error weighting
    Vis_point = np.ones(len(visdata), dtype=np.complex128)
    if errweight != 0:
        sigma = np.float64(visdata["sigma"])
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


def calc_bparms(visdata):
    '''
    Infer beam parameters (major size, minor size, position angle)

    keywords:
      visdata: input visibility data
    '''
    import numpy as np
    import pandas as pd

    # read uv information
    U = np.float64(visdata["u"])
    V = np.float64(visdata["v"])

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


def gauss_func(X, Y, maja, mina, PA, x0=0., y0=0., scale=1.):
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
    import numpy as np
    import pandas as pd

    # scaling
    maja *= scale
    mina *= scale

    # calculate gauss function
    cosPA = np.cos(np.radians(PA))
    sinPA = np.sin(np.radians(PA))
    L = ((X * sinPA + Y * cosPA)**2) / (maja**2) + \
        ((X * cosPA - Y * sinPA)**2) / (mina**2)
    return np.exp(-L * 4 * np.log(2))


def fit_chisq(parms, X, Y, dbeam):
    '''
    Calculate residuals of two 2-D array

    keywords:
      parms: information of clean beam
      X: 2-D array of x-axis
      Y: 2-D array of y-axis
      dbeam: an array of dirty beam
    '''
    import numpy as np

    # get parameters of clean beam
    (maja, mina, angle) = parms

    # calculate clean beam and residuals
    cbeam = gauss_func(X, Y, maja, mina, angle)
    cbeam /= np.max(cbeam)
    if cbeam.size == dbeam.size:
        return (dbeam - cbeam).reshape(dbeam.size)
    else:
        print("not equal the size of two beam array")


def calc_cbeam(fitsdata, visdata, angunit="mas", errweight=0., ftsign=+1):
    '''
    This function calculates an array and total flux of dirty beam
    from the input visibility data

    keywords:
      fitsdata:
        input imdata.IMFITS object
      visdata:
        input visibility data
      angunit (string):
        Angular unit (uas, mas, asec or arcsec, amin or arcmin, degree)
      errweight (float):
        index for errer weighting
      ftsign (integer):
        a sign for fourier matrix
    '''
    import numpy as np
    import pandas as pd
    import copy
    from scipy import optimize

    # create output fits
    dbfitsdata, dbflux = calc_dbeam(
        fitsdata, visdata, errweight=errweight, ftsign=ftsign)

    # infer the parameters of clean beam
    parm0 = calc_bparms(visdata)
    X, Y = fitsdata.get_xygrid(angunit="deg", twodim=True)
    dbeam = dbfitsdata.data[0, 0]
    dbeam /= np.max(dbeam)

    parms = optimize.leastsq(fit_chisq, parm0, args=(X, Y, dbeam))

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
