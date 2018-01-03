#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
This is a submodule of sparselab.modelfit, containing functions to 
calculate visibilities and images of some geometric models 
'''
import numpy as np
import astropy.constants as ac
from .. import util


def vis_cgauss(u,v,x0=0.,y0=0.,totalflux=1.,size=1.,angunit="uas"):
    '''
    u,v : array-like lambda
    flux: Jy
    size: unit specified with angunit
    angunit: unit of angular size
    '''
    conv = util.angconv(angunit, "rad")
    
    sigma=size/np.sqrt(8*np.log(2))*conv
    gamma = np.square(u)
    gamma += np.square(v)
    gamma *= 2*np.pi*np.pi*sigma*sigma
    
    phase=2j*np.pi*(u*x0+v*y0)*conv
    
    Vcmp=totalflux*np.exp(-gamma+phase)
    
    return Vcmp


def img_cgauss(x,y,x0=0.,y0=0.,totalflux=1.,size=1.,dx=1.,dy=1.):
    '''
    x,y: x,y coordinates of the image
    x0, y0: the centroid coordinate of the specified circular Gaussian
    totalflux: total flux of the specified circular Gaussian
    size: FWHM size of the specified circular Gaussian
    dx, dy: the pixel size of the image
    '''
    twosigmasq = 2*np.square(size/np.sqrt(8*np.log(2)))
    gamma =  np.square(x-x0)
    gamma += np.square(y-y0)
    gamma /= twosigmasq
    
    I = totalflux/np.pi/twosigmasq * np.exp(-gamma) * dx * dy
    return I



def vis_egauss(u,v,x0=0.,y0=0.,totalflux=1.,majsize=1.,minsize=None,pa=0.0,angunit="uas"):
    
    # conversion factor from the specified unit to radian
    conv = util.angconv(angunit, "rad")
    
    # Calculate standard deviations
    fwhm2sig = conv/np.sqrt(8*np.log(2))
    sigmaj = majsize*fwhm2sig     # sigma_maj in radian
    if minsize is None:
        sigmin = sigmaj           # sigma_min = sigma_maj (in radian)
    else:
        sigmin = minsize*fwhm2sig # sigma_minor in radian
    
    # Calculate Gaussian
    parad = np.deg2rad(pa)
    cospa=np.cos(parad)
    sinpa=np.sin(parad)  
    #  Rotation
    u1=u*cospa-v*sinpa
    v1=u*sinpa+v*cospa
    #  Exponent
    gamma = np.square(u1*sigmin)
    gamma += np.square(v1*sigmaj)
    gamma *= 2*np.pi*np.pi
    #  Complex Exponent (phase)
    phase=2j*np.pi*(u*x0+v*y0)*conv
    
    Vcmp=totalflux*np.exp(-gamma+phase)
    
    return Vcmp


def img_egauss(x,y,x0=0.,y0=0.,totalflux=1.,
               majsize=1.,minsize=None,pa=0.0,dx=1.,dy=1.):
    
    # Calculate standard deviations
    sig2fwhm = np.sqrt(8*np.log(2))
    sigmaj = majsize/sig2fwhm
    if minsize is None:
        sigmin = sigmaj
    else:
        sigmin = minsize/sig2fwhm
    
    # Calculate Gaussian
    parad = np.deg2rad(pa)
    cospa=np.cos(parad)
    sinpa=np.sin(parad)  
    
    x1=x-x0    
    y1=y-y0
    
    x2=x1*cospa-y1*sinpa
    y2=x1*sinpa+y1*cospa
    
    # Normalization Factor
    factor = totalflux/2./np.pi/sigmaj/sigmin*dx*dy
    
    gamma = (x2/sigmin)**2
    gamma+= (y2/sigmaj)**2
    gamma*= 0.5
      
    I = factor*np.exp(-gamma)
    
    return I


def vis_scatt(u,v,nu=None,lamb=None,
              thetamaj=1.309,thetamin=0.64,alpha=2.,pa=78.0):
    '''
    Args:
        u,v (float, array-like):
            u,v coordinates in lambda
        nu=None, lamb=None (float):
            observing frequency in Hz or wavelength in m.
            One of them must be specified. If both are specified,
            frequency will be used to calculate the scattering kernel.
        thetamaj=1.309, thetamin=0.64 (float):
            Factors for the scattering power law in mas/cm^(alpha)
        alpha=2.0 (float):
            Index of the scattering power law
        pa=78.0 (float)
            Position angle of the scattering kernel in degree
    Return:
        A float or array of the kernel visibility amplitude
    '''
    # calculate the observaitonal wavelength in meter
    c_si = ac.c.si.value
    #
    if (nu is None) and (lamb is None):
        raise ValueError("please specify observing frequency or wavelength")
    elif nu is not None:
        # calc lambda from nu
        lambcm = c_si/nu * 1e2
    else:
        lambcm = lamb * 1e2
    
    # derive Gaussian Kernel parameters
    majsize = thetamaj * lambcm**alpha # in mas (Bower et al. 2006)
    minsize = thetamin * lambcm**alpha #(Bower et al. 2006)
    totalflux = 1.
    x0 = 0.
    y0 = 0.
    
    Vcmp = vis_egauss(u,v,x0=x0,y0=y0,
                      totalflux=totalflux,
                      majsize=majsize,minsize=minsize,pa=pa,
                      angunit="mas")
    Vamp = np.abs(Vcmp)
    return Vamp


def img_scatt(x,y,dx=1.,dy=1.,
              nu=None,lamb=None,
              thetamaj=1.309,thetamin=0.64,alpha=2.,pa=78.0,
              angunit="uas"):
    '''
    Args:
        x,y (float, array-like):
            x,y coordinates of the image in angunit
        nu=None, lamb=None (float):
            observing frequency in Hz or wavelength in m.
            One of them must be specified. If both are specified,
            frequency will be used to calculate the scattering kernel.
        thetamaj=1.309, thetamin=0.64 (float):
            Factors for the scattering power law in mas/cm^(alpha)
        alpha=2.0 (float):
            Index of the scattering power law
        pa=78.0 (float)
            Position angle of the scattering kernel in degree
        angunit="uas" (string):
            angular unit of x,y
    Return:
        A float or array of the kernel brightness
    '''
    # calculate the observaitonal wavelength in meter
    c_si = ac.c.si.value
    #
    if (nu is None) and (lamb is None):
        raise ValueError("please specify observing frequency or wavelength")
    elif nu is not None:
        # calc lambda from nu
        lambcm = c_si/nu * 1e2
    else:
        lambcm = lamb * 1e2
    
    # derive Gaussian Kernel parameters
    conv = util.angconv("mas", angunit)
    majsize = thetamaj * lambcm**alpha * conv # in angunit (Bower et al. 2006)
    minsize = thetamin * lambcm**alpha * conv # in angunit (Bower et al. 2006)
    totalflux = 1.
    x0 = 0.
    y0 = 0.
    
    I = img_egauss(x,y,x0=x0,y0=y0,
                   totalflux=totalflux,
                   majsize=majsize,minsize=minsize,pa=pa,
                   dx=dx,dy=dy)
    return I
