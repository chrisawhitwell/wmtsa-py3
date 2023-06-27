###########################################################################
#                                                                         #
#    Copyright 2016 Andrea Cimatoribus                                    #
#    ECOL, ENAC, IIE, EPFL                                                #
#    GR A1 435, Station 2, CH-1015 Lausanne, Switzerland                  #
#    andrea.cimatoribus@epfl.ch                                           #
#                                                                         #
#    This file is part of wmtsa toolbox.                                  #
#                                                                         #
#    wmtsa toolbox is free software: you can redistribute it              #
#    and/or modify it under the terms of the GNU General Public           #
#    License as published by the Free Software Foundation, either         #
#    version 3 of the License, or (at your option) any later version.     #
#                                                                         #
#    wmtsa toolbox is distributed in the hope that it will be             #
#    useful, but WITHOUT ANY WARRANTY; without even the implied warranty  #
#    of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the     #
#    GNU General Public License for more details.                         #
#                                                                         #
#    You should have received a copy of the GNU General Public            #
#    License along with wmtsa.                                            #
#    If not, see <http://www.gnu.org/licenses/>.                          #
#                                                                         #
###########################################################################
'''
Created on May 16, 2013

@author: cimatori
'''

from wmtsa import dwtArray, wtfilter

import numpy as np

def modwt(X, wtf='la8', nlevels='conservative', boundary='reflection', RetainVJ=False):
    """
    function modwt(X, wtf='la8', nlevels='conservative', boundary='reflection', RetainVJ=False)
       modwt -- Compute the (partial) maximal overlap discrete wavelet transform (MODWT).
    
    NAME
       modwt -- Compute the (partial) maximal overlap discrete wavelet transform (MODWT).
    
    INPUTS
       X          -- set of observations 
                    (vector of length NX)
       wtf        -- (optional) wavelet transform filter name
                     (string, case-insensitve or wtf struct).
                     Default:  'la8'
       nlevels    -- (optional) maximum level J0 (integer) 
                     or method of calculating J0 (string).
                     Valid values: integer>0 or a valid method name
                     Default:  'conservative'
       boundary   -- (optional) boundary conditions to use (string)
                     Valid values: 'circular' or 'reflection'
                     Default: 'reflection'
       RetainVJ   -- (optional) boolean flag to retain V at each
                     decomposition level
                     Default: False
    
    OUTPUTS
       WJt        -- MODWT wavelet coefficents (J x NW array) dwtArray
       VJt        -- MODWT scaling coefficients ((none or J) x NW vector) dwtArray
    
    DESCRIPTION
       modwt calculates the wavelet and scaling coefficients using the maximal
       overlap discrete wavelet transform (MODWT).
    
       The optional input arguments have default values:
       * wtf      -- 'la8' filter
       * nlevels  -- 'convservative' --> J0 < log2( N / (L-1) + 1)
       * boundary -- 'reflection'.
    
       The output arguments include an info attribute with metadata.
       info is a dictionary with the following fields:
       * Transform  -- name of transform ('MODWT')
       * WTF        -- name of wavelet transform filter or a wtf_s struct.
       * NX         -- number of observations in original series (= length(X))
       * NW         -- number of wavelet coefficients
       * J0         -- number of levels of partial decompsition.
       * Boundary   -- boundary conditions applied.
       * Aligned    -- Boolean flag indicating whether coefficients are aligned
                       with original series (1 = true) or not (0 = false).
       * RetainVJ   -- Boolean flag indicating whether VJ scaling coefficients
                       at all levels have been retained (1= true) or not (0 = false).
       * BCs        -- ((1 or J0)x 2) array with indices of last element to the left,
                       and first element to the right which are influenced to some 
                       extent by boundary conditions. Given at each decomposition level
    
    EXAMPLE

       WJt, VJt = modwt(x, 'la8', 6, 'reflection')
    
    ALGORITHM
       See pages 177-178 of WMTSA for description of Pyramid Algorithm for
       the MODWT.
    
    REFERENCES
       Percival, D. B. and A. T. Walden (2000) Wavelet Methods for
       Time Series Analysis. Cambridge: Cambridge University Press.
    """

    # Get a valid wavelet transform filter coefficients struct.
    wtf_s = wtfilter(wtf)
  
    wtfname = wtf_s.Name
    gt = wtf_s.g
    ht = wtf_s.h
    L  = wtf_s.L

    # Make sure that X is a numpy array
    X = np.array(X)
    if len(X.shape)>1:
        raise ValueError('Input array must be one-dimensional')
    #  N    = length of original series
    N = X.size
        
    #  If nlevels is an integer > 0, set J0 = nlevels.
    #  otherwise, select J0 based on choice method specified.
    if isinstance(nlevels, str):
        J0 = choose_nlevels(nlevels, wtfname, N)
    elif isinstance(nlevels, int):
        if nlevels > 0:
            J0 = nlevels
        else:
            raise ValueError('NegativeJ0, nlevels must be an integer greater than 0.')
    else:
        raise ValueError('Invalid NLevels Value')
    
    if (J0 < 0):
        raise ValueError('Negative J0')
    
    if (2**J0 > N):
        print ('Warning (MODWT):Large J0, JO > log2(Number of samples).')

    # Initialize the scale (Vin) for first level by setting it equal to X
    # using specified  boundary conditions
    if boundary=='reflection':
        Xin = np.hstack((X, X[::-1]))
    elif boundary in ('circular', 'periodic'):
        Xin = X
    else:
        raise ValueError('Invalid Boundary Conditions')

    # NW = length of the extended series = number of coefficients
    NW = Xin.size
    
    # Pre-allocate memory.
    WJt = np.ndarray((J0, NW), dtype=np.float64)*np.nan
    if RetainVJ:
        VJt = np.ndarray((J0, NW), dtype=np.float64)*np.nan
    else:
        VJt = np.ndarray((NW), dtype=np.float64)*np.nan

    # Do the MODWT.
    import pyximport; pyximport.install()
    from .modwtj import modwtj
    Vin = Xin; bw = np.ndarray((J0,2))*np.nan
    for j in range(J0):
        Wt_j, Vout = modwtj(Vin, j+1, ht, gt)
        WJt[j,:]   = Wt_j
        Vin        = Vout
        if RetainVJ:
            VJt[j,:] = Vout
            
        # boundary values pag.198 WMTSA
        L_j     = equivalent_filter_width(L, j+1)
        bw[j,0] = min(L_j - 2, NW-1) #max index to the left
        #bw[j,1] = np.nan #Bc are only at the beginning
    if not RetainVJ:
        VJt[:] = Vout
        bv     = bw[-1,:]
    else:
        bv     = bw

    # Update attributes
    att = {'Transform':'MODWT',
           'WTF'      : wtfname,
           'N'        : N,
           'NW'       : NW,
           'J0'       : J0,
           'Boundary' : boundary,
           'Aligned'  : False,
           'RetainVJ' : RetainVJ,
           'Type'     : 'Wavelet',
           'BCs'      : np.int32(bw)
           }
    WJt = dwtArray(WJt, info=att)

    att = {'Transform':'MODWT',
           'WTF'      : wtfname,
           'N'        : N,
           'NW'       : NW,
           'J0'       : J0,
           'Boundary' : boundary,
           'Aligned'  : False,
           'RetainVJ' : RetainVJ,
           'Type'     : 'Scaling',
           'BCs'      : np.int32(bv)
           }
    VJt = dwtArray(VJt, info=att)
    
    return WJt, VJt


def imodwt_details(WJt):
    """
    imodwt_details -- Calculate details via inverse maximal overlap discrete wavelet transform (IMODWT).

    NAME
       imodwt_details -- Calculate details via inverse maximal overlap discrete wavelet transform (IMODWT).
    
    INPUTS
       WJt          -  NxJ array of MODWT wavelet coefficents
                       where N  = number of time points
                             J = number of levels.
                       The array must be a dwtArray (containing the information on the transform)
    
    OUTPUT
       DJt          -  JxN dwtArray of reconstituted details of data series for J0 scales.
       att          -  structure containing IMODWT transform attributes.
    
    SIDE EFFECTS
       1.  wavelet is a WMTSA-supported MODWT wavelet filter; otherwise error.
    
    DESCRIPTION
       The output parameter att is a structure with the following fields:
           name      - name of transform (= 'MODWT')
           wtfname   - name of MODWT wavelet filter
           npts      - number of observations (= length(X))
           J0        - number of levels 
           boundary  - boundary conditions
    
    EXAMPLE
       DJt = imodwt_details(WJt)
    
    ALGORITHM
       See pages 177-179 of WMTSA for description of Pyramid Algorithm for
       the inverse MODWT multi-resolution analysis.
    
    REFERENCES
       Percival, D. B. and A. T. Walden (2000) Wavelet Methods for
       Time Series Analysis. Cambridge: Cambridge University Press.
    
    SEE ALSO
       imodwtj, imodwt, imodwt_smooth, modwt_filter, modwt
    """
    
    # Get a the wavelet transform filter coefficients.
    if type(WJt) is not dwtArray:
        raise TypeError('Input must be a dwtArray')
    if WJt.info['Type'] is not 'Wavelet':
        raise TypeError('Input array does not contain Wavelet coefficients but {} coefficients'.format(WJt.info['Type']))
    wtfname = WJt.info['WTF']
    wtf_s   = wtfilter(wtfname)
  
    gt = wtf_s.g
    ht = wtf_s.h
    L  = wtf_s.L

    J,N = WJt.shape
    J0  = J

    # initialise arrays
    zeroj = np.zeros(N)
    DJt   = np.zeros((J, N))

    import pyximport; pyximport.install()
    from .modwtj import imodwtj 
    for j in range(J0-1,-1,-1):
        Vin = zeroj
        Win = WJt[j,:]
        for jj in range(j,-1,-1):
            Vout = imodwtj(Win, Vin, jj+1, ht, gt)
            Win = zeroj
            Vin = Vout
        DJt[j,:] = Vout
    
    # boundary values pag.199 WMTSA
    bw = np.ndarray((J0,2), dtype=np.int16)*np.nan
    for j in range(J0):
        #Calculate circularly shifted wavelet coefficient boundary indices at jth level
        L_j     = equivalent_filter_width(L, j+1)
        bw[j,0] = L_j-2; bw[j,1]= -L_j+1
    # Update attributes
    att = dict(list(WJt.info.items()) +
               list({'Transform':'IMODWT',
                'Type'     :'Detail',
                'BCs'      :np.int32(bw)}.items()))
    
    DJt = dwtArray(DJt, info=att)

    return DJt

def imodwt_smooth(VJt):
    """
    imodwt_smooth -- Calculate smooths at J0 level via inverse maximal overlap discrete wavelet transform (IMODWT).

    NAME
       imodwt_smooth -- Calculate smooths at J0 level via inverse maximal overlap discrete wavelet transform (IMODWT).
    
    INPUTS
       VJt          =  N dwtArray of MODWT scaling coefficients at J0 level.
    
    OUTPUT
       SJOt         =  dwtArray of reconstituted smoothed data series.
    
    SIDE EFFECTS
    
    DESCRIPTION
    
    EXAMPLE
       SJt = imodwt_smooth(VJt)
 
    NOTES
    
    ALGORITHM
       See pages 177-179 of WMTSA for description of Pyramid Algorithm for
       the inverse MODWT multi-resolution analysis.
    
    REFERENCES
       Percival, D. B. and A. T. Walden (2000) Wavelet Methods for
       Time Series Analysis. Cambridge: Cambridge University Press.
    
    SEE ALSO
       imodwtj, imodwt, imodwt_details, modwt_filter, modwt
    """

    # Get the wavelet transform filter coefficients.
    if type(VJt) is not dwtArray:
        raise TypeError('Input must be a dwtArray')
    if VJt.info['Type'] is not 'Scaling':
        raise TypeError('Input array does not contain Scaling coefficients but {} coefficients'.format(VJt.info['Type']))
    wtfname = VJt.info['WTF']
    J0      = VJt.info['J0']
    wtf_s   = wtfilter(wtfname)
    gt = wtf_s.g
    ht = wtf_s.h
    L  = wtf_s.L

    if len(VJt.shape)>1:
        if VJt.info['RetainVJ']:
            VJt = VJt[-1,:]
        else:
            raise TypeError('The input is a multidimensional array but {} MODWT\
             has been computed with RetainVJ=False. \nThis does not make sense')
    N = VJt.size

    # initialize arrays
    zeroj = np.zeros(N)

    Vin = VJt

    import pyximport; pyximport.install()
    from .modwtj import imodwtj
    for j in range(J0-1,-1,-1):
        Vout = imodwtj(zeroj, Vin, j+1, ht, gt)
        Vin  = Vout

    SJt = Vout
    
    # boundary values pag.199 WMTSA
    L_j = equivalent_filter_width(L, J0)
    bv  = np.array([L_j-2, -L_j+1])

    # Update attributes
    att = dict(list(VJt.info.items()) +
               list({'Transform':'IMODWT',
                'Type'     :'Smooth',
                'BCs'      :np.int32(bv)}.items()))

    SJt = dwtArray(SJt, info=att)

    return SJt

def imodwt_mra(WJt, VJt):
    """
    imodwt_mra -- Calculate MODWT multi-resolution details and smooths from wavelet coefficients via IMODWT transform.

    NAME
    imodwt_mra -- Calculate MODWT multi-resolution details and smooths from wavelet coefficients via IMODWT transform.
    
    INPUTS
       * WJt        -- MODWT wavelet coefficents (J x N).
       * VJt        -- MODWT scaling coefficients ((1 or J) x N)
    
    OUTPUT
       * DJt        -- MODWT details coefficents (J x N).
       * SJt        -- MODWT smooth coefficients ((1 or J) x N)
    
    DESCRIPTION
       modwt_mra computes the multi-resolution detail and smooth coefficients
       fomr the MODWT wavelet and scaling coefficients.
    
    EXAMPLE
       DJt, SJt = imodwt_smooth(WJt, VJt)
    
    NOTES

    ALGORITHM
       See pages 177-179 of WMTSA for description of Pyramid Algorithm for
       the inverse MODWT multi-resolution analysis.
    
    REFERENCES
       Percival, D. B. and A. T. Walden (2000) Wavelet Methods for
       Time Series Analysis. Cambridge: Cambridge University Press.
    
    SEE ALSO
       imodwt_details, imodwt_smooth, imodwtj, modwt, modwt_filter
    """

    # check input
    if type(WJt) is not dwtArray:
        raise TypeError('Input must be a dwtArray')
    if WJt.info['Type'] is not 'Wavelet':
        raise TypeError('First input array does not contain Wavelet coefficients but {} coefficients'.format(WJt.info['Type']))
    if type(VJt) is not dwtArray:
        raise TypeError('Input must be a dwtArray')
    if VJt.info['Type'] is not 'Scaling':
        raise TypeError('Second input array does not contain Scaling coefficients but {} coefficients'.format(VJt.info['Type']))

    DJt = imodwt_details(WJt)
    SJt = imodwt_smooth(VJt)

    return DJt, SJt

def cir_shift(WJt, VJ0t, subtract_mean_VJ0t=True):
    """
    shift_modwt_coef -- shift the MODWT wavelet and scaling coefficients.
    
    NAME
        shift_modwt_coef -- Shift the MODWT wavelet and scaling coefficients.

    INPUTS
        WJt          =  JxN dwtArray of MODWT wavelet coefficents
                        where N = number of time intervals,
                              J = number of levels
        VJ0t         =  N dwtArray of MODWT scaling coefficients at level J0.

        subtract_mean_VJ0t = (optional) subtract mean value of scaling coefficient 
                        from itself
                        Default: True

    OUTPUTS
        W  = shifted wavelet coefficients with boundary conditions (dwtArray)
        V  = shifted scaling coefficients with boundary conditions (dwtArray)

    DESCRIPTION
       The MODWT coefficients are circularly shifted at each level so as to 
       properly align the coefficients with the original data series.

    REFERENCES
       See figure 183 of WMTSA.

    SEE ALSO
       modwt, modwt_filter, overplot_modwt_cir_shift_coef_bdry,
       multi_yoffset_plot
    """
    
    # check input
    if type(WJt) is not dwtArray:
        raise TypeError('Input must be a dwtArray')
    if WJt.info['Type'] is not 'Wavelet':
        raise TypeError('First input array does not contain Wavelet coefficients but {} coefficients'.format(WJt.info['Type']))
    if type(VJ0t) is not dwtArray:
        raise TypeError('Input must be a dwtArray')
    if VJ0t.info['Type'] is not 'Scaling':
        raise TypeError('Second input array does not contain Scaling coefficients but {} coefficients'.format(VJ0t.info['Type']))
    
    wtf_s = wtfilter(WJt.info['WTF'])
    L  = wtf_s.L

    wtfname       = WJt.info['WTF']
    N             = WJt.info['N']
    NW            = WJt.info['NW']
    J0            = WJt.info['J0']
    
    Nm = min(N,NW)

    if WJt.info['Aligned']:
        print ('WARNING (wmtsa.modwt.cir_shift): Wavelet coefficients are already aligned')
        W = WJt
    else:
        if WJt.shape !=dwtArray([]).shape:
            W = np.ndarray(WJt.shape)*np.nan; bw = np.ndarray((J0,2))*np.nan
            for j in range(J0):
                # shift wavelet coefficients
                nuHj = advance_wavelet_filter(wtfname, j+1)
                W[j,:] = np.roll(WJt[j,:], nuHj)
                
                #Calculate circularly shifted wavelet coefficient boundary indices at jth level
                L_j   = equivalent_filter_width(L, j+1)
        
                bw[j,0] = L_j - 2 - np.abs(nuHj) #max index to the left
                bw[j,1] = Nm - abs(nuHj) #min index to the right
            W = W[:,:Nm]
            # Update attributes
            att = dict(list(WJt.info.items()) +
                       list({'Aligned':True,
                        'BCs'    :np.int32(bw)
                        }.items()))
            W = dwtArray(W, info=att)
        else: # if an empty array was given, do nothing and return it
            W = WJt

    if VJ0t.info['Aligned']:
        print ('WARNING (wmtsa.modwt.cir_shift): Wavelet coefficients are already aligned')
        V = VJ0t
    else:
        V = np.ndarray(VJ0t.shape)*np.nan; bv = np.ndarray((2,))*np.nan
        # shift scaling coefficients
        if VJ0t.shape !=dwtArray([]).shape:
            nuGj = advance_scaling_filter(wtfname, J0)
            if subtract_mean_VJ0t:
                VJ0t = VJ0t - VJ0t.mean()
            V    = np.roll(VJ0t, nuGj)
    
            bv[0] = L_j - 2 - np.abs(nuGj) #max index to the left
            bv[1] = Nm - np.abs(nuGj) #min index to the right
            # Update attributes
            att = dict(list(VJ0t.info.items()) +
                       list({'Aligned':True,
                        'BCs'    :np.int32(bv)
                        }.items()))
            V = dwtArray(V[:Nm], info=att)
        else: # if an empty array was given, do nothing and return it
            V = VJ0t

    return W,V

def rot_cum_wav_svar(WJt, VJ0t, method='cumsc'):
    """
    rot_cum_wav_svar -- Calculate cumulative sample variance of MODWT wavelet coefficients.
    
     NAME
       rot_cum_wav_svar -- Calculate cumulative sample variance of 
             MODWT wavelet coefficients.
    
     INPUTS
       WJt          -  JxN dwtArray of MODWT wavelet coefficents
                       where N = number of time intervals
                             J = number of levels
                       they can be already rotated or not
       VJ0t         -  N dwtArray of MODWT J0 scaling coefficents
                       they can be already rotated or not
       method       - variance estimate returned
                       'cum'   = cumulative variance
                       'cumsc' = cumulative "scaled" (see pag.189)
                       Default: 'power' 
    
     OUTPUTS
       cwsvar       -  cumulative wavelet sample variance (dwtArray).
    
     DESCRIPTION
       'cumsc' methods is equivalent to the one on pag.189 of WMTSA
    
     EXAMPLE
    
    
     ALGORITHM
    
       cwsvar[j,t] = 1/N * sum( WJt^2 subscript(j,u+nuH_j mod N)) 
                        for t = 0,N-1 at jth level
       for j in range(J):
        rcwsvar[j,:] = cswvar[j,:] - t*cwsvarN[j]/(N-1.)
    
       For details, see page 189 of WMTSA.   
    
     REFERENCES
       Percival, D. B. and A. T. Walden (2000) Wavelet Methods for
         Time Series Analysis. Cambridge: Cambridge University Press.
    
     SEE ALSO
       cir_shift, modwt
    """
    
    if method not in ('power','cum','cumsc'):
        raise ValueError('Valid methods are only: "power", "cum" or "cumsc"')
    
    # check input
    if type(WJt) is not dwtArray:
        raise TypeError('Input must be a dwtArray')
    if WJt.info['Type'] is not 'Wavelet':
        raise TypeError('First input array does not contain Wavelet coefficients but {} coefficients'.format(WJt.info['Type']))
    if type(VJ0t) is not dwtArray:
        raise TypeError('Input must be a dwtArray')
    if VJ0t.info['Type'] is not 'Scaling':
        raise TypeError('Second input array does not contain Scaling coefficients but {} coefficients'.format(VJ0t.info['Type']))
            
    # rotate if they are not yet aligned
    WJt,VJ0t = cir_shift(WJt, VJ0t) # the check for rotation is done in cir_shift

    # get dimensions
    try:
        J, N   = WJt.shape
    except ValueError:
        N = WJt.size
        J=1
    if len(VJ0t.shape)>1:
        raise ValueError('Only J0 level scaling coefficient should be given')

    pWJt = WJt**2; pVJ0t = VJ0t**2
    
    cwsvar = np.cumsum(pWJt, axis=1)/N
    swsvar = np.cumsum(pVJ0t)/N
    cwsvarN = cwsvar[:,-1]
    swsvarN = swsvar[-1]
    
    if method=='cum':
        Watt = dict(list(WJt.info.items()) + list({'Type':'WavRotCumVar'}.items()))
        Vatt = dict(list(VJ0t.info.items()) + list({'Type':'ScalRotCumVar'}.items()))
        return dwtArray(cwsvar/cwsvarN[:,np.newaxis], info=Watt),dwtArray(swsvar/swsvarN, info=Vatt)

    # compute rotated cumulative variance
    t = np.arange(N, dtype=np.float64)

    rcwsvar = cwsvar - t*cwsvarN[:,np.newaxis]/(N-1)
    rswsvar = swsvar - t*swsvarN/(N-1)

    Watt = dict(list(WJt.info.items()) + list({'Type':'WavRotCumScVar'}.items()))
    Vatt = dict(list(VJ0t.info.items()) + list({'Type':'ScalRotCumScVar'}.items()))
    return dwtArray(rcwsvar, info=Watt),dwtArray(rswsvar, info=Vatt)

def running_wvar(WJt, Ns=3, overlap=False, ci_method='chi2eta3', p=0.05):
    """
     running_wvar -- Calculate running wavelet variance of circularly-shifted MODWT wavelet coefficients.

     INPUTS
       WJt         = JxN dwtArray of circularly shifted (advanced) MODWT 
                      wavelet coefficents
       Ns           = (optional) number of points in running segment 
                      over which to calculate wavelet variance. An odd number
                      makes more sense.
                      Default value: 3
       overlap      = (optional) return overlapping estimates if True.
                      Default: False
       ci_method    = (optional) method for calculating confidence interval
                      valid values:  'gaussian', 'chi2eta1', 'chi2eta3' or 'none'
                      If None is passed, no confidence interval is computed
                      Default value: 'chi2eta3'
       p            = (optional) percentage point for chi2square distribute
                      Default value: 0.05 ==> 95% confidence interval
    
     OUTPUTS
       rwvar        = J x N_rm array containing running wavelet variances, 
                      where N_rm is number of runnning means.
                      N_rm depends on overlap, if overlap is True, then N_rm=N
                      if overlap is False, N_rm=N/Ns (+1 if N%Ns>0)
       CI_rwvar     = J x 2 x N_rm array containing confidence intervals of
                      running wavelet variances with lower bound (column 1) 
                      and upper bound (column 2).
       indices      = Indices of time points in original data series for which
                      rwvar values are calculated.
    
     DESCRIPTION
       Function calculates the running wavelet variance from the translated
       (circularly shifted) MODWT wavelet coefficients.  User may specify
       the range and steps of time points to over which to calculate wavelet
       variances and number of continguous values (span) to calculate each
       variance.  The running variance is computed for a span of values
       center at the particular time point.
       See Eq. 324 WMTSA
    
     EXAMPLE
       WJt,  VJ0t      = modwt(X, 'la8', 9)
       WJt, TVJ0t     = modwt_cir_shift(WJt, VJ0t)
       rwvar, CI_rwvar, indices = running_wvar(WJt)
    
     NOTES
       1.  User must use circularly shift MODWT wavelet coefficients.
           Use modwt_cir_shift prior to calculating running wavelet variances.
       2.  The biased estimation is used

     REFERENCES
       Percival, D. B. and A. T. Walden (2000) Wavelet Methods for
         Time Series Analysis. Cambridge: Cambridge University Press.
    """
    valid_ci_methods = ('gaussian', 'chi2eta1', 'chi2eta3', 'none')
    # check input
    if ci_method.lower() not in valid_ci_methods:
        raise ValueError('Bad C.I. method: "{}". Valid methods for confidence interval are only: {}'.format(ci_method,valid_ci_methods))
        
    if type(WJt) is not dwtArray:
        raise TypeError('Input must be a dwtArray')
    if WJt.info['Type'] is not 'Wavelet':
        raise TypeError('Input array does not contain Wavelet coefficients but {} coefficients'.format(WJt.info['Type']))
    if not WJt.info['Aligned']:
        raise TypeError('Input coefficients must be aligned')
    
    J,N = WJt.shape

    Ns = np.int(Ns)
    Ns2 = Ns/2
        
    if overlap:
        rwvar = np.zeros((J,N))*np.nan
        for j in range(J):
            # overlapping running average
            rwvar[j,:] = np.convolve(WJt[j,:]**2, np.ones(Ns), mode='same')
        indices    = np.arange(N)
    else:
        Split  = np.arange(0, N, Ns)
        Denom  = np.hstack((np.diff(Split),N-Split[-1]))
        rwvar  = np.add.reduceat((WJt**2), Split, axis=1)/np.float64(Denom)
        indices = Split + Ns2
        indices[-1] = Split[-1] + (N-Split[-1])/2 # to avoid going after last element

    Watt = dict(list(WJt.info.items()) + list({'Type':'MODWT WVAR'}.items())) # we need to update the meta data since now it contains wavelt variance ("power")
    rwvar = dwtArray(rwvar, info=Watt)

    if ci_method.lower()=='none':  
        CI_rwvar = None
    else:
        CI_rwvar = np.zeros((J,2,rwvar.shape[1]))*np.nan
        if overlap:
            if Ns2%2==0:
                rend = Ns2
                tend = N-Ns2
            else:
                rend = Ns2+1   # if Ns is odd, we need to take one more variance estimate on the right
                tend = N-Ns2-1 # but one less point before finishing the time series
            for t in range(Ns2,tend):
                CI_rwvar[:,:,t],_ = wvar_ci(rwvar[:,t], np.ones(J)*Ns,\
                                                lbound=np.zeros(J), ubound=np.ones(J)*Ns,\
                                                ci_method=ci_method, WJt=WJt, p=p)
        else:
            for t in range(rwvar.shape[1]-1):
                CI_rwvar[:,:,t],_ = wvar_ci(rwvar[:,t], np.ones(J)*Ns,\
                                                lbound=np.zeros(J), ubound=np.ones(J)*Ns,\
                                                ci_method=ci_method, WJt=WJt, p=p)
            # the last element can come from a different number of
            # averaging points
            CI_rwvar[:,:,-1],_ = wvar_ci(rwvar[:,-1], np.ones(J)*(N-Split[-1]),\
                                             lbound=np.zeros(J), ubound=np.ones(J)*(N-Split[-1]),\
                                             ci_method=ci_method, WJt=WJt, p=p)
        
    Watt = dict(list(WJt.info.items()) + list({'Type':'RunWavVar'}.items()))
  
    return dwtArray(rwvar, info=Watt), CI_rwvar, indices
    
def wvar(WJt, ci_method='chi2eta3', estimator='biased', p=0.05):
    """
    wvar -- Calculate wavelet variance of MODWT wavelet coefficients.
    
     INPUTS
       WJt          = MODWT wavelet coefficients (JxN dwtArray).
                      where N = number of time intervals
                            J = number of levels
       ci_method    = (optional) method for calculating confidence interval
                      valid values:  'gaussian', 'chi2eta1', 'chi2eta3'
                      default: 'chi2eta3'
       estimator    = (optional) type of estimator
                      valid values:  'biased', 'unbiased', 'weaklybiased'
                      default: 'biased'
       p            = (optional) percentage point for chi2square distribution.
                      default: 0.05 ==> 95% confidence interval

     OUTPUTS
       wvar         = wavelet variance (Jx1 dwtArray).
       CI_wvar      = confidence interval of wavelet variance (Jx2 array).
                      lower bound (column 1) and upper bound (column 2).
       edof         = equivalent degrees of freedom (Jx1 vector).
       MJ           = number of coefficients used calculate the wavelet variance at
                      each level (integer).

     NOTES
       MJ is vector containing the number of coefficients used to calculate the 
       wavelet variance at each level. 
       For the unbiased estimator, MJ = MJ for j=1:J0, where MJ is the number 
       of nonboundary MODWT coefficients at each level.
       For the biased estimator, MJ = N for all levels.
       For the weaklybiased estimator, MJ = MJ(Haar), for j=1:J0, where MJ(Haar) 
       is the number of nonboundary MODWT coefficients for Haar filter at each level.
    
     ALGORITHM
       See section 8.3 of WMTSA on page 306.
       For unbiased estimator of wavelet variance, see equation 306b. 
       For biased estimator of wavelet variance, see equation 306c. 
    
     REFERENCES
       Percival, D. B. and A. T. Walden (2000) Wavelet Methods for
         Time Series Analysis. Cambridge: Cambridge University Press.
    """
    
    valid_estimator_methods = ('unbiased', 'biased', 'weaklybiased')
    
    # check input
    if estimator.lower() not in valid_estimator_methods:
        raise ValueError('Bad estimator: "{}". Valid estimator methods are only: {}'.format(estimator,valid_estimator_methods))    
        
    if type(WJt) is not dwtArray:
        raise TypeError('Input must be a dwtArray')
    if WJt.info['Type'] is not 'Wavelet':
        raise TypeError('Input array does not contain Wavelet coefficients but {} coefficients'.format(WJt.info['Type']))
    if WJt.info['Aligned'] & (estimator=='unbiased'):
        raise TypeError('The unbiased method required wavelet coefficients not circularly shifted')
    
    wtfname = WJt.info['WTF']
    N  = WJt.info['N']
    J  = WJt.info['J0']
    
    if N>WJt.shape[1]:
        if estimator=='biased':
            N = WJt.shape[1] #in this case we assume to be using the function for computing running mean
        else:
            print ('WARNING modwt.wvar: Computing unbiased wavelet variance with less points than the original time series!')
    
    # Do we need to initialize?
    wvar    = np.zeros(J)*np.nan
    CI_wvar = None; edof = None
    
    wtf_s = wtfilter(wtfname)
    L = wtf_s.L

    # prepare attributes to be updated later
    Watt = dict(list(WJt.info.items()) + list({'Type':'MODWT WVAR',
                                    'CI Method':ci_method,
                                    'Estimator':estimator,
                                    'p-level'  : p}.items()))
        
    if estimator=='unbiased':
        #Unbiased estimator of wavelet variance (equation 306b)
        #Sum of squares of coefficients from L_j to N
        #For unbiased estimator, do not include wavelets coefficients affected by 
        #circularity assumption.

        LJ = equivalent_filter_width(L, np.arange(1,J+1))
        MJ = num_nonboundary_coef(wtfname, N, np.arange(1,J+1))
        if np.any(MJ==0):
            jmin = np.where(MJ==0)[0][0] + 1
            print ('WARNING (modwt.wvar): Coefficients of order {} and higher are influenced by boundary conditions on all points'.format(jmin))
    
        for j in range(J):
            # we take only the coefficients up to N-1 (after that, they would be due to reflection)
            # This can only work with non rotated coefficients
            # since it considers BCs to be only at the beginning
            wvar[j] = np.sum(WJt[j,(LJ[j]-1):N]**2) / float(MJ[j])

        wvar = dwtArray(wvar, info=Watt)
    
        if ci_method!=None:
            lb = LJ-1
            ub = np.ones(J)*N
            CI_wvar, edof = wvar_ci(wvar, MJ=MJ, ci_method=ci_method, WJt=WJt, lbound=lb, ubound=ub, p=p)

    elif estimator=='biased':
        # Biased estimator of wavelet variance (equation 306c)
        # Use all coefficients.
        
        MJ = np.ones(J)*N
        wvar = np.sum(WJt[:,:N]**2, axis=1) / float(N)

        wvar = dwtArray(wvar, info=Watt)
    
        if ci_method!=None:
            lb = np.zeros(J)
            ub = np.ones(J)*N
            CI_wvar, edof = wvar_ci(wvar, MJ=MJ, ci_method=ci_method, WJt=WJt, lbound=lb, ubound=ub, p=p)
    
    elif estimator in ('weaklybiased',):
        # Weakly Biased estimator
        # Use wavelet coefficients that are 1/2 the filter autocorrelation width
        # away from time series boundary.   Over half of signal contribution comes
        # from coefficients within autocorrelation width.
        # This is less retrictive than unbiased estimator but more so biased one.
        # This is equivalent to using circular shifting wavelet coefficients for
        # time alignment and then not including boundary coefficients for Haar
        # filter, since L_j = w_a_j for Haar.

        #TODO: implement         
        raise ValueError ('WEAKLY BIASED NOT IMPLEMENTED')    
  
    return wvar, CI_wvar, edof, MJ

def wspec(WJt, dt=1, estimator='biased', ci_method='chi2eta3', p=0.05):
    """
        wspec -- Convenience function to obtain the spectral estimate from wavelet variance
    INPUTS
       WJt          = MODWT wavelet coefficients (JxN dwtArray).
                      where N = number of time intervals
                            J = number of levels
       dt           = time step of the time series
                      Default: 1
       estimator,ci_method,p  = optional arguments of wvar (see wvar)

     OUTPUTS
       wspec        = wavelet "spectrum" (Jx1 dwtArray).

     NOTES
    
     ALGORITHM
       Cj = 2**j * wvarj *dt
       Based on Eq. 316 of WMTSA
       
     REFERENCES
       Percival, D. B. and A. T. Walden (2000) Wavelet Methods for
         Time Series Analysis. Cambridge: Cambridge University Press.
    """
    
    J = WJt.info['J0']
    
    wv,CI_wvar, edof, MJ = wvar(WJt, estimator=estimator, ci_method=ci_method, p=p)

    wspec   = 2**np.arange(1,J+1) * wv * dt
    try:
        CI_wvar = (2**np.arange(1,J+1))[:,np.newaxis] * CI_wvar * dt
    except TypeError:
        CI_wvar = None
    
    return wspec, CI_wvar, edof, MJ 

def wvar_ci(wvar, MJ, lbound=None, ubound=None, ci_method='chi2eta3', WJt=None, p=0.05, moreOut=False):
    """
        wvar_ci -- Calculate confidence interval of MODWT wavelet variance.
    
     INPUTS
       wvar         = wavelet variance (J dwtArray).
       MJ           = number of coefficients used calculate the wavelet variance at
                      each level (J).
       ci_method    = (optional) method for calculating confidence interval
                      valid values:  'gaussian', 'chi2eta1', 'chi2eta3'
                      default: 'chi2eta3'
       WJt          = MODWT wavelet coefficients (JxN array).
                      where N = number of time intervals
                            J = number of levels
                      required for 'gaussian' and 'chi2eta1' methods.
       lbound       = lower bound of range of WJt for calculating ACVS for each
                      level (J vector).
       ubound       = upper bound of range of WJt for calculating ACVS for each
                      level (J vector).
       p            = (optional) percentage point for chi2square distribution.
                      default: 0.05 ==> 95% confidence interval
       moreOut      = Return also Qeta and AJ
                      default: false
    
     OUTPUTS
       CI_wvar      = confidence interval of wavelet variance  (Jx2 array).
                      lower bound (column 1) and upper bound (column 2).
       edof         = equivalent degrees of freedom (Jx1 vector).
       Qeta         = p x 100% percentage point of chi-square(eta) distribution (Jx2 array).
                      lower bound (column 1) and upper bound (column 2).
       AJ           = integral of squared SDF for WJt (Jx1 vector).
    
     DESCRIPTION
       MJ is vector containing the number of coefficients used to calculate the 
       wavelet variance at each level. 
       For the unbiased estimator, MJ = MJ for j=1:J0, where MJ is the number 
       of nonboundary MODWT coefficients at each level.
       For the biased estimator, MJ = N for all levels.
       For the weaklybiased estimator, MJ = MJ(Haar), for j=1:J0, where MJ(Haar) 
       is the number of nonboundary MODWT coefficients for Haar filter at each level.
    
     EXAMPLE
    
     NOTES
       The output argument edof (equivalent degrees of freedom) is returned for
       the chi2 confidence interval methods.  For the gaussian method, a null
       value is returned for edof.
      
     ALGORITHM
       See section 8.4 of WMTSA on pages 311-315.
    
     REFERENCES
       Percival, D. B. and A. T. Walden (2000) Wavelet Methods for
         Time Series Analysis. Cambridge: Cambridge University Press.
    """

    # define a function we will need later
    def ci(wvar, eta, p):
        #eq. 313c
        from scipy.stats import chi2
        Qeta1 = chi2.ppf(1.0-p, eta)
        Qeta2 = chi2.ppf(p, eta)
        CI_wvar1 = eta * wvar / Qeta1
        CI_wvar2 = eta * wvar / Qeta2
        return  np.vstack((Qeta1,Qeta2)).T, np.vstack((CI_wvar1,CI_wvar2)).T
        
    valid_ci_methods = ('gaussian', 'chi2eta1', 'chi2eta3')
    
    # check input
    if ci_method.lower() not in valid_ci_methods:
        raise ValueError('Bad C.I. method: "{}". Valid methods for confidence interval are only: {}'.format(ci_method,valid_ci_methods))

    # if WJt!=None:        
    #     if type(WJt) is not dwtArray:
    #         raise TypeError('Input wavelet coefficients must be a dwtArray')
    #     if WJt.info['Type']!='Wavelet':
    #         raise TypeError('Input array does not contain Wavelet coefficients but {} coefficients'.format(WJt.info['Type']))
    
    if type(wvar) is not dwtArray:
            raise TypeError('Wavelet variance must be a dwtArray')
    if wvar.info['Type']!='MODWT WVAR':
        raise TypeError('Input array does not contain Wavelet Variance but {}'.format(WJt.info['Type']))

    if ci_method in ('gaussian', 'chi2eta1'):
        if (WJt==None) | (lbound==None) | (ubound==None):
            raise Exception('Missing required argument WJt, lbound or ubound using "gaussian" or "chi2eta1" method')
        elif np.any((ubound-lbound)!=MJ):
            raise ValueError('MJ is inconsistent with lower and/or upper bounds passed to wvar_ci for some j')
            
    
    J = wvar.size
    CI_wvar = np.ndarray((J,2))
    edof=None; Qeta=None; AJ=None
    
    # Calculate ACVS and AJ used by gaussian and chi2eta1 ci methods.
    if ci_method in ('gaussian', 'chi2eta1'):
        # For 'gaussian', 'chi2eta1' ci methods, 
        # compute AJ (integral of squared SDF for WJt) 
        # via calculating ACVS of WJt. (see page 312 of WMTSA).
    
        AJ = np.ndarray(J)
        for j in range(J):
            if MJ[j]>0:
                from wmtsa import acvs
                ACVS  = acvs(WJt[j, lbound[j]:ubound[j]], estimator='biased', subtract_mean=False);
                # Get ACVS for positive lags (there are MJ coefficients in WJt)
                ACVS  = ACVS[(MJ[j]-1):]
                AJ[j] = 0.5*ACVS[0]**2 + np.sum(ACVS[1:]**2)
    
    #Calculate confidence intervals
    if ci_method=='gaussian':
        # Gaussian method, Eq. 311 of WMSTA
        VARwvar = 2.0 * AJ / MJ
        from scipy.stats import norm
        phi1 = norm.ppf(1-p)
        CI_wvar[:,0] = wvar - phi1 * VARwvar**0.5
        CI_wvar[:,1] = wvar + phi1 * VARwvar**0.5
     
    if ci_method=='chi2eta1':
        # Chi-squar equivalent degree of freedom method 1
        edof = MJ * (wvar**2) / AJ #this is eta1 of eq. 313d WMTSA
        Qeta, CI_wvar = ci(wvar, edof, p) #this gives c.i. in 313c WMTSA
    
    elif ci_method=='chi2eta2':
        #TODO: implement chi2eta2 method
        #Eq.314b and 313c
        raise ValueError('Not implemented, it needs an assumption of the shape of the spectrum.')
     
    elif ci_method=='chi2eta3':
        # Chi-square wquivalent degree of freedom method 3
        j_range = np.arange(1,J+1)
        edof = np.maximum((MJ/(2.0**j_range)), 1.0) #this is eta3 of eq. 314c WMTSA
        Qeta, CI_wvar = ci(wvar, edof, p)

    if moreOut:    
        return CI_wvar, edof, Qeta, AJ
    else:
        return CI_wvar, edof
  
def advance_time_series_filter(wtfname):
    """
    advance_time_series_filter -- Calculate the advance of the time series or filter for a given wavelet.
    
    NAME
       advance_time_series_filter -- Calculate the advance of the time series or filter for a given wavelet.
    
    SYNOPSIS
       nu = advance_time_series_filter(wtfname)
    
    INPUTS
       wtfname      -  string containing name of WMTSA-supported wavelet filter.
    
    OUTPUTS
       nu           -  advance of time series for specified wavelet filter.
    
    SIDE EFFECTS
       wavelet is a WMTSA-supported wavelet filter; otherwise error.
    
    DESCRIPTION
    
    EXAMPLE
    
    ALGORITHM
      
      For Least Asymmetric filters, equation 112e of WMTSA:
       nu =   -L/2 + 1,   for L/2 is even;
          =   -L/2,       for L = 10 or 18;
          =   -L/2 + 2,   for L = 14.  
    
      For Best Localized filter, page 119 of WMTSA.
       nu =   -5,         for L = 14;
          =   -11,        for L = 18;
          =   -9,         for L = 20.
    
      For Coiflet filters, page 124 and equation 124 of WMTSA:
       nu =   -2*L/3 + 1
    
    REFERENCES
       Percival, D. B. and A. T. Walden (2000) Wavelet Methods for
         Time Series Analysis. Cambridge: Cambridge University Press.
    
    SEE ALSO
       dwt_filter    
    """

    
    def advance_least_asymetric_filter(L):
        #Equation 112c of WMTSA.
        if L in(10, 18):
            nu = -(L/2)
        elif L==14:
            nu = -(L/2) + 2
        elif (L/2)%2 == 0:
            #L/2 is even, i.e. L = 8, 12, 16, 20
            nu = -(L/2) + 1
        else:
            raise ValueError('Invalid filter length (L = {}) specified.'.format(L))
        return nu

    def advance_best_localized_filter(L):
        #Page 119 of WMTSA.
        if L==14:
            nu = -5
        elif L==18:
            nu = -11
        elif L==20:
            nu = -9
        else:
            pass
        return nu
  
    def advance_coiflet_filter(L):
        #Page 124 and equation 124 of WMTSA
        nu = -(2 * L / 3.) + 1
        return nu

    # Get a valid wavelet transform filter coefficients struct.
    wtf_s = wtfilter(wtfname)
    L     = wtf_s.L

    nu = np.nan

    #Haar
    if wtfname.lower() in ('haar','d2'):
        nu = 0
    # Haar filter
    # Value from Figure 115
    elif wtfname.lower() in ('d4'):
        nu = -1
    #Extremal Phase filters
    #case {'haar', 'd4', 'd6', 'd8', 'd12', 'd14', 'd16', 'd18', 'd20'}
    elif wtfname.lower() in ('d6', 'd8', 'd12', 'd14', 'd16', 'd18', 'd20'):
        raise ValueError('Need to determine nu for Extremal Phase filters  -  Is it -1 for all filters?.')
    #Least Asymmetric filters
    elif wtfname.lower() in ('la8', 'la10', 'la12', 'la14', 'la18', 'la16', 'la20'):
        nu = advance_least_asymetric_filter(L)
    #Best Localized filters
    elif wtfname.lower() in ('bl14', 'bl18', 'bl20'):
        nu = advance_best_localized_filter(L)
    #Coiflet filters
    elif wtfname.lower() in ('c6', 'c12', 'c18', 'c24', 'c30'):
        nu = advance_coiflet_filter(L)
    #otherwise
    else:
        pass
    
    return np.int16(np.around(nu))

def advance_wavelet_filter(wtfname, j):
    """
    advance_wavelet_filter -- Calculate the advance of the wavelet filter at jth level for a given wavelet.
    
    NAME
       advance_wavelet_filter -- Calculate the advance of the wavelet filter at jth level for a given wavelet.

    INPUTS
       wtfname      = string containing name of WMTSA-supported wavelet filter.
       j            = jth level (index) of scale or a range of j levels of scales
                      (integer or Jx1 vector of integers).
    
    OUTPUTS
       nuHj         = advance of wavelet filter at jth level
                      (integer or vector of integers).
    
    SIDE EFFECTS
       wavelet is a WMTSA-supported wavelet filter; otherwise error.
    
    ALGORITHM
       nuHj = - (2^(j-1) * (L-1) + nu);
    
       For details, see equation 114b of WMTSA.
    
    REFERENCES
       Percival, D. B. and A. T. Walden (2000) Wavelet Methods for
         Time Series Analysis. Cambridge: Cambridge University Press.
    
    SEE ALSO
       advance_time_series_filter, dwt_filter
    """

    # Get a valid wavelet transform filter coefficients struct.
    wtf_s = wtfilter(wtfname)
    L     = wtf_s.L

    nu = advance_time_series_filter(wtfname)

    nuHj = -(2**(j-1) * (L-1) + nu)

    return nuHj

def advance_scaling_filter(wtfname, j):
    """
     advance_scaling_filter -- Calculate the value to advance scaling filter at jth level for a given wavelet.
    
     NAME
       advance_scaling_filter -- Calculate the value to advance scaling filter at jth level for a given wavelet.
    
     SYNOPSIS
       nuGj = advance_scaling_filter(wtfname, level)
    
     INPUTS
       wtfname      = string containing name of WMTSA-supported wavelet filter.
       j            = jth level (index) of scale or a range of j levels of scales
                      (integer or vector of integers).
    
     OUTPUTS
       nuGj         = advance of scaling filter at specified levels.
    
     SIDE EFFECTS
       wavelet is a WMTSA-supported scaling filter; otherwise error.
    
     ALGORITHM
       nuGj = (2^j - 1) * nu
    
       For details, see equation 114a of WMTSA.
    
     REFERENCES
       Percival, D. B. and A. T. Walden (2000) Wavelet Methods for
         Time Series Analysis. Cambridge: Cambridge University Press.
    
     SEE ALSO
       advance_time_series_filter, dwt_filter
    """

    if wtfname.lower()=='haar':
        nuGj = advance_wavelet_filter('haar', j)
    else:
        nu = advance_time_series_filter(wtfname)
        nuGj = (2**j - 1) * nu

    return nuGj

def equivalent_filter_width(L, j):
    """
     equivalent_filter_width -- Calculate width of the equivalent wavelet or scaling filter.
    
     NAME
       equivalent_filter_width -- Calculate width of the equivalent wavelet or scaling filter.
    
     INPUTS
       * L          --  width of wavelet or scaling filter (unit scale).
       * j          --  jth level (index) of scale or a range of j levels of scales. 
                        (integer or vector of integers).
    
     OUTPUTS
       * Lj         -- equivalent width of wavelet or scaling filter for specified
                       levels (integer or J vector of integers).
    
     SIDEEFFECTS
       1.  L > 0, otherwise error.
       2.  j > 0, otherwise error.
    
     DESCRIPTION
       Given the length of a wavelet or scaling filter, the function calculates the
       width of the equivalent filter a level or range of levels j for the specified
       base filter width L.
    
     ALGORITHM
        Lj = (2^j - 1) * (L - 1) + 1  (equation 96a of WMTSA)
    
     REFERENCES
       Percival, D. B. and A. T. Walden (2000) Wavelet Methods for
         Time Series Analysis. Cambridge: Cambridge University Press.
    """
    #Check input arguments and set defaults.
    if L<1:
        raise ValueError('L must be positive')
    if np.any(j<1):
        raise ValueError('j must be positive')

    Lj = (2.0**j - 1) * (L - 1) + 1

    return Lj



def choose_nlevels(choice, wtfname, N):
    """
    choose_nlevels -- Select J0 based on choice, wavelet filter and data series length.
    
    NAME
      choose_nlevels -- Select J0 based on choice, wavelet filter and data series length.
    
    USAGE
       J0 = choose_nlevels(choice, wtfname, N)
    
    INPUTS
    choice      -- choice for method for calculating J0 (string)
                   Valid Values:
                   'conservative'
                   'max', 'maximum'
                   'supermax', 'supermaximum'
    wtfname     -- wavelet transform filter name (string)
                   Valid Values:  see modwt_filter
    N           -- number of observations.
    
    OUTPUT
    J0          -- number of levels (J0) based selection criteria.
    
    SIDE EFFECTS
    1.  wtfname is a WMTSA-supported MODWT wtfname, otherwise error.
    2.  N > 0, otherwise error.
    
    DESCRIPTION
    
    
    EXAMPLE
    J0 = choose_nlevels('convservative', 'la8', N)
    
    ERRORS  
    WMTSA:MODWT:InvalidNumLevels    =  Invalid type/value specified for nlevels.
    
    ALGORITHM
    for 'conservative':              J0  < log2( N / (L-1) + 1)
    for 'max', 'maximum':            J0 =< log2(N)
    for 'supermax', 'supermaximum':  J0 =< log2(1.5 * N)
    
    For further details, see page 200 of WMTSA.
    
    REFERENCES
    Percival, D. B. and A. T. Walden (2000) Wavelet Methods for
    Time Series Analysis. Cambridge: Cambridge University Press.
    
    SEE ALSO
    modwt_filter
    """
      
    available_choices = ('conservative', 'max', 'supermax')

    #Check for valid wtfname and get wavelet filter coefficients

    wtf = wtfilter(wtfname)

    L = wtf.L

    if choice=='conservative':
        J0 = np.floor(np.log2( (float(N) / (L - 1)) - 1))
    elif choice in ('max', 'maximum'):
        J0 = np.floor(np.log2(N))
    elif ('supermax', 'supermaximum'):
        J0 = np.floor(np.log2(1.5 * N))
    else:
        raise ValueError('WMTSA:invalidNLevelsValue: available choices are {}'.format(available_choices))
    return np.int(J0)

def num_nonboundary_coef(wtfname, N, j):
    """
     modwt_num_nonboundary_coef -- Calculate number of nonboundary MODWT coefficients for jth level.
    
     INPUTS
       * wtfname    -- name of wavelet transform filter (string).
       * N          -- number of samples (integer).
       * j          -- jth level (index) of scale or range of j levels.
                       (integer or vector of integers).
    
     OUTPUTS
       * MJ         -- number of nonboundary MODWT coefficients for specified
                       levels (integer or Jx1 vector of integers).
    
     DESCRIPTION
       N-Lj+1 can become negative for large j, so set MJ = min(MJ, 0).
    
     EXAMPLE
    
     ALGORITHM
       M_j = N - Lj + 1
    
       see page 306 of WMTSA for definition.
    
     REFERENCES
       Percival, D. B. and A. T. Walden (2000) Wavelet Methods for
         Time Series Analysis. Cambridge: Cambridge University _roPress.
    """
    # Check for valid wtf and get wtf filter coefficients
    wtf = wtfilter(wtfname)
    
    L = wtf.L
    
    N = np.int(N); j = np.int32(j)
     
    # Calculate MJ
    Lj = equivalent_filter_width(L, j)
    MJ = N - Lj + 1
    MJ = np.maximum(MJ, 0)
    
    return MJ
    
    
    
    
    
    
    
    
