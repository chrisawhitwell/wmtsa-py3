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
from wmtsa.modwt import modwt_choose_nlevels, equivalent_filter_width

import numpy as np

def modwpt(X, wtf='la8', nlevels='conservative', boundary='reflection'):
    """
    function modwpt(X, wtf='la8', nlevels='conservative', boundary='reflection')
       modwt -- Compute the (partial) maximal overlap discrete wavelet packet transform (MODWPT).
    
    NAME
       modwpt -- Compute the (partial) maximal overlap discrete wavelet packet transform (MODWPT).
    
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
    
    OUTPUTS
       WJnt       -- MODWPT wavelet coefficents (2**J0 x NW array) dwtArray
    
    DESCRIPTION
       modwpt calculates the wavelet packet coefficients using the MODWPT.
    
       The output arguments include an info attribute with metadata.
       info is a dictionary with the following fields:
       * Transform  -- name of transform ('MODWPT')
       * WTF        -- name of wavelet transform filter or a wtf_s struct.
       * NX         -- number of observations in original series (= length(X))
       * NW         -- number of wavelet coefficients
       * J0         -- number of levels of partial decompsition.
       * Boundary   -- boundary conditions applied.
       * Aligned    -- Boolean flag indicating whether coefficients are aligned
                       with original series (1 = true) or not (0 = false).
       * BCs        -- (n x 2) array with indices of last element to the left,
                       and first element to the right which are influenced to some 
                       extent by boundary conditions. Given at each decomposition level

    
    EXAMPLE

       WJnt = modwpt(x, 'la8', 6, 'reflection')
    
    NOTES
    
    ALGORITHM
       See pag. 231 of WMTSA for description of Pyramid Algorithm for
       the MODWPT.
    
    REFERENCES
       Percival, D. B. and A. T. Walden (2000) Wavelet Methods for
       Time Series Analysis. Cambridge: Cambridge University Press.
    """

    # Get a valid wavelet transform filter coefficients struct.
    wtf_s = wtfilter(wtf, transform = 'MODWPT')
  
    wtfname = wtf_s.Name
    gt = wtf_s.g
    ht = wtf_s.h

    # Make sure that X is a numpy array
    X = np.array(X)
    if len(X.shape)>1:
        raise ValueError('Input array must be one-dimensional')
    #  N    = length of original series
    N = X.size
        
    #  If nlevels is an integer > 0, set J0 = nlevels.
    #  otherwise, select J0 based on choice method specified.
    if isinstance(nlevels, str):
        J0 = modwt_choose_nlevels(nlevels, wtfname, N)
    elif isinstance(nlevels, int):
        if nlevels > 0:
            J0 = nlevels
        else:
            raise ValueError('Negative J0, nlevels must be an integer greater than 0.')
    else:
        raise ValueError('Invalid NLevels Value')
    
    if (J0 < 0):
        raise ValueError('Negative J0')
    
    if (2**J0 > N):
        print ('Warning (MODWT) Large J0, > log2(Number of samples).')

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
    
    # Do the MODWPT.
    import pyximport; pyximport.install()
    from modwptjn import modwptjn
    # We have to add a dimension to Win in order to
    # use modptjn
    Win = np.ndarray((1,NW)); Win[0,:] = Xin
    bw = np.ndarray((2**J0,2))*np.nan
    Wout = modwptjn(Win, J0, ht, gt)

    # Update attributes
    att = {'Transform':'MODWPT',
           'WTF'      : wtfname,
           'N'        : N,
           'NW'       : NW,
           'J0'       : J0,
           'Boundary' : boundary,
           'Aligned'  : False,
           'Type'     : 'Wavelet',
           'BCs'      : np.int32(bw)
           }
    WJnt = dwtArray(Wout, info=att)
    
    return WJnt

def modwpt_cir_shift(WJt):
    """
    shift_modwt_coef -- shift the MODWT wavelet and scaling coefficients.
    
    NAME
        shift_modwt_coef -- Shift the MODWT wavelet and scaling coefficients.

    INPUTS
        WJt          =  (2**J)xN dwtArray of MODWPT wavelet coefficents
                        where N = number of time intervals,
                              J = number of levels

    OUTPUTS
        W  = shifted wavelet coefficients with boundary conditions (dwtArray)

    DESCRIPTION
       The MODWPT coefficients are circularly shifted at each level so as to 
       properly align the coefficients with the original data series.

    REFERENCES
       See pag.230 of WMTSA.
    """
    
    # check input
    if type(WJt) is not dwtArray:
        raise TypeError('Input must be a dwtArray')
    if WJt.info['Type'] is not 'Wavelet':
        raise TypeError('Input array does not contain Wavelet coefficients but {} coefficients'.format(WJt.info['Type']))
    if WJt.info['Transform'] is not 'MODWPT':
        raise TypeError('Input array does not contain MODWPT coefficients but {} coefficients'.format(WJt.info['Transform']))
    
    wtf_s = wtfilter(WJt.info['WTF'])
    L  = wtf_s.L

    wtfname       = WJt.info['WTF']
    N             = WJt.info['N']
    NW            = WJt.info['NW']
    J0            = WJt.info['J0']
    
    Nm = min(N,NW)

    if WJt.info['Aligned']:
        print ('WARNING (wmtsa.modwpt.modwpt_cir_shift): Coefficients are already aligned')
        W = WJt
    else:
        if WJt!=dwtArray([]):
            W = np.ndarray(WJt.shape)*np.nan; bw = np.ndarray((2**J0,2))*np.nan
            for n in range(2**J0):
                # shift wavelet coefficients
                nujn = advance_filter(wtfname, J0)
                W[n,:] = np.roll(WJt[n,:], nujn)
                
                #Calculate circularly shifted wavelet coefficient boundary indices at jth level
                L_j   = equivalent_filter_width(L, J0)
        
                bw[n,0] = L_j - 2 - np.abs(nujn) #max index to the left
                bw[n,1] = Nm - abs(nujn) #min index to the right
            W = W[:,:Nm]
            # Update attributes
            att = dict(WJt.info.items() +
                       {'Aligned':True,
                        'BCs'    :np.int32(bw)
                        }.items())
            W = dwtArray(W, info=att)
        else: # if an empty array was given, do nothing and return it
            W = WJt

    return W

def advance_filter(wtfname, j):
    """
    advance_time_series_filter -- Calculate the advance of the time series or filter for a given wavelet.
    
    NAME
       advance_time_series_filter -- Calculate the advance of the time series or filter for a given wavelet.
    
    SYNOPSIS
       nujn = advance_time_series_filter(wtfname,j)
    
    INPUTS
       wtfname      -  string containing name of WMTSA-supported wavelet filter.
    
    OUTPUTS
       nu           -  advance of time series for specified wavelet filter.
    
    SIDE EFFECTS
       wavelet is a WMTSA-supported wavelet filter; otherwise error.
    
    DESCRIPTION
    
    EXAMPLE
    
    ALGORITHM
      
    REFERENCES
       Percival, D. B. and A. T. Walden (2000) Wavelet Methods for
         Time Series Analysis. Cambridge: Cambridge University Press.
    
    SEE ALSO
       dwt_filter    
    """

    def Sjn1(J0):
        """
        See pag. 229 and pag. 215 WMTSA
        """
        cjIn = np.array([0,1]) #cj1,n
        for j in range(2,J0):
            for n in range(2**j):
                mod = n%4
                if (mod==0)|(mod==3):
                    cjOut = np.hstack((cjIn,0))
                else:
                    cjOut = np.hstack((cjIn,1))
            cjIn = cjOut
        Sjn1 = np.sum(cjOut*(2**np.arange(J0)))
        return Sjn1
        
    
    def advance_least_asymetric_filter(L, L_j, j):
        #Equation 230 of WMTSA
        if L in(10, 18):
            nu = -(L_j/2.) - (2**(j-1) - Sjn1(j)) +1
        elif L==14:
            nu = -(L_j/2.) + 3*(2**(j-1) - Sjn1(j)) -1
        elif (L/2)%2 == 0:
            #L/2 is even, i.e. L = 8, 12, 16, 20
            nu = -(L_j/2.) + (2**(j-1) - Sjn1(j))
        else:
            raise ValueError('Invalid filter length (L = {}) specified.'.format(L))
        return nu

    def advance_coiflet_filter(L, L_j, j):
        #Page 124 and equation 124 of WMTSA
        nu = -(L_j/2.) - (L_j-3)/3.0*(2**(j-1) - Sjn1(j) - 0.5) + 0.5
        return nu

    # Get a valid wavelet transform filter coefficients struct.
    wtf_s = wtfilter(wtfname)
    L     = wtf_s.L
    L_j   = equivalent_filter_width(L, j)

    nujn = np.nan

    #Haar
    if wtfname.lower() in ('haar','d2'):
        nujn = 0
    #Least Asymmetric filters
    elif wtfname.lower() in ('la8', 'la10', 'la12', 'la14', 'la18', 'la16', 'la20'):
        nujn = advance_least_asymetric_filter(L, L_j, j)
    #Coiflet filters
    elif wtfname.lower() in ('c6', 'c12', 'c18', 'c24', 'c30'):
        nujn = advance_coiflet_filter(L, L_j, j)
    #otherwise
    else:
        pass
    
    return np.int16(np.around(nujn))
