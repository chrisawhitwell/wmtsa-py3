###########################################################################
#                                                                         #
#    Copyright 2013 Andrea Cimatoribus                                    #
#    NIOZ, PO Box 59, 1790AB, Den Burg (Texel), Netherlands               #
#    Andrea.Cimatoribus@nioz.nl                                           #
#                                                                         #
#    This file is part of wmtsa toolbox.                                  #
#                                                                         #
#    wmtsa toolbox is free software: you can redistribute it              #
#    and/or modify it under the terms of the GNU Lesser General Public    #
#    License as published by the Free Software Foundation, either         #
#    version 3 of the License, or (at your option) any later version.     #
#                                                                         #
#    wmtsa toolbox is distributed in the hope that it will be             #
#    useful, but WITHOUT ANY WARRANTY; without even the implied warranty  #
#    of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the     #
#    GNU Lesser General Public License for more details.                  #
#                                                                         #
#    You should have received a copy of the GNU Lesser General Public     #
#    License along with wmtsa.                                            #
#    If not, see <http://www.gnu.org/licenses/>.                          #
#                                                                         #
###########################################################################
'''
Created on May 16, 2013

@author: cimatori
'''

import numpy as np
# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport numpy as np
# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE = np.float64
# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.float64_t DTYPE_t

def modwtj(np.ndarray[DTYPE_t, ndim=1] Vin, int j, \
            np.ndarray[DTYPE_t, ndim=1] ht, np.ndarray[DTYPE_t, ndim=1] gt):
    """
    MODWT transform pyramid algorithm
    see pag.178 WMTSA
    """

    cdef int k, n, t
    cdef int N = Vin.shape[0]
    cdef int L = ht.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] Wout = np.ndarray(N, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] Vout = np.ndarray(N, dtype=DTYPE)

    if L!=gt.shape[0]: raise ValueError('filters ht and gt must have the same length')
    
    for t in range(N):
        k = t
        Wout[t] = ht[0] * Vin[k]
        Vout[t] = gt[0] * Vin[k]
        for n in range(1,L):
            k -= int(2**(j - 1))
            if (k < 0):
                k = k%N
            Wout[t] += ht[n] * Vin[k]
            Vout[t] += gt[n] * Vin[k]
    return Wout, Vout

def imodwtj(np.ndarray[DTYPE_t, ndim=1] Win, np.ndarray[DTYPE_t, ndim=1] Vin, \
            int j, np.ndarray[DTYPE_t, ndim=1] ht, np.ndarray[DTYPE_t, ndim=1] gt):
    """
    MODWT inverse transform pyramid algorithm
    see pag.178 WMTSA
    """

    cdef int k, n, t
    cdef int N = Vin.shape[0]
    cdef int L = ht.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] Vout = np.zeros(N, dtype=DTYPE)

    if L!=gt.shape[0]: raise ValueError('filters ht and gt must have the same length')
    
    for t in range(N):
        k = t
        Vout[t] = (ht[0] * Win[k]) + (gt[0] * Vin[k])
        for n in range(1,L):
            k += int(2**(j - 1))
            if (k >= N):
                k = k%N
            Vout[t] += (ht[n] * Win[k]) + (gt[n] * Vin[k])
    return Vout
