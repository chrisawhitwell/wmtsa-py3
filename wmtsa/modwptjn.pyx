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

def modwptjn(np.ndarray[DTYPE_t, ndim=2] Win, int j0, \
            np.ndarray[DTYPE_t, ndim=1] ht, np.ndarray[DTYPE_t, ndim=1] gt):
    """
    MODWPT transform pyramid algorithm
    see pag.231 WMTSA
    """

    cdef int j, k, n, t, l
    cdef int N = Win.shape[1]
    cdef int L = ht.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] Wout
    
    if L!=gt.shape[0]: raise ValueError('filters ht and gt must have the same length')

    for j in range(1,j0+1):
        Wout = np.ndarray((2**j,N), dtype=DTYPE)
        for n in range(0,2**j,4):
            for t in range(N):
                k = t
                Wout[n,t] = gt[0] * Win[n/2,k]
                for l in range(1,L):
                    k -= int(2**(j - 1))
                    if (k < 0):
                        k += N
                    Wout[n,t] += gt[l] * Win[n/2,k]
        for n in range(3,2**j,4):
            for t in range(N):
                k = t
                Wout[n,t] = gt[0] * Win[n/2,k]
                for l in range(1,L):
                    k -= int(2**(j - 1))
                    if (k < 0):
                        k += N
                    Wout[n,t] += gt[l] * Win[n/2,k]
        for n in range(1,2**j,4):
            for t in range(N):
                k = t
                Wout[n,t] = ht[0] * Win[n/2,k]
                for l in range(1,L):
                    k -= int(2**(j - 1))
                    if (k < 0):
                        k += N
                    Wout[n,t] += ht[l] * Win[n/2,k]
        for n in range(2,2**j,4):
            for t in range(N):
                k = t
                Wout[n,t] = ht[0] * Win[n/2,k]
                for l in range(1,L):
                    k -= int(2**(j - 1))
                    if (k < 0):
                        k += N
                    Wout[n,t] += ht[l] * Win[n/2,k]
        Win = Wout
    return Wout
