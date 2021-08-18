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

@author: Andrea Cimatoribus
'''

__all__ = ['modwt']

import numpy as np

class dwtArray(np.ndarray):
    """
    class dwtArray(numpy.ndarray)
    
    NAME
       dwtArray -- generic array containing meta data on the transformation
    
    INPUTS
       same as numpy.ndarray
       
       info -- (optional) pass a dictionary of meta data
       
    DESCRIPTION
       An array class that contains meta data.
       This is the basic building block of the package, since it enables to 
       easily pass data with information on the transformations applied to
       the various functions
       
    EXAMPLE
       a = np.arange(10)
       b = a.view(dwtArray)
       b.info = {'metadata':'info'}
       
    WARNINGS
    
    ERRORS  
    
    NOTES
    
    ALGORITHM
    
    REFERENCES
       http://docs.scipy.org/doc/numpy/user/basics.subclassing.html
        
    SEE ALSO
    """
    
    stdinfo = {'Transform': None,
               'Type'     : None,
               'WTF'      : None,
               'N'        : None,
               'NW'       : None,
               'J0'       : None,
               'Boundary' : None,
               'Aligned'  : None,
               'RetainVJ' : None,
               'BCs'      : None
               }
    
    def __new__(cls, input_array, info=stdinfo):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.info = info
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self.info = getattr(obj, 'info', self.stdinfo)

class wtfilter(object):
    """
    wtfilter -- Class defining the wavelet filter, see __init__
    """
    
    def __init__(self, wtfilter, transform='MODWT'):
        """
         wtfilter -- Define wavelet transform filter coefficients.
        
         NAME
           wtfilter -- Define wavelet transform filter coefficients.
        
         INPUTS
           * wtfname    -- name of wavelet transform filter (string, case-insenstive).
           * transform  -- name of wavelet transform  (string, case-insenstive).
        
         OUTPUTS
           * wtf        -- wavelet tranform filter class (wtf_s).
        
         DESCRIPTION
           wtfilter returns a class with the wavelet (high-pass) and
           scaling (low-pass) filter coefficients, and associated attributes. 
        
           The wtf_s class has attributes:
           * g         -- scaling (low-pass) filter coefficients (vector).
           * h         -- wavelet (high-pass) filter coefficients (vector).
           * L         -- filter length (= number of coefficients) (integer).
           * Name      -- name of wavelet filter (character string).
           * WTFclass  -- class of wavelet filters (character string).
           * Transform -- name of transform (character string).
        
           The MODWT filter coefficients are calculated from the DWT filter 
           coefficients:
        
              ht = h / sqrt(2)
              gt = g / sqrt(2)
        
           The wavelet filter coefficients (h) are calculated from the scaling
           filter coefficients via the QMF function (wmtsa_qmf).
         
         EXAMPLE
            wtf = wtfilter('LA8', 'modwt')
            wtf = wtfilter('haar', 'dwt')
        
         ALGORITHM
        
        
         REFERENCES
           Percival, D. B. and A. T. Walden (2000) Wavelet Methods for
             Time Series Analysis. Cambridge: Cambridge University Press.
        
         SEE ALSO
          wtf_s, wtf_qmf
        """

#TODO: implement other wavelet filters

        if wtfilter.lower()=='la8':
            # **********************************
            # ***  Least asymmetric filters  ***
            # **********************************
            self.Name ='LA8'
            self.g = np.array(\
                        [-0.0757657147893407,-0.0296355276459541, 0.4976186676324578,\
                          0.8037387518052163,  0.2978577956055422,-0.0992195435769354,\
                         -0.0126039672622612,  0.0322231006040713])
            self.h = wmtsa_qmf(self.g)
            self.L = 8
            self.SELFClass = 'LeastAsymmetric'
            self.Transform = 'DWT'
        elif wtfilter.lower()=='la10':
            self.Name ='LA10'
            self.g = np.array(\
                             [0.0195388827353869, -0.0211018340249298, -0.1753280899081075,\
                              0.0166021057644243,  0.6339789634569490,  0.7234076904038076,\
                              0.1993975339769955, -0.0391342493025834,  0.0295194909260734,\
                              0.0273330683451645])
            self.h = wmtsa_qmf(self.g)
            self.L = 10
            self.WTFClass = 'LeastAsymmetric'
            self.Transform = 'DWT'
        elif wtfilter.lower()=='la12':
            self.Name ='LA12'
            self.g = np.array(\
                              [0.0154041093273377, 0.0034907120843304, -0.1179901111484105,
                              -0.0483117425859981, 0.4910559419276396,  0.7876411410287941,
                               0.3379294217282401,-0.0726375227866000, -0.0210602925126954,
                               0.0447249017707482, 0.0017677118643983, -0.0078007083247650])
            self.h = wmtsa_qmf(self.g)
            self.L = 12
            self.WTFClass = 'LeastAsymmetric'
            self.Transform = 'DWT'
        elif wtfilter.lower()=='la14':
            self.Name ='LA14'
            self.g = np.array(\
                              [0.0102681767084968, 0.0040102448717033, -0.1078082377036168,
                              -0.1400472404427030, 0.2886296317509833,  0.7677643170045710,
                               0.5361019170907720, 0.0174412550871099, -0.0495528349370410,
                               0.0678926935015971, 0.0305155131659062, -0.0126363034031526,
                              -0.0010473848889657, 0.0026818145681164])
            self.h = wmtsa_qmf(self.g)
            self.L = 14
            self.WTFClass = 'LeastAsymmetric'
            self.Transform = 'DWT'
        elif wtfilter.lower()=='la16':
            self.Name ='LA16'
            self.g = np.array(\
                              [-0.0033824159513594, -0.0005421323316355, 0.0316950878103452,
                                0.0076074873252848, -0.1432942383510542, -0.0612733590679088,
                                0.4813596512592012,  0.7771857516997478,  0.3644418948359564,
                               -0.0519458381078751, -0.0272190299168137,  0.0491371796734768,
                                0.0038087520140601, -0.0149522583367926, -0.0003029205145516,
                                0.0018899503329007])
            self.h = wmtsa_qmf(self.g)
            self.L = 16
            self.WTFClass = 'LeastAsymmetric'
            self.Transform = 'DWT'
        elif wtfilter.lower()=='la18':
            self.Name ='LA18'
            self.g = np.array(\
                              [0.0010694900326538, -0.0004731544985879, -0.0102640640276849,
                               0.0088592674935117,  0.0620777893027638, -0.0182337707798257,
                              -0.1915508312964873,  0.0352724880359345,  0.6173384491413523,
                               0.7178970827642257,  0.2387609146074182, -0.0545689584305765,
                               0.0005834627463312,  0.0302248788579895, -0.0115282102079848,
                              -0.0132719677815332,  0.0006197808890549,  0.0014009155255716])
            self.h = wmtsa_qmf(self.g)
            self.L = 18
            self.WTFClass = 'LeastAsymmetric'
            self.Transform = 'DWT'
        elif wtfilter.lower()=='la20':
            self.Name ='LA20'
            self.g = np.array(\
                              [0.0007701598091030, 0.0000956326707837, -0.0086412992759401,
                              -0.0014653825833465, 0.0459272392237649,  0.0116098939129724,
                              -0.1594942788575307,-0.0708805358108615,  0.4716906668426588,
                               0.7695100370143388, 0.3838267612253823, -0.0355367403054689,
                              -0.0319900568281631, 0.0499949720791560,  0.0057649120455518,
                              -0.0203549398039460,-0.0008043589345370,  0.0045931735836703,
                               0.0000570360843390,-0.0004593294205481])
            self.h = wmtsa_qmf(self.g)
            self.L = 20
            self.WTFClass = 'LeastAsymmetric'
            self.Transform = 'DWT'
        elif wtfilter.lower()in('haar','d2'):
            # **********************************
            # ***  Extremal phase filters    ***
            # **********************************
            self.Name = 'Haar'
            self.g = np.array(\
                              [0.7071067811865475, 0.7071067811865475])
            self.h = wmtsa_qmf(self.g)
            self.L = 2
            self.WTFClass = 'ExtremalPhase'
            self.Transform = 'DWT'
        elif wtfilter.lower()=='d4':
            self.Name = 'D4'
            self.g    = np.array(\
                                 [0.4829629131445341, 0.8365163037378077, 0.2241438680420134, -0.1294095225512603])
            self.h = wmtsa_qmf(self.g)
            self.L = 4
            self.WTFClass = 'ExtremalPhase'
            self.Transform = 'DWT'
        elif wtfilter.lower()=='d6':
            self.Name = 'D6'
            self.g = np.array(\
                              [0.3326705529500827, 0.8068915093110928, 0.4598775021184915,
                              -0.1350110200102546,-0.0854412738820267, 0.0352262918857096])
            self.h = wmtsa_qmf(self.g)
            self.L = 6
            self.WTFClass = 'ExtremalPhase'
            self.Transform = 'DWT'
        elif wtfilter.lower()=='d8':
            self.Name ='D8'
            self.g = np.array(\
                              [0.2303778133074431, 0.7148465705484058, 0.6308807679358788,
                              -0.0279837694166834,-0.1870348117179132, 0.0308413818353661,
                               0.0328830116666778,-0.0105974017850021])
            self.h = wmtsa_qmf(self.g)
            self.L = 8
            self.WTFClass = 'ExtremalPhase'
            self.Transform = 'DWT'
        elif wtfilter.lower()=='d10':
            self.Name ='D10'
            self.g = np.array(\
                              [0.1601023979741930, 0.6038292697971898, 0.7243085284377729,
                               0.1384281459013204,-0.2422948870663824,-0.0322448695846381,
                               0.0775714938400459,-0.0062414902127983,-0.0125807519990820,
                               0.0033357252854738])
            self.h = wmtsa_qmf(self.g)
            self.L = 10
            self.WTFClass = 'ExtremalPhase'
            self.Transform = 'DWT'
        elif wtfilter.lower()=='d12':
            self.Name ='D12'
            self.g = np.array(\
                              [0.1115407433501094, 0.4946238903984530, 0.7511339080210954,
                               0.3152503517091980,-0.2262646939654399,-0.1297668675672624,
                               0.0975016055873224, 0.0275228655303053,-0.0315820393174862,
                               0.0005538422011614, 0.0047772575109455,-0.0010773010853085])
            self.h = wmtsa_qmf(self.g)
            self.L = 12
            self.WTFClass = 'ExtremalPhase'
            self.Transform = 'DWT'
        elif wtfilter.lower()=='d14':
            self.Name ='D14'
            self.g = np.array(\
                              [0.0778520540850081, 0.3965393194819136, 0.7291320908462368,
                               0.4697822874052154,-0.1439060039285293,-0.2240361849938538,
                               0.0713092192668312, 0.0806126091510820,-0.0380299369350125,
                              -0.0165745416306664, 0.0125509985560993, 0.0004295779729214,
                              -0.0018016407040474, 0.0003537137999745])
            self.h = wmtsa_qmf(self.g)
            self.L = 14
            self.WTFClass = 'ExtremalPhase'
            self.Transform = 'DWT'
        elif wtfilter.lower()=='d16':
            self.Name ='D16'
            self.g = np.array(\
                              [0.0544158422431049, 0.3128715909143031, 0.6756307362972904,
                               0.5853546836541907,-0.0158291052563816,-0.2840155429615702,
                               0.0004724845739124, 0.1287474266204837,-0.0173693010018083,
                              -0.0440882539307952, 0.0139810279173995, 0.0087460940474061,
                              -0.0048703529934518,-0.0003917403733770, 0.0006754494064506,
                              -0.0001174767841248])
            self.h = wmtsa_qmf(self.g)
            self.L = 16
            self.WTFClass = 'ExtremalPhase'
            self.Transform = 'DWT'
        elif wtfilter.lower()=='d18':
            self.Name ='D18'
            self.g = np.array(\
                              [0.0380779473638791, 0.2438346746125939, 0.6048231236901156,
                               0.6572880780512955, 0.1331973858249927,-0.2932737832791761,
                              -0.0968407832229524, 0.1485407493381306, 0.0307256814793395,
                              -0.0676328290613302, 0.0002509471148340, 0.0223616621236805,
                              -0.0047232047577520,-0.0042815036824636, 0.0018476468830564,
                               0.0002303857635232,-0.0002519631889427, 0.0000393473203163])
            self.h = wmtsa_qmf(self.g)
            self.L = 18
            self.WTFClass = 'ExtremalPhase'
            self.Transform = 'DWT'
        elif wtfilter.lower()=='d20':
            self.Name ='D20'
            self.g = np.array(\
                              [0.0266700579005546, 0.1881768000776863, 0.5272011889317202,
                               0.6884590394536250, 0.2811723436606485,-0.2498464243272283,
                              -0.1959462743773399, 0.1273693403357890, 0.0930573646035802,
                              -0.0713941471663697,-0.0294575368218480, 0.0332126740593703,
                               0.0036065535669880,-0.0107331754833036, 0.0013953517470692,
                               0.0019924052951930,-0.0006858566949566,-0.0001164668551285,
                               0.0000935886703202,-0.0000132642028945])
            self.h = wmtsa_qmf(self.g)
            self.L = 20
            self.WTFClass = 'ExtremalPhase'
            self.Transform = 'DWT'
        elif wtfilter.lower()=='c6':
            # **********************************
            # ***  Coiflet filters           ***
            # **********************************
            self.Name ='C6'
            self.g = np.array(\
                              [-0.0156557285289848, -0.0727326213410511, 0.3848648565381134,\
                               0.8525720416423900,  0.3378976709511590,-0.0727322757411889])
            self.h = wmtsa_qmf(self.g)
            self.L = 6
            self.WTFClass = 'Coiflet'
            self.Transform = 'DWT'
        else:
            raise ValueError ('Unrecognised wavelet filter name')

        if transform.lower()=='modwt':
            self.Transform = 'MODWT'
            self.g = self.g / (2**0.5)
            self.h = self.h / (2**0.5)

        elif transform.lower()=='modwpt':
            self.Transform = 'MODWPT'
            self.g = self.g / (2**0.5)
            self.h = self.h / (2**0.5)
            
def wmtsa_qmf(a, inverse=False):
    """
     wmtsa_qmf -- Calculate quadrature mirror filter (QMF).
    
     NAME
       wmtsa_qmf -- Calculate quadrature mirror filter (QMF).
    
     INPUTS
       * a           -- filter coefficients (vector).
       * inverse     -- (optional) flag for calculating inverse QMF (Boolean).
                        Default: inverse = False
    
     OUTPUTS
        b            - QMF coefficients (vector).
    
     DESCRIPTION
        wmtsa_qmf calculates the quadrature mirror filter (QMF) of
        for the specified filter coefficients.  If a is a vector,
        the QMF of the vector is calculated. If a is an array, an
        error is raised
    
       The inverse flag, if set, calculates the inverse QMF.  inverse
       is a Boolean values specified as (1/0, y/n, T/F or true/false).
    
     EXAMPLE
        # h is the QMF of g.
        g = [0.7071067811865475 0.7071067811865475];
        h = wmtsa_qmf(g);
    
        # g is the inverse QMF of h.
        h = [0.7071067811865475 -0.7071067811865475];
        g = wmtsa_qmf(h, 1);
    
     ALGORITHM
          g_l = (-1)^(l+1) * h_L-1-l
          h_l = (-1)^l * g_L-1-l
        See pages 75 of WMTSA for additional details.
    
     REFERENCES
       Percival, D. B. and A. T. Walden (2000) Wavelet Methods for
       Time Series Analysis. Cambridge: Cambridge University Press.
    
     SEE ALSO
       yn
    """

    a = np.array(a)
    
    # we must deep copy a, otherwise we will modify a too
    b = a.copy()
    
    if len(b.shape)>1:
        raise ValueError('Input array must be 1-dimensional')
    
    b = b[::-1]

    if inverse:
        first = 0
    else:
        first = 1
    
    b[first::2] = -b[first::2]
  
    return b

def acvs(X, estimator='biased', subtract_mean=True, method='fft', dim=-1):
    """
    wmtsa_acvs -- Calculate the autocovariance sequence (ACVS) of a data series.

     INPUTS
       X           -- time series (array).
       estimator   -- (optional) type of estimator
                        Valid values:  'biased', 'unbiased', 'none'
                        Default: 'biased'
       subtract_mean -- (optional) flag whether to subtract mean
                        Default: True = subtract mean
       method      -- (optional) method used to calculate ACVS (character string).
       dim         -- (optional) dimension to compute ACVS along (integer).
                        Default: last dimension
     OUTPUTS
       ACVS        -- autocovariance sequence (ACVS) (vector or matrix).
    
     DESCRIPTION
       wmtsa_acvs calculates the autocovariance sequence (ACVS) for a real valued
       series.
     
       By default, the function calculates the ACVS over the last dimension.
       For the current implementation X must be a 1 or 2d array.
       If X is a vector, ACVS is returned with dimensions as X.  If X is a 2d array,
       ACVS is calculated for the rows.
       If input argument 'dim' is supplied, the ACVS is calculated over that dim.
       
       The estimator option normalizes the ACVS estimate as follows:
       * 'biased'   -- divide by N
       * 'unbiased' -- divide by N - tau
       * 'none'     -- unnormalized.
    
       The 'subtract_mean' input argument specifies whether to subtract
       the mean from the prior to calculating the ACVS. The default is to
       subtract them mean.
    
       The 'method' input argument specifies the method used to calculate
       the ACVS:
        'lag'     -- Calculate taking lag products.
        'fft'     -- Calculate via FFT.
       The default is 'fft'.
    
    
     ALGORITHM
       See page 266 of WMTSA for definition of ACVS.
       See page 269 of WMTSA for definition of biased ACVS estimator.
    
     REFERENCES
       Percival, D. B. and A. T. Walden (2000) Wavelet Methods for
         Time Series Analysis. Cambridge: Cambridge University Press.
    """
    
    valid_estimators = ('biased', 'unbiased', 'none')
    
    # check input
    if estimator.lower() not in valid_estimators:
        raise ValueError('Bad estimator: "{}". Valid estimator methods are only: {}'.format(estimator,valid_estimators))    
    try:
        sz   = X.shape
        ndim = len(sz)
        if ndim>2:
            raise TypeError('Can handle onle 1 or 2d arrays')
        else:
            N    = sz[dim]
            ACVS = np.zeros(sz)
    except AttributeError:
        # a scalar
        return 0.0

    if subtract_mean:
        X = X - X.mean(axis=dim)

    if method=='lag':
        if ndim==1:
            for tau in range(N):
                ACVS[tau] = np.sum(X[:N-tau]*X[tau:])
        elif ndim==2:
            if dim==0:
                for tau in range(N):
                    ACVS[tau,:] = np.sum(X[:N-tau,:]*X[tau:,:], axis=0)
            elif dim in (-1,1):
                for tau in range(N):
                    ACVS[:,tau] = np.sum(X[:,:N-tau]*X[:,tau:], axis=0)
    elif method=='fft':
        Nft  = np.int(2**(np.ceil(np.log2(N))))
        Xhat = np.fft.fft(X, Nft, axis=dim)
        ACVS = np.real(np.fft.ifft(Xhat*np.conjugate(Xhat), axis=dim))
        ACVS = ACVS[:N]
    else:
        raise ValueError('Unrecognised method for computing autocovariance')

    if estimator==None:
        pass
    elif estimator.lower()=='biased':
        ACVS = ACVS / np.float(N)
    elif estimator.lower()=='unbiased':
        tau = np.arange(N)
        if (dim==0) & (ndim>1):
            ACVS = ACVS / np.float64(N - tau)[:,np.newaxis]
        else:
            ACVS = ACVS / np.float64(N - tau)
    else:
        raise ValueError('Estimator "{}" is not a valid one'.format(estimator))

    # create even ACVS
    if ndim==2:
        if dim==0:
            ACVS = np.vstack((ACVS[::-1,:],ACVS[1:,:]))
        else:
            ACVS = np.hstack((ACVS[:,::-1],ACVS[:,1:]))
    elif ndim==1:
        ACVS = np.hstack((ACVS[::-1],ACVS[1:]))

    return ACVS
