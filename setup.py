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

from setuptools import setup,Extension, find_packages
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("wmtsa.modwtj",["wmtsa/modwtj.pyx"],libraries = [], include_dirs=['.',numpy.get_include()]),
    Extension("wmtsa.modwptjn",["wmtsa/modwptjn.pyx"],libraries = [], include_dirs=['.',numpy.get_include()])
    ]

ext_modules =  cythonize(extensions)

setup(	name = "wmtsapy3",
		author = 'Chris Whitwell',
		author_email = 'chris.whitwell@research.uwa.edu.au',
		cmdclass = {},
		ext_modules = ext_modules,
		packages=find_packages()
		)

