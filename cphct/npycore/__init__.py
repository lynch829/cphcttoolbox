#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# __init__ - global numpy core engine module init
# Copyright (C) 2011-2013  The Cph CT Toolbox Project lead by Brian Vinter
#
# This file is part of Cph CT Toolbox.
#
# Cph CT Toolbox is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# Cph CT Toolbox is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301,
# USA.
#
# -- END_HEADER ---
#

"""Cph CT Toolbox global numpy core engine module initializer. Used as a
shared helper for all numpy-based engines. Please only use npycore code
internally in the cphct package and instead rely on the actual npy, cu or ocl
interfaces in apps, plugins and tools.
"""

__dummy = \
    '''This dummy right after the module doc string prevents PythonTidy
from incorrectly moving following comments above module doc string'''

# Expose everything from numpy

from numpy import __all__ as __numpy_all__
from numpy import *

allowed_data_types = {
    'float32': float32,
    'float64': float64,
    'uint16': uint16,
    'uint32': uint32,
    'uint64': uint64,
    'complex64': complex64,
    }

allowed_cdata_types = {
    'float32': 'float *',
    'float64': 'double *',
    'uint16': 'unsigned short *',
    'uint32': 'unsigned int *',
    'uint64': 'unsigned long *',
    }

# All sub modules to load in case of 'from X import *'

__priv = [i for i in locals().keys() \
          if not i in __numpy_all__ and not i.startswith('_')]
__all__ = __numpy_all__ + __priv
