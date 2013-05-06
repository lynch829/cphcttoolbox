#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# __init__ - shared numpy engine lib module init
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

"""Cph CT Toolbox global numpy engine module initializer.

Clients should *only* import numpy
stuff through here to ease future back end replacement.

I.e. please *DO NOT* mix cphct.npy imports with direct numpy
imports like:
from numpy import zeros
from cphct.npy import ones

Instead use cphct.npy as the *only* source:
from cphct.npy import zeros, ones
"""

__dummy = \
    '''This dummy right after the module doc string prevents PythonTidy
from incorrectly moving following comments above module doc string'''

# Expose everything from npycore

from cphct.npycore import __all__ as __npycore_all__
from cphct.npycore import *

# All sub modules to load in case of 'from X import *'

__priv = [i for i in locals().keys() \
          if not i in __npycore_all__ and not i.startswith('_')]
__all__ = __npycore_all__ + __priv

