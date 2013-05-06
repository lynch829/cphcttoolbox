#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# __init__ - global pycuda engine module init
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

"""Cph CT Toolbox global pycuda engine module initializer.

Clients should *only* import pycuda
stuff through here to ease future back end replacement.

I.e. please *DO NOT* mix cphct.cu imports with direct pycuda
imports like:
from pycuda import driver
from cphct.cu import compiler

Instead use cphct.cu as the *only* source:
from cphct.cu import driver, compiler
"""

__dummy = \
    '''This dummy right after the module doc string prevents PythonTidy
from incorrectly moving following comments above module doc string'''

# Expose everything from pycuda

from pycuda import *
from pycuda import compiler, driver, tools, gpuarray, elementwise

# All sub modules to load in case of 'from X import *'

__all__ = ['core']
