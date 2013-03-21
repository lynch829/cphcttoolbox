#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# base - cuda back end functions shared by plugin and tools
# Copyright (C) 2012  The Cph CT Toolbox Project lead by Brian Vinter
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

"""Cph CT Toolbox cuda back end functions shared by plugins and tools.
We separate I/O from the actual handlers so that they can be used inside apps
and in separate tools scripts."""

from cphct.cu import gpuarray
from cphct.cu.misc import gpuarray_square


def prepare_output(shape, conf):
    """Shared helper to create output matrix for manipulation

    Parameters
    ----------
    shape : tuple of int
        Tuple with integer dimensions of output matrix.
    conf : dict
        Configuration dictionary.

    Returns
    -------
    output : ndarray
        Returns output ndarray.
    """

    fdt = conf['output_data_type']
    return gpuarray.zeros(shape, dtype=fdt)


def square_array(data, out=None):
    """Square all values in data.

    Parameters
    ----------
    conf : dict
        Configuration dictionary.
    data : gpuarray
        Data matrix to square.
    out : gpuarray, optional
        Output argument. 
        This must have the exact kind that would be returned 
        if it was not used.
        In particular, it must have the right type, must be C-contiguous,
        and its dtype must be the dtype that would be returned 
        for without *out* set.
        This is a performance feature. 
        Therefore, if these conditions are not met,
        an exception is raised,  instead of attempting to be flexible.
    
    Returns
    -------
    output : gpuarray
        Returns squared ndarray.
    """

    return gpuarray_square(data, out)


