#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# misc - cuda core misc helpers
# Copyright (C) 2011-2012  The Cph CT Toolbox Project lead by Brian Vinter
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

"""Cuda core misc helper functions"""

from cphct.npycore import allowed_cdata_types
from cphct.cu import gpuarray, elementwise


def __dtype_npy_to_cu(data):
    """Translate numpy dtype to CUDA data type string
    
    Parameters
    ----------
    data : gpuarray
        GPU data array
    
    Returns
    -------
    output : str
        CUDA data type string for *data.dtype*

    Raises
    ------
    ValueError
        If *data.dtype* is unsupported
    """

    if data.dtype.name not in allowed_cdata_types:
        ValueError('Unsupported CUDA data type: %s' % data.dtype)

    return allowed_cdata_types[data.dtype.name]


def gpuarray_copy(conf, data, out=None):
    """Copy gpuarray

    Parameters
    ----------
    conf : dict                      
        Configuration dictionary.
    data : gpuarray
        Input gpuarray to copy
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
        A copy of *data*
    """

    gpu_module = conf['gpu']['module']

    if out is None:
        out = gpuarray.empty_like(data)

    gpu_module.memcpy_dtod(out.gpudata, data.gpudata, data.nbytes)

    return out


def gpuarray_square(data, out=None):
    """Squares gpuarray
    
    Parameters
    ----------
    data : gpuarray
        Input gpuarray to square
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
        A squared *data*
    """

    if out is None:
        out = gpuarray.empty_like(data)

    kernel_args = '%sdata, %sout' % (__dtype_npy_to_cu(data),
            __dtype_npy_to_cu(data))
    kernel_code = 'out[i] = data[i] * data[i]'
    kernel_name = 'square_kernel'

    square_kernel = elementwise.ElementwiseKernel(kernel_args,
            kernel_code, kernel_name)

    square_kernel(data, out)

    return out


