#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# misc - OpenCL core misc helpers
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

"""OpenCL core misc helper functions"""

from cphct.npycore import allowed_cdata_types
from cphct.cl import gpuarray, elementwise, opencl
from cphct.cl.core import get_active_gpu_context, gpu_alloc_from_array


def __dtype_npy_to_cl(data):
    """Translate numpy dtype to OpenCL data type string
    
    Parameters
    ----------
    data : gpuarray
        GPU data array
    
    Returns
    -------
    output : str
        OpenCL data type string for *data.dtype*

    Raises
    ------
    ValueError
        If *data.dtype* is unsupported
    """

    if data.dtype.name not in allowed_cdata_types:
        ValueError('Unsupported OpenCL data type: %s' % data.dtype)

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

    gpu_queue = conf['gpu']['queue']

    if out is None:
        out = gpuarray.empty_like(data)

    opencl.enqueue_copy(gpu_queue, gpu_alloc_from_array(out),
                        gpu_alloc_from_array(data))

    return out


def gpuarray_square(conf, data, out=None):
    """Squares gpuarray
    
    Parameters
    ----------
    conf : dict                      
        Configuration dictionary.
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

    gpu_ctx = get_active_gpu_context(conf)
    if out is None:
        out = gpuarray.empty_like(data)

    kernel_args = '%sdata, %sout' % (__dtype_npy_to_cl(data),
            __dtype_npy_to_cl(data))
    kernels_code = 'out[i] = data[i] * data[i]'
    kernel_name = 'square_kernel'

    square_kernel = elementwise.ElementwiseKernel(gpu_ctx, kernel_args,
            kernels_code, kernel_name)

    square_kernel(data, out)

    return out


