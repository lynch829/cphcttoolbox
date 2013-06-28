#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# flux2proj - plugin to convert measured intensities to attenuation projections
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

"""Flux to projection plugin to convert measured intensities to actual
attenuation projections.
"""

import os
from cphct.npycore import log, allowed_data_types
from cphct.npycore.io import load_helper_proj, get_npy_data
from cphct.npycore.utils import check_norm
from cphct.plugins import get_plugin_var
from cphct.cu.core import gpuarray, get_gpu_layout, generate_gpu_init, \
    load_kernel_source, compile_kernels, cuda, gpu_pointer_from_array
from cphct.cu.misc import gpuarray_copy

# Internal plugin state for individual plugin instances

__plugin_state__ = {}


def __make_gpu_kernels(conf, air_ref_pixel_idx=None):
    """Make the plugin GPU kernels based on *conf* and plugin kernel source
    
    Parameters
    ----------
    conf : dict
       A dictionary of configuration options.
    air_ref_pixel_idx : int, optional
       Flat projection pixel posistion containing air value
       
    Returns
    -------
    output : pycuda.compiler.SourceModule
       Compiled plugin CUDA kernel
       
    """

    rt_const = {}
    rt_const['int'] = ['detector_rows', 'detector_columns']
    rt_const['float'] = []
    rt_const['str'] = []

    cu_kernel_path = '%s/base.cu' % os.path.dirname(__file__)

    kernel_code = generate_gpu_init(conf, rt_const)

    kernel_code += '\n'
    kernel_code += \
        '/* --- BEGIN AUTOMATIC PLUGIN RUNTIME CONFIGURATION --- */\n'
    kernel_code += '\n'

    if air_ref_pixel_idx is not None:
        kernel_code += '#define plugin_rt_air_ref_pixel_idx %s\n' \
            % air_ref_pixel_idx

    if 'max_rows' in conf['app_state']['projs']:
        projection_rows = conf['app_state']['projs']['max_rows']
    else:
        projection_rows = conf['detector_rows']

    kernel_code += '#define plugin_rt_proj_size %s\n' \
        % (projection_rows * conf['detector_columns'])

    kernel_code += '\n'
    kernel_code += \
        '/* --- END AUTOMATIC PLUGIN RUNTIME CONFIGURATION --- */\n'
    kernel_code += '\n'

    kernel_code += load_kernel_source(cu_kernel_path)

    (_, kernels, _) = compile_kernels(conf, kernel_code)

    return kernels


def plugin_init(
    conf,
    zero_norm,
    air_norm,
    air_ref_pixel=None,
    dtype_norm='float32',
    ):
    """Plugin init function called once with full configuration upon plugin
    load. Called before any hooks, so it may be useful for global
    preparations of the plugin.
    Any values for use in subsequent hooks can be saved in the plugin-specific
    __plugin_state__ dictionary.

    Set up air and zero norm helper arrays

    Parameters
    ----------
    conf : dict
        A dictionary of configuration options.
    zero_norm : str
        Background intensity normalization 'zero_norm', value or file path
        If zero_norm='zero_norm' the zero norm matrix is extracted using
        get_plugin_var
    air_norm : str
        Pure air intensity normalization 'air_norm', value or file path.
        If air_norm='air_norm' the air norm matrix is extracted using
        get_plugin_var
    air_ref_pixel : str, optional
        Tuble of pixel posistion (y,x) in projection containing air value
    dtype_norm : str, optional
        Norm martrices dtype

    Raises
    ------
    ValueError
        If provided dtype_norm is not a valid data type,
        if provided zero_norm value is neither 'zero_norm', 
        a suitable projection file nor
        a single value compatible with dtype_norm
        if provided air_norm value is neither 'air_norm',
        a suitable projection file nor
        a single value compatible with dtype_norm,
        if zero norm is greater than air norm or
        if air_ref_pixel is set not a valid (y,x) index
    pycuda.driver.CompileError
        If CUDA kernel didn't compile 
    """

    # Transform dtype_norm string to dtype

    dtype_norm = allowed_data_types[dtype_norm]

    # Fill zero and air norm

    if zero_norm == 'zero_norm':
        zero_norm_matrix = get_plugin_var(conf, 'zero_norm')
    else:
        zero_norm_matrix = load_helper_proj(zero_norm, conf, dtype_norm)

    if air_norm == 'air_norm':
        air_norm_matrix = get_plugin_var(conf, 'air_norm')
    else:
        air_norm_matrix = load_helper_proj(air_norm, conf, dtype_norm)

    if air_ref_pixel is not None:

        # Create flat air reference pixel index

        air_ref_list = air_ref_pixel.split(',')
        air_ref_pixel_idx = int(air_ref_list[0].strip('(').strip()) \
            * conf['detector_columns'] + int(air_ref_list[1].strip(')'
                ).strip())

        # Generate tmp input data gpu array
        # Due to GPU out-of-order execution we need a
        # proj copy when using air_ref_pixel

        projs_data = get_npy_data(conf, 'projs_data')

        gpu_proj_ref_pixel_vals = gpuarray.zeros((projs_data.shape[0],
                1), projs_data.dtype)
    else:
        air_ref_pixel_idx = None
        gpu_proj_ref_pixel_vals = None

    # Check norm values

    check_norm(zero_norm_matrix, air_norm_matrix)

    # Convert zero_norm_matrix to fdt

    fdt = conf['data_type']

    # Convert zero_norm_matrix to fdt

    zero_norm_matrix = fdt(zero_norm_matrix)

    air_norm_matrix = fdt(air_norm_matrix - zero_norm_matrix)

    # If air_ref_pixel is defined log is posponed to GPU

    if air_ref_pixel == None:
        log(air_norm_matrix, air_norm_matrix)

    __plugin_state__['gpu_air_norm'] = gpuarray.to_gpu(air_norm_matrix)
    __plugin_state__['gpu_proj_count'] = gpuarray.zeros(1,
            dtype=allowed_data_types['uint32'])
    __plugin_state__['gpu_zero_norm'] = \
        gpuarray.to_gpu(zero_norm_matrix)
    __plugin_state__['gpu_layout'] = conf['app_state']['gpu']['layouts'
            ]['proj']

    __plugin_state__['air_ref_pixel_idx'] = air_ref_pixel_idx
    __plugin_state__['gpu_proj_ref_pixel_vals'] = \
        gpu_proj_ref_pixel_vals

    gpu_kernels = __make_gpu_kernels(conf, air_ref_pixel_idx)

    __plugin_state__['gpu_kernels'] = gpu_kernels

    if not __plugin_state__['gpu_kernels']:
        raise cuda.CompileError('no valid gpu compute kernels found!')


def plugin_exit(
    conf,
    zero_norm,
    air_norm,
    air_ref_pixel=None,
    dtype_norm='float32',
    ):
    """Plugin exit function called once with full configuration at the end of
    execution. Called after all hooks are finished, so it may be useful
    for global clean up after the plugin.
    Any memory allocations that need to be released can be handled here.

    Clean up after helper arrays.

    Parameters
    ----------
    conf : dict
        A dictionary of configuration options.
    zero_norm : str
        Dummy argument
    air_norm : str
        Dummy argument
    air_ref_pixel : str, optional
        Dummy argument
    dtype_norm : str, optional
        Dummy argument
    """

    __plugin_state__.clear()


def preprocess_input(
    gpu_input_data,
    input_meta,
    conf,
    zero_norm,
    air_norm,
    air_ref_pixel=None,
    dtype_norm='float32',
    ):
    """Convert measured intensity input values to attenuation values.

    Parameters
    ----------
    gpu_input_data : gpuarray
        array to process.
    input_meta : list of dict
        List of meta data dictionaries matching input_data.
    conf : dict
        A dictionary of configuration options.
    zero_norm : str
        Dummy argument
    air_norm : str
        Dummy argument
    air_ref_pixel : str, optional
        Dummy argument
    dtype_norm : str, optional
        Dummy argument
    Returns
    -------
    output : tuple of gpuarray and dict
        Returns a 2-tuple of the array of stacked projections and input_meta.
    """

    gpu_module = conf['gpu']['module']

    # Retrieve initialized variables

    gpu_proj_count = __plugin_state__['gpu_proj_count']
    gpu_proj_ref_pixel_vals = __plugin_state__['gpu_proj_ref_pixel_vals'
            ]
    gpu_zero_norm = __plugin_state__['gpu_zero_norm']
    gpu_air_norm = __plugin_state__['gpu_air_norm']
    gpu_layout = __plugin_state__['gpu_layout']
    gpu_kernels = __plugin_state__['gpu_kernels']
    air_ref_pixel_idx = __plugin_state__['air_ref_pixel_idx']

    # Raise error if input is not a numpy array

    if not hasattr(gpu_input_data, 'dtype'):
        raise ValueError('invalid flux_to_proj preprocess input array')

    gpu_module.memset_d32(gpu_proj_count.gpudata,
                          gpu_input_data.shape[0], 1)

    gpu_flux2proj = gpu_kernels.get_function('flux2proj')

    if air_ref_pixel_idx is None:
        gpu_flux2proj(
            gpu_input_data,
            gpu_proj_count,
            gpu_zero_norm,
            gpu_air_norm,
            block=gpu_layout[0],
            grid=gpu_layout[1],
            )
    else:

        # if 'air_ref_pixel_idx' we need to move projection reference pixels to the GPU

        gpu_pointer_from_array
        gpu_proj_ref_pixel_vals_ptr = \
            gpu_pointer_from_array(gpu_proj_ref_pixel_vals)
        projs_data = get_npy_data(conf, 'projs_data')
        for i in xrange(projs_data.shape[0]):
            gpu_offset = i * gpu_proj_ref_pixel_vals.dtype.itemsize
            gpu_module.memcpy_htod(int(gpu_proj_ref_pixel_vals_ptr
                                   + gpu_offset),
                                   projs_data[i].flat[air_ref_pixel_idx])

        gpu_flux2proj(
            gpu_input_data,
            gpu_proj_ref_pixel_vals,
            gpu_proj_count,
            gpu_zero_norm,
            gpu_air_norm,
            block=gpu_layout[0],
            grid=gpu_layout[1],
            )
    return (gpu_input_data, input_meta)


if __name__ == '__main__':
    print 'no unit tests!'
