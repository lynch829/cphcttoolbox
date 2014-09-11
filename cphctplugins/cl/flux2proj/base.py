#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# flux2proj - plugin to convert measured intensities to attenuation projections
# Copyright (C) 2012-2014  The Cph CT Toolbox Project lead by Brian Vinter
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
from cphct.cl import enqueue_nd_range_kernel
from cphct.cl.core import generate_gpu_init, load_kernels_source, \
    compile_kernels, gpu_alloc_from_array, gpuarray

# Internal plugin state for individual plugin instances

__plugin_state__ = {}


def __make_gpu_kernels(conf, air_ref_pixel):
    """Make the plugin GPU kernels based on *conf* and plugin kernel source
    
    Parameters
    ----------
    conf : dict
       A dictionary of configuration options.
    air_ref_pixel : tuple
        Tuble of pixel posistion (y,x) in projection containing air value
       
    Returns
    -------
    output : pyopencl.Program
       Compiled plugin OpenCL kernel
       
    """

    rt_const = {}
    rt_const['int'] = ['detector_columns']
    rt_const['float'] = []
    rt_const['str'] = []

    cl_kernels_path = '%s/base.cl' % os.path.dirname(__file__)

    kernels_code = generate_gpu_init(conf, rt_const)

    kernels_code += '\n'
    kernels_code += \
        '/* --- BEGIN AUTOMATIC PLUGIN RUNTIME CONFIGURATION --- */\n'
    kernels_code += '\n'

    if air_ref_pixel is not None:
        kernels_code += \
            '#define plugin_rt_air_ref_pixel_flat_idx ((int)%d)\n' \
            % (conf['detector_columns'] * air_ref_pixel[0]
               + air_ref_pixel[1])

    if 'max_rows' in conf['app_state']['projs']:
        projection_rows = conf['app_state']['projs']['max_rows']
    else:
        projection_rows = conf['detector_rows']

    kernels_code += '#define plugin_rt_proj_size %s\n' \
        % (projection_rows * conf['detector_columns'])

    kernels_code += '\n'
    kernels_code += \
        '/* --- END AUTOMATIC PLUGIN RUNTIME CONFIGURATION --- */\n'
    kernels_code += '\n'

    kernels_code += load_kernels_source(cl_kernels_path)

    (_, kernels, _) = compile_kernels(conf, kernels_code)

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
        if air_ref_pixel is set and chunk 0 is not in chunks_enabled or
        if air_ref_pixel is set but not a valid (y,x) index or 
        if air_ref_pixel is outside the first projection bounding box

    pyopencl.LogicError
        If OpenCL kernel didn't compile 
    """

    gpu_module = conf['gpu']['module']
    gpu_queue = conf['gpu']['queue']

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

        air_ref_pixel = (int(air_ref_list[0].strip('(').strip()),
                         int(air_ref_list[1].strip(')').strip()))

        # Check if air_ref_pixel is within the first detector bounding box

        detector_boundingboxes = get_npy_data(conf,
                'detector_boundingboxes')

        # NOTE: Right now we only support air ref pixels
        # that is inside the first projection bounding box

        if not 0 in conf['chunks_enabled']:
            msg_ln1 = 'air_ref_pixel support require chunk: 0'
            raise ValueError('%s\n%s' % msg_ln1)

        if detector_boundingboxes is not None and (air_ref_pixel[0]
                < detector_boundingboxes[0, 0, 0] or air_ref_pixel[0]
                > detector_boundingboxes[0, 0, 1] or air_ref_pixel[1]
                < detector_boundingboxes[0, 1, 0] or air_ref_pixel[1]
                > detector_boundingboxes[0, 1, 1]):

            msg_ln1 = 'air_ref_pixel: %s must be inside' \
                % str(air_ref_pixel)
            msg_ln2 = 'first projection chunk: (%s, %s)' \
                % (str(detector_boundingboxes[0, 0, 1]),
                   str(detector_boundingboxes[0, 1, 1]))
            raise ValueError('%s\n%s' % (msg_ln1, msg_ln2))

        # Allocate GPU memory for air ref pixels

        projs_data = get_npy_data(conf, 'projs_data')

        gpu_proj_ref_pixel_vals = gpuarray.zeros(gpu_queue,
                conf['projs_per_turn'], projs_data.dtype)
    else:
        air_ref_pixel = None
        gpu_proj_ref_pixel_vals = None

    # Check norm values

    check_norm(zero_norm_matrix, air_norm_matrix)

    # Convert zero_norm_matrix to fdt

    fdt = conf['data_type']

    # Convert zero_norm_matrix to fdt

    zero_norm_matrix = fdt(zero_norm_matrix)

    air_norm_matrix = fdt(air_norm_matrix - zero_norm_matrix)

    # If air_ref_pixel is defined log is posponed to GPU

    if air_ref_pixel is None:
        log(air_norm_matrix, air_norm_matrix)

    gpu_kernels = __make_gpu_kernels(conf, air_ref_pixel)
    gpu_flux2proj = gpu_kernels.flux2proj

    # Set kernel arguments

    gpu_air_norm = gpuarray.to_device(gpu_queue, air_norm_matrix)
    gpu_zero_norm = gpuarray.to_device(gpu_queue, zero_norm_matrix)

    gpu_flux2proj.set_arg(1, gpu_alloc_from_array(gpu_zero_norm))
    gpu_flux2proj.set_arg(2, gpu_alloc_from_array(gpu_air_norm))

    if air_ref_pixel is not None:
        gpu_flux2proj.set_arg(5,
                              gpu_alloc_from_array(gpu_proj_ref_pixel_vals))

    __plugin_state__['gpu_flux2proj'] = gpu_flux2proj

    __plugin_state__['gpu_layout'] = conf['app_state']['gpu']['layouts'
            ]['proj']

    __plugin_state__['air_ref_pixel'] = air_ref_pixel
    __plugin_state__['gpu_proj_ref_pixel_vals'] = \
        gpu_proj_ref_pixel_vals

    # NOTE:
    # We need to keep a handle to gpu_{zero|air}_norm otherwise
    # they are garbage collected even though set as gpu_flux2proj
    # arguments.

    __plugin_state__['gpu_zero_norm'] = gpu_zero_norm
    __plugin_state__['gpu_air_norm'] = gpu_air_norm


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
    gpu_queue = conf['gpu']['queue']

    # Retrieve initialized variables

    gpu_proj_ref_pixel_vals = __plugin_state__['gpu_proj_ref_pixel_vals'
            ]
    gpu_layout = __plugin_state__['gpu_layout']
    gpu_flux2proj = __plugin_state__['gpu_flux2proj']
    air_ref_pixel = __plugin_state__['air_ref_pixel']

    # Raise error if input is not a numpy array

    if not hasattr(gpu_input_data, 'dtype'):
        raise ValueError('invalid flux_to_proj preprocess input array')

    first_proj = conf['app_state']['projs']['first']
    last_proj = conf['app_state']['projs']['last']

    if air_ref_pixel is not None:

        # if 'air_ref_pixel'
        # we need to copy projection reference pixels on the GPU
        # because out-of-order execution prevents us from using
        # the ref pixels in-line
        #
        # NOTE: For now the air_ref_pixel
        #       must be within the first projection chunk

        nr_projs = last_proj - first_proj + 1

        if not 'boundingbox' in conf['app_state']['projs'] \
            or 'boundingbox' in conf['app_state']['projs'] \
            and (get_npy_data(conf, 'detector_boundingboxes')[0]
                 == conf['app_state']['projs']['boundingbox']).all():

            for i in xrange(nr_projs):
                ref_pixel_idx = first_proj + i

                gpu_proj_ref_pixel_vals[ref_pixel_idx] = \
                    gpu_input_data[i, air_ref_pixel[0],
                                   air_ref_pixel[1]]

    # Set kernel arguments and call GPU flux2proj kernel

    gpu_flux2proj.set_arg(0, gpu_alloc_from_array(gpu_input_data))
    gpu_flux2proj.set_arg(3, allowed_data_types['uint32'](first_proj))
    gpu_flux2proj.set_arg(4, allowed_data_types['uint32'](last_proj))

    __plugin_state__['gpu_flux2proj_event'] = \
        enqueue_nd_range_kernel(gpu_queue, gpu_flux2proj,
                                gpu_layout[1], gpu_layout[0])

    return (gpu_input_data, input_meta)


if __name__ == '__main__':
    print 'no unit tests!'
