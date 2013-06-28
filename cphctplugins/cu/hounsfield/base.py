#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# hounsfield - hounsfield plugin to scale voxel data to hounsfield units (HU)
# Copyright (C) 2012-2013  The Cph CT Toolbox Project lead by Brian Vinter
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

"""hounsfield plugin to scale voxel data to hounsfield units (HU)"""

import os

from cphct.npycore import zeros
from cphct.plugins import get_plugin_var
from cphct.cu.core import gpuarray, get_gpu_layout, generate_gpu_init, \
    load_kernel_source, compile_kernels, cuda

# Internal plugin state for individual plugin instances

__plugin_state__ = {}


def __make_gpu_kernels(conf):
    """Make the plugin GPU kernels based on *conf* and plugin kernel source
    
    Parameters
    ----------
    conf : dict
       A dictionary of configuration options.
        
    Returns
    -------
    output : pycuda.compiler.SourceModule
       Compiled plugin CUDA kernel
       
    """

    rt_const = {}
    rt_const['int'] = ['x_voxels', 'y_voxels', 'z_voxels', 'chunk_size']
    rt_const['float'] = []
    rt_const['str'] = []

    cu_kernel_path = '%s/base.cu' % os.path.dirname(__file__)

    kernel_code = generate_gpu_init(conf, rt_const)
    kernel_code += load_kernel_source(cu_kernel_path)

    (_, kernels, _) = compile_kernels(conf, kernel_code)

    return kernels


def plugin_init(conf, raw_voxel_water):
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
    raw_voxel_water : float
        The raw voxel value of distilled water

    Raises
    ------
    ValueError
        If provided raw_voxel_water is neither 'raw_voxel_water' 
        nor a valid floating point number.
    pycuda.driver.CompileError
        If CUDA kernel didn't compile 
    """

    fdt = conf['data_type']
    npy_raw_voxel_water = zeros(1, dtype=fdt)
    if raw_voxel_water == 'raw_voxel_water':
        npy_raw_voxel_water[:] = get_plugin_var(conf, 'raw_voxel_water')
    else:
        npy_raw_voxel_water[:] = float(raw_voxel_water)

    __plugin_state__['gpu_raw_voxel_water'] = \
        gpuarray.to_gpu(npy_raw_voxel_water)
    __plugin_state__['gpu_layout'] = get_gpu_layout(conf['y_voxels'],
            conf['x_voxels'], conf['gpu_target_threads'])
    __plugin_state__['gpu_kernels'] = __make_gpu_kernels(conf)
    if not __plugin_state__['gpu_kernels']:
        raise cuda.CompileError('no valid gpu compute kernels found!')


def plugin_exit(conf, raw_voxel_water):
    """Plugin exit function called once with full configuration at the end of
    execution. Called after all hooks are finished, so it may be useful
    for global clean up after the plugin.
    Any memory allocations that need to be released can be handled here.

    Clean up after helper arrays.

    Parameters
    ----------
    conf : dict
        A dictionary of configuration options.
    raw_voxel_water : float
        The raw voxel value of distilled water
    """

    __plugin_state__.clear()


def postprocess_output(
    gpu_output_data,
    output_meta,
    conf,
    raw_voxel_water,
    ):
    """Convert reconstructed data to the hounsfield scale 
    based on the raw voxel value of distilled water.

    Parameters
    ----------
    gpu_output_data : ndarray
        array to process.
    output_meta : list of dict
        List of meta data dictionaries matching *gpu_output_data*.
    conf : dict
        A dictionary of configuration options.
    raw_voxel_water : float
        The raw voxel value of distilled water

    Returns
    -------
    output : tuple of gpuarray and dict
        Returns a tuple of the data array scaled to hounsfield units
        and meta list.
    """

    gpu_module = conf['gpu']['module']

    # Retrieve initialized variables

    gpu_raw_voxel_water = __plugin_state__['gpu_raw_voxel_water']
    gpu_layout = __plugin_state__['gpu_layout']
    gpu_kernels = __plugin_state__['gpu_kernels']

    # Raise error if output is not a gpu array

    if not hasattr(gpu_output_data, 'dtype'):
        raise ValueError('invalid hounsfield postprocess output array')

    gpu_hounsfield_scale = gpu_kernels.get_function('hounsfield_scale')

    gpu_hounsfield_scale(gpu_output_data, gpu_raw_voxel_water,
                         block=gpu_layout[0], grid=gpu_layout[1])

    return (gpu_output_data, output_meta)


if __name__ == '__main__':
    print 'no unit tests!'
