#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# maskvolumefov - maskvolumefov plugin to mask volume FoV
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

"""Base maskvolumefov plugin functions to mask FoV to specified radius"""

import os

from cphct.npycore import zeros, sqrt
from cphct.npycore.misc import linear_coordinates
from cphct.plugins import get_plugin_var
from cphct.cu.core import gpuarray, get_gpu_layout, generate_gpu_init, \
    load_kernels_source, compile_kernels

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
    rt_const['int'] = ['x_voxels', 'y_voxels', 'chunk_size']
    rt_const['float'] = []
    rt_const['str'] = []

    cu_kernels_path = '%s/base.cu' % os.path.dirname(__file__)

    kernels_code = generate_gpu_init(conf, rt_const)
    kernels_code += load_kernels_source(cu_kernels_path)

    (_, kernels, _) = compile_kernels(conf, kernels_code)

    return kernels


def plugin_init(conf, fov_radius=None):
    """Plugin init function called once with full configuration upon plugin
    load. Called before any hooks, so it may be useful for global
    preparations of the plugin.
    Any values for use in subsequent hooks can be saved in the plugin-specific
    __plugin_state__ dictionary.

    Just check args in this case.

    Parameters
    ----------
    conf : dict
        A dictionary of configuration options.
    fov_radius : float, optional
        The included FoV radius in cm

    Raises
    ------
    pycuda.driver.CompileError
        If CUDA kernel didn't compile 
    """

    __plugin_state__['name'] = __name__

    gpu_module = conf['gpu']['module']
    fdt = conf['data_type']
    x_voxels = conf['x_voxels']
    y_voxels = conf['y_voxels']

    x_min = conf['x_min']
    x_max = conf['x_max']
    y_min = conf['y_min']
    y_max = conf['y_max']

    x_coords_2 = linear_coordinates(x_min, x_max, x_voxels, True, fdt) \
        ** 2
    y_coords_2 = linear_coordinates(y_min, y_max, y_voxels, True, fdt) \
        ** 2

    if fov_radius is None:
        fov_radius = max(abs(x_min), abs(x_max), abs(y_min), abs(y_max))
    fov_radius = fdt(fov_radius)

    npy_fov_mask = zeros((y_voxels, x_voxels), dtype=fdt)

    for y in xrange(y_voxels):
        fov = sqrt(y_coords_2[y] + x_coords_2)
        npy_fov_mask[y, fov <= fov_radius] = 1.0

    __plugin_state__['gpu_fov_mask'] = gpuarray.to_gpu(npy_fov_mask)
    __plugin_state__['gpu_layout'] = get_gpu_layout(conf['chunk_size'],
                                                    conf['y_voxels'],
                                                    conf['x_voxels'],
                                                    conf['gpu_target_threads'])
    __plugin_state__['gpu_kernels'] = __make_gpu_kernels(conf)

    if not __plugin_state__['gpu_kernels']:
        raise gpu_module.LogicError('no valid gpu compute kernels found!'
                                    )


def plugin_exit(conf, fov_radius=None):
    """Plugin exit function called once with full configuration at the end of
    execution. Called after all hooks are finished, so it may be useful
    for global clean up after the plugin.
    Any memory allocations that need to be released can be handled here.

    Nothing to do in this case.

    Parameters
    ----------
    conf : dict
        A dictionary of configuration options.
    fov_radius : int, optional
        The included FoV radius in cm
    """

    __plugin_state__.clear()


def postprocess_output(
    gpu_output_data,
    output_meta,
    conf,
    fov_radius=None,
    ):
    """
    Mask reconstructed data to specified FoV radius.

    Parameters
    ----------
    gpu_output_data : gpuarray
        array to process.
    output_meta : list of dict
        List of meta data dictionaries matching output_data.
    conf : dict
        A dictionary of configuration options.
    fov_radius : int, optional
        The included FoV radius in cm
    
    Returns
    -------
    output : tuple of gpuarray and dict
        Returns a tuple of the data array masked to specified FoV
        radius and meta list.
    """

    # Retrieve initialized variables

    gpu_fov_mask = __plugin_state__['gpu_fov_mask']
    gpu_layout = __plugin_state__['gpu_layout']
    gpu_kernels = __plugin_state__['gpu_kernels']

    # Raise error if output is not a gpu array

    if not hasattr(gpu_output_data, 'dtype'):
        raise ValueError('invalid maskvolumefov postprocess output array'
                         )

    gpu_mask_volume_fov = gpu_kernels.get_function('mask_volume_fov')

    gpu_mask_volume_fov(gpu_output_data, gpu_fov_mask,
                        block=gpu_layout[0], grid=gpu_layout[1])

    return (gpu_output_data, output_meta)


if __name__ == '__main__':
    print 'no unit tests!'

