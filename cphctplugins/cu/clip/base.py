#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# clip - clip plugin to truncate data values outside a given range
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


"""Base clip plugin to truncate input and output data values outside a range"""

import os

from cphct.plugins import get_plugin_var
from cphct.cu.core import get_gpu_layout, generate_gpu_init, \
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


def plugin_init(conf, clip_min, clip_max):
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
    clip_min : float
        The minimum value to truncate input array to.
    clip_max : float
        The maximum value to truncate input array to.

    Raises
    ------
    ValueError
        If provided clip_min or clip_max is not a valid floating point number
        or if clip_min is greater than clip_max.
    pycuda.driver.CompileError
        If CUDA kernel didn't compile  
    """

    __plugin_state__['name'] = __name__

    gpu_module = conf['gpu']['module']
    fdt = conf['data_type']

    min_val = fdt(clip_min)
    max_val = fdt(clip_max)
    if min_val > max_val:
        raise ValueError('clip_min is greater than clip_max')

    __plugin_state__['clip_min'] = min_val
    __plugin_state__['clip_max'] = max_val
    __plugin_state__['gpu_layout'] = get_gpu_layout(conf['chunk_size'],
                                                    conf['y_voxels'],
                                                    conf['x_voxels'],
                                                    conf['gpu_target_threads'])
    __plugin_state__['gpu_kernels'] = __make_gpu_kernels(conf)

    if not __plugin_state__['gpu_kernels']:
        raise gpu_module.LogicError('no valid gpu compute kernels found!')
        

def plugin_exit(conf, clip_min, clip_max):
    """Plugin exit function called once with full configuration at the end of
    execution. Called after all hooks are finished, so it may be useful
    for global clean up after the plugin.
    Any memory allocations that need to be released can be handled here.

    Nothing to do in this case.

    Parameters
    ----------
    conf : dict
        A dictionary of configuration options.
    clip_min : float
        The minimum value to truncate input array to.
    clip_max : float
        The maximum value to truncate input array to.
    """

    __plugin_state__.clear()


def postprocess_output(
    gpu_output_data,
    output_meta,
    conf,
    clip_min, 
    clip_max,
    ):
    """Clip output using args for range information.

    Parameters
    ----------
    gpu_output_data : gpuarray
        array to process.
    output_meta : list of dict
        List of meta data dictionaries matching output_data.
    conf : dict
        A dictionary of configuration options.
    clip_min : float
        The minimum value to truncate input array to.
    clip_max : float
        The maximum value to truncate input array to.
    
    Returns
    -------
    output : tuple of gpuarray and dict
        Returns a tuple of the data array limited to [clip_min:clip_max]
        range and meta list.
    """

    # Retrieve initialized variables

    clip_min = __plugin_state__['clip_min']
    clip_max = __plugin_state__['clip_max']
    gpu_layout = __plugin_state__['gpu_layout']
    gpu_kernels = __plugin_state__['gpu_kernels']

    # Raise error if output is not a gpu array

    if not hasattr(gpu_output_data, 'dtype'):
        raise ValueError('invalid clip postprocess input array')

    gpu_clip = gpu_kernels.get_function('clip')

    gpu_clip(gpu_output_data, clip_min, clip_max,
             block=gpu_layout[0], grid=gpu_layout[1])

    return (gpu_output_data, output_meta)


if __name__ == '__main__':
    print 'no unit tests!'

