#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# square - plugin to square all data values in array
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

"""Square plugin to square all input and output data values"""

from cphct.cu.core import gpu_init_mod, gpu_init_ctx, gpu_exit
from cphct.cu import gpuarray
from cphct.cu.utils import square_array
from cphct.npycore import float32, float64, uint16, uint32, uint64, \
    int16, int32, int64

# Internal plugin state for individual plugin instances

__plugin_state__ = {}


def plugin_init(conf):
    """Plugin init function called once with full configuration upon plugin
    load. Called before any hooks, so it may be useful for global
    preparations of the plugin.
    Any values for use in subsequent hooks can be saved in the plugin-specific
    __plugin_state__ dictionary.

    Nothing to do in this case.

    Parameters
    ----------
    conf : dict
        A dictionary of configuration options.
    """


def plugin_exit(conf):
    """Plugin exit function called once with full configuration at the end of
    execution. Called after all hooks are finished, so it may be useful
    for global clean up after the plugin.
    Any memory allocations that need to be released can be handled here.

    Nothing to do in this case.

    Parameters
    ----------
    conf : dict
        A dictionary of configuration options.
    """

    __plugin_state__.clear()


def preprocess_input(gpu_input_data, input_meta, conf):
    """Square all input values

    Parameters
    ----------
    gpu_input_data : gpuarray
        array to process.
    input_meta : list of dict
        List of meta data dictionaries matching input_data.
    conf : dict
        A dictionary of configuration options.

    Returns
    -------
    output : tuple of gpuarray and list
        Returns a tuple of the data array squared and meta list.
    """

    # Raise error if input is not a gpu array

    if not hasattr(gpu_input_data, 'dtype'):
        raise ValueError('invalid square preprocess input array')
    return (square_array(gpu_input_data, out=gpu_input_data),
            input_meta)


def postprocess_output(gpu_output_data, output_meta, conf):
    """Square all output values

    Parameters
    ----------
    gpu_output_data : gpuarray
        array to process.
    output_meta : list of dict
        List of meta data dictionaries matching output_data.
    conf : dict
        A dictionary of configuration options.

    Returns
    -------
    output : tuple of gpuarray and dict
        Returns a tuple of the data array squared and meta list.
    """

    # Raise error if input is not a gpu array

    if not hasattr(gpu_output_data, 'dtype'):
        raise ValueError('invalid square postprocess output array')

    return (square_array(gpu_output_data, out=gpu_output_data),
            output_meta)


if __name__ == '__main__':
    conf = {}
    gpu_init_mod(conf)
    gpu_module = conf['gpu']['module']
    gpu_count = gpu_module.Device.count()
    if gpu_count < 1:
        raise Exception('No GPUs available!')
    conf['cuda_device_index'] = 0
    gpu_init_ctx(conf)

    arange = gpuarray.arange
    for data_type in [
        float32,
        float64,
        uint16,
        uint32,
        uint64,
        int16,
        int32,
        int64,
        ]:
        data = arange(3, 8, dtype=data_type)
        print 'Square data %s (%s)' % (data, data.dtype.name)
        out = square_array(data)
        print 'Squared to %s (%s)' % (out, out.dtype.name)
        data = arange(3, 8, dtype=data_type)
        out = square_array(data)
        print 'Squared to byte range %s (%s)' % (out, out.dtype.name)
        data = arange(-3, 7, dtype=data_type)
        print 'Square data %s (%s)' % (data, data.dtype.name)
        out = square_array(data)
        print 'Squared to %s (%s)' % (out, out.dtype.name)
        data = arange(-3, 7, dtype=data_type)
        out = square_array(data)
        print 'Squared to byte range %s (%s)' % (out, out.dtype.name)
    gpu_exit(conf)
