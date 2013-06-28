#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# savestacked - plugin to save/append data values to a file
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

"""Save stacked plugin to save 3D volumes in stacks of 2D slices"""

from cphct.io import create_path_dir, expand_path, temporary_file
from cphct.npycore import memmap
from cphct.npycore.io import save_auto

# Internal plugin state for individual plugin instances

__plugin_state__ = {}


def plugin_init(
    conf,
    dim,
    save_path,
    stack_size=None,
    ):
    """Plugin init function called once with full configuration upon plugin
    load. Called before any hooks, so it may be useful for global
    preparations of the plugin.
    Any values for use in subsequent hooks can be saved in the plugin-specific
    __plugin_state__ dictionary.

    Calculate helpers and prepare memory mapped temp file for volume.

    Parameters
    ----------
    conf : dict
        A dictionary of configuration optionS.
    dim : str
        Dimension to slice ('x', 'y' or 'z'):
        x: 2D slices with z dim as rows and y dim as columns
        y: 2D slices with z dim as rows and x dim as columns
        z: 2D slices with y dim as rows and x dim as columns
    save_path : str
        Path where stacked slices are saved. Must contain a substition
        argument for index replacement e.g. stack.%.4d.png
    stack_size : str or None
        Optional max width of saved stack

    Raises
    ------
    ValueError
        If provided dim is not a valid dimension name (x, y or z).
    """

    if not dim in ('x', 'y', 'z'):
        raise ValueError('invalid savestacked dim value: %s' % dim)

    abs_save_path = expand_path(conf['working_directory'], save_path)
    __plugin_state__['abs_save_path'] = abs_save_path
    __plugin_state__['slice_dim'] = dim

    # Step and Shoot algos reconstruct a FoV in each step

    if conf['scanning_path'] == 'step':
        full_scan_chunk_count = conf['total_turns'] * conf['chunk_count']
        full_z_voxels = conf['total_turns'] * conf['z_voxels']
    else:
        full_scan_chunk_count = conf['chunk_count']
        full_z_voxels = conf['z_voxels']

    __plugin_state__['full_scan_chunk_count'] = full_scan_chunk_count
    __plugin_state__['full_z_voxels'] = full_z_voxels

    # Define shape of memory map ndarray

    if dim == 'x':
        map_shape = (full_z_voxels, conf['x_voxels'] * conf['y_voxels'])
    elif dim == 'y':
        map_shape = (full_z_voxels, conf['y_voxels'] * conf['x_voxels'])
    elif dim == 'z':
        map_shape = (conf['y_voxels'], conf['x_voxels'] * full_z_voxels)

    # Define output data type

    odt = conf['output_data_type']

    # Create save_path

    create_path_dir(abs_save_path)

    # Create memory mapping

    tmp_file = temporary_file(conf, mode='r+b')

    __plugin_state__['tmp_file'] = tmp_file
    __plugin_state__['memmap_data'] = memmap(tmp_file, dtype=odt,
            mode='r+', shape=map_shape)

    # Initialize min/max values for image contrast scaling

    __plugin_state__['min_value'] = odt(1 << odt(0.0).itemsize * 8)
    __plugin_state__['max_value'] = 0.0


def plugin_exit(
    conf,
    dim,
    save_path,
    stack_size=None,
    ):
    """Plugin exit function called once with full configuration at the end of
    execution. Called after all hooks are finished, so it may be useful
    for global clean up after the plugin.
    Any memory allocations that need to be released can be handled here.

    Closes temp file descriptors for dim='x' and dim='y'

    Parameters
    ----------
    conf : dict
        A dictionary of configuration optionS.
    dim : str
        Dimension to slice ('x', 'y' or 'z'):
        x: 2D slices with z dim as rows and y dim as columns
        y: 2D slices with z dim as rows and x dim as columns
        z: 2D slices with y dim as rows and x dim as columns
    save_path : str
        Path where stacked slices are saved. Must contain a substition
        argument for index replacement e.g. stack.%.4d.png
    stack_size : str or None
        Optional max width of saved stack
    """

    __plugin_state__['tmp_file'].close()
    __plugin_state__.clear()

def __eager_write(conf):
    """Write output parts as soon as they are ready. Reads app progress from
    conf and compares it with __plugin_state__ to decide when to write.

    Parameters
    ----------
    conf : dict
        A dictionary of configuration options.

    Returns
    -------
    output : list
        Returns a list of path names written.
    """

    written = []

    # We should write when ready but we wait until last enabled chunk for now

    if conf['app_state']['chunk']['idx'] != conf["chunk_range"][-1]:
        return written

    # Create slices based on memmap_data

    dynamic_range = (__plugin_state__['min_value'],
                     __plugin_state__['max_value'])
    abs_save_path = __plugin_state__['abs_save_path']
    memmap_data = __plugin_state__['memmap_data']
    stack_steps = __plugin_state__['stack_steps']
    stack_size = __plugin_state__['stack_size']
    stack_range = __plugin_state__['stack_range']
    stack_start = 0
    stack_end = stack_size
    for _ in xrange(0, stack_range, stack_size):
        col_start = stack_start * stack_steps
        col_end = stack_end * stack_steps
        if col_end > memmap_data.shape[1]:
            col_end = memmap_data.shape[1]
            stack_end = col_end / stack_steps

        save_data = memmap_data[:, col_start:col_end]
        stack_save_path = abs_save_path % (stack_start, stack_end - 1)
        save_auto(stack_save_path, save_data, dynamic_range)
        written.append(stack_save_path)

        stack_start = stack_end
        stack_end += stack_size
    return written

def __shared_save_output(
    output_data,
    output_meta,
    conf,
    dim,
    save_path,
    stack_size=None,
    ):
    """Save stacked output slices to *save_path*

    Parameters
    ----------
    output_data : ndarray
        array to process.
    output_meta : list of dict
        List of meta data dictionaries matching output_data.
    conf : dict
        A dictionary of configuration options.
    dim : str
        Dimension to slice ('x', 'y' or 'z'):
        x: 2D slices with z dim as rows and y dim as columns
        y: 2D slices with z dim as rows and x dim as columns
        z: 2D slices with y dim as rows and x dim as columns 
    save_path : str
        Path where stacked slices are saved. Must contain a substition
        argument for index replacement e.g. stack.%.4d.png
    stack_size : str or None
        Optional max width of saved stack

    Returns
    -------
    output : tuple of ndarray and list
        Returns a tuple of the same data array and meta list.

    Raises
    ------
    ValueError
        If *output_data* is not a numpy array
    """

    if not hasattr(output_data, 'dtype'):
        raise ValueError('invalid savestacked plugin input array')

    x_voxels = conf['x_voxels']
    y_voxels = conf['y_voxels']

    chunk_size = conf['chunk_size']
    chunk_index = conf['app_state']['chunk']['idx']

    full_scan_chunk_count = __plugin_state__['full_scan_chunk_count']
    memmap_data = __plugin_state__['memmap_data']

    # Note: we leave disabled chunks blank but still write them
    
    if not chunk_index in conf['chunks_enabled']:
        return (output_data, output_meta)

    # Step and Shoot algos reconstruct a FoV in each step

    if conf['scanning_path'] == 'step':
        full_scan_chunk_idx = chunk_index + conf['app_state']['scanner_step'] \
                              * conf['chunk_count']
    else:
        full_scan_chunk_idx = chunk_index

    # Record minimum value for image contrast scaling

    min_value = output_data.min()
    if min_value < __plugin_state__['min_value']:
        __plugin_state__['min_value'] = min_value

    # Record maximum value for image contrast scaling

    max_value = output_data.max()
    if max_value > __plugin_state__['max_value']:
        __plugin_state__['max_value'] = max_value

    z_offset = full_scan_chunk_idx * chunk_size

    if dim == 'x':
        stack_steps = y_voxels
        row_start = z_offset
        row_end = row_start + chunk_size
        for index in xrange(x_voxels):
            col_start = y_voxels * index
            col_end = col_start + y_voxels
            memmap_data[row_start:row_end, col_start:col_end] = \
                output_data[:, :, index]
    elif dim == 'y':
        stack_steps = x_voxels
        row_start = z_offset
        row_end = row_start + chunk_size

        for index in xrange(y_voxels):
            col_start = x_voxels * index
            col_end = col_start + x_voxels
            memmap_data[row_start:row_end, col_start:col_end] = \
                output_data[:, index, :]
    elif dim == 'z':
        stack_steps = x_voxels
        for index in xrange(chunk_size):
            col_start = z_offset * x_voxels + x_voxels * index
            col_end = col_start + x_voxels
            memmap_data[:, col_start:col_end] = output_data[index, :, :]

    stack_range = memmap_data.shape[1] / stack_steps
    __plugin_state__['stack_steps'] = stack_steps
    __plugin_state__['stack_range'] = stack_range
    if stack_size:
        __plugin_state__['stack_size'] = int(stack_size)
    else:
        __plugin_state__['stack_size'] = stack_range
    __eager_write(conf)

    return (output_data, output_meta)


# Map postprocess and save to shared function

postprocess_output = __shared_save_output
save_output = __shared_save_output


if __name__ == '__main__':
    print 'No test available'
