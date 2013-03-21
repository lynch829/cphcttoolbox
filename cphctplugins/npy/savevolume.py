#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# savevolume - plugin to save/append data values to a binary file
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

"""Save volume plugin to save 3D volumes in binary format"""

from cphct.io import create_path_dir, expand_path
from cphct.npycore import memmap

# Internal plugin state for individual plugin instances

__plugin_state__ = {}


def plugin_init(conf, save_path, flush=True):
    """Plugin init function called once with full configuration upon plugin
    load. Called before any hooks, so it may be useful for global
    preparations of the plugin.
    Any values for use in subsequent hooks can be saved in the plugin-specific
    __plugin_state__ dictionary.

    Open a memory mapped volume output file descriptor.

    Parameters
    ----------
    conf : dict
        A dictionary of configuration options.
    save_path : str
        Path where volume is saved
    flush : bool, optional
        If True data is flushed to disk after each write
        
    Raises
    ------
    ValueError
        If flush is not compatible with boolean
    """

    abs_save_path = expand_path(conf['working_directory'], save_path)
    __plugin_state__['abs_save_path'] = abs_save_path

    # Step and Shoot algos reconstruct a FoV in each step

    if conf['scanning_path'] == 'step':
        full_scan_chunk_count = conf['total_turns'] * conf['chunk_count'
                ]
        full_z_voxels = conf['total_turns'] * conf['z_voxels']
    else:
        full_scan_chunk_count = conf['chunk_count']
        full_z_voxels = conf['z_voxels']

    __plugin_state__['full_scan_chunk_count'] = full_scan_chunk_count
    __plugin_state__['full_z_voxels'] = full_z_voxels

    # Define shape of memory map ndarray

    map_shape = (full_z_voxels, conf['y_voxels'], conf['x_voxels'])

    # Define output data type

    odt = conf['output_data_type']

    # Create save_path

    create_path_dir(abs_save_path)

    # Create memory mapping

    volume_file = open(abs_save_path, mode='w+b')

    __plugin_state__['volume_file'] = volume_file
    __plugin_state__['memmap_data'] = memmap(volume_file, dtype=odt,
            mode='w+', shape=map_shape)
    __plugin_state__['flush'] = bool(flush)


def plugin_exit(conf, save_path, flush=True):
    """Plugin exit function called once with full configuration at the end of
    execution. Called after all hooks are finished, so it may be useful
    for global clean up after the plugin.
    Any memory allocations that need to be released can be handled here.

    Closes volume file descriptor.

    Parameters
    ----------
    conf : dict
        A dictionary of configuration options.
    save_path : str
        Dummy argument
    flush : bool, optional
        Dummy argument
    """

    __plugin_state__['volume_file'].close()
    __plugin_state__.clear()


def save_output(
    output_data,
    output_meta,
    conf,
    save_path,
    flush=True,
    ):
    """Save output volume to *save_path*

    Parameters
    ----------
    output_data : ndarray
        array to process.
    output_meta : list of dict
        List of meta data dictionaries matching output_data.
    conf : dict
        A dictionary of configuration options.
    save_path : str
        Dummy argument
    flush : bool, optional
        Dummy argument
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
        raise ValueError('invalid savevolume save input array')

    full_z_voxels = __plugin_state__['full_z_voxels']

    chunk_size = conf['chunk_size']
    chunk_index = conf['app_state']['chunk']['idx']

    full_scan_chunk_count = __plugin_state__['full_scan_chunk_count']
    memmap_data = __plugin_state__['memmap_data']
    flush = __plugin_state__['flush']

    # Note: we leave disabled chunks blank but still write them

    if not chunk_index in conf['chunks_enabled']:
        return (output_data, output_meta)

    # Step and Shoot algos reconstruct a FoV in each step

    if conf['scanning_path'] == 'step':
        full_scan_chunk_idx = chunk_index + conf['app_state'
                ]['scanner_step'] * conf['chunk_count']
    else:
        full_scan_chunk_idx = chunk_index

    z_offset = full_scan_chunk_idx * chunk_size

    memmap_data[z_offset:z_offset + chunk_size, :, :] = output_data[:
            chunk_size, :, :]

    if flush == True:
        memmap_data.flush()

    return (output_data, output_meta)


if __name__ == '__main__':
    print 'No test available'
