#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# savesinogramrows - plugin to save input projections as individual sinograms
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

"""Save input projections as sinograms one file per row"""

from cphct.io import create_path_dir, expand_path, temporary_file
from cphct.npycore import memmap
from cphct.npycore.io import save_auto

# Internal plugin state for individual plugin instances

__plugin_state__ = {}


def plugin_init(conf, save_path):
    """Plugin init function called once with full configuration upon plugin
    load. Called before any hooks, so it may be useful for global
    preparations of the plugin.
    Any values for use in subsequent hooks can be saved in the plugin-specific
    __plugin_state__ dictionary.

    Open a memory mapped singram tmpfile descriptor.

    Parameters
    ----------
    conf : dict
        A dictionary of configuration options.
    save_path : str
        Path where volume is saved
    """

    # Create save_path

    abs_save_path = expand_path(conf['working_directory'], save_path)
    __plugin_state__['abs_save_path'] = abs_save_path

    # Define shape of memory map ndarray

    total_projs = conf['total_turns'] * conf['projs_per_turn']
    __plugin_state__['total_projs'] = total_projs
    map_shape = (conf['detector_rows'], conf['detector_columns'],
                 total_projs)

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


def plugin_exit(conf, save_path):
    """Plugin exit function called once with full configuration at the end of
    execution. Called after all hooks are finished, so it may be useful
    for global clean up after the plugin.
    Any memory allocations that needb to be released can be handled here.

    Clean up after helper arrays.

    Parameters
    ----------
    conf : dict
        Dummy argument
    save_path : str
        Dummy argument
    """

    __plugin_state__['tmp_file'].close()
    __plugin_state__.clear()


def preprocess_input(
    gpu_input_data,
    input_meta,
    conf,
    save_path,
    ):
    """Save measured intensity input in individual sinogram layout.

    Parameters
    ----------
    gpu_input_data : gpuarray
        array to process.
    input_meta : list of dict
        List of meta data dictionaries matching gpu_input_data.
    conf : dict
        A dictionary of configuration options.
    save_path : str
        Dummy argument

    Returns
    -------
    output : tuple of ndarray and dict
        Returns a 2-tuple of gpu_input_data and input_meta.
    """

    total_projs = __plugin_state__['total_projs']
    memmap_data = __plugin_state__['memmap_data']
    abs_save_path = __plugin_state__['abs_save_path']
    min_value = __plugin_state__['min_value']
    max_value = __plugin_state__['max_value']
    detector_rows = conf['detector_rows']

    # Write projection data to tmpfile

    first_proj = conf['app_state']['projs']['first']
    last_proj = conf['app_state']['projs']['last']

    start_row = 0
    end_row = detector_rows
    if 'boundingbox' in conf['app_state']['projs']:
        start_row = conf['app_state']['projs']['boundingbox'][0, 0]
        end_row = conf['app_state']['projs']['boundingbox'][0, 1]
    sinogram_row_count = end_row - start_row

    input_data = gpu_input_data.get()

    for proj_idx in xrange(first_proj, last_proj + 1):
        proj_data = input_data[proj_idx - first_proj, :sinogram_row_count]

        proj_min_value = proj_data.min()
        if proj_min_value < min_value:
            min_value = proj_min_value

        proj_max_value = proj_data.max()
        if proj_max_value > max_value:
            max_value = proj_max_value

        memmap_data[start_row:end_row, :, proj_idx] = proj_data

    # If last projection save images

    if last_proj == total_projs - 1:
        dynamic_range = (min_value, max_value)
        for row in xrange(detector_rows):
            sinogram_save_path = abs_save_path % row
            save_auto(sinogram_save_path, memmap_data[row],
                      dynamic_range)

    __plugin_state__['min_value'] = min_value
    __plugin_state__['max_value'] = max_value

    return (gpu_input_data, input_meta)


if __name__ == '__main__':
    print 'no unit tests!'
