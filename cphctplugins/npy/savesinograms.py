#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# savesinogramrows - plugin to save input projections as sinograms
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

from cphct.io import create_path_dir, expand_path
from cphct.npycore import memmap

# Internal plugin state for individual plugin instances

__plugin_state__ = {}


def plugin_init(conf, save_path, flush=False):
    """Plugin init function called once with full configuration upon plugin
    load. Called before any hooks, so it may be useful for global
    preparations of the plugin.
    Any values for use in subsequent hooks can be saved in the plugin-specific
    __plugin_state__ dictionary.

    Open a memory mapped singram file descriptor.

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

    # Create save_path

    abs_save_path = expand_path(conf['working_directory'], save_path)
    __plugin_state__['abs_save_path'] = abs_save_path

    # Define shape of memory map ndarray

    total_projs = conf['total_turns'] * conf['projs_per_turn']
    map_shape = (conf['detector_rows'], conf['detector_columns'],
                 total_projs)

    # Define output data type

    odt = conf['output_data_type']

    # Create save_path

    create_path_dir(abs_save_path)

    # Create memory mapping

    sinogram_file = open(abs_save_path, mode='w+b')

    __plugin_state__['sinogram_file'] = sinogram_file
    __plugin_state__['memmap_data'] = memmap(sinogram_file, dtype=odt,
            mode='w+', shape=map_shape)
    __plugin_state__['flush'] = bool(flush)


def plugin_exit(conf, save_path, flush=False):
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
    flush : bool, optional
        Dummy argument
    """

    __plugin_state__['sinogram_file'].close()
    __plugin_state__.clear()


def preprocess_input(
    input_data,
    input_meta,
    conf,
    save_path,
    flush=False,
    ):
    """Convert measured intensity input values to attenuation values.

    Parameters
    ----------
    input_data : ndarray
        array to process.
    input_meta : list of dict
        List of meta data dictionaries matching input_data.
    conf : dict
        A dictionary of configuration options.
    save_path : str
        Dummy argument
    flush : bool, optional
        Dummy argument

    Returns
    -------
    output : tuple of ndarray and dict
        Returns a 2-tuple of input_data and input_meta.
    """
    
    detector_rows = conf['detector_rows']
    memmap_data = __plugin_state__['memmap_data']
    flush = __plugin_state__['flush']

    # Write projection data to memory mapped singram file 

    first_proj = conf['app_state']['projs']['first']
    last_proj = conf['app_state']['projs']['last']

    start_row = 0
    end_row = detector_rows
    if 'boundingbox' in conf['app_state']['projs']:
        start_row = conf['app_state']['projs']['boundingbox'][0, 0]
        end_row = conf['app_state']['projs']['boundingbox'][0, 1]

    for proj_idx in xrange(first_proj, last_proj + 1):
        memmap_data[start_row:end_row, :, proj_idx] = \
            input_data[proj_idx - first_proj, start_row:end_row]
            
    if flush == True:
        memmap_data.flush()

    return (input_data, input_meta)


if __name__ == '__main__':
    print 'no unit tests!'
