#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# loadraw - plugin to load input data from a raw projection file
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

"""Simulate load of input data for reconstruction with shape
(total_projs, rows, cols), related angle and progress 
information of shape (total_projs) 
are calculated based on the scanner conf settings."""

from cphct.npycore import allowed_data_types, arange

# Internal plugin state for individual plugin instances

__plugin_state__ = {}


def plugin_init(conf, angles_dtype='float32', progress_dtype='float32'):
    """Plugin init function called once with full configuration upon plugin
    load. Called before any hooks, so it may be useful for global
    preparations of the plugin.
    Any values for use in subsequent hooks can be saved in the plugin-specific
    __plugin_state__ dictionary.

    Check input paths.

    Parameters
    ----------
    conf : dict
        A dictionary of configuration options.
    angles_dtype : str, optional
        Angle file data type
    progress_dtype : str, optional
        Progress file data type
        
    """

    # Define angles

    total_projs = conf['total_turns'] * conf['projs_per_turn']
    angles_per_proj = 360.0 / conf['projs_per_turn']
    __plugin_state__['angles_data'] = arange(0, total_projs,
            dtype=angles_dtype) * angles_per_proj

    # Define progress_per_turn

    # NOTE: Katsevich doesn't actually use 'progress' from input_meta yet,
    #       Ticket: #70

    if 'progress_per_turn' in conf:
        progress_dtype = allowed_data_types[progress_dtype]
        progress_per_proj = conf['progress_per_turn'] \
            / conf['projs_per_turn']
        __plugin_state__['progress_data'] = arange(0, total_projs,
                dtype=progress_dtype) * progress_per_proj


def plugin_exit(conf, angles_dtype='float32', progress_dtype='float32'):
    """Plugin exit function called once with full configuration at the end of
    execution. Called after all hooks are finished, so it may be useful
    for global clean up after the plugin.
    Any memory allocations that need to be released can be handled here.

    Clean up after helper arrays.
    
    Parameters
    ----------
    conf : dict
        A dictionary of configuration options.
    angles_dtype : str, optional
        Dummy argument
    progress_dtype : str, optional
        Dummy argument
    """

    # Close files again

    __plugin_state__.clear()


def load_input(
    input_data,
    input_meta,
    conf,
    angles_dtype='float32',
    progress_dtype='float32',
    ):
    """Load projections with meta data

    Parameters
    ----------
    input_data : ndarray
        Array to load projections into.
    input_meta : list of dict
        List of meta data to fill with dictionaries matching each projection.
    conf : dict
        A dictionary of configuration options.
    angles_dtype : str, optional
        Dummy argument
    progress_dtype : str, optional
        Dummy argument

    Raises
    -------
    ValueError :
        If projections, angles or progress can't be loaded.
    """

    angles_data = __plugin_state__['angles_data']
    progress_data = None
    if 'progress_data' in __plugin_state__:
        progress_data = __plugin_state__['progress_data']

    first_proj = conf['app_state']['projs']['first']
    last_proj = conf['app_state']['projs']['last']
    nr_projs = last_proj - first_proj + 1

    # Reset data (input_data is allready allocated)

    input_data[:nr_projs] = 0.0

    # Generate meta data

    input_meta[:] = []
    for meta_idx in xrange(last_proj - first_proj + 1):
        input_meta.append({})
        proj_idx = first_proj + meta_idx
        input_meta[meta_idx]['angle'] = angles_data[proj_idx]
        input_meta[meta_idx]['filtered'] = False
        if progress_data is not None:
            input_meta[meta_idx]['progress'] = progress_data[proj_idx]

    return (input_data, input_meta)


if __name__ == '__main__':
    print 'no unit tests!'
