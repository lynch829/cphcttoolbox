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

"""Loads input data for reconstruction from a raw projection file 
with shape (total_projs, rows, cols), related angle and progress 
information of shape (total_projs) are optionally loaded. 
If angle and/or progress files are omitted these values
are calculated based on the scanner conf settings."""

from cphct.io import expand_path
from cphct.log import logging
from cphct.npycore import allowed_data_types, memmap, arange
from cphct.npycore.io import load_projs

# Internal plugin state for individual plugin instances

__plugin_state__ = {}


def plugin_init(
    conf,
    projs_filepath,
    projs_dtype='float32',
    prefiltered=False,
    angles_filepath=None,
    angles_dtype='float32',
    progress_filepath=None,
    progress_dtype='float32',
    ):
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
    projs_filepath : str
        Path to raw projection file
    projs_dtype : str, optional
        Projection file data type
    prefiltered : str formatted bool
        Optional marker to point out that loaded projections are already on
        prefiltered form. The projection meta data will include this
        information so that the prefiltering step is skipped later.
    angles_filepath : str, optional
        Path to file with scan angles
    angles_dtype : str, optional
        Angle file data type
    progress_filepath : str, optional
        Path to file with scan progress (in cm)
    progress_dtype : str, optional
        Progress file data type
        
    Raises
    ------
    IOError
        If *projs_filepath* does not point to a file,
        if *angles_filepath* is set but does not point to a file,
        or if *progress_filepath* is set but does not point to a file.
    """

    # Parse prefiltered *string* value to a boolean

    if str(prefiltered).lower() in ('1', 't', 'true', 'y', 'yes'):
        __plugin_state__['prefiltered_projs'] = True
    elif str(prefiltered).lower() in (
        '',
        '0',
        'f',
        'false',
        'n',
        'no',
        ):
        __plugin_state__['prefiltered_projs'] = False
    else:
        logging.warning('unexpected "prefiltered" argument "%s"!')
        __plugin_state__['prefiltered_projs'] = False

    # Define data type of projections

    projs_dtype = allowed_data_types[projs_dtype]

    __plugin_state__['projs_dtype'] = projs_dtype

    # Define itemsize of projections

    projs_itemsize = projs_dtype(0.0).itemsize

    __plugin_state__['projs_itemsize'] = projs_itemsize

    # Define projs_shape

    projs_shape = (conf['detector_rows'], conf['detector_columns'])

    __plugin_state__['projs_shape'] = projs_shape

    # Define projs_items

    projs_items = conf['detector_rows'] * conf['detector_columns']

    __plugin_state__['projs_items'] = projs_items

    # Open projs data file

    abs_projs_filepath = expand_path(conf['working_directory'],
            projs_filepath)

    projs_file = open(abs_projs_filepath, mode='rb')

    __plugin_state__['projs_file'] = projs_file

    # If angles_filpath is defined create memory mapping,
    # otherwise define angles from conf settings

    total_projs = conf['total_turns'] * conf['projs_per_turn']
    angles_dtype = allowed_data_types[angles_dtype]
    if angles_filepath is None:
        angles_per_proj = 360.0 / conf['projs_per_turn']
        __plugin_state__['angles_data'] = arange(0, total_projs,
                dtype=angles_dtype) * angles_per_proj
    else:
        abs_angles_filepath = expand_path(conf['working_directory'],
                angles_filepath)

        angles_file = open(abs_angles_filepath, mode='rb')

        __plugin_state__['angles_file'] = angles_file
        __plugin_state__['angles_data'] = memmap(angles_file,
                dtype=angles_dtype, mode='r')

    # If progress_filpath is defined create memory mapping,
    # otherwise define progress from conf settings

    # NOTE: Katsevich doesn't actually use 'progress' from input_meta yet,
    #       Ticket: #70

    progress_dtype = allowed_data_types[progress_dtype]
    if progress_filepath is None:
        if 'progress_per_turn' in conf:
            progress_per_proj = conf['progress_per_turn'] \
                / conf['projs_per_turn']
            __plugin_state__['progress_data'] = arange(0, total_projs,
                    dtype=progress_dtype) * progress_per_proj
    else:
        abs_progress_filepath = expand_path(conf['working_directory'],
                progress_filepath)

        progress_file = open(abs_progress_filepath, mode='rb')

        __plugin_state__['progress_file'] = progress_file
        __plugin_state__['progress_data'] = memmap(progress_file,
                dtype=progress_dtype, mode='r')


def plugin_exit(
    conf,
    projs_filepath,
    projs_dtype='float32',
    prefiltered=False,
    angles_filepath=None,
    angles_dtype='float32',
    progress_filepath=None,
    progress_dtype='float32',
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
    projs_filepath : str
        Dummy argument
    projs_dtype : str
        Dummy argument
    prefiltered : str formatted bool
        Dummy argument
    angles_filepath : str, optional
        Dummy argument
    angles_dtype : str, optional
        Dummy argument
    progress_filepath : str, optional
        Dummy argument
    progress_dtype : str, optional
        Dummy argument
    """

    # Close files again

    __plugin_state__['projs_file'].close()

    if 'angles_file' in __plugin_state__:
        __plugin_state__['angles_file'].close()

    if 'progress_file' in __plugin_state__:
        __plugin_state__['progress_file'].close()

    __plugin_state__.clear()


def load_input(
    input_data,
    input_meta,
    conf,
    projs_filepath,
    projs_dtype='float32',
    prefiltered=False,
    angles_filepath=None,
    angles_dtype='float32',
    progress_filepath=None,
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
    projs_filepath : str
        Dummy argument
    projs_dtype : str
        Dummy argument
    prefiltered : str formatted bool
        Dummy argument
    angles_filepath : str, optional
        Dummy argument
    angles_dtype : str, optional
        Dummy argument
    progress_filepath : str, optional
        Dummy argument
    progress_dtype : str, optional
        Dummy argument

    Raises
    -------
    ValueError :
        If projections, angles or progress can't be loaded.
    """

    projs_dtype = __plugin_state__['projs_dtype']
    projs_shape = __plugin_state__['projs_shape']
    projs_itemsize = __plugin_state__['projs_itemsize']
    projs_items = __plugin_state__['projs_items']
    projs_file = __plugin_state__['projs_file']
    prefiltered_projs = __plugin_state__['prefiltered_projs']
    angles_data = __plugin_state__['angles_data']
    progress_data = None
    if 'progress_data' in __plugin_state__:
        progress_data = __plugin_state__['progress_data']

    first_proj = conf['app_state']['projs']['first']
    last_proj = conf['app_state']['projs']['last']

    # Fetch data

    filepos = first_proj * projs_items * projs_itemsize
    projs_file.seek(filepos)

    nr_projs = last_proj - first_proj + 1
    load_shape = (nr_projs, projs_shape[0], projs_shape[1])
    load_items = nr_projs * projs_items

    input_data[:nr_projs] = load_projs(projs_file, load_shape,
            projs_dtype, projs_dtype, items=load_items)

    # Generate meta data

    input_meta[:] = []
    for meta_idx in xrange(last_proj - first_proj + 1):
        input_meta.append({})
        proj_idx = first_proj + meta_idx
        input_meta[meta_idx]['angle'] = angles_data[proj_idx]
        input_meta[meta_idx]['filtered'] = prefiltered_projs
        if progress_data is not None:
            input_meta[meta_idx]['progress'] = progress_data[proj_idx]

    return (input_data, input_meta)


if __name__ == '__main__':
    print 'no unit tests!'
