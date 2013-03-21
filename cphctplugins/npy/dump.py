#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# dump - dump plugin to save/append data values to a file
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

"""Dump plugin to save input and output data values to a file"""

from cphct.io import create_path_dir, expand_path
from cphct.npycore import arange
from cphct.npycore.utils import dump_array

# Internal plugin state for individual plugin instances

__plugin_state__ = {}


def plugin_init(conf, dump_path):
    """Plugin init function called once with full configuration upon plugin
    load. Called before any hooks, so it may be useful for global
    preparations of the plugin.
    Any values for use in subsequent hooks can be saved in the plugin-specific
    __plugin_state__ dictionary.

    Always truncate file before use and prepare file descriptor.

    Parameters
    ----------
    conf : dict
        A dictionary of configuration options.
    dump_path : str
        Path of file to save/append input array to.
    Raises
    ------
    OSError
        If provided dump_path cannot be created or truncated.
    """

    abs_dump_path = expand_path(conf['working_directory'], dump_path)
    __plugin_state__['abs_dump_path'] = abs_dump_path

    create_path_dir(abs_dump_path)
    dump_file = open(abs_dump_path, 'wb')
    __plugin_state__['dump_file'] = dump_file


def plugin_exit(conf, dump_path):
    """Plugin exit function called once with full configuration at the end of
    execution. Called after all hooks are finished, so it may be useful
    for global clean up after the plugin.
    Any memory allocations that need to be released can be handled here.

    Close open file descriptor.

    Parameters
    ----------
    conf : dict
        A dictionary of configuration options.
    dump_path : str
        Path of file to save/append input array to.
    """

    __plugin_state__['dump_file'].close()
    __plugin_state__.clear()


def preprocess_input(
    input_data,
    input_meta,
    conf,
    dump_path,
    ):
    """Dump input to dump_path using append if file exists

    Parameters
    ----------
    input_data : ndarray
        array to process.
    input_meta : list of dict
        List of meta data dictionaries matching input_data.
    conf : dict
        A dictionary of configuration options.
    dump_path : str
        Path of file to save/append input array to.

    Returns
    -------
    output : tuple of ndarray and list
        Returns a tuple of the same data array and meta list.

    Raises
    ------
    ValueError
        If input is not a numpy array
    """

    if not hasattr(input_data, 'dtype'):
        raise ValueError('invalid dump preprocess input array')

    dump_file = __plugin_state__['dump_file']

    return (dump_array(input_data, dump_file), input_meta)


def postprocess_output(
    output_data,
    output_meta,
    conf,
    dump_path,
    ):
    """Dump input to dump_path using append if file exists

    Parameters
    ----------
    output_data : ndarray
        array to process.
    output_meta : list of dict
        List of meta data dictionaries matching output_data.
    conf : dict
        A dictionary of configuration options.
    dump_path : str
        Path of file to save/append input array to.

    Returns
    -------
    output : tuple of ndarray and dict
        Returns a tuple of the same data array and meta list.

    Raises
    ------
    ValueError
        If input is not a numpy array
    """

    if not hasattr(output_data, 'dtype'):
        raise ValueError('invalid dump postprocess input array')

    dump_file = __plugin_state__['dump_file']

    return (dump_array(output_data, dump_file), output_meta)


if __name__ == '__main__':
    import tempfile
    data = arange(3, 8)
    tmp_fd = tempfile.NamedTemporaryFile()
    path = tmp_fd.name
    print 'Dump data %s to %s' % (data, path)
    out = dump_array(data, path)
    tmp_fd.close()
