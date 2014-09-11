#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# verify - verify plugin to validate input/output against file of known values
# Copyright (C) 2013  The Cph CT Toolbox Project lead by Brian Vinter
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

"""Verify plugin to validate input or output data values against a file"""

from cphct.io import create_path_dir, expand_path
from cphct.log import logging
from cphct.npycore import arange
from cphct.npycore.utils import verify_array

# Internal plugin state for individual plugin instances

__plugin_state__ = {}


def plugin_init(conf, verify_path):
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
    verify_path : str
        Path of file to verify values against.
    Raises
    ------
    OSError
        If provided verify_path cannot be created or truncated.
    """

    abs_verify_path = expand_path(conf['working_directory'], verify_path)
    __plugin_state__['abs_verify_path'] = abs_verify_path

    verify_file = open(abs_verify_path, 'rb')
    __plugin_state__['verify_file'] = verify_file
    __plugin_state__['verify_success'] = True
    __plugin_state__['verify_failed'] = []


def plugin_exit(conf, verify_path):
    """Plugin exit function called once with full configuration at the end of
    execution. Called after all hooks are finished, so it may be useful
    for global clean up after the plugin.
    Any memory allocations that need to be released can be handled here.

    Close open file descriptor.

    Parameters
    ----------
    conf : dict
        A dictionary of configuration options.
    verify_path : str
        Path of file to verify values against.
    """

    __plugin_state__['verify_file'].close()
    if __plugin_state__['verify_success']:
        logging.info('Verify succeeded for all chunks')
    else:
        failed_str = ', '.join([str(i) for i in \
                                __plugin_state__['verify_failed']])
        logging.error('Verify failed for chunks %s' % failed_str)
    __plugin_state__.clear()


def preprocess_input(
    input_data,
    input_meta,
    conf,
    verify_path,
    ):
    """Verify input using values from verify_path. Logs errors and sets
    app_state exit_code on failures.

    Parameters
    ----------
    input_data : ndarray
        array to process.
    input_meta : list of dict
        List of meta data dictionaries matching input_data.
    conf : dict
        A dictionary of configuration options.
    verify_path : str
        Path of file to verify input against.

    Returns
    -------
    output : tuple of ndarray and list
        Returns a tuple of the same data array and meta list.

    Raises
    ------
    ValueError
        If input is not a numpy array or if input did not match verify data
    """

    if not hasattr(input_data, 'dtype'):
        raise ValueError('invalid verify preprocess input array')

    verify_file = __plugin_state__['verify_file']

    chunk_index = conf['app_state']['chunk']['idx']
    if not verify_array(input_data, verify_file, chunk_index):
        logging.error('Verify failed for chunk %d' % chunk_index)
        __plugin_state__['verify_success'] = False
        __plugin_state__['verify_failed'].append(chunk_index)
        conf['app_state']['exit_code'] = 3
        
    return (input_data, input_meta)


def postprocess_output(
    output_data,
    output_meta,
    conf,
    verify_path,
    ):
    """Verify input using values from verify_path. Logs errors and sets
    app_state exit_code on failures.

    Parameters
    ----------
    output_data : ndarray
        array to process.
    output_meta : list of dict
        List of meta data dictionaries matching output_data.
    conf : dict
        A dictionary of configuration options.
    verify_path : str
        Path of file to verify values against.

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
        raise ValueError('invalid verify postprocess input array')

    verify_file = __plugin_state__['verify_file']

    chunk_index = conf['app_state']['chunk']['idx']
    if not verify_array(output_data, verify_file, chunk_index):
        logging.error('Verify failed for chunk %d' % chunk_index)
        __plugin_state__['verify_success'] = False
        __plugin_state__['verify_failed'].append(chunk_index)
        conf['app_state']['exit_code'] = 3

    return (output_data, output_meta)


if __name__ == '__main__':
    import tempfile
    data = arange(3, 8)
    tmp_fd = tempfile.NamedTemporaryFile()
    path = tmp_fd.name
    data.tofile(path)
    print 'Verify data %s to %s' % (data, path)
    out = verify_array(data, path, 0)
    tmp_fd.close()
